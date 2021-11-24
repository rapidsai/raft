/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <stdint.h>
#include <cub/cub.cuh>
#include <limits>
#include <raft/cuda_utils.cuh>
#include <raft/distance/pairwise_distance_base.cuh>
#include <raft/linalg/contractions.cuh>

namespace raft {
namespace distance {

#if (ENABLE_MEMCPY_ASYNC == 1)
#include <cuda_pipeline.h>
using namespace nvcuda::experimental;
#endif

template <typename LabelT, typename DataT>
struct KVPMinReduce {
  typedef cub::KeyValuePair<LabelT, DataT> KVP;

  DI KVP operator()(LabelT rit, const KVP& a, const KVP& b) {
    return b.value < a.value ? b : a;
  }

};  // KVPMinReduce

template <typename LabelT, typename DataT>
struct MinAndDistanceReduceOp {
  typedef typename cub::KeyValuePair<LabelT, DataT> KVP;
  DI void operator()(LabelT rid, KVP* out, const KVP& other) {
    if (other.value < out->value) {
      out->key = other.key;
      out->value = other.value;
    }
  }

  DI void init(KVP* out, DataT maxVal) {
    out->key = -1;
    out->value = maxVal;
  }
};

template <typename LabelT, typename DataT>
struct MinReduceOp {
  typedef typename cub::KeyValuePair<LabelT, DataT> KVP;
  DI void operator()(LabelT rid, DataT* out, const KVP& other) {
    if (other.value < *out) {
      *out = other.value;
    }
  }

  DI void init(DataT* out, DataT maxVal) { *out = maxVal; }
};

template <typename DataT, typename OutT, typename IdxT, typename ReduceOpT>
__global__ void initKernel(OutT* min, IdxT m, DataT maxVal, ReduceOpT redOp) {
  auto tid = IdxT(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid < m) {
    redOp.init(min + tid, maxVal);
  }
}

// TODO: specialize this function for MinAndDistanceReduceOp<int, float>
// with atomicCAS of 64 bit which will eliminate mutex and shfls
template <typename P, typename OutT, typename IdxT, typename KVPair,
          typename ReduceOpT>
DI void updateReducedVal(int* mutex, OutT* min, KVPair* val, ReduceOpT red_op,
                         IdxT m, IdxT gridStrideY) {
  const auto lid = threadIdx.x % raft::WarpSize;
  const auto accrowid = threadIdx.x / P::AccThCols;

  // for now have first lane from each warp update a unique output row. This
  // will resolve hang issues with pre-Volta architectures
#pragma unroll
  for (int j = 0; j < (raft::WarpSize / P::AccThCols); j++) {
    if (lid == 0) {
#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
        auto rid = gridStrideY + accrowid + j + i * P::AccThRows;
        if (rid < m) {
          auto value = val[i];
          while (atomicCAS(mutex + rid, 0, 1) == 1)
            ;
          __threadfence();
          red_op(rid, min + rid, value);
          __threadfence();
          atomicCAS(mutex + rid, 1, 0);
        }
      }
    }
    if (j < (raft::WarpSize / P::AccThCols) - 1) {
#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
        auto tmpkey = raft::shfl(val[i].key, (j + 1) * P::AccThCols);
        auto tmpvalue = raft::shfl(val[i].value, (j + 1) * P::AccThCols);
        val[i] = {tmpkey, tmpvalue};
      }
    }
  }
}

template <typename DataT, typename OutT, typename IdxT, bool Sqrt, typename P,
          typename ReduceOpT, typename KVPReduceOpT, typename CoreLambda,
          typename FinalLambda>
__global__ __launch_bounds__(P::Nthreads, 2) void fusedL2NNkernel(
  OutT* min, const DataT* x, const DataT* y, const DataT* xn, const DataT* yn,
  IdxT m, IdxT n, IdxT k, DataT maxVal, int* mutex, ReduceOpT redOp,
  KVPReduceOpT pairRedOp, CoreLambda core_op, FinalLambda fin_op) {
  extern __shared__ char smem[];

  typedef cub::KeyValuePair<IdxT, DataT> KVPair;
  KVPair val[P::AccRowsPerTh];
#pragma unroll
  for (int i = 0; i < P::AccRowsPerTh; ++i) {
    val[i] = {-1, maxVal};
  }

  // epilogue operation lambda for final value calculation
  auto epilog_lambda = [n, pairRedOp, &val, maxVal] __device__(
                         DataT acc[P::AccRowsPerTh][P::AccColsPerTh],
                         DataT * regxn, DataT * regyn, IdxT gridStrideX,
                         IdxT gridStrideY) {
    KVPReduceOpT pairRed_op(pairRedOp);

#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < P::AccColsPerTh; ++j) {
        acc[i][j] = regxn[i] + regyn[j] - (DataT)2.0 * acc[i][j];
      }
    }
    if (Sqrt) {
#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
        for (int j = 0; j < P::AccColsPerTh; ++j) {
          acc[i][j] = raft::mySqrt(acc[i][j]);
        }
      }
    }

    // intra thread reduce
    const auto acccolid = threadIdx.x % P::AccThCols;
    const auto accrowid = threadIdx.x / P::AccThCols;
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < P::AccColsPerTh; ++j) {
        auto tmpkey = acccolid + j * P::AccThCols + gridStrideX;
        KVPair tmp = {tmpkey, acc[i][j]};
        if (tmpkey < n) {
          val[i] =
            pairRed_op(accrowid + i * P::AccThRows + gridStrideY, tmp, val[i]);
        }
      }
    }
  };

  auto rowEpilog_lambda = [m, mutex, min, pairRedOp, redOp, &val,
                           maxVal] __device__(IdxT gridStrideY) {
    KVPReduceOpT pairRed_op(pairRedOp);
    ReduceOpT red_op(redOp);

    const auto accrowid = threadIdx.x / P::AccThCols;
    const auto lid = raft::laneId();

    // reduce
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = P::AccThCols / 2; j > 0; j >>= 1) {
        auto tmpkey = raft::shfl(val[i].key, lid + j);
        auto tmpvalue = raft::shfl(val[i].value, lid + j);
        KVPair tmp = {tmpkey, tmpvalue};
        val[i] =
          pairRed_op(accrowid + i * P::AccThRows + gridStrideY, tmp, val[i]);
      }
    }

    updateReducedVal<P, OutT, IdxT, KVPair, ReduceOpT>(mutex, min, val, red_op,
                                                       m, gridStrideY);

    // reset the val array.
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
      val[i] = {-1, maxVal};
    }
  };

  IdxT lda = k, ldb = k, ldd = n;
  PairwiseDistances<true, DataT, DataT, DataT, IdxT, P, CoreLambda,
                    decltype(epilog_lambda), FinalLambda,
                    decltype(rowEpilog_lambda), true, false>
    obj(x, y, m, n, k, lda, ldb, ldd, xn, yn, nullptr, smem, core_op,
        epilog_lambda, fin_op, rowEpilog_lambda);
  obj.run();
}

template <typename DataT, typename OutT, typename IdxT, int VecLen,
          typename ReduceOpT, typename KVPReduceOpT>
void fusedL2NNImpl(OutT* min, const DataT* x, const DataT* y, const DataT* xn,
                   const DataT* yn, IdxT m, IdxT n, IdxT k, int* workspace,
                   ReduceOpT redOp, KVPReduceOpT pairRedOp, bool sqrt,
                   bool initOutBuffer, cudaStream_t stream) {
  typedef typename linalg::Policy4x4<DataT, VecLen>::Policy P;

  dim3 blk(P::Nthreads);
  auto nblks = raft::ceildiv<int>(m, P::Nthreads);
  constexpr auto maxVal = std::numeric_limits<DataT>::max();
  typedef cub::KeyValuePair<IdxT, DataT> KVPair;

  // Accumulation operation lambda
  auto core_lambda = [] __device__(DataT & acc, DataT & x, DataT & y) {
    acc += x * y;
  };

  CUDA_CHECK(cudaMemsetAsync(workspace, 0, sizeof(int) * m, stream));
  if (initOutBuffer) {
    initKernel<DataT, OutT, IdxT, ReduceOpT>
      <<<nblks, P::Nthreads, 0, stream>>>(min, m, maxVal, redOp);
    CUDA_CHECK(cudaGetLastError());
  }

  auto fin_op = [] __device__(DataT d_val, int g_d_idx) { return d_val; };

  constexpr size_t shmemSize =
    P::SmemSize + ((P::Mblk + P::Nblk) * sizeof(DataT));
  if (sqrt) {
    auto fusedL2NNSqrt =
      fusedL2NNkernel<DataT, OutT, IdxT, true, P, ReduceOpT, KVPReduceOpT,
                      decltype(core_lambda), decltype(fin_op)>;
    dim3 grid = launchConfigGenerator<P>(m, n, shmemSize, fusedL2NNSqrt);

    fusedL2NNSqrt<<<grid, blk, shmemSize, stream>>>(
      min, x, y, xn, yn, m, n, k, maxVal, workspace, redOp, pairRedOp,
      core_lambda, fin_op);
  } else {
    auto fusedL2NN =
      fusedL2NNkernel<DataT, OutT, IdxT, false, P, ReduceOpT, KVPReduceOpT,
                      decltype(core_lambda), decltype(fin_op)>;
    dim3 grid = launchConfigGenerator<P>(m, n, shmemSize, fusedL2NN);
    fusedL2NN<<<grid, blk, shmemSize, stream>>>(min, x, y, xn, yn, m, n, k,
                                                maxVal, workspace, redOp,
                                                pairRedOp, core_lambda, fin_op);
  }

  CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Fused L2 distance and 1-nearest-neighbor computation in a single call.
 *
 * The benefits of such a call are 2-fold: 1) eliminate the need for an
 * intermediate buffer to store the output of gemm 2) reduce the memory read
 * traffic on this intermediate buffer, otherwise needed during the reduction
 * phase for 1-NN.
 *
 * @tparam DataT     data type
 * @tparam OutT      output type to either store 1-NN indices and their minimum
 *                   distances or store only the min distances. Accordingly, one
 *                   has to pass an appropriate `ReduceOpT`
 * @tparam IdxT      indexing arithmetic type
 * @tparam ReduceOpT A struct to perform the final needed reduction operation
 *                   and also to initialize the output array elements with the
 *                   appropriate initial value needed for reduction.
 *
 * @param[out] min           will contain the reduced output (Length = `m`)
 *                           (on device)
 * @param[in]  x             first matrix. Row major. Dim = `m x k`.
 *                           (on device).
 * @param[in]  y             second matrix. Row major. Dim = `n x k`.
 *                           (on device).
 * @param[in]  xn            L2 squared norm of `x`. Length = `m`. (on device).
 * @param[in]  yn            L2 squared norm of `y`. Length = `n`. (on device)
 * @param[in]  m             gemm m
 * @param[in]  n             gemm n
 * @param[in]  k             gemm k
 * @param[in]  workspace     temp workspace. Size = sizeof(int)*m. (on device)
 * @param[in]  redOp         reduction operator in the epilogue
 * @param[in]  sqrt          Whether the output `minDist` should contain L2-sqrt
 * @param[in]  initOutBuffer whether to initialize the output buffer before the
 *                           main kernel launch
 * @param[in]  stream        cuda stream
 */
template <typename DataT, typename OutT, typename IdxT, typename ReduceOpT,
          typename KVPReduceOpT>
void fusedL2NN(OutT* min, const DataT* x, const DataT* y, const DataT* xn,
               const DataT* yn, IdxT m, IdxT n, IdxT k, void* workspace,
               ReduceOpT redOp, KVPReduceOpT pairRedOp, bool sqrt,
               bool initOutBuffer, cudaStream_t stream) {
  size_t bytes = sizeof(DataT) * k;
  if (16 % sizeof(DataT) == 0 && bytes % 16 == 0) {
    fusedL2NNImpl<DataT, OutT, IdxT, 16 / sizeof(DataT), ReduceOpT>(
      min, x, y, xn, yn, m, n, k, (int*)workspace, redOp, pairRedOp, sqrt,
      initOutBuffer, stream);
  } else if (8 % sizeof(DataT) == 0 && bytes % 8 == 0) {
    fusedL2NNImpl<DataT, OutT, IdxT, 8 / sizeof(DataT), ReduceOpT>(
      min, x, y, xn, yn, m, n, k, (int*)workspace, redOp, pairRedOp, sqrt,
      initOutBuffer, stream);
  } else {
    fusedL2NNImpl<DataT, OutT, IdxT, 1, ReduceOpT>(
      min, x, y, xn, yn, m, n, k, (int*)workspace, redOp, pairRedOp, sqrt,
      initOutBuffer, stream);
  }
}

}  // namespace distance
}  // namespace raft
