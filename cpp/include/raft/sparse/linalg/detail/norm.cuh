/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cusparse_v2.h>
#include <raft/common/nvtx.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/norm_types.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cuda_runtime.h>
#include <stdio.h>

#include <iostream>
#include <limits>

#include <raft/sparse/detail/utils.h>

namespace raft {
namespace sparse {
namespace linalg {
namespace detail {

template <int TPB_X = 64, typename T>
__global__ void csr_row_normalize_l1_kernel(
  // @TODO: This can be done much more parallel by
  // having threads in a warp compute the sum in parallel
  // over each row and then divide the values in parallel.
  const int* ia,  // csr row ex_scan (sorted by row)
  const T* vals,
  int nnz,  // array of values and number of non-zeros
  int m,    // num rows in csr
  T* result)
{  // output array

  // row-based matrix 1 thread per row
  int row = (blockIdx.x * TPB_X) + threadIdx.x;

  // sum all vals_arr for row and divide each val by sum
  if (row < m) {
    int start_idx = ia[row];
    int stop_idx  = 0;
    if (row < m - 1) {
      stop_idx = ia[row + 1];
    } else
      stop_idx = nnz;

    T sum = T(0.0);
    for (int j = start_idx; j < stop_idx; j++) {
      sum = sum + fabs(vals[j]);
    }

    for (int j = start_idx; j < stop_idx; j++) {
      if (sum != 0.0) {
        T val     = vals[j];
        result[j] = val / sum;
      } else {
        result[j] = 0.0;
      }
    }
  }
}

/**
 * @brief Perform L1 normalization on the rows of a given CSR-formatted sparse matrix
 *
 * @param ia: row_ind array
 * @param vals: data array
 * @param nnz: size of data array
 * @param m: size of row_ind array
 * @param result: l1 normalized data array
 * @param stream: cuda stream to use
 */
template <int TPB_X = 64, typename T>
void csr_row_normalize_l1(const int* ia,  // csr row ex_scan (sorted by row)
                          const T* vals,
                          int nnz,  // array of values and number of non-zeros
                          int m,    // num rows in csr
                          T* result,
                          cudaStream_t stream)
{  // output array

  dim3 grid(raft::ceildiv(m, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  csr_row_normalize_l1_kernel<TPB_X, T><<<grid, blk, 0, stream>>>(ia, vals, nnz, m, result);
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <int TPB_X = 64, typename T>
__global__ void csr_row_normalize_max_kernel(
  // @TODO: This can be done much more parallel by
  // having threads in a warp compute the sum in parallel
  // over each row and then divide the values in parallel.
  const int* ia,  // csr row ind array (sorted by row)
  const T* vals,
  int nnz,  // array of values and number of non-zeros
  int m,    // num total rows in csr
  T* result)
{  // output array

  // row-based matrix 1 thread per row
  int row = (blockIdx.x * TPB_X) + threadIdx.x;

  // find max across columns and divide
  if (row < m) {
    int start_idx = ia[row];
    int stop_idx  = 0;
    if (row < m - 1) {
      stop_idx = ia[row + 1];
    } else
      stop_idx = nnz;

    T max = std::numeric_limits<float>::min();
    for (int j = start_idx; j < stop_idx; j++) {
      if (vals[j] > max) max = vals[j];
    }

    // divide nonzeros in current row by max
    for (int j = start_idx; j < stop_idx; j++) {
      if (max != 0.0 && max > std::numeric_limits<float>::min()) {
        T val     = vals[j];
        result[j] = val / max;
      } else {
        result[j] = 0.0;
      }
    }
  }
}

/**
 * @brief Perform L_inf normalization on a given CSR-formatted sparse matrix
 *
 * @param ia: row_ind array
 * @param vals: data array
 * @param nnz: size of data array
 * @param m: size of row_ind array
 * @param result: l1 normalized data array
 * @param stream: cuda stream to use
 */

template <int TPB_X = 64, typename T>
void csr_row_normalize_max(const int* ia,  // csr row ind array (sorted by row)
                           const T* vals,
                           int nnz,  // array of values and number of non-zeros
                           int m,    // num total rows in csr
                           T* result,
                           cudaStream_t stream)
{
  dim3 grid(raft::ceildiv(m, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  csr_row_normalize_max_kernel<TPB_X, T><<<grid, blk, 0, stream>>>(ia, vals, nnz, m, result);
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <int warpSize, int rpb>
struct CsrReductionPolicy {
  static constexpr int LogicalWarpSize = warpSize;
  static constexpr int RowsPerBlock    = rpb;
  static constexpr int ThreadsPerBlock = LogicalWarpSize * RowsPerBlock;
};

template <typename Policy,
          typename Type,
          typename IdxType,
          typename MainLambda,
          typename ReduceLambda,
          typename FinalLambda>
__global__ void __launch_bounds__(Policy::ThreadsPerBlock)
  csrReductionKernel(Type* norm,
                     const IdxType* ia,
                     const Type* data,
                     IdxType N,
                     Type init,
                     MainLambda main_op,
                     ReduceLambda reduce_op,
                     FinalLambda final_op)
{
  IdxType i = threadIdx.y + (Policy::RowsPerBlock * static_cast<IdxType>(blockIdx.x));
  if (i >= N) return;

  Type acc = init;
  for (IdxType j = ia[i] + threadIdx.x; j < ia[i + 1]; j += Policy::LogicalWarpSize) {
    acc = reduce_op(acc, main_op(data[j]));
  }
  acc = raft::logicalWarpReduce<Policy::LogicalWarpSize>(acc, reduce_op);
  if (threadIdx.x == 0) { norm[i] = final_op(acc); }
}

template <typename Policy,
          typename Type,
          typename IdxType      = int,
          typename MainLambda   = raft::identity_op,
          typename ReduceLambda = raft::add_op,
          typename FinalLambda  = raft::identity_op>
void csrReduction(Type* norm,
                  const IdxType* ia,
                  const Type* data,
                  IdxType N,
                  Type init,
                  cudaStream_t stream,
                  MainLambda main_op     = raft::identity_op(),
                  ReduceLambda reduce_op = raft::add_op(),
                  FinalLambda final_op   = raft::identity_op())
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "csrReduction<%d,%d>", Policy::LogicalWarpSize, Policy::RowsPerBlock);
  dim3 threads(Policy::LogicalWarpSize, Policy::RowsPerBlock, 1);
  dim3 blocks(ceildiv<IdxType>(N, Policy::RowsPerBlock), 1, 1);
  csrReductionKernel<Policy>
    <<<blocks, threads, 0, stream>>>(norm, ia, data, N, init, main_op, reduce_op, final_op);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename Type, typename IdxType, typename Lambda>
void rowNormCsrCaller(Type* norm,
                      const IdxType* ia,
                      const Type* data,
                      IdxType nnz,
                      IdxType N,
                      raft::linalg::NormType type,
                      cudaStream_t stream,
                      Lambda fin_op)
{
  // TODO: dispatch nnz to Policy?
  switch (type) {
    case raft::linalg::NormType::L1Norm:
      csrReduction<CsrReductionPolicy<32, 4>>(
        norm, ia, data, N, (Type)0, stream, raft::abs_op(), raft::add_op(), fin_op);
      break;
    case raft::linalg::NormType::L2Norm:
      csrReduction<CsrReductionPolicy<32, 4>>(
        norm, ia, data, N, (Type)0, stream, raft::sq_op(), raft::add_op(), fin_op);
      break;
    case raft::linalg::NormType::LinfNorm:
      csrReduction<CsrReductionPolicy<32, 4>>(
        norm, ia, data, N, (Type)0, stream, raft::abs_op(), raft::max_op(), fin_op);
      break;
    default: THROW("Unsupported norm type: %d", type);
  };
}

};  // end NAMESPACE detail
};  // end NAMESPACE linalg
};  // end NAMESPACE sparse
};  // end NAMESPACE raft