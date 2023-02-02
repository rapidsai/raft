/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <raft/core/operators.hpp>
#include <raft/linalg/contractions.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/vectorized.cuh>

#include <cstddef>

namespace raft {
namespace distance {
namespace detail {

/**
 * @brief Device class for L1, L2 and cosine distance metrics.
 * @tparam useNorms       whether norms are needed
 * @tparam DataT          input data-type (for A and B matrices)
 * @tparam AccT           accumulation data-type
 * @tparam OutT           output data-type (for C and D matrices)
 * @tparam IdxT           index data-type
 * @tparam Policy         struct which tunes the Contraction kernel
 * @tparam CoreLambda     tells how to accumulate an x and y into
                          acc. its signature:
    template <typename AccT, typename DataT> void core_lambda(AccT& acc,
      const DataT& x, const DataT& y)
 * @tparam EpilogueLambda applies an elementwise function to compute final
    values. Its signature is:
    template <typename AccT, typename DataT> void epilogue_lambda
    (AccT acc[][], DataT* regxn, DataT* regyn);
 * @tparam FinalLambda the final lambda called on final distance value
 * @param[in] x input matrix
 * @param[in] y input matrix
 * @param[in] m number of rows of A and C/D
 * @param[in] n number of columns of B and C/D
 * @param[in] k number of cols of A and rows of B
 * @param[in] lda leading dimension of A
 * @param[in] ldb leading dimension of B
 * @param[in] ldd leading dimension of C/D
 * @param[in] xn row norms of input matrix A. Required for expanded L2, cosine
 * @param[in] yn row norms of input matrix B. Required for expanded L2, cosine
 * @param[output] pD output matrix
 * @param[in] smem shared mem buffer for intermediate storage of A, B, xn & yn.
 * @param core_op the core accumulation operation lambda
 * @param epilog_op the epilog operation lambda
 * @param fin_op the final gemm epilogue lambda
 */

template <bool useNorms,
          typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT,
          typename Policy,
          typename CoreLambda,
          typename EpilogueLambda,
          typename FinalLambda,
          typename rowEpilogueLambda,
          bool isRowMajor    = true,
          bool writeOut      = true,
          typename BaseClass = raft::linalg::Contractions_NT<DataT, IdxT, Policy, isRowMajor>>
struct PairwiseDistances : public BaseClass {
 private:
  typedef Policy P;
  const DataT* xn;
  const DataT* yn;
  const DataT* const yBase;
  OutT* dOutput;
  char* smem;
  CoreLambda core_op;
  EpilogueLambda epilog_op;
  FinalLambda fin_op;
  rowEpilogueLambda rowEpilog_op;

  AccT acc[P::AccRowsPerTh][P::AccColsPerTh];

 public:
  // Constructor
  DI PairwiseDistances(const DataT* _x,
                       const DataT* _y,
                       IdxT _m,
                       IdxT _n,
                       IdxT _k,
                       IdxT _lda,
                       IdxT _ldb,
                       IdxT _ldd,
                       const DataT* _xn,
                       const DataT* _yn,
                       OutT* _dOutput,
                       char* _smem,
                       CoreLambda _core_op,
                       EpilogueLambda _epilog_op,
                       FinalLambda _fin_op,
                       rowEpilogueLambda _rowEpilog_op)
    : BaseClass(_x, _y, _m, _n, _k, _lda, _ldb, _ldd, _smem),
      xn(_xn),
      yn(_yn),
      yBase(_y),
      dOutput(_dOutput),
      smem(_smem),
      core_op(_core_op),
      epilog_op(_epilog_op),
      fin_op(_fin_op),
      rowEpilog_op(_rowEpilog_op)
  {
  }

  DI void run()
  {
    for (auto gridStrideY = blockIdx.y * P::Mblk; gridStrideY < this->m;
         gridStrideY += P::Mblk * gridDim.y) {
      for (auto gridStrideX = blockIdx.x * P::Nblk; gridStrideX < this->n;
           gridStrideX += P::Nblk * gridDim.x) {
        prolog(gridStrideX, gridStrideY);
        loop();
        epilog(gridStrideX, gridStrideY);
      }
      rowEpilog_op(gridStrideY);
    }
  }

 private:
  DI void updateIndicesY()
  {
    const auto stride = P::Nblk * gridDim.x;
    if (isRowMajor) {
      this->y += stride * this->ldb;
    } else {
      this->y += stride;
    }
    this->yrowid += stride;
  }

  DI void updateIndicesXY()
  {
    const auto stride = P::Mblk * gridDim.y;
    if (isRowMajor) {
      this->x += stride * this->lda;
      this->yrowid = IdxT(blockIdx.x) * P::Nblk + this->srowid;
      this->y      = yBase + this->yrowid * this->ldb;
    } else {
      this->x += stride;
      this->yrowid = IdxT(blockIdx.x) * P::Nblk;
      this->y      = yBase + this->yrowid + this->srowid * this->ldb;
    }
    this->xrowid += stride;
  }

  DI void ldgNextGridStride(IdxT gridStrideX, IdxT gridStrideY)
  {
    // Fetch next grid stride ldg if within range
    if ((gridStrideX + gridDim.x * P::Nblk) < this->n) {
      updateIndicesY();
      this->ldgXY(0);
    } else if ((gridStrideY + gridDim.y * P::Mblk) < this->m) {
      updateIndicesXY();
      this->ldgXY(0);
    }
  }

  DI void prolog(IdxT gridStrideX, IdxT gridStrideY)
  {
    if (gridStrideX == blockIdx.x * P::Nblk) { this->ldgXY(0); }

#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < P::AccColsPerTh; ++j) {
        acc[i][j] = BaseClass::Zero;
      }
    }

    this->stsXY();
    __syncthreads();
    this->pageWr ^= 1;
  }

  DI void loop()
  {
    for (int kidx = P::Kblk; kidx < this->k; kidx += P::Kblk) {
      this->ldgXY(kidx);
      accumulate();  // on the previous k-block
      this->stsXY();
      __syncthreads();
      this->pageWr ^= 1;
      this->pageRd ^= 1;
    }
    accumulate();  // last iteration
    // This is needed for making sure next grid stride of
    // non-norm based metrics uses previously accumulated buffer so
    // it doesn't make shmem dirty until previous iteration
    // is complete.
    this->pageRd ^= 1;
  }

  DI void accumulate()
  {
#pragma unroll
    for (int ki = 0; ki < P::Kblk; ki += P::Veclen) {
      this->ldsXY(ki);
#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
        for (int j = 0; j < P::AccColsPerTh; ++j) {
#pragma unroll
          for (int v = 0; v < P::Veclen; ++v) {
            core_op(acc[i][j], this->regx[i][v], this->regy[j][v]);
          }
        }
      }
    }
  }

  DI void epilog(IdxT gridStrideX, IdxT gridStrideY)
  {
    if (useNorms) {
      DataT* sxNorm = (DataT*)(&smem[P::SmemSize]);
      DataT* syNorm = (&sxNorm[P::Mblk]);

      // Load x & y norms required by this threadblock in shmem buffer
      if (gridStrideX == blockIdx.x * P::Nblk) {
        for (int i = threadIdx.x; i < P::Mblk; i += P::Nthreads) {
          auto idx  = gridStrideY + i;
          sxNorm[i] = idx < this->m ? xn[idx] : 0;
        }
      }

      for (int i = threadIdx.x; i < P::Nblk; i += P::Nthreads) {
        auto idx  = gridStrideX + i;
        syNorm[i] = idx < this->n ? yn[idx] : 0;
      }

      __syncthreads();

      DataT regxn[P::AccRowsPerTh], regyn[P::AccColsPerTh];
#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
        regxn[i] = sxNorm[i * P::AccThRows + (threadIdx.x / P::AccThCols)];
      }
#pragma unroll
      for (int i = 0; i < P::AccColsPerTh; ++i) {
        regyn[i] = syNorm[i * P::AccThCols + (threadIdx.x % P::AccThCols)];
      }

      // Overlap ldg with epilog computation
      ldgNextGridStride(gridStrideX, gridStrideY);
      epilog_op(acc, regxn, regyn, gridStrideX, gridStrideY);
    } else {
      // Overlap ldg with epilog computation
      ldgNextGridStride(gridStrideX, gridStrideY);
      epilog_op(acc, nullptr, nullptr, gridStrideX, gridStrideY);
    }

    if (writeOut) {
      IdxT starty = gridStrideY + this->accrowid;
      IdxT startx = gridStrideX + this->acccolid;

#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
        auto rowId = starty + i * P::AccThRows;
#pragma unroll
        for (int j = 0; j < P::AccColsPerTh; ++j) {
          auto colId = startx + j * P::AccThCols;
          if (rowId < this->m && colId < this->n) {
            // Promote to 64 bit index for final write, as output array can be > 2^31
            dOutput[std::size_t(rowId) * this->n + colId] = fin_op(acc[i][j], 0);
          }
        }
      }
    }
  }
};  // struct PairwiseDistances

/**
 * @brief the distance matrix calculation kernel for L1, L2 and cosine
 * @tparam useNorms       whether norms are needed
 * @tparam DataT          input data-type (for A and B matrices)
 * @tparam AccT           accumulation data-type
 * @tparam OutT           output data-type (for C and D matrices)
 * @tparam IdxT           index data-type
 * @tparam Policy         struct which tunes the Contraction kernel
 * @tparam CoreLambda     lambda which implements accumulation operation
 * @tparam EpilogueLambda lambda which implements operation for calculating
                          final value.
 * @tparam FinalLambda    final lambda called on final distance value
 * @tparam isRowMajor     true if input/output is row major(default),
                          false for column major
 *
 * @param[in]       x input matrix
 * @param[in]       y input matrix
 * @param[in]       xn row norms of input matrix A.
 * @param[in]       yn row norms of input matrix B.
 * @param[in]       m number of rows of A and C/D
 * @param[in]       n number of columns of B and C/D
 * @param[in]       k number of cols of A and rows of B
 * @param[in]       lda leading dimension of A
 * @param[in]       ldb leading dimension of B
 * @param[in]       ldd leading dimension of C/D
 * @param[output]   pD output matrix
 * @param core_op   the core lambda
 * @param epilog_op the epilogue lambda
 * @param fin_op    the final gemm epilogue lambda
 */

template <bool useNorms,
          typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT,
          typename Policy,
          typename CoreLambda,
          typename EpilogueLambda,
          typename FinalLambda,
          bool isRowMajor = true,
          bool writeOut   = true>
__global__ __launch_bounds__(Policy::Nthreads, 2)

  void pairwiseDistanceMatKernel(const DataT* x,
                                 const DataT* y,
                                 const DataT* _xn,
                                 const DataT* _yn,
                                 IdxT m,
                                 IdxT n,
                                 IdxT k,
                                 IdxT lda,
                                 IdxT ldb,
                                 IdxT ldd,
                                 OutT* dOutput,
                                 CoreLambda core_op,
                                 EpilogueLambda epilog_op,
                                 FinalLambda fin_op)
{
  extern __shared__ char smem[];
  auto rowEpilog = [] __device__(IdxT starty) { return; };

  PairwiseDistances<useNorms,
                    DataT,
                    AccT,
                    OutT,
                    IdxT,
                    Policy,
                    CoreLambda,
                    EpilogueLambda,
                    FinalLambda,
                    decltype(rowEpilog),
                    isRowMajor,
                    writeOut>
    obj(
      x, y, m, n, k, lda, ldb, ldd, _xn, _yn, dOutput, smem, core_op, epilog_op, fin_op, rowEpilog);
  obj.run();
}

/**
 * @brief the distance matrix calculation kernel for L2 and cosine
 * for GPU arch < SM 8.0, this version is to make sure we don't recompile
 * these kernels for ampere or higher as we use cutlass kernel for it.
 * @tparam useNorms       whether norms are needed
 * @tparam DataT          input data-type (for A and B matrices)
 * @tparam AccT           accumulation data-type
 * @tparam OutT           output data-type (for C and D matrices)
 * @tparam IdxT           index data-type
 * @tparam Policy         struct which tunes the Contraction kernel
 * @tparam CoreLambda     lambda which implements accumulation operation
 * @tparam EpilogueLambda lambda which implements operation for calculating
                          final value.
 * @tparam FinalLambda    final lambda called on final distance value
 * @tparam isRowMajor     true if input/output is row major(default),
                          false for column major
 *
 * @param[in]       x input matrix
 * @param[in]       y input matrix
 * @param[in]       xn row norms of input matrix A.
 * @param[in]       yn row norms of input matrix B.
 * @param[in]       m number of rows of A and C/D
 * @param[in]       n number of columns of B and C/D
 * @param[in]       k number of cols of A and rows of B
 * @param[in]       lda leading dimension of A
 * @param[in]       ldb leading dimension of B
 * @param[in]       ldd leading dimension of C/D
 * @param[output]   pD output matrix
 * @param core_op   the core lambda
 * @param epilog_op the epilogue lambda
 * @param fin_op    the final gemm epilogue lambda
 */

template <bool useNorms,
          typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT,
          typename Policy,
          typename CoreLambda,
          typename EpilogueLambda,
          typename FinalLambda,
          bool isRowMajor = true,
          bool writeOut   = true>
__global__ __launch_bounds__(Policy::Nthreads, 2)

  void pairwiseDistanceMatKernelPriorToAmpere(const DataT* x,
                                              const DataT* y,
                                              const DataT* _xn,
                                              const DataT* _yn,
                                              IdxT m,
                                              IdxT n,
                                              IdxT k,
                                              IdxT lda,
                                              IdxT ldb,
                                              IdxT ldd,
                                              OutT* dOutput,
                                              CoreLambda core_op,
                                              EpilogueLambda epilog_op,
                                              FinalLambda fin_op)
{
  //#if __CUDA_ARCH__ < 800
  // TODO: re-enable the CUDA_ARCH guard for below Ampere once cutlass based
  //  kernels are enabled for CUDA 12.0
  extern __shared__ char smem[];
  auto rowEpilog = [] __device__(IdxT starty) { return; };

  PairwiseDistances<useNorms,
                    DataT,
                    AccT,
                    OutT,
                    IdxT,
                    Policy,
                    CoreLambda,
                    EpilogueLambda,
                    FinalLambda,
                    decltype(rowEpilog),
                    isRowMajor,
                    writeOut>
    obj(
      x, y, m, n, k, lda, ldb, ldd, _xn, _yn, dOutput, smem, core_op, epilog_op, fin_op, rowEpilog);
  obj.run();
  //#endif
}

template <typename P, typename IdxT, typename T>
dim3 launchConfigGenerator(IdxT m, IdxT n, std::size_t sMemSize, T func)
{
  const auto numSMs  = raft::getMultiProcessorCount();
  int numBlocksPerSm = 0;
  dim3 grid;

  RAFT_CUDA_TRY(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, func, P::Nthreads, sMemSize));
  std::size_t minGridSize = numSMs * numBlocksPerSm;
  std::size_t yChunks     = raft::ceildiv<int>(m, P::Mblk);
  std::size_t xChunks     = raft::ceildiv<int>(n, P::Nblk);
  grid.y                  = yChunks > minGridSize ? minGridSize : yChunks;
  grid.x                  = (minGridSize - grid.y) <= 0 ? 1 : xChunks;
  if (grid.x != 1) {
    std::size_t i = 1;
    while (grid.y * i < minGridSize) {
      i++;
    }
    grid.x = i >= xChunks ? xChunks : i;
  }

  return grid;
}

};  // namespace detail
};  // namespace distance
};  // namespace raft
