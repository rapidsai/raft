/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <raft/distance/detail/pairwise_distance_base.cuh>
#include <raft/linalg/contractions.cuh>
#include <raft/util/cuda_utils.cuh>

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
          typename BaseClass = raft::linalg::Contractions_NT<DataT, IdxT, Policy, isRowMajor>>
struct MaskedDistances : public BaseClass {
 private:
  typedef Policy P;
  const DataT* xn;
  const DataT* yn;
  const DataT* const yBase;
  const uint64_t* adj;
  const IdxT* group_idxs;
  IdxT num_groups;
  char* smem;
  CoreLambda core_op;
  EpilogueLambda epilog_op;
  FinalLambda fin_op;
  rowEpilogueLambda rowEpilog_op;

  AccT acc[P::AccRowsPerTh][P::AccColsPerTh];

 public:
  // Constructor
  DI MaskedDistances(const DataT* _x,
                     const DataT* _y,
                     IdxT _m,
                     IdxT _n,
                     IdxT _k,
                     IdxT _lda,
                     IdxT _ldb,
                     IdxT _ldd,
                     const DataT* _xn,
                     const DataT* _yn,
                     const uint64_t* _adj,
                     const IdxT* _group_idxs,
                     IdxT _num_groups,
                     char* _smem,
                     CoreLambda _core_op,
                     EpilogueLambda _epilog_op,
                     FinalLambda _fin_op,
                     rowEpilogueLambda _rowEpilog_op)
    : BaseClass(_x, _y, _m, _n, _k, _lda, _ldb, _ldd, _smem),
      xn(_xn),
      yn(_yn),
      yBase(_y),
      adj(_adj),
      group_idxs(_group_idxs),
      num_groups(_num_groups),
      smem(_smem),
      core_op(_core_op),
      epilog_op(_epilog_op),
      fin_op(_fin_op),
      rowEpilog_op(_rowEpilog_op)
  {
  }

  DI void run()
  {
    const auto grid_stride_m = (P::Mblk * gridDim.y);
    const auto grid_offset_m = (P::Mblk * blockIdx.y);

    const auto grid_stride_g = gridDim.x;
    const auto grid_offset_g = blockIdx.x;

    for (auto tile_idx_m = grid_offset_m; tile_idx_m < this->m; tile_idx_m += grid_stride_m) {
      // Start loop over groups
      for (auto idx_g = grid_offset_g; idx_g < this->num_groups; idx_g += grid_stride_g) {
        // The __syncthreads() ensures that loading the block flag occurs at
        // the same time in all threads of the block. Since all threads load
        // the same address, this speeds up the code.
        __syncthreads();
        const uint64_t block_adj = get_block_adjacency(adj, tile_idx_m, idx_g);
        // block_adj is a bitfield that contains a 1 if a row is adjacent to the
        // current group. All zero means we can skip this group.
        if (block_adj == 0) { continue; }

        // Determine which results, that are computed by this thread, have to
        // be taken into account. This information is stored in a bitfield,
        // thread_adj. If all results computed by this thread can be ignored,
        // then we can also skip some computations (thread_adj == 0).

        // We precompute this information because it is used in various
        // locations to skip thread-local computations.
        int thread_adj = compute_thread_adjacency(block_adj);

        auto tile_idx_n       = idx_g == 0 ? 0 : group_idxs[idx_g - 1];
        const auto tile_end_n = group_idxs[idx_g];
        for (; tile_idx_n < tile_end_n; tile_idx_n += P::Nblk) {
          // We provide tile_end_n to limit the number of unnecessary data
          // points that are loaded from y.
          this->ldgXY(tile_idx_m, tile_idx_n, 0, tile_end_n);

          reset_accumulator();
          this->stsXY();
          __syncthreads();
          this->switch_write_buffer();

          for (int kidx = P::Kblk; kidx < this->k; kidx += P::Kblk) {
            this->ldgXY(tile_idx_m, tile_idx_n, kidx, tile_end_n);
            // Process all data in shared memory (previous k-block) and
            // accumulate in registers.
            if (thread_adj != 0) { accumulate(); }
            this->stsXY();
            __syncthreads();
            this->switch_write_buffer();
            this->switch_read_buffer();
          }
          if (thread_adj != 0) {
            accumulate();  // last iteration
          }
          // This is needed for making sure next grid stride of
          // non-norm based metrics uses previously accumulated buffer so
          // it doesn't make shmem dirty until previous iteration
          // is complete.
          this->switch_read_buffer();

          if (useNorms) {
            DataT regxn[P::AccRowsPerTh], regyn[P::AccColsPerTh];
            load_norms(tile_idx_m, tile_idx_n, tile_end_n, regxn, regyn);
            if (thread_adj != 0) {
              epilog_op(acc, thread_adj, regxn, regyn, tile_idx_n, tile_idx_m, tile_end_n);
            }
          } else {
            if (thread_adj != 0) {
              epilog_op(acc, thread_adj, nullptr, nullptr, tile_idx_n, tile_idx_m, tile_end_n);
            }
          }
        }  // tile_idx_n
      }    // idx_g
      rowEpilog_op(tile_idx_m);
    }  // tile_idx_n
  }

 private:
  DI uint64_t get_block_adjacency(const uint64_t* adj, IdxT tile_idx_m, IdxT idx_group)
  {
    IdxT block_flag_idx = tile_idx_m / P::Mblk;
    return adj[block_flag_idx * this->num_groups + idx_group];
  }

  DI uint32_t compute_thread_adjacency(const uint64_t block_adj)
  {
    uint32_t thread_adj = 0;
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
      const uint64_t read_mask  = 1ull << (this->accrowid + i * P::AccThRows);
      const uint32_t write_mask = 1 << i;
      if ((block_adj & read_mask) != 0) { thread_adj |= write_mask; }
    }
    return thread_adj;
  }

  DI void reset_accumulator()
  {
    // Reset accumulator registers to zero.
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < P::AccColsPerTh; ++j) {
        acc[i][j] = BaseClass::Zero;
      }
    }
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

  DI void load_norms(IdxT tile_idx_m,
                     IdxT tile_idx_n,
                     IdxT tile_end_n,
                     DataT (&regxn)[P::AccRowsPerTh],
                     DataT (&regyn)[P::AccColsPerTh])
  {
    DataT* sxNorm = (DataT*)(&smem[P::SmemSize]);
    DataT* syNorm = (&sxNorm[P::Mblk]);

    // Load x & y norms required by this threadblock in shmem buffer
    for (int i = threadIdx.x; i < P::Mblk; i += P::Nthreads) {
      auto idx  = tile_idx_m + i;
      sxNorm[i] = idx < this->m ? xn[idx] : 0;
    }

    for (int i = threadIdx.x; i < P::Nblk; i += P::Nthreads) {
      auto idx  = tile_idx_n + i;
      syNorm[i] = idx < tile_end_n ? yn[idx] : 0;
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
      regxn[i] = sxNorm[i * P::AccThRows + (threadIdx.x / P::AccThCols)];
    }
#pragma unroll
    for (int i = 0; i < P::AccColsPerTh; ++i) {
      regyn[i] = syNorm[i * P::AccThCols + (threadIdx.x % P::AccThCols)];
    }
  }
};  // struct MaskedDistances

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
          bool isRowMajor = true>
__global__ __launch_bounds__(Policy::Nthreads, 2)

  void maskedDistanceMatKernel(const DataT* x,
                               const DataT* y,
                               const DataT* _xn,
                               const DataT* _yn,
                               const bool* adj,
                               IdxT m,
                               IdxT n,
                               IdxT k,
                               IdxT lda,
                               IdxT ldb,
                               IdxT ldd,
                               CoreLambda core_op,
                               EpilogueLambda epilog_op,
                               FinalLambda fin_op)
{
  extern __shared__ char smem[];
  auto rowEpilog = [] __device__(IdxT starty) { return; };

  MaskedDistances<useNorms,
                  DataT,
                  AccT,
                  OutT,
                  IdxT,
                  Policy,
                  CoreLambda,
                  EpilogueLambda,
                  FinalLambda,
                  decltype(rowEpilog),
                  isRowMajor>
    obj(x, y, m, n, k, lda, ldb, ldd, _xn, _yn, smem, core_op, epilog_op, fin_op, rowEpilog);
  obj.run();
}

};  // namespace detail
};  // namespace distance
};  // namespace raft
