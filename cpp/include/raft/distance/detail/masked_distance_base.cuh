/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
 * @brief Device class for masked nearest neighbor computations.
 *
 * @tparam useNorms       whether norms are needed
 * @tparam DataT          input data-type (for x and y matrices)
 * @tparam AccT           accumulation data-type
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
 * @tparam rowEpilogueLambda epilog lambda that executes when a full row has
 * been processed.
 *
 * @param[in] x input matrix
 * @param[in] y input matrix
 * @param[in] m number of rows of x
 * @param[in] n number of columns of y
 * @param[in] k number of cols of x and y
 * @param[in] lda leading dimension of x
 * @param[in] ldb leading dimension of y
 * @param[in] ldd parameter to keep Contractions_NT happy..
 * @param[in] xn row norms of input matrix A. Required for expanded L2, cosine
 * @param[in] yn row norms of input matrix B. Required for expanded L2, cosine
 * @param[in]  adj           An adjacency matrix encoded as a bitfield indicating for each
 *                           row of `x` and each group in `y` whether to compute the
 *                           distance. Dim = `(m / 64) x num_groups`.
 * @param[in]  group_idxs    An array containing the *end* indices of each group
 *                           in `y`. The value of group_idxs[j] indicates the
 *                           start of group j + 1, i.e., it is the inclusive
 *                           scan of the group lengths. The first group is
 *                           always assumed to start at index 0 and the last
 *                           group typically ends at index `n`. Length =
 *                           `num_groups`.
 * @param[in] num_groups     The number of groups in group_idxs.
 * @param[in] smem shared mem buffer for intermediate storage of x, y, xn & yn.
 * @param core_op the core accumulation operation lambda
 * @param epilog_op the epilog operation lambda
 * @param fin_op the final gemm epilogue lambda
 * @param rowEpilog_op epilog lambda that executes when a full row has been processed.
 */
template <bool useNorms,
          typename DataT,
          typename AccT,
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
        const uint64_t block_adj = get_block_adjacency(adj, tile_idx_m, idx_g);
        // block_adj is a bitfield that contains a 1 if a row is adjacent to the
        // current group. All zero means we can skip this group.
        if (block_adj == 0) { continue; }

        // thread_adj is a bitfield that contains a 1 at location i iff we must
        // compute row i of acc (the accumulator register tile). That is,
        // for i = 0,.., AccRowsPerTh and j = 0,.., AccColsPerTh:
        //
        //   ((1 << i) & thread_adj) > 0 <=> acc[i][j] must be computed.
        //
        // We precompute this information because it is used in various
        // locations to skip thread-local computations, specifically:
        //
        // 1. To skip computations if thread_adj == 0, i.e., none of the values
        //    of `acc` have to be computed.
        //
        // 2. In epilog_op, to consider only values of `acc` to be reduced that
        //    are not masked of.
        //
        // Note 1: Even when the computation can be skipped for a specific thread,
        // the thread still participates in synchronization operations.
        //
        // Note 2: In theory, it should be possible to skip computations for
        // specific rows of `acc`. In practice, however, this does not improve
        // performance.
        int thread_adj = compute_thread_adjacency(block_adj);

        auto tile_idx_n        = idx_g == 0 ? 0 : group_idxs[idx_g - 1];
        const auto group_end_n = group_idxs[idx_g];
        for (; tile_idx_n < group_end_n; tile_idx_n += P::Nblk) {
          // We provide group_end_n to limit the number of unnecessary data
          // points that are loaded from y.
          this->ldgXY(tile_idx_m, tile_idx_n, 0, group_end_n);

          reset_accumulator();
          this->stsXY();
          __syncthreads();
          this->switch_write_buffer();

          for (int kidx = P::Kblk; kidx < this->k; kidx += P::Kblk) {
            this->ldgXY(tile_idx_m, tile_idx_n, kidx, group_end_n);
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
          // The pre-condition for the loop over tile_idx_n is that write_buffer
          // and read_buffer point to the same buffer. This flips read_buffer
          // back so that it satisfies the pre-condition of this loop.
          this->switch_read_buffer();

          if (useNorms) {
            DataT regxn[P::AccRowsPerTh], regyn[P::AccColsPerTh];
            load_norms(tile_idx_m, tile_idx_n, group_end_n, regxn, regyn);
            if (thread_adj != 0) {
              epilog_op(acc, thread_adj, regxn, regyn, tile_idx_n, tile_idx_m, group_end_n);
            }
          } else {
            if (thread_adj != 0) {
              epilog_op(acc, thread_adj, nullptr, nullptr, tile_idx_n, tile_idx_m, group_end_n);
            }
          }
        }  // tile_idx_n
      }    // idx_g
      rowEpilog_op(tile_idx_m);
    }  // tile_idx_m
  }

 private:
  DI uint64_t get_block_adjacency(const uint64_t* adj, IdxT tile_idx_m, IdxT idx_group)
  {
    // A single element of `adj` contains exactly enough bits to indicate which
    // rows in the current tile to skip and which to compute.
    static_assert(P::Mblk == 8 * sizeof(adj[0]),
                  "masked_l2_nn only supports a policy with 64 rows per block.");
    IdxT block_flag_idx = tile_idx_m / P::Mblk;
    // Index into adj at row tile_idx_m / 64 and column idx_group.
    return adj[block_flag_idx * this->num_groups + idx_group];
  }

  DI uint32_t compute_thread_adjacency(const uint64_t block_adj)
  {
    // thread_adj is a bitfield that contains a 1 at location i iff we must
    // compute row i of acc (the accumulator register tile). It is described in
    // more detail in the run() method.
    uint32_t thread_adj = 0;
#pragma unroll
    for (int thread_row_idx = 0; thread_row_idx < P::AccRowsPerTh; ++thread_row_idx) {
      // Index `thread_row_idx` refers to a row of the current threads' register
      // tile `acc`, i.e., acc[i][:]. Index `block_row_idx` refers to the
      // corresponding row of the current block tile in shared memory.
      const int block_row_idx = this->accrowid + thread_row_idx * P::AccThRows;

      // block_row_is_adjacent is true if the current block_row_idx is adjacent
      // to the current group.
      const uint64_t block_mask        = 1ull << block_row_idx;
      const bool block_row_is_adjacent = (block_adj & block_mask) != 0;
      if (block_row_is_adjacent) {
        // If block row is adjacent, write a 1 bit to thread_adj at location
        // `thread_row_idx`.
        const uint32_t thread_mask = 1 << thread_row_idx;
        thread_adj |= thread_mask;
      }
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
                     IdxT end_n,
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
      syNorm[i] = idx < end_n ? yn[idx] : 0;
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

};  // namespace detail
};  // namespace distance
};  // namespace raft
