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
#include <raft/linalg/contractions.cuh>       // raft::linalg::Contractions_NT
#include <raft/util/cuda_dev_essentials.cuh>  // ceildiv
#include <raft/util/cuda_rt_essentials.hpp>   // RAFT_CUDA_TRY

#include <cstddef>  // size_t

namespace raft {
namespace distance {
namespace detail {

/**
 * @brief Device class for L1, L2 and cosine distance metrics.
 * @tparam DataT          input data-type (for A and B matrices)
 * @tparam AccT           accumulation data-type
 * @tparam OutT           output data-type (for C and D matrices)
 * @tparam IdxT           index data-type
 * @tparam Policy         struct which tunes the Contraction kernel
 * @tparam OpT            A distance operation, e.g., cosine_distance_op.
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
 * @param distance_op the distance operation, e.g. cosine_distance_op
 * @param epilog_op the epilog operation lambda
 * @param fin_op the final gemm epilogue lambda
 * @param rowEpilog_op epilog lambda that executes when a full row has been processed
 */

template <typename DataT,
          typename OutT,
          typename IdxT,
          typename Policy,
          typename OpT,
          typename EpilogueLambda,
          typename FinalLambda,
          typename rowEpilogueLambda,
          bool isRowMajor    = true,
          bool writeOut      = true,
          typename BaseClass = raft::linalg::Contractions_NT<DataT, IdxT, Policy, isRowMajor>>
struct PairwiseDistances : public BaseClass {
  // Get accumulation type from distance_op
  using AccT = typename OpT::AccT;

 private:
  typedef Policy P;
  const DataT* xn;
  const DataT* yn;
  const DataT* const yBase;
  OutT* dOutput;
  char* smem;
  OpT distance_op;
  EpilogueLambda epilog_op;
  FinalLambda fin_op;
  rowEpilogueLambda rowEpilog_op;

  const IdxT grid_stride_m;
  const IdxT grid_stride_n;
  const IdxT grid_offset_m;
  const IdxT grid_offset_n;

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
                       OpT _distance_op,
                       EpilogueLambda _epilog_op,
                       FinalLambda _fin_op,
                       rowEpilogueLambda _rowEpilog_op)
    : BaseClass(_x, _y, _m, _n, _k, _lda, _ldb, _ldd, _smem),
      xn(_xn),
      yn(_yn),
      yBase(_y),
      dOutput(_dOutput),
      smem(_smem),
      distance_op(_distance_op),
      epilog_op(_epilog_op),
      fin_op(_fin_op),
      rowEpilog_op(_rowEpilog_op),
      grid_stride_m(P::Mblk * gridDim.y),
      grid_stride_n(P::Nblk * gridDim.x),
      grid_offset_m(P::Mblk * blockIdx.y),
      grid_offset_n(P::Nblk * blockIdx.x)
  {
  }

  DI void run()
  {
    for (auto tile_idx_m = grid_offset_m; tile_idx_m < this->m; tile_idx_m += grid_stride_m) {
      this->ldgXY(tile_idx_m, grid_offset_n, 0);
      for (auto tile_idx_n = grid_offset_n; tile_idx_n < this->n; tile_idx_n += grid_stride_n) {
        // Prolog:
        reset_accumulator();
        this->stsXY();
        __syncthreads();
        this->switch_write_buffer();

        // Main loop:
        for (int kidx = P::Kblk; kidx < this->k; kidx += P::Kblk) {
          this->ldgXY(tile_idx_m, tile_idx_n, kidx);
          // Process all data in shared memory (previous k-block) and
          // accumulate in registers.
          accumulate();
          this->stsXY();
          __syncthreads();
          this->switch_write_buffer();
          this->switch_read_buffer();
        }
        accumulate();  // last iteration
        // The pre-condition for the loop over tile_idx_n is that write_buffer
        // and read_buffer point to the same buffer. This flips read_buffer back
        // so that it satisfies the pre-condition of this loop.
        this->switch_read_buffer();

        // Epilog:
        if (distance_op.use_norms) {
          DataT regxn[P::AccRowsPerTh], regyn[P::AccColsPerTh];
          load_norms(tile_idx_m, tile_idx_n, regxn, regyn);
          // Overlap ldg with epilog computation
          ldgNextGridStride(tile_idx_m, tile_idx_n);
          // Calculate distance_op epilog.
          // Use .template to disambiguate (See:
          // https://en.cppreference.com/w/cpp/language/dependent_name)
          distance_op.template epilog<Policy>(acc, regxn, regyn, tile_idx_n, tile_idx_m);
          // And any possible additional epilogs
          epilog_op(acc, regxn, regyn, tile_idx_n, tile_idx_m);
        } else {
          // Overlap ldg with epilog computation
          ldgNextGridStride(tile_idx_m, tile_idx_n);
          // Calculate distance_op epilog.
          // Use .template to disambiguate (See:
          // https://en.cppreference.com/w/cpp/language/dependent_name)
          distance_op.template epilog<Policy>(acc, nullptr, nullptr, tile_idx_n, tile_idx_m);
          // And any possible additional epilogs
          epilog_op(acc, nullptr, nullptr, tile_idx_n, tile_idx_m);
        }
        if (writeOut) { store_output(tile_idx_m, tile_idx_n); }
      }
      rowEpilog_op(tile_idx_m);
    }
  }

 private:
  DI void ldgNextGridStride(IdxT tile_idx_m, IdxT tile_idx_n)
  {
    // Fetch next grid stride ldg if within range
    const auto next_tile_tile_idx_n = tile_idx_n + grid_stride_n;
    const auto next_tile_tile_idx_m = tile_idx_m + grid_stride_m;
    if ((next_tile_tile_idx_n) < this->n) {
      this->ldgXY(tile_idx_m, next_tile_tile_idx_n, 0);
    } else if ((next_tile_tile_idx_m) < this->m) {
      this->ldgXY(next_tile_tile_idx_m, grid_offset_n, 0);
    }
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

  DI void accumulate_reg_tile(DataT (&reg_x)[P::AccRowsPerTh][P::Veclen],
                              DataT (&reg_y)[P::AccColsPerTh][P::Veclen])
  {
#pragma unroll
    for (int v = 0; v < P::Veclen; ++v) {
#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
        for (int j = 0; j < P::AccColsPerTh; ++j) {
          distance_op.core(acc[i][j], reg_x[i][v], reg_y[j][v]);
        }
      }
    }
  }

  DI void accumulate()
  {
    // We have a separate ldsXY and accumulate_reg_tile outside the loop body,
    // so that these separated calls can be interspersed with preceding and
    // following instructions, thereby hiding latency.
    this->ldsXY(0);

    // If expensive inner loop, do not unroll loop.
    constexpr int num_iterations = P::Kblk / P::Veclen - 1;
    constexpr int unroll_count   = decltype(distance_op)::expensive_inner_loop ? 1 : num_iterations;
#pragma unroll unroll_count
    for (int ki = P::Veclen; ki < P::Kblk; ki += P::Veclen) {
      accumulate_reg_tile(this->regx, this->regy);
      this->ldsXY(ki);
    }

    // Accumulate last loaded tile.
    accumulate_reg_tile(this->regx, this->regy);
  }

  DI void load_norms(IdxT tile_idx_m,
                     IdxT tile_idx_n,
                     DataT (&regxn)[P::AccRowsPerTh],
                     DataT (&regyn)[P::AccColsPerTh])
  {
    DataT* sxNorm = (DataT*)(&smem[P::SmemSize]);
    DataT* syNorm = (&sxNorm[P::Mblk]);

    // Load x & y norms required by this threadblock in shmem buffer
    if (tile_idx_n == blockIdx.x * P::Nblk) {
      for (int i = threadIdx.x; i < P::Mblk; i += P::Nthreads) {
        auto idx  = tile_idx_m + i;
        sxNorm[i] = idx < this->m ? xn[idx] : 0;
      }
    }

    for (int i = threadIdx.x; i < P::Nblk; i += P::Nthreads) {
      auto idx  = tile_idx_n + i;
      syNorm[i] = idx < this->n ? yn[idx] : 0;
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

  DI void store_output(IdxT tile_idx_m, IdxT tile_idx_n)
  {
    IdxT starty = tile_idx_m + this->accrowid;
    IdxT startx = tile_idx_n + this->acccolid;

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
};  // struct PairwiseDistances

template <typename P, typename IdxT, typename T>
dim3 launchConfigGenerator(IdxT m, IdxT n, std::size_t sMemSize, T func)
{
  int devId;
  RAFT_CUDA_TRY(cudaGetDevice(&devId));
  int numSMs;
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId));

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
