/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

/*! \file
  \brief Epilogue for threadblock scoped GEMMs using Tensor Ops.

This file contains a customized version of PredicatedTileIterator from CUTLASS 2.9.0
(https://github.com/NVIDIA/cutlass/blob/v2.9.0/include/cutlass/epilogue/threadblock/predicated_tile_iterator.h#L75)

Changes:
- added `Layout_` template param
- PredicatedTileIteratorParams() is customized to not stride by layout.stride(0).
- makes use of `SharedStorage` to store reduced values across warps to gmem in coalesced manner.
- customized the store_with_byte_offset() to perform reduction per row and write final value to
gmem.
- customized the Params() struct to take user inputs from epilogueOp params.

*/

#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cutlass/arch/arch.h>
#include <cutlass/arch/memory.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/threadblock/output_tile_thread_map.h>
#include <cutlass/epilogue/threadblock/predicated_tile_iterator_params.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/matrix_shape.h>
#include <cutlass/numeric_types.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/transform/pitch_linear_thread_map.h>

namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////

namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Tile iterator used to load and store output tile from global memory in epilogue.
///
/// Satisfies: ReadableTileIterator | PredicatedTileIterator | ForwardTileIterator
///
template <typename ThreadMap_,  ///< Thread map (conept: OutputTileThreadMap)
          typename Element_,    ///< Element data type
          typename Layout_,
          typename EpilogueOpParams_,
          bool ScatterD     = false,  ///< Scatter D operand or not
          bool UseCUDAStore = false>
class PredicatedTileIteratorReducedVec {
 public:
  using ThreadMap = ThreadMap_;
  using Shape     = typename ThreadMap::Shape;

  using Element = Element_;

  using Layout         = Layout_;
  using TensorRef      = TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index            = typename Layout::Index;
  using LongIndex        = typename Layout::LongIndex;
  using TensorCoord      = MatrixCoord;
  using EpilogueOpParams = EpilogueOpParams_;
  using OutIdxT          = typename EpilogueOpParams::CGReduceT::IndexT;
  using OutValT          = typename EpilogueOpParams::CGReduceT::AccTypeT;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kThreads           = ThreadMap::kThreads;
  static int const kIterations        = ThreadMap::Count::kTile;

  static_assert(ThreadMap::Iterations::kRow > 0, "ThreadMap::Iterations::kRow must be > 0");
  static_assert(ThreadMap::Iterations::kGroup > 0, "ThreadMap::Iterations::kGroup must be > 0");
  static_assert(ThreadMap::Iterations::kCluster > 0, "ThreadMap::Iterations::kCluster must be > 0");
  static_assert(ThreadMap::Iterations::kColumn > 0, "ThreadMap::Iterations::kColumn must be > 0");
  static_assert(!UseCUDAStore, "UseCUDAStore path is not supported");

  static int const total_rows = ThreadMap::kWarpCount * ThreadMap::Iterations::kRow *
                                ThreadMap::Iterations::kGroup * ThreadMap::Iterations::kCluster *
                                ThreadMap::Count::kTile * ThreadMap::Delta::kRow;
  /// Fragment object
  using Fragment =
    Array<OutValT,
          ThreadMap::Iterations::kColumn * ThreadMap::Iterations::kRow *
            ThreadMap::Iterations::kGroup * ThreadMap::Iterations::kCluster * kElementsPerAccess>;

  // Memory access size
  using AccessType     = AlignedArray<Element, kElementsPerAccess>;
  using AccessTypeValT = AlignedArray<OutValT, kElementsPerAccess>;

  //
  // Parameters struct
  //

  /// Uses a non-template class
  struct Params : PredicatedTileIteratorParams {
    using Base = PredicatedTileIteratorParams;

    EpilogueOpParams user_param;
    CUTLASS_HOST_DEVICE
    Params() {}

    CUTLASS_HOST_DEVICE
    Params(Layout const& layout)
      : PredicatedTileIteratorParams(
          layout.stride(0) * int(sizeof(AccessType)) / kElementsPerAccess,
          make_OutputTileThreadMapDesc<ThreadMap>())
    {
    }

    CUTLASS_HOST_DEVICE
    Params(Layout const& layout, EpilogueOpParams const& user_param_)
      : PredicatedTileIteratorParams(int(sizeof(AccessType)) / kElementsPerAccess,
                                     make_OutputTileThreadMapDesc<ThreadMap>()),
        user_param(user_param_)
    {
    }

    CUTLASS_HOST_DEVICE
    Params(Base const& base) : Base(base) {}
  };

  /// Mask object
  struct Mask {
    // static int const kCount = ThreadMap::Iterations::kColumn;
    static int const kCount = ThreadMap::Iterations::kColumn * kElementsPerAccess;

    /// Predicate state
    bool predicates[kCount];

    //
    // Mask
    //
    CUTLASS_HOST_DEVICE
    Mask() { enable(); }

    ///< Efficiently disables all accesses guarded by mask
    CUTLASS_HOST_DEVICE void clear()
    {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        predicates[i] = false;
      }
    }

    ///< CUTLASS_HOST_DEVICE enables all accesses guarded by mask
    CUTLASS_DEVICE void enable()
    {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        predicates[i] = true;
      }
    }
  };

  /// Shared storage allocation needed by the predicated tile
  //  iterator for reduction.
  struct SharedStorage {
    //
    // Type definitions
    //
    using Shape = MatrixShape<total_rows, 1>;

    /// Shape of the shared memory allocation for the reduced values store
    using StorageShape = MatrixShape<Shape::kRow, Shape::kColumn>;

    //
    // Data members

    //
    // Methods
    //
    AlignedBuffer<Element, StorageShape::kCount> storage;

    CUTLASS_DEVICE
    Element* data() { return storage.data(); }

    SharedStorage() {}

    CUTLASS_DEVICE
    void initSmem(EpilogueOpParams const& user_params)
    {
      Element* shared_elem_arr = data();
      constexpr auto maxVal    = std::numeric_limits<OutValT>::max();

      for (int row = threadIdx.x; row < total_rows; row += blockDim.x) {
        user_params.red_op_.init(&shared_elem_arr[row], maxVal);
      }
    }
  };

  template <typename cg_reduce_op_t,
            typename cg_group_t,
            typename IdxT,
            typename ValT,
            typename OutT>
  struct select_reduce {
    /// Performs warp level reduction and stores a reduced output to memory
    CUTLASS_DEVICE
    select_reduce(OutT value,
                  ValT prev_red_val,
                  cg_reduce_op_t reduce_op,
                  cg_group_t cg_warp_group,
                  OutT& shmem_ptr)
    {
      if (cg_warp_group.any(reduce_op.isAmin(value, prev_red_val))) {
        OutT reduced_val = cg::reduce(cg_warp_group, value, reduce_op);
        if (cg_warp_group.thread_rank() == 0) { shmem_ptr = reduced_val; }
      }
    }
  };

  template <typename cg_reduce_op_t, typename cg_group_t, typename IdxT>
  struct select_reduce<cg_reduce_op_t, cg_group_t, IdxT, float, raft::KeyValuePair<IdxT, float>> {
    using ValT = float;
    using Ty   = raft::KeyValuePair<IdxT, ValT>;
    /// Performs warp level reduction of key value pair and stores a reduced output to memory
    CUTLASS_DEVICE
    select_reduce(Ty val_to_red,
                  float prev_red_val,
                  cg_reduce_op_t cg_reduce_op,
                  cg_group_t cg_warp_group,
                  Ty& shmem_ptr)
    {
      ValT val = val_to_red.value;

      if (cg_warp_group.any(cg_reduce_op.isAmin(val, prev_red_val))) {
        ValT reduced_val = cg::reduce(cg_warp_group, val, cg_reduce_op);
        bool pred        = (reduced_val == val);
        auto subTile     = cg::binary_partition(cg_warp_group, pred);
        if (pred) {
          if (subTile.thread_rank() == 0) { shmem_ptr = val_to_red; }
        }
      }
    }
  };

  template <typename cg_reduce_op_t, typename cg_group_t, typename IdxT>
  struct select_reduce<cg_reduce_op_t, cg_group_t, IdxT, double, raft::KeyValuePair<IdxT, double>> {
    using ValT = double;
    using Ty   = raft::KeyValuePair<IdxT, ValT>;
    /// Performs warp level reduction of key value pair and stores a reduced output to memory
    CUTLASS_DEVICE
    select_reduce(Ty val_to_red,
                  double prev_red_val,
                  cg_reduce_op_t cg_reduce_op,
                  cg_group_t cg_warp_group,
                  Ty& shmem_ptr)
    {
      ValT val = val_to_red.value;

      if (cg_warp_group.any(cg_reduce_op.isAmin(val, prev_red_val))) {
        ValT reduced_val = cg::reduce(cg_warp_group, val, cg_reduce_op);
        bool pred        = (reduced_val == val);
        auto subTile     = cg::binary_partition(cg_warp_group, pred);
        if (pred) {
          if (subTile.thread_rank() == 0) { shmem_ptr = val_to_red; }
        }
      }
    }
  };

 private:
  //
  // Data members
  //

  /// Parameters structure containing reference and precomputed state.
  Params params_;

  /// Byte-level pointer first tile offset of this threadblock.
  volatile uint8_t* first_tile_byte_pointer_;

  /// Array of boolean values to contain steady-state predicates
  Mask mask_;

  /// Extent of the matrix tile in rows
  Index extent_row_;

  /// Extent of the matrix tile in rows
  Index extent_column_;

  /// A thread's starting row position (assuming steady-state predicates have been computed)
  Index thread_start_row_;
  Index block_start_row_first_tile_;

  /// A thread's starting column
  Index thread_start_column_;

  /// Internal state counter
  int state_[3];
  // mutable int shared_tile_id;

  /// Scatter indices
  int const* indices_;

  const int do_gmem_reduction_;

  //
  // Static asserts about internal strides
  //

  static_assert(sizeof(extent_row_) == 4, "Expected 32b extents");
  static_assert(sizeof(thread_start_row_) == 4, "Expected 32b extents");
  static_assert(sizeof(Params::stride) == 8, "Expected 64b strides");

 protected:
  SharedStorage& shared_storage_;

 private:
  //
  // Methods
  //
 public:
  //
  // Methods
  //
  /// Constructor
  CUTLASS_DEVICE
  PredicatedTileIteratorReducedVec(SharedStorage& shared_storage,
                                   Params const& params,
                                   volatile Element* pointer,
                                   TensorCoord extent,
                                   int thread_idx,
                                   const bool do_gmem_reduction,
                                   TensorCoord threadblock_offset = TensorCoord(),
                                   int const* indices             = nullptr)
    : params_(params),
      indices_(indices),
      shared_storage_(shared_storage),
      do_gmem_reduction_(do_gmem_reduction)
  {
    TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx) + threadblock_offset;

    extent_row_    = extent.row();
    extent_column_ = extent.column();

    thread_start_row_    = thread_offset.row();
    thread_start_column_ = thread_offset.column();

    TensorCoord block_offset    = ThreadMap::initial_offset(0) + threadblock_offset;
    block_start_row_first_tile_ = block_offset.row();

    // Initialize predicates
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kColumn * kElementsPerAccess; ++c) {
      int columnPerAccess       = (c / kElementsPerAccess);
      int columnWithinPerAccess = c % kElementsPerAccess;
      mask_.predicates[c] = ((thread_offset.column() + ThreadMap::Delta::kColumn * columnPerAccess +
                              columnWithinPerAccess) < extent.column());
    }

    if (threadblock_offset.column() == 0) {
      EpilogueOpParams const& user_params = params_.user_param;
      shared_storage_.initSmem(user_params);
    }
    __syncthreads();

    // Null pointer performs no accesses
    if (!pointer) { mask_.clear(); }

    if (ScatterD && !indices) { mask_.clear(); }

    // Initialize pointer
    first_tile_byte_pointer_ = reinterpret_cast<volatile uint8_t*>(pointer) +
                               LongIndex(block_offset.row()) * LongIndex(params_.stride);

    // Initialize internal state counter
    state_[0] = state_[1] = state_[2] = 0;
  }

  CUTLASS_DEVICE void dumpToGmem()
  {
    if (block_start_row_first_tile_ >= extent_row_) { return; }

    if (do_gmem_reduction_) {
      EpilogueOpParams const& user_params = params_.user_param;
      const uint32_t mutex_id             = (block_start_row_first_tile_ / total_rows);
      const bool useGmemMutex  = (gridDim.x != ((extent_row_ - 1 + total_rows) / total_rows));
      int row                  = threadIdx.x;
      Element* shared_elem_arr = shared_storage_.data();
      Element row_local_min;
      if (row < total_rows) { row_local_min = shared_elem_arr[row]; }

      // single lock per block for multiple rows
      if (useGmemMutex && threadIdx.x == 0) { user_params.bin_mutex_[mutex_id].acquire(); }
      __syncthreads();

      if (row < total_rows) {
        volatile Element* gmem_ptr = reinterpret_cast<volatile Element*>(first_tile_byte_pointer_);

        if ((block_start_row_first_tile_ + row) < extent_row_) {
          user_params.red_op_(block_start_row_first_tile_ + row, (gmem_ptr + row), row_local_min);
        }
      }

      __syncthreads();
      __threadfence();

      if (useGmemMutex && (threadIdx.x == 0)) {
        // release mutex lock.
        user_params.bin_mutex_[mutex_id].release();
      }
      shared_storage_.initSmem(user_params);
      __syncthreads();
    }
  }

  /// Destructor
  CUTLASS_DEVICE
  ~PredicatedTileIteratorReducedVec() {}

  /// Performs reduction and Stores a reduced output to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment& frag, int64_t byte_offset) const
  {
    AccessTypeValT* frag_ptr = reinterpret_cast<AccessTypeValT*>(&frag);

    cg::thread_block cta = cg::this_thread_block();
    // tile_width 16 is required if kElementPerAccess > 1
    constexpr int tile_width                 = (32 / ThreadMap::Delta::kColumn) ? 32 : 16;
    cg::thread_block_tile<tile_width> tile32 = cg::tiled_partition<tile_width>(cta);
    EpilogueOpParams const& user_params      = params_.user_param;

    using cg_reduce_t = decltype(user_params.cg_reduce_op);
    using tile32_t    = decltype(tile32);

    Element* shared_elem_arr = shared_storage_.data();
    constexpr auto maxVal    = std::numeric_limits<OutValT>::max();

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {
      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
          int frag_row_idx =
            (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          int row_offset = row * ThreadMap::Delta::kRow + group * ThreadMap::Delta::kGroup +
                           cluster * ThreadMap::Delta::kCluster;

          const OutIdxT row_id = row_offset + thread_start_row_;
          bool row_guard       = (row_id < extent_row_);

          const int frag_idx = frag_row_idx * ThreadMap::Iterations::kColumn * kElementsPerAccess;
          Element red_val;
          user_params.red_op_.init(&red_val, maxVal);

          if (row_guard) {
            CUTLASS_PRAGMA_UNROLL
            for (int column = 0; column < ThreadMap::Iterations::kColumn * kElementsPerAccess;
                 ++column) {
              int columnPerAccess     = column / kElementsPerAccess;
              int columnWithPerAccess = column % kElementsPerAccess;
              bool guard              = mask_.predicates[column];
              if (guard) {
                const OutIdxT key_id = thread_start_column_ +
                                       ThreadMap::Delta::kColumn * columnPerAccess +
                                       columnWithPerAccess;
                const int frag_col_idx = frag_idx + column;

                Element this_val;
                user_params.red_op_.init(&this_val, (*frag_ptr)[frag_col_idx]);
                user_params.red_op_.init_key(this_val, key_id);
                user_params.red_op_(row_id, &red_val, this_val);
              }
            }
          }
          const int iter_row      = (row_id % total_rows);
          const auto prev_red_val = user_params.red_op_.get_value(shared_elem_arr[iter_row]);
          if (row_guard) {
            // select_reduce doesn't need to use `red_op_` as at the warp level we use cg_reduce_op,
            // this satisfies the requirement of mst/single linkage of checking colors buffer.
            select_reduce<cg_reduce_t, tile32_t, OutIdxT, OutValT, Element> red_obj(
              red_val, prev_red_val, user_params.cg_reduce_op, tile32, shared_elem_arr[iter_row]);
          }
        }
      }
    }
    __syncthreads();
  }

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment& frag) const { store_with_byte_offset(frag, 0); }

  CUTLASS_DEVICE
  MatrixCoord thread_start() const { return MatrixCoord(thread_start_row_, thread_start_column_); }

  /// Need to get the thread start row from the tile iterator
  CUTLASS_DEVICE
  int32_t thread_start_row() const { return thread_start_row_; }

  /// Need to get the thread start row from the tile iterator
  CUTLASS_DEVICE
  int32_t thread_start_column() const { return thread_start_column_; }

  /// Extent of the matrix in rows
  CUTLASS_DEVICE
  Index extent_row() const { return extent_row_; }

  /// Extent of the matrix in columns
  CUTLASS_DEVICE
  Index extent_column() const { return extent_column_; }

  /// Advances to the next position to load or store
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorReducedVec& operator++()
  {
    ++state_[0];

    thread_start_row_ += ThreadMap::Shape::kRow;

    if (state_[0] == ThreadMap::Count::kRow) {
      state_[0] = 0;
      ++state_[1];

      thread_start_row_ +=
        (ThreadMap::Shape::kGroup - 1) * ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

      if (state_[1] == ThreadMap::Count::kGroup) {
        state_[1] = 0;
        ++state_[2];

        thread_start_row_ += ThreadMap::Count::kGroup * ThreadMap::Shape::kGroup *
                             ThreadMap::Count::kRow * ThreadMap::Shape::kRow;

        if (state_[2] == ThreadMap::Count::kCluster) { state_[2] = 0; }
      }
    }
    return *this;
  }

  ///< Efficiently disables all accesses guarded by mask
  CUTLASS_DEVICE void clear_mask() { mask_.clear(); }

  ///< Efficiently enables all accesses guarded by mask
  CUTLASS_DEVICE void enable_mask() { mask_.enable(); }

  ///< Sets the mask
  CUTLASS_DEVICE void get_mask(Mask& mask) const { mask = mask_; }

  ///< Sets the mask
  CUTLASS_DEVICE void set_mask(Mask const& mask) { mask_ = mask; }
};

///////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
