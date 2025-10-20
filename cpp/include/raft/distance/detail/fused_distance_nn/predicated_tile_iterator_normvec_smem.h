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
- Only the row index is used to load the data in load_with_byte_offset().
  This way the same normalization data is used across all columns in a row.

*/
#pragma once

#include <raft/util/device_loads_stores.cuh>

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
          bool ScatterD     = false,  ///< Scatter D operand or not
          bool UseCUDAStore = false>
class PredicatedTileIteratorNormVecSmem {
 public:
  using ThreadMap = ThreadMap_;
  using Shape     = typename ThreadMap::Shape;

  using Element = Element_;

  using Layout         = Layout_;
  using TensorRef      = TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index       = typename Layout::Index;
  using LongIndex   = typename Layout::LongIndex;
  using TensorCoord = MatrixCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kThreads           = ThreadMap::kThreads;
  static int const kIterations        = ThreadMap::Count::kTile;

  static int const total_rows = ThreadMap::kWarpCount * ThreadMap::Iterations::kRow *
                                ThreadMap::Iterations::kGroup * ThreadMap::Iterations::kCluster *
                                ThreadMap::Count::kTile * ThreadMap::Delta::kRow;

  static_assert(ThreadMap::Iterations::kRow > 0, "ThreadMap::Iterations::kRow must be > 0");
  static_assert(ThreadMap::Iterations::kGroup > 0, "ThreadMap::Iterations::kGroup must be > 0");
  static_assert(ThreadMap::Iterations::kCluster > 0, "ThreadMap::Iterations::kCluster must be > 0");
  static_assert(ThreadMap::Iterations::kColumn > 0, "ThreadMap::Iterations::kColumn must be > 0");

  using Fragment = Array<Element,
                         ThreadMap::Iterations::kRow * ThreadMap::Iterations::kGroup *
                           ThreadMap::Iterations::kCluster * ThreadMap::kElementsPerAccess>;

  /// Memory access size
  using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;

  //
  // Parameters struct
  //

  /// Uses a non-template class
  struct Params : PredicatedTileIteratorParams {
    using Base = PredicatedTileIteratorParams;

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
    Params(Base const& base) : Base(base) {}
  };

  /// Mask object
  struct Mask {
    static int const kCount = ThreadMap::Iterations::kColumn;

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
  //  iterator for storing rowNorm chunk.
  struct SharedStorage {
    //
    // Type definitions
    //
    using Shape = MatrixShape<total_rows, 1>;

    /// Shape of the shared memory allocation
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
    void initSmem(void* pointer,
                  const Index& num_rows,
                  const Index& tb_row_offset,
                  const LongIndex& stride)
    {
      Element* shared_elem_arr = data();
      uint8_t* first_tile_byte_pointer_ =
        reinterpret_cast<uint8_t*>(pointer) + LongIndex(tb_row_offset) * LongIndex(stride);
      const auto gmem_ptr = reinterpret_cast<Element*>(first_tile_byte_pointer_);

      for (int row = threadIdx.x; row < total_rows; row += blockDim.x) {
        bool guard = (tb_row_offset + row) < num_rows;
        cutlass::arch::cp_async<sizeof(Element)>(shared_elem_arr + row, gmem_ptr + row, guard);
        cutlass::arch::cp_async_wait<0>();
      }
    }
  };

 private:
  //
  // Data members
  //

  /// Parameters structure containing reference and precomputed state.
  PredicatedTileIteratorParams params_;

  /// Byte-level pointer
  uint8_t* byte_pointer_;

  /// Array of boolean values to contain steady-state predicates
  Mask mask_;

  /// Extent of the matrix tile in rows
  Index extent_row_;

  /// Extent of the matrix tile in rows
  Index extent_column_;

  /// A thread's starting row position (assuming steady-state predicates have been computed)
  Index thread_start_row_;

  /// A thread's starting column
  Index thread_start_column_;

  /// Internal state counter
  int state_[3];

  /// Scatter indices
  int const* indices_;

  //
  // Static asserts about internal strides
  //

  static_assert(sizeof(extent_row_) == 4, "Expected 32b extents");
  static_assert(sizeof(thread_start_row_) == 4, "Expected 32b extents");
  static_assert(sizeof(PredicatedTileIteratorParams::stride) == 8, "Expected 64b strides");

 private:
  //
  // Methods
  //

 protected:
  SharedStorage& shared_storage_;

 public:
  //
  // Methods
  //

  /// Constructor
  CUTLASS_DEVICE
  PredicatedTileIteratorNormVecSmem(SharedStorage& shared_storage,
                                    PredicatedTileIteratorParams const& params,
                                    Element* pointer,
                                    TensorCoord extent,
                                    int thread_idx,
                                    TensorCoord& threadblock_offset,
                                    int const* indices = nullptr)
    : params_(params), indices_(indices), shared_storage_(shared_storage)
  {
    TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx) + threadblock_offset;

    extent_row_    = extent.row();
    extent_column_ = extent.column();

    thread_start_row_    = thread_offset.row();
    thread_start_column_ = thread_offset.column();

    // Initialize predicates
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
      mask_.predicates[c] =
        ((thread_offset.column() + ThreadMap::Delta::kColumn * c) < extent.column());
    }

    // Null pointer performs no accesses
    if (!pointer) {
      mask_.clear();
      return;
    }

    if (ScatterD && !indices) { mask_.clear(); }

    // Initialize pointer
    byte_pointer_ = reinterpret_cast<uint8_t*>(pointer) +
                    LongIndex(thread_offset.row()) * LongIndex(params_.stride);

    if (ScatterD) {
      byte_pointer_ = reinterpret_cast<uint8_t*>(pointer) +
                      LongIndex(thread_offset.column()) * sizeof(AccessType) / kElementsPerAccess;
    }

    if (threadblock_offset.column() == 0) {
      shared_storage_.initSmem(pointer, extent_row_, threadblock_offset.row(), params_.stride);
    }

    // Initialize internal state counter
    state_[0] = state_[1] = state_[2] = 0;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset)
  {
    byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment& frag, int64_t byte_offset) const
  {
    AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

    Element* shared_elem_arr = shared_storage_.data();

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
          int iter_row = ((row_offset + thread_start_row_) % total_rows);
          Element val  = shared_elem_arr[iter_row];

          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < kElementsPerAccess; ++i) {
            (*frag_ptr)[frag_row_idx + i] = val;
          }
        }
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment& frag) const { load_with_byte_offset(frag, 0); }

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
  PredicatedTileIteratorNormVecSmem& operator++()
  {
    ++state_[0];

    if (!ScatterD) { byte_pointer_ += params_.advance_row; }

    thread_start_row_ += ThreadMap::Shape::kRow;

    if (state_[0] == ThreadMap::Count::kRow) {
      state_[0] = 0;
      ++state_[1];
      byte_pointer_ += params_.advance_group;

      thread_start_row_ +=
        (ThreadMap::Shape::kGroup - 1) * ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

      if (state_[1] == ThreadMap::Count::kGroup) {
        state_[1] = 0;
        ++state_[2];
        byte_pointer_ += params_.advance_cluster;

        thread_start_row_ += ThreadMap::Count::kGroup * ThreadMap::Shape::kGroup *
                             ThreadMap::Count::kRow * ThreadMap::Shape::kRow;

        if (state_[2] == ThreadMap::Count::kCluster) {
          state_[2] = 0;
          byte_pointer_ += params_.advance_tile;
        }
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
