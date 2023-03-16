/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

  // static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kElementsPerAccess = 1;
  static int const kThreads           = ThreadMap::kThreads;
  static int const kIterations        = ThreadMap::Count::kTile;

  static_assert(ThreadMap::Iterations::kRow > 0, "ThreadMap::Iterations::kRow must be > 0");
  static_assert(ThreadMap::Iterations::kGroup > 0, "ThreadMap::Iterations::kGroup must be > 0");
  static_assert(ThreadMap::Iterations::kCluster > 0, "ThreadMap::Iterations::kCluster must be > 0");
  static_assert(ThreadMap::Iterations::kColumn > 0, "ThreadMap::Iterations::kColumn must be > 0");
  static_assert(!UseCUDAStore, "UseCUDAStore path is not supported");

  /// Fragment object
  using Fragment =
    Array<Element,
          ThreadMap::Iterations::kColumn * ThreadMap::Iterations::kRow *
            ThreadMap::Iterations::kGroup * ThreadMap::Iterations::kCluster * kElementsPerAccess>;

  /// Memory access size
  using AccessType = AlignedArray<Element, kElementsPerAccess>;

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
  //  iterator for reduction.
  struct SharedStorage {
    //
    // Type definitions
    //
    using Shape = MatrixShape<ThreadMap::kWarpCount * ThreadMap::Iterations::kRow *
                                ThreadMap::Iterations::kGroup * ThreadMap::Iterations::kCluster *
                                ThreadMap::Count::kTile,
                              1>;

    /// Shape of the shared memory allocation for the reduced values store
    using StorageShape = MatrixShape<Shape::kRow, Shape::kColumn>;

    //
    // Data members
    //
    static const int warp_row_stride =
      ThreadMap::Iterations::kRow * ThreadMap::Iterations::kGroup * ThreadMap::Iterations::kCluster;
    static const int tile_row_stride = ThreadMap::kWarpCount * ThreadMap::Iterations::kRow *
                                       ThreadMap::Iterations::kGroup *
                                       ThreadMap::Iterations::kCluster;

    //
    // Methods
    //
    AlignedBuffer<Element, StorageShape::kCount> storage;

    CUTLASS_DEVICE
    Element* data() { return storage.data(); }

    SharedStorage() {}
  };

  template <typename reduce_op_t, typename cg_group_t, typename IdxT, typename ValT, typename OutT>
  struct select_reduce {
    /// Performs reduction and stores a reduced output to memory
    CUTLASS_DEVICE
    select_reduce(OutT red_value, reduce_op_t reduce_op, cg_group_t cg_warp_group, OutT& shmem_ptr)
    {
      OutT reduced_val = cg::reduce(cg_warp_group, red_value, reduce_op);
      if (cg_warp_group.thread_rank() == 0) { shmem_ptr = reduced_val; }
    }
  };

  template <typename reduce_op_t, typename cg_group_t, typename IdxT>
  struct select_reduce<reduce_op_t, cg_group_t, IdxT, float, raft::KeyValuePair<IdxT, float>> {
    using ValT = float;
    using Ty   = raft::KeyValuePair<IdxT, ValT>;

    CUTLASS_DEVICE
    select_reduce(Ty val_to_red, reduce_op_t reduce_op, cg_group_t cg_warp_group, Ty& shmem_ptr)
    {
      ValT val         = val_to_red.value;
      ValT reduced_val = cg::reduce(cg_warp_group, val, reduce_op);
      bool pred        = (reduced_val == val);
      auto subTile     = cg::binary_partition(cg_warp_group, pred);
      if (pred) {
        if (subTile.thread_rank() == 0) { shmem_ptr = val_to_red; }
      }
    }
  };

  template <typename reduce_op_t, typename cg_group_t, typename IdxT>
  struct select_reduce<reduce_op_t, cg_group_t, IdxT, double, raft::KeyValuePair<IdxT, double>> {
    using ValT = double;
    using Ty   = raft::KeyValuePair<IdxT, ValT>;

    CUTLASS_DEVICE
    select_reduce(Ty val_to_red, reduce_op_t reduce_op, cg_group_t cg_warp_group, Ty& shmem_ptr)
    {
      ValT val         = val_to_red.value;
      ValT reduced_val = cg::reduce(cg_warp_group, val, reduce_op);
      bool pred        = (reduced_val == val);
      auto subTile     = cg::binary_partition(cg_warp_group, pred);
      if (pred) {
        if (subTile.thread_rank() == 0) { shmem_ptr = val_to_red; }
      }
    }
  };

 private:
  //
  // Data members
  //

  /// Parameters structure containing reference and precomputed state.
  Params params_;

  /// Byte-level pointer
  uint8_t* byte_pointer_;
  /// Byte-level pointer first tile offset of this threadblock.
  uint8_t* first_tile_byte_pointer_;

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
  mutable int shared_tile_id;

  /// Scatter indices
  int const* indices_;

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
                                   Element* pointer,
                                   TensorCoord extent,
                                   int thread_idx,
                                   TensorCoord threadblock_offset = TensorCoord(),
                                   int const* indices             = nullptr)
    : params_(params), indices_(indices), shared_storage_(shared_storage)
  {
    TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx) + threadblock_offset;

    extent_row_    = extent.row();
    extent_column_ = extent.column();

    thread_start_row_    = thread_offset.row();
    thread_start_column_ = thread_offset.column();

    TensorCoord block_offset    = ThreadMap::initial_offset(0) + threadblock_offset;
    block_start_row_first_tile_ = block_offset.row();
    shared_tile_id              = 0;

    // Initialize predicates
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
      mask_.predicates[c] =
        ((thread_offset.column() + ThreadMap::Delta::kColumn * c) < extent.column());
    }

    // Null pointer performs no accesses
    if (!pointer) { mask_.clear(); }

    if (ScatterD && !indices) { mask_.clear(); }

    // Initialize pointer
    byte_pointer_ = reinterpret_cast<uint8_t*>(pointer) +
                    LongIndex(thread_offset.row()) * LongIndex(params_.stride);

    first_tile_byte_pointer_ = reinterpret_cast<uint8_t*>(pointer) +
                               LongIndex(block_offset.row()) * LongIndex(params_.stride);

    if (ScatterD) {
      byte_pointer_ = reinterpret_cast<uint8_t*>(pointer) +
                      LongIndex(thread_offset.column()) * sizeof(AccessType) / kElementsPerAccess;
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
    uint8_t* byte_pointer = byte_pointer_;
    AccessType* frag_ptr  = reinterpret_cast<AccessType*>(&frag);

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

          bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

          AccessType* memory_pointer = reinterpret_cast<AccessType*>(byte_pointer + byte_offset);

          if (ScatterD && row_guard) {
            assert(indices_);

            memory_pointer = reinterpret_cast<AccessType*>(
              byte_pointer + byte_offset +
              LongIndex(indices_[row_offset + thread_start_row_]) * LongIndex(params_.stride));
          }

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
            bool guard = row_guard && mask_.predicates[column];

            cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
              frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
              (void*)&memory_pointer[0],
              guard);
          }

          if (row + 1 < ThreadMap::Iterations::kRow) {
            if (!ScatterD) { byte_pointer += params_.increment_row; }
          }
        }

        if (group + 1 < ThreadMap::Iterations::kGroup) { byte_pointer += params_.increment_group; }
      }

      if (cluster + 1 < ThreadMap::Iterations::kCluster) {
        byte_pointer += params_.increment_cluster;
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment& frag) const { load_with_byte_offset(frag, 0); }

  /// Performs reduction and Stores a reduced output to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment& frag, int64_t byte_offset) const
  {
    AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

    cg::thread_block cta                = cg::this_thread_block();
    cg::thread_block_tile<32> tile32    = cg::tiled_partition<32>(cta);
    EpilogueOpParams const& user_params = params_.user_param;

    using cg_reduce_t = decltype(user_params.cg_reduce_op);
    using tile32_t    = decltype(tile32);

    Element* shared_elem_arr = shared_storage_.data();

    static int const total_rows = ThreadMap::kWarpCount * ThreadMap::Iterations::kRow *
                                  ThreadMap::Iterations::kGroup * ThreadMap::Iterations::kCluster *
                                  ThreadMap::Count::kTile;

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

          bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

          const int frag_idx = frag_row_idx * ThreadMap::Iterations::kColumn;
          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
            bool guard = row_guard && mask_.predicates[column];

            if (guard) {
              const auto key_id      = thread_start_column_ + ThreadMap::Delta::kColumn * column;
              const int frag_col_idx = frag_idx + column;
              user_params.red_op_.init_key((*frag_ptr)[frag_col_idx], key_id);
              user_params.red_op_(key_id, &(*frag_ptr)[frag_idx], (*frag_ptr)[frag_col_idx]);
            }
          }
          bool col_guard  = row_guard && mask_.predicates[0];
          auto subTile    = cg::binary_partition(tile32, col_guard);
          using subTile_t = decltype(subTile);

          if (col_guard) {
            int iter_row = ((row_offset + thread_start_row_) % total_rows);
            if (subTile.size() == 32) {
              select_reduce<cg_reduce_t, tile32_t, OutIdxT, OutValT, Element> red_obj(
                (*frag_ptr)[frag_idx], user_params.cg_reduce_op, tile32, shared_elem_arr[iter_row]);
            } else {
              select_reduce<cg_reduce_t, subTile_t, OutIdxT, OutValT, Element> red_obj(
                (*frag_ptr)[frag_idx],
                user_params.cg_reduce_op,
                subTile,
                shared_elem_arr[iter_row]);
            }
          }
        }
      }
    }

    // If this is last tile then perform reduction in gmem.
    if (shared_tile_id == (ThreadMap::Count::kTile - 1)) {
      const auto mutex_id = (block_start_row_first_tile_ / total_rows);
      // single lock per block for multiple rows
      if (threadIdx.x == 0 && block_start_row_first_tile_ < extent_row_) {
        // acquire mutex lock.
        while (atomicCAS(user_params.mutexes_ + mutex_id, 0, 1) == 1)
          ;
      }
      __syncthreads();

      auto gmem_ptr = reinterpret_cast<Element*>(first_tile_byte_pointer_);

      for (int row = threadIdx.x; row < total_rows; row += blockDim.x) {
        if (block_start_row_first_tile_ + row < extent_row_) {
          user_params.red_op_(0, &gmem_ptr[row], shared_elem_arr[row]);
        }
      }

      __threadfence();
      __syncthreads();
      if (threadIdx.x == 0 && block_start_row_first_tile_ < extent_row_) {
        // release mutex lock.
        atomicCAS(user_params.mutexes_ + mutex_id, 1, 0);
      }
    }
  }

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment& frag) const { store_with_byte_offset(frag, 0); }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void downsample_load_with_byte_offset(Fragment& frag,
                                        int64_t byte_offset,
                                        int convolution_P,
                                        int convolution_Q,
                                        int add_P,
                                        int add_Q,
                                        int problem_N) const
  {
    uint8_t* byte_pointer = byte_pointer_;
    AccessType* frag_ptr  = reinterpret_cast<AccessType*>(&frag);

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

          bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

          int output_row = row_offset + thread_start_row_;
          int output_N   = output_row / (convolution_P * convolution_Q);
          int output_PQ  = output_row % (convolution_P * convolution_Q);
          int output_P   = output_PQ / convolution_Q;
          int output_Q   = output_PQ % convolution_Q;

          int input_row = output_N * 2 * convolution_P * 2 * convolution_Q +
                          (2 * output_P + add_P) * 2 * convolution_Q + 2 * output_Q + add_Q;

          int64_t byte_offset = (input_row - output_row) * problem_N * sizeof(float);

          AccessType* memory_pointer = reinterpret_cast<AccessType*>(byte_pointer + byte_offset);

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
            bool guard = row_guard && mask_.predicates[column];

            cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
              frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
              (void*)&memory_pointer[column * ThreadMap::Delta::kColumn / kElementsPerAccess],
              guard);
          }

          if (row + 1 < ThreadMap::Iterations::kRow) { byte_pointer += params_.increment_row; }
        }

        if (group + 1 < ThreadMap::Iterations::kGroup) { byte_pointer += params_.increment_group; }
      }

      if (cluster + 1 < ThreadMap::Iterations::kCluster) {
        byte_pointer += params_.increment_cluster;
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void upsample_load_with_byte_offset(Fragment& frag,
                                      int64_t byte_offset,
                                      int convolution_P,
                                      int convolution_Q,
                                      int add_P,
                                      int add_Q,
                                      int problem_N) const
  {
    uint8_t* byte_pointer = byte_pointer_;
    AccessType* frag_ptr  = reinterpret_cast<AccessType*>(&frag);

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

          bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

          int output_row = row_offset + thread_start_row_;
          int output_N   = output_row / (convolution_P * convolution_Q);
          int output_PQ  = output_row % (convolution_P * convolution_Q);
          int output_P   = output_PQ / convolution_Q;
          int output_Q   = output_PQ % convolution_Q;
          int row_add_P  = add_P;
          int row_add_Q  = add_Q;
          if (output_P > convolution_P - 2) row_add_P = 0;
          if (output_Q > convolution_Q - 2) row_add_Q = 0;

          int input_row = output_N * (convolution_P / 2) * (convolution_Q / 2) +
                          ((output_P + row_add_P) / 2) * (convolution_Q / 2) +
                          (output_Q + row_add_Q) / 2;

          int64_t byte_offset = (input_row - output_row) * problem_N * sizeof(float);

          AccessType* memory_pointer = reinterpret_cast<AccessType*>(byte_pointer + byte_offset);

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
            bool guard = row_guard && mask_.predicates[column];

            cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
              frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
              (void*)&memory_pointer[column * ThreadMap::Delta::kColumn / kElementsPerAccess],
              guard);
          }

          if (row + 1 < ThreadMap::Iterations::kRow) { byte_pointer += params_.increment_row; }
        }

        if (group + 1 < ThreadMap::Iterations::kGroup) { byte_pointer += params_.increment_group; }
      }

      if (cluster + 1 < ThreadMap::Iterations::kCluster) {
        byte_pointer += params_.increment_cluster;
      }
    }
  }

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
    shared_tile_id++;  // tile iteration.

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
