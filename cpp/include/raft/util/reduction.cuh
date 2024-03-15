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

#pragma once

#include <raft/core/cudart_utils.hpp>
#include <raft/core/operators.hpp>
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/warp_primitives.cuh>

#include <stdint.h>

namespace raft {

/**
 * @brief Logical-warp-level reduction
 * @tparam logicalWarpSize Logical warp size (2, 4, 8, 16 or 32)
 * @tparam T Value type to be reduced
 * @tparam ReduceLambda Reduction operation type
 * @param val input value
 * @param reduce_op Reduction operation
 * @return Reduction result. All lanes will have the valid result.
 */
template <int logicalWarpSize, typename T, typename ReduceLambda>
DI T logicalWarpReduce(T val, ReduceLambda reduce_op)
{
#pragma unroll
  for (int i = logicalWarpSize / 2; i > 0; i >>= 1) {
    T tmp = shfl_xor(val, i);
    val   = reduce_op(val, tmp);
  }
  return val;
}

/**
 * @brief Warp-level reduction
 * @tparam T Value type to be reduced
 * @tparam ReduceLambda Reduction operation type
 * @param val input value
 * @param reduce_op Reduction operation
 * @return Reduction result. All lanes will have the valid result.
 * @note Why not cub? Because cub doesn't seem to allow working with arbitrary
 *       number of warps in a block. All threads in the warp must enter this
 *       function together
 */
template <typename T, typename ReduceLambda>
DI T warpReduce(T val, ReduceLambda reduce_op)
{
  return logicalWarpReduce<WarpSize>(val, reduce_op);
}

/**
 * @brief Warp-level reduction
 * @tparam T Value type to be reduced
 * @param val input value
 * @return Reduction result. All lanes will have the valid result.
 * @note Why not cub? Because cub doesn't seem to allow working with arbitrary
 *       number of warps in a block. All threads in the warp must enter this
 *       function together
 */
template <typename T>
DI T warpReduce(T val)
{
  return warpReduce(val, raft::add_op{});
}

/**
 * @brief 1-D block-level reduction
 * @param val input value
 * @param smem shared memory region needed for storing intermediate results. It
 *             must alteast be of size: `sizeof(T) * nWarps`
 * @param reduce_op a binary reduction operation.
 * @return only the thread0 will contain valid reduced result
 * @note Why not cub? Because cub doesn't seem to allow working with arbitrary
 *       number of warps in a block. All threads in the block must enter this
 *       function together. cub also uses too many registers
 */
template <typename T, typename ReduceLambda = raft::add_op>
DI T blockReduce(T val, char* smem, ReduceLambda reduce_op = raft::add_op{})
{
  auto* sTemp = reinterpret_cast<T*>(smem);
  int nWarps  = (blockDim.x + WarpSize - 1) / WarpSize;
  int lid     = laneId();
  int wid     = threadIdx.x / WarpSize;
  val         = warpReduce(val, reduce_op);
  if (lid == 0) sTemp[wid] = val;
  __syncthreads();
  val = lid < nWarps ? sTemp[lid] : T(0);
  return warpReduce(val, reduce_op);
}

/**
 * @brief 1-D warp-level ranked reduction which returns the value and rank.
 * thread 0 will have valid result and rank(idx).
 * @param val input value
 * @param idx index to be used as rank
 * @param reduce_op a binary reduction operation.
 */
template <typename T, typename ReduceLambda, typename i_t = int>
DI void warpRankedReduce(T& val, i_t& idx, ReduceLambda reduce_op = raft::min_op{})
{
#pragma unroll
  for (i_t offset = WarpSize / 2; offset > 0; offset /= 2) {
    T tmpVal   = shfl(val, laneId() + offset);
    i_t tmpIdx = shfl(idx, laneId() + offset);
    if (reduce_op(tmpVal, val) == tmpVal) {
      val = tmpVal;
      idx = tmpIdx;
    }
  }
}

/**
 * @brief 1-D block-level ranked reduction which returns the value and rank.
 * thread 0 will have valid result and rank(idx).
 * @param val input value
 * @param shbuf shared memory region needed for storing intermediate results. It
 *             must alteast be of size: `(sizeof(T) + sizeof(i_t)) * WarpSize`
 * @param idx index to be used as rank
 * @param reduce_op binary min or max operation.
 * @return only the thread0 will contain valid reduced result
 */
template <typename T, typename ReduceLambda, typename i_t = int>
DI std::pair<T, i_t> blockRankedReduce(T val,
                                       T* shbuf,
                                       i_t idx                = threadIdx.x,
                                       ReduceLambda reduce_op = raft::min_op{})
{
  T* values    = shbuf;
  i_t* indices = (i_t*)&shbuf[WarpSize];
  i_t wid      = threadIdx.x / WarpSize;
  i_t nWarps   = (blockDim.x + WarpSize - 1) / WarpSize;
  warpRankedReduce(val, idx, reduce_op);  // Each warp performs partial reduction
  i_t lane = laneId();
  if (lane == 0) {
    values[wid]  = val;  // Write reduced value to shared memory
    indices[wid] = idx;  // Write reduced value to shared memory
  }

  __syncthreads();  // Wait for all partial reductions

  // read from shared memory only if that warp existed
  if (lane < nWarps) {
    val = values[lane];
    idx = indices[lane];
  } else {
    // get the lower_bound of the type if it is a max op,
    // get the upper bound of the type if it is a min op
    val = reduce_op(lower_bound<T>(), upper_bound<T>()) == lower_bound<T>() ? upper_bound<T>()
                                                                            : lower_bound<T>();
    idx = -1;
  }
  __syncthreads();
  if (wid == 0) warpRankedReduce(val, idx, reduce_op);
  return std::pair<T, i_t>{val, idx};
}

/**
 * @brief Executes a 1d binary block reduce
 * @param val binary value to be reduced across the thread block
 * @param shmem memory needed for the reduction. It should be at least of size blockDim.x/WarpSize
 * @return only the thread0 will contain valid reduced result
 */
template <int BLOCK_SIZE, typename i_t>
DI i_t binaryBlockReduce(i_t val, i_t* shmem)
{
  static_assert(BLOCK_SIZE <= 1024);
  assert(val == 0 || val == 1);
  const uint32_t mask    = __ballot_sync(~0, val);
  const uint32_t n_items = __popc(mask);

  // Each first thread of the warp
  if (threadIdx.x % WarpSize == 0) { shmem[threadIdx.x / WarpSize] = n_items; }
  __syncthreads();

  val = (threadIdx.x < BLOCK_SIZE / WarpSize) ? shmem[threadIdx.x] : 0;

  if (threadIdx.x < WarpSize) {
    return warpReduce(val);
  }
  // Only first warp gets the results
  else {
    return -1;
  }
}

}  // namespace raft
