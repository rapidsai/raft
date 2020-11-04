
/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "utils.cuh"

#include <limits>

#include <raft/device_atomics.cuh>

namespace raft {
namespace mst {
namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
__global__ void kernel_min_edge_per_vertex(
  const vertex_t* offsets, const edge_t* indices, const weight_t* weights,
  vertex_t* color, vertex_t* successor, bool* mst_edge, edge_t* new_mst_edge,
  weight_t* min_edge_color, const vertex_t v) {
  edge_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  unsigned warp_id = tid / 32;
  unsigned lane_id = tid % 32;

  __shared__ edge_t min_edge_index[32];
  __shared__ weight_t min_edge_weight[32];
  __shared__ vertex_t min_color[32];

  min_edge_index[lane_id] = std::numeric_limits<edge_t>::max();
  min_edge_weight[lane_id] = std::numeric_limits<weight_t>::max();
  min_color[lane_id] = std::numeric_limits<vertex_t>::max();

  // TODO: Find a way to set limits
  // Above does not work as it is host code
  // min_edge_index[lane_id] = 100;
  // min_edge_weight[lane_id] = 100;
  // min_color[lane_id] = 100;
  __syncthreads();

  vertex_t self_color = color[warp_id];

  if (warp_id < v) {
    // one row is associated with one warp
    edge_t row_start = offsets[warp_id];
    edge_t row_end = offsets[warp_id + 1];

    // assuming one warp per row
    // find min for each thread in warp
    for (edge_t e = row_start + lane_id; e < row_end; e += 32) {
      weight_t curr_edge_weight = weights[e];
      vertex_t successor_color = color[indices[e]];

      if (!mst_edge[e] && self_color != successor_color) {
        if (curr_edge_weight < min_edge_weight[lane_id]) {
          min_color[lane_id] = successor_color;
          min_edge_weight[lane_id] = curr_edge_weight;
          min_edge_index[lane_id] = e;
          // theta = abs(curr_edge_weight - min_edge_weight[lane_id]);
        } else if (curr_edge_weight == min_edge_weight[lane_id]) {
          // tie break
          if (min_color[lane_id] > successor_color) {
            min_color[lane_id] = successor_color;
            min_edge_weight[lane_id] = curr_edge_weight;
            min_edge_index[lane_id] = e;
          }
        }
      }
    }
  }
  __syncthreads();

  // reduce across threads in warp
  for (int offset = 16; offset > 0; offset >>= 1) {
    if (lane_id < offset) {
      if (min_edge_weight[lane_id] > min_edge_weight[lane_id + offset]) {
        min_color[lane_id] = min_color[lane_id + offset];
        min_edge_weight[lane_id] = min_edge_weight[lane_id + offset];
        min_edge_index[lane_id] = min_edge_index[lane_id + offset];
      } else if (min_edge_weight[lane_id] ==
                 min_edge_weight[lane_id + offset]) {
        if (min_color[lane_id] > min_color[lane_id + offset]) {
          min_color[lane_id] = min_color[lane_id + offset];
          min_edge_weight[lane_id] = min_edge_weight[lane_id + offset];
          min_edge_index[lane_id] = min_edge_index[lane_id + offset];
        }
      }
    }
    __syncthreads();
  }

  // min edge may now be found in first thread
  if (lane_id == 0) {
    if (min_edge_weight[0] != std::numeric_limits<weight_t>::max()) {
      // successor[warp_id] = indices[min_edge_index[0]];

      new_mst_edge[warp_id] = min_edge_index[0];

      // atomically set min edge per color
      // takes care of super vertex case
      atomicMin(&min_edge_color[self_color], min_edge_weight[0]);
    }
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
__global__ void min_edge_per_supervertex(
  const vertex_t* color, edge_t* new_mst_edge, const vertex_t* indices,
  const weight_t* weights, vertex_t* successor, bool* mst_edge,
  vertex_t* temp_src, vertex_t* temp_dest, const weight_t* min_edge_color,
  const vertex_t v) {
  vertex_t tid = get_1D_idx();

  if (tid < v) {
    vertex_t vertex_color = color[tid];
    edge_t edge_idx = new_mst_edge[tid];

    // check if valid outgoing edge was found
    if (edge_idx != std::numeric_limits<edge_t>::max()) {
      weight_t vertex_weight = weights[edge_idx];
      if (min_edge_color[vertex_color] == vertex_weight) {
        temp_src[tid] = tid;
        temp_dest[tid] = indices[edge_idx];

        mst_edge[edge_idx] = true;

        // successor[tid] = indices[edge_idx];
      } else {
        new_mst_edge[tid] = std::numeric_limits<edge_t>::max();
      }
    }
  }
}

// executes for each vertex and updates the colors of both vertices to the lower color
template <typename vertex_t>
__global__ void min_pair_colors(const vertex_t mst_edge_count, const vertex_t *mst_src, const vertex_t* mst_dest,
                                vertex_t* color, vertex_t* next_color) {
  vertex_t i = get_1D_idx();
  if (i < mst_edge_count) {
    auto src = mst_src[i];
    auto dest = mst_dest[i];

    atomicMin(&next_color[src], color[dest]);
    atomicMin(&next_color[dest], color[src]);
  }
}

template <typename vertex_t>
__global__ void check_color_change(const vertex_t v, vertex_t* color,
                                   vertex_t* next_color, bool* done) {
  //This kernel works on the global_colors[] array
  vertex_t i = get_1D_idx();
  if (i < v) {
    if (color[i] > next_color[i]) {
      //Termination for label propagation
      done[0] = false;
      color[i] = next_color[i];
    }
  }
  // Notice that some degree >1 and we run in parallel
  // min_pair_colors kernel may result in pair color inconsitencies
  // resolving here for next iteration
  // TODO check experimentally
  next_color[i] = color[i];
}

// template <typename vertex_t>
// __global__ void kernel_check_termination(const vertex_t& e, bool* mst_edge,
//                                          bool* prev_mst_edge, bool* done) {
//   vertex_t tid = get_1D_idx();

//   // count > 0 values in block
//   bool predicate = tid < e && (mst_edge[tid] ^ prev_mst_edge[tid]);
//   vertex_t block_count = __syncthreads_count(predicate);

//   if (threadIdx.x == 0 && block_count > 0) {
//     *done = false;
//   }
// }

// Alterate the weights, make all undirected edge weight unique while keeping Wuv == Wvu
// Consider using curand device API instead of precomputed random_values array
template <typename vertex_t, typename edge_t, typename weight_t>
__global__ void alteration_kernel(const vertex_t v, const edge_t e,
                                  const edge_t* offsets,
                                  const vertex_t* indices,
                                  const weight_t* weights, weight_t max,
                                  weight_t* random_values,
                                  weight_t* alterated_weights) {
  auto row = get_1D_idx();
  if (row < v) {
    auto row_begin = offsets[row];
    auto row_end = offsets[row + 1];
    for (auto i = row_begin; i < row_end; i++) {
      auto column = indices[i];
      alterated_weights[i] =
        weights[i] + max * (random_values[row] + random_values[column]);
    }
  }
}

template <typename vertex_t>
__global__ void kernel_count_new_mst_edges(const vertex_t* mst_src,
                                           vertex_t* mst_edge_count,
                                           const vertex_t v) {
  vertex_t tid = get_1D_idx();

  bool predicate =
    tid < v && (mst_src[tid] != std::numeric_limits<vertex_t>::max());
  vertex_t block_count = __syncthreads_count(predicate);

  if (threadIdx.x == 0 && block_count > 0) {
    atomicAdd(mst_edge_count, block_count);
  }
}

}  // namespace detail
}  // namespace mst
}  // namespace raft
