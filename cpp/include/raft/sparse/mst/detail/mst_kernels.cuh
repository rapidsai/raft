
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

namespace raft {
namespace mst {
namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
__global__ void kernel_min_edge_per_vertex(const vertex_t* offsets,
                                           const edge_t* indices,
                                           const weight_t* weights,
                                           vertex_t* color, vertex_t* successor,
                                           bool* mst_edge, const vertex_t v) {
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
      successor[warp_id] = indices[min_edge_index[0]];
    }
  }
}

// executes for each vertex and updates the colors of both vertices to the lower color
template <typename vertex_t>
__global__ void min_pair_colors(const vertex_t v, const vertex_t* successor,
                                vertex_t* color, vertex_t* next_color) {
  int i = get_1D_idx();
  if (i < v) {
    atomicMin(&next_color[i], color[successor[i]]);
    atomicMin(&next_color[successor[i]], color[i]);
  }
}

template <typename vertex_t>
__global__ void check_color_change(const vertex_t v, vertex_t* color,
                                   vertex_t* next_color, bool* done) {
  //This kernel works on the global_colors[] array
  int i = get_1D_idx();
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

}  // namespace detail
}  // namespace mst
}  // namespace raft
