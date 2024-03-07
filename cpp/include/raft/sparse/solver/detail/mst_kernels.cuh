
/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <raft/sparse/solver/detail/mst_utils.cuh>
#include <raft/util/device_atomics.cuh>

#include <limits>

namespace raft::sparse::solver::detail {

template <typename vertex_t, typename edge_t, typename alteration_t>
RAFT_KERNEL kernel_min_edge_per_vertex(const edge_t* offsets,
                                       const vertex_t* indices,
                                       const alteration_t* weights,
                                       const vertex_t* color,
                                       const vertex_t* color_index,
                                       edge_t* new_mst_edge,
                                       const bool* mst_edge,
                                       alteration_t* min_edge_color,
                                       const vertex_t v)
{
  edge_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  unsigned warp_id = tid / 32;
  unsigned lane_id = tid % 32;

  __shared__ edge_t min_edge_index[32];
  __shared__ alteration_t min_edge_weight[32];
  __shared__ vertex_t min_color[32];

  min_edge_index[lane_id]  = std::numeric_limits<edge_t>::max();
  min_edge_weight[lane_id] = std::numeric_limits<alteration_t>::max();
  min_color[lane_id]       = std::numeric_limits<vertex_t>::max();

  __syncthreads();

  vertex_t self_color_idx = color_index[warp_id];
  vertex_t self_color     = color[self_color_idx];

  // find the minimum edge associated per row
  // each thread in warp holds the minimum edge for
  // only the edges that thread scanned
  if (warp_id < v) {
    // one row is associated with one warp
    edge_t row_start = offsets[warp_id];
    edge_t row_end   = offsets[warp_id + 1];

    // assuming one warp per row
    // find min for each thread in warp
    for (edge_t e = row_start + lane_id; e < row_end; e += 32) {
      alteration_t curr_edge_weight = weights[e];
      vertex_t successor_color_idx  = color_index[indices[e]];
      vertex_t successor_color      = color[successor_color_idx];

      if (!mst_edge[e] && self_color != successor_color) {
        if (curr_edge_weight < min_edge_weight[lane_id]) {
          min_color[lane_id]       = successor_color;
          min_edge_weight[lane_id] = curr_edge_weight;
          min_edge_index[lane_id]  = e;
        }
      }
    }
  }
  __syncthreads();

  // reduce across threads in warp
  // each thread in warp holds min edge scanned by itself
  // reduce across all those warps
  for (int offset = 16; offset > 0; offset >>= 1) {
    if (lane_id < offset) {
      if (min_edge_weight[lane_id] > min_edge_weight[lane_id + offset]) {
        min_color[lane_id]       = min_color[lane_id + offset];
        min_edge_weight[lane_id] = min_edge_weight[lane_id + offset];
        min_edge_index[lane_id]  = min_edge_index[lane_id + offset];
      }
    }
    __syncthreads();
  }

  // min edge may now be found in first thread
  if (lane_id == 0) {
    if (min_edge_weight[0] != std::numeric_limits<alteration_t>::max()) {
      new_mst_edge[warp_id] = min_edge_index[0];

      // atomically set min edge per color
      // takes care of super vertex case
      atomicMin(&min_edge_color[self_color], min_edge_weight[0]);
    }
  }
}

template <typename vertex_t, typename edge_t, typename weight_t, typename alteration_t>
RAFT_KERNEL min_edge_per_supervertex(const vertex_t* color,
                                     const vertex_t* color_index,
                                     edge_t* new_mst_edge,
                                     bool* mst_edge,
                                     const vertex_t* indices,
                                     const weight_t* weights,
                                     const alteration_t* altered_weights,
                                     vertex_t* temp_src,
                                     vertex_t* temp_dst,
                                     weight_t* temp_weights,
                                     const alteration_t* min_edge_color,
                                     const vertex_t v,
                                     bool symmetrize_output)
{
  auto tid = get_1D_idx<vertex_t>();
  if (tid < v) {
    vertex_t vertex_color_idx = color_index[tid];
    vertex_t vertex_color     = color[vertex_color_idx];
    edge_t edge_idx           = new_mst_edge[tid];

    // check if valid outgoing edge was found
    // find minimum edge is same as minimum edge of whole supervertex
    // if yes, that is part of mst
    if (edge_idx != std::numeric_limits<edge_t>::max()) {
      alteration_t vertex_weight = altered_weights[edge_idx];

      bool add_edge = false;
      if (min_edge_color[vertex_color] == vertex_weight) {
        add_edge = true;

        auto dst = indices[edge_idx];
        if (!symmetrize_output) {
          auto dst_edge_idx = new_mst_edge[dst];
          auto dst_color    = color[color_index[dst]];

          // vertices added each other
          // only if destination has found an edge
          // the edge points back to source
          // the edge is minimum edge found for dst color
          if (dst_edge_idx != std::numeric_limits<edge_t>::max() && indices[dst_edge_idx] == tid &&
              min_edge_color[dst_color] == altered_weights[dst_edge_idx]) {
            if (vertex_color > dst_color) { add_edge = false; }
          }
        }

        if (add_edge) {
          temp_src[tid]      = tid;
          temp_dst[tid]      = dst;
          temp_weights[tid]  = weights[edge_idx];
          mst_edge[edge_idx] = true;
        }
      }

      if (!add_edge) { new_mst_edge[tid] = std::numeric_limits<edge_t>::max(); }
    }
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
RAFT_KERNEL add_reverse_edge(const edge_t* new_mst_edge,
                             const vertex_t* indices,
                             const weight_t* weights,
                             vertex_t* temp_src,
                             vertex_t* temp_dst,
                             weight_t* temp_weights,
                             const vertex_t v,
                             bool symmetrize_output)
{
  auto tid = get_1D_idx<vertex_t>();

  if (tid < v) {
    bool reverse_needed = false;

    edge_t edge_idx = new_mst_edge[tid];
    if (edge_idx != std::numeric_limits<edge_t>::max()) {
      vertex_t neighbor_vertex = indices[edge_idx];
      edge_t neighbor_edge_idx = new_mst_edge[neighbor_vertex];

      // if neighbor picked no vertex then reverse edge is
      // definitely needed
      if (neighbor_edge_idx == std::numeric_limits<edge_t>::max()) {
        reverse_needed = true;
      } else {
        // check what vertex the neighbor vertex picked
        if (symmetrize_output) {
          vertex_t neighbor_vertex_neighbor = indices[neighbor_edge_idx];

          // if vertices did not pick each other
          // add a reverse edge
          if (tid != neighbor_vertex_neighbor) { reverse_needed = true; }
        }
      }

      // if reverse was needed, add the edge
      if (reverse_needed) {
        // it is assumed the each vertex only picks one valid min edge
        // per cycle
        // hence, we store at index tid + v for the reverse edge scenario
        temp_src[tid + v]     = neighbor_vertex;
        temp_dst[tid + v]     = tid;
        temp_weights[tid + v] = weights[edge_idx];
      }
    }
  }
}

// executes for newly added mst edges and updates the colors of both vertices to the lower color
template <typename vertex_t, typename edge_t>
RAFT_KERNEL min_pair_colors(const vertex_t v,
                            const vertex_t* indices,
                            const edge_t* new_mst_edge,
                            const vertex_t* color,
                            const vertex_t* color_index,
                            vertex_t* next_color)
{
  auto i = get_1D_idx<vertex_t>();

  if (i < v) {
    edge_t edge_idx = new_mst_edge[i];

    if (edge_idx != std::numeric_limits<edge_t>::max()) {
      vertex_t neighbor_vertex = indices[edge_idx];
      // vertex_t self_color = color[i];
      vertex_t self_color_idx       = color_index[i];
      vertex_t self_color           = color[self_color_idx];
      vertex_t neighbor_color_idx   = color_index[neighbor_vertex];
      vertex_t neighbor_super_color = color[neighbor_color_idx];

      // update my own color as source of edge
      // update neighbour color index directly
      // this will ensure v1 updates supervertex color
      // while v2 will update the color of its supervertex
      // thus, allowing the colors to progress towards 0
      atomicMin(&next_color[self_color_idx], neighbor_super_color);
      atomicMin(&next_color[neighbor_color_idx], self_color);
    }
  }
}

// for each vertex, update color if it was changed in min_pair_colors kernel
template <typename vertex_t>
RAFT_KERNEL update_colors(const vertex_t v,
                          vertex_t* color,
                          const vertex_t* color_index,
                          const vertex_t* next_color,
                          bool* done)
{
  auto i = get_1D_idx<vertex_t>();

  if (i < v) {
    vertex_t self_color     = color[i];
    vertex_t self_color_idx = color_index[i];
    vertex_t new_color      = next_color[self_color_idx];

    // update self color to new smaller color
    if (self_color > new_color) {
      color[i] = new_color;
      *done    = false;
    }
  }
}

// point vertices to their final color index
template <typename vertex_t>
RAFT_KERNEL final_color_indices(const vertex_t v, const vertex_t* color, vertex_t* color_index)
{
  auto i = get_1D_idx<vertex_t>();

  if (i < v) {
    vertex_t self_color_idx = color_index[i];
    vertex_t self_color     = color[self_color_idx];

    // if self color is not equal to self color index,
    // it means self is not supervertex
    // in which case, iterate until we can find
    // parent supervertex
    while (self_color_idx != self_color) {
      self_color_idx = color_index[self_color];
      self_color     = color[self_color_idx];
    }

    // point to new supervertex
    color_index[i] = self_color_idx;
  }
}

// Alterate the weights, make all undirected edge weight unique while keeping Wuv == Wvu
// Consider using curand device API instead of precomputed random_values array
template <typename vertex_t, typename edge_t, typename weight_t, typename alteration_t>
RAFT_KERNEL alteration_kernel(const vertex_t v,
                              const edge_t e,
                              const edge_t* offsets,
                              const vertex_t* indices,
                              const weight_t* weights,
                              alteration_t max,
                              alteration_t* random_values,
                              alteration_t* altered_weights)
{
  auto row = get_1D_idx<vertex_t>();
  if (row < v) {
    auto row_begin = offsets[row];
    auto row_end   = offsets[row + 1];
    for (auto i = row_begin; i < row_end; i++) {
      auto column        = indices[i];
      altered_weights[i] = weights[i] + max * (random_values[row] + random_values[column]);
    }
  }
}

template <typename vertex_t, typename edge_t>
RAFT_KERNEL kernel_count_new_mst_edges(const vertex_t* mst_src,
                                       edge_t* mst_edge_count,
                                       const vertex_t v)
{
  auto tid = get_1D_idx<vertex_t>();

  // count number of new mst edges added
  bool predicate       = tid < v && (mst_src[tid] != std::numeric_limits<vertex_t>::max());
  vertex_t block_count = __syncthreads_count(predicate);

  if (threadIdx.x == 0 && block_count > 0) { atomicAdd(mst_edge_count, block_count); }
}

}  // namespace raft::sparse::solver::detail
