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

#include "mst_kernels.cuh"
#include "utils.cuh"

namespace raft {
namespace mst {

template <typename vertex_t, typename edge_t, typename weight_t>
MST_solver<vertex_t, edge_t, weight_t>::MST_solver(
  const raft::handle_t& handle_, vertex_t const* offsets_,
  vertex_t const* indices_, weight_t const* weights_, vertex_t const v_,
  vertex_t const e_)
  : handle(handle_),
    offsets(offsets_),
    indices(indices_),
    weights(weights_),
    v(v_),
    e(e_),
    color(v_),
    next_color(v_),
    active_color(v_),
    successor(v_),
    mst_edge(e_, false) {
  max_blocks = handle_.get_device_properties().maxGridSize[0];
  max_threads = handle_.get_device_properties().maxThreadsPerBlock;
  sm_count = handle_.get_device_properties().multiProcessorCount;

  //Initially, color holds the vertex id as color
  thrust::sequence(color.begin(), color.end());
  //Initially, each next_color redirects to its own color
  thrust::sequence(next_color.begin(), next_color.end());
  //Initially, each edge is not in the mst
}

template <typename vertex_t, typename edge_t, typename weight_t>
void MST_solver<vertex_t, edge_t, weight_t>::solve() {
  RAFT_EXPECTS(v > 0, "0 vertices");
  RAFT_EXPECTS(e > 0, "0 edges");
  RAFT_EXPECTS(offsets != nullptr, "Null offsets.");
  RAFT_EXPECTS(indices != nullptr, "Null indices.");
  RAFT_EXPECTS(weights != nullptr, "Null weights.");

  min_edge_per_color();

  detail::printv(successor);

  label_prop();

  detail::printv(color);

  // Theorem : the minimum incident edge to any vertex has to be in the MST
  // This is a segmented min scan/reduce
  // cub::KeyValuePair<vertex_t, weight_t>* d_out = nullptr;
  // void* cub_temp_storage = nullptr;
  // size_t cub_temp_storage_bytes = 0;
  // cub::DeviceSegmentedReduce::ArgMin(cub_temp_storage, cub_temp_storage_bytes,
  //                                    weights, d_out, v, offsets, offsets + 1);
  // // FIXME RMM Allocate temporary storage
  // cudaMalloc(&cub_temp_storage, cub_temp_storage_bytes);
  // // Run argmin-reduction
  // cub::DeviceSegmentedReduce::ArgMin(cub_temp_storage, cub_temp_storage_bytes,
  //                                    weights, d_out, v, offsets, offsets + 1);
  //
  // TODO: mst[offset[i]+key[i]]=true; (thrust)?
  // Extract MST edge list by just filtering with the mask generated above?

  // bool mst_edge_found = true;
  // Boruvka original formulation says "while more than 1 supervertex remains"
  // Here we adjust it to support disconnected components (spanning forest)
  // track completion with mst_edge_found status.
  // should have max_iter ensure it always exits.
  // for (auto i = 0; i < v; i++) {
  //   {
  // updates colors of supervertices by propagating the lower color to the higher
  // TODO

  // Finds the minimum outgoing edge from each supervertex to the lowest outgoing color
  // by working at each vertex of the supervertex
  // TODO
  // segmented min with an extra check to discard edges leading to the same color

  // filter internal edges / remove cycles
  // TODO

  // done
  //     if (!mst_edge_found) break;
  //   }
  // }
}

template <typename vertex_t, typename edge_t, typename weight_t>
void MST_solver<vertex_t, edge_t, weight_t>::label_prop() {
  // update the colors of both ends its until there is no change in colors
  int nthreads = std::min(v, max_threads);
  int nblocks = std::min((v + nthreads - 1) / nthreads, max_blocks);
  auto stream = handle.get_stream();

  rmm::device_vector<bool> done(1, false);
  vertex_t* color_ptr = thrust::raw_pointer_cast(color.data());
  vertex_t* next_color_ptr = thrust::raw_pointer_cast(next_color.data());
  vertex_t* successor_ptr = thrust::raw_pointer_cast(successor.data());

  bool* done_ptr = thrust::raw_pointer_cast(done.data());

  auto i = 0;
  std::cout << "==================" << std::endl;
  detail::printv(color);
  while (!done[0]) {
    done[0] = true;
    detail::min_pair_colors<<<nblocks, nthreads, 0, stream>>>(
      v, successor_ptr, color_ptr, next_color_ptr);
    detail::printv(next_color);
    detail::check_color_change<<<nblocks, nthreads, 0, stream>>>(
      v, color_ptr, next_color_ptr, done_ptr);
    detail::printv(color);
    i++;
  }
  std::cout << "Label prop iterations : " << i << std::endl;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void MST_solver<vertex_t, edge_t, weight_t>::min_edge_per_color() {
  auto stream = handle.get_stream();
  int n_threads = 32;

  vertex_t* color_ptr = thrust::raw_pointer_cast(color.data());
  vertex_t* successor_ptr = thrust::raw_pointer_cast(successor.data());
  bool* mst_edge_ptr = thrust::raw_pointer_cast(mst_edge.data());

  detail::kernel_min_edge_per_color<<<v, n_threads, 0, stream>>>(
    offsets, indices, weights, color_ptr, successor_ptr, mst_edge_ptr, v);
}

}  // namespace mst
}  // namespace raft