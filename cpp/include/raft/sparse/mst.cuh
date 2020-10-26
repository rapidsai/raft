
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
#include <raft/cudart_utils.h>
#include <rmm/thrust_rmm_allocator.h>
#include <algorithm>
#include <cstddef>
#include <cub/cub.cuh>
#include <raft/error.hpp>
#include <raft/handle.hpp>

namespace raft {
namespace mst {

template <typename vertex_t, typename edge_t, typename weight_t>
class MST_solver {
 private:
  raft::handle_t const& handle;

  //CSR
  const vertex_t* offsets;
  const vertex_t* indices;
  const weight_t* weights;
  const vertex_t v;
  const vertex_t e;

  int max_blocks;
  int max_threads;
  int sm_count;

  rmm::device_vector<vertex_t> color;  // represent each supervertex as a color
  rmm::device_vector<vertex_t> color_idx;  //index of v color in color array
  rmm::device_vector<bool> active_color;   // track active supervertex color
  rmm::device_vector<vertex_t> degree;     // supervertices degrees
  rmm::device_vector<vertex_t> cycle;      // edges to be excluded from mst_edge
  rmm::device_vector<vertex_t> new_mst_edge_idx;  // current mst iteration
  rmm::device_vector<vertex_t> mst_edge_idx;      // mst output

 public:
  MST_solver(const raft::handle_t& handle_, vertex_t const* offsets_,
             vertex_t const* indices_, weight_t const* weights_,
             vertex_t const v_, vertex_t const e_);

  void solve(rmm::device_vector<vertex_t>& mst_src,
             rmm::device_vector<vertex_t>& mst_dst);

  ~MST_solver() {}
};

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
    color_idx(v_),
    active_color(v_),
    degree(v_),
    cycle(v_),
    new_mst_edge_idx(v_),
    mst_edge_idx(e_) {
  max_blocks = handle_.get_device_properties().maxGridSize[0];
  max_threads = handle_.get_device_properties().maxThreadsPerBlock;
  sm_count = handle_.get_device_properties().multiProcessorCount;

  //Initially, color holds the vertex id as color
  thrust::sequence(color.begin(), color.end());
  //Initially, each color_idx redirects to its own color
  thrust::sequence(color_idx.begin(), color_idx.end());
}

template <typename vertex_t, typename edge_t, typename weight_t>
__global__ void kernel_min_edge_per_vertex() {

  unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;

  unsigned warp_id = tid / 32;
  unsigned lane_id = tid % 32;

  if (tid < size) {
    // one row is associated with one warp
    row_start = offsets[warp_id];
    row_end = offsets[warp_id + 1];

    // reduce for min edge across warp
    edge_t min_edge_index = maxval;
    weight_t min_edge_weight = maxval;

    // assuming one warp per row
    for (unsigned e = row_start + lane_id, mid = 16; e < row_end; e += 32, mid >>= 1) {
      if (lane_id < mid) {
        if (weights[e] < weights[e + mid]) {
          min_edge_weight = weights[e];
          min_edge_index = e;

          successor[warp_id] = indices[e];
        }
        else if (weights[e] == weights[e + mid]) {
          // tie break
        }
      }
    }
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
void MST_solver<vertex_t, edge_t, weight_t>::solve(
  rmm::device_vector<vertex_t>& mst_src,
  rmm::device_vector<vertex_t>& mst_dst) {
  RAFT_EXPECTS(v > 0, "0 vertices");
  RAFT_EXPECTS(e > 0, "0 edges");
  RAFT_EXPECTS(offsets != nullptr, "Null offsets.");
  RAFT_EXPECTS(indices != nullptr, "Null indices.");
  RAFT_EXPECTS(weights != nullptr, "Null weights.");

  int nthreads = std::min(v, max_threads);
  int nblocks = std::min((v + nthreads - 1) / nthreads, max_blocks);
  auto stream = handle.get_stream();

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
  // TODO set component color : color[i] = min(i, offset[i]+key[i])
  // TODO: mst_idx[offset[i]+key[i]]=true; (thrust)

  bool mst_edge_found = true;
  // Boruvka original formulation says "while more than 1 supervertex remains"
  // Here we adjust it to support disconnected components (spanning forest)
  // track completion with mst_edge_found status.
  // should have max_iter ensure it always exits.
  for (auto i = 0; i < v; i++) {
    {
      // updates colors of supervertices by propagating the lower color to the higher
      // TODO

      // Finds the minimum outgoing edge from each supervertex to the lowest outgoing color
      // by working at each vertex of the supervertex
      // TODO
      // segmented min with an extra check to discard edges leading to the same color

      // filter internal edges / remove cycles
      // TODO

      // done
      if (!mst_edge_found) break;
    }
  }
}
}  // namespace mst
}  // namespace raft