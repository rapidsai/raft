
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
#include <limits>
#include <raft/error.hpp>
#include <raft/handle.hpp>

namespace raft {
namespace mst {

//FIXME this should live elswhere
template <typename T>
void printv(rmm::device_vector<T>& vec) {
  std::cout.precision(15);
  std::cout << "Size = " << vec.size() << std::endl;
  thrust::copy(vec.begin(), vec.end(),
               std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl;
}

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
  rmm::device_vector<vertex_t> next_color;  //index of v color in color array
  rmm::device_vector<bool> active_color;    // track active supervertex color
  //rmm::device_vector<vertex_t> degree;     // supervertices degrees
  //rmm::device_vector<vertex_t> cycle;      // edges to be excluded from mst_edge
  rmm::device_vector<vertex_t>
    successor;  // current mst iteration. edge being added is (src=i, dst=successor[i])
  rmm::device_vector<bool>
    mst_edge;  // mst output -  true if the edge belongs in mst
  rmm::device_vector<edge_t> min_edge_color;  // minimum incident edge per color

  void label_prop();

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
    next_color(v_),
    active_color(v_),
    successor(v_),
    mst_edge(e_, false),
    min_edge_color(v_, 100) {
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

  // min_edge_index[lane_id] = std::numeric_limits<edge_t>::max();
  // min_edge_weight[lane_id] = std::numeric_limits<weight_t>::max();
  // min_color[lane_id] = std::numeric_limits<vertex_t>::max();

  // TODO: Find a way to set limits
  // Above does not work as it is host code
  min_edge_index[lane_id] = 100;
  min_edge_weight[lane_id] = 100;
  min_color[lane_id] = 100;
  __syncthreads();

  vertex_t self_color = color[tid];

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
    if (min_edge_weight[0] != 100) {
      successor[warp_id] = indices[min_edge_index[0]];
    }
  }
}

// TODO make this work in 64bit
__device__ int get_1D_idx() { return blockIdx.x * blockDim.x + threadIdx.x; }

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
  printv(color);
  while (!done[0]) {
    done[0] = true;
    min_pair_colors<<<nblocks, nthreads, 0, stream>>>(
      v, successor_ptr, color_ptr, next_color_ptr);
    printv(next_color);
    check_color_change<<<nblocks, nthreads, 0, stream>>>(
      v, color_ptr, next_color_ptr, done_ptr);
    printv(color);
    i++;
  }
  std::cout << "Label prop iterations : " << i << std::endl;
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

  auto stream = handle.get_stream();

  kernel_min_edge_per_vertex<<<v, 32, 0, stream>>>(
    offsets, indices, weights, thrust::raw_pointer_cast(color.data()),
    thrust::raw_pointer_cast(successor.data()),
    thrust::raw_pointer_cast(mst_edge.data()), v);

  printv(successor);

  label_prop();

  printv(color);

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
}  // namespace mst
}  // namespace raft