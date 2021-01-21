
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

#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

namespace raft {

template <typename vertex_t, typename edge_t, typename weight_t>
struct Graph_COO {
  rmm::device_uvector<vertex_t> src;
  rmm::device_uvector<vertex_t> dst;
  rmm::device_uvector<weight_t> weights;
  edge_t n_edges;

  Graph_COO(vertex_t size, cudaStream_t stream)
    : src(size, stream), dst(size, stream), weights(size, stream) {}
};

namespace mst {

template <typename vertex_t, typename edge_t, typename weight_t>
class MST_solver {
 public:
  MST_solver(const raft::handle_t& handle_, const edge_t* offsets_,
             const vertex_t* indices_, const weight_t* weights_,
             const vertex_t v_, const edge_t e_, vertex_t* color_,
             cudaStream_t stream_);

  raft::Graph_COO<vertex_t, edge_t, weight_t> solve();

  ~MST_solver() {}

 private:
  const raft::handle_t& handle;
  cudaStream_t stream;

  //CSR
  const edge_t* offsets;
  const vertex_t* indices;
  const weight_t* weights;
  const vertex_t v;
  const edge_t e;

  vertex_t max_blocks;
  vertex_t max_threads;
  vertex_t sm_count;

  vertex_t* color;  // represent each supervertex as a color
  rmm::device_vector<weight_t>
    min_edge_color;  // minimum incident edge weight per color
  rmm::device_vector<edge_t> new_mst_edge;       // new minimum edge per vertex
  rmm::device_vector<weight_t> altered_weights;  // weights to be used for mst
  rmm::device_vector<edge_t>
    mst_edge_count;  // total number of edges added after every iteration
  rmm::device_vector<edge_t>
    prev_mst_edge_count;  // total number of edges up to the previous iteration
  rmm::device_vector<bool>
    mst_edge;  // mst output -  true if the edge belongs in mst
  rmm::device_vector<vertex_t> next_color;  //  next iteration color
  rmm::device_vector<vertex_t>
    color_index;  // index of color that vertex points to

  // new src-dst pairs found per iteration
  rmm::device_vector<vertex_t> temp_src;
  rmm::device_vector<vertex_t> temp_dst;
  rmm::device_vector<weight_t> temp_weights;

  void label_prop(vertex_t* mst_src, vertex_t* mst_dst);
  void min_edge_per_vertex();
  void min_edge_per_supervertex();
  void check_termination();
  void alteration();
  weight_t alteration_max();
  void append_src_dst_pair(vertex_t* mst_src, vertex_t* mst_dst,
                           weight_t* mst_weights);
};

}  // namespace mst
}  // namespace raft

#include "detail/mst_solver_inl.cuh"
