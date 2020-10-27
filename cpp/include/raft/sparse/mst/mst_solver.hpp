
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

namespace raft {
namespace mst {

template <typename vertex_t, typename edge_t, typename weight_t>
class MST_solver {
 public:
  MST_solver(const raft::handle_t& handle_, vertex_t const* offsets_,
             vertex_t const* indices_, weight_t const* weights_,
             vertex_t const v_, vertex_t const e_);

  void solve();

  ~MST_solver() {}

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
  void min_edge_per_color();
};

}  // namespace mst
}  // namespace raft

#include "detail/mst_solver_inl.hpp"
