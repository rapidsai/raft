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

#include <bits/stdc++.h>

#include <gtest/gtest.h>
#include <rmm/thrust_rmm_allocator.h>
#include <iostream>
#include <vector>

#include <raft/handle.hpp>
#include <raft/cudart_utils.h>

#include <raft/sparse/mst.cuh>

template <typename vertex_t, typename edge_t, typename value_t>
struct CSRHost {
  std::vector<vertex_t> offsets;
  std::vector<edge_t> indices;
  std::vector<value_t> weights;
};

template <typename vertex_t, typename edge_t, typename value_t>
struct CSRDevice {
  rmm::device_vector<vertex_t> offsets;
  rmm::device_vector<edge_t> indices;
  rmm::device_vector<value_t> weights;
};

namespace raft {
namespace mst {

// Sequential prims function
// Returns total weight of MST
template <typename vertex_t, typename edge_t, typename value_t>
value_t prims(CSRHost<vertex_t, edge_t, value_t> &csr_h) {

  auto n_vertices = csr_h.offsets.size() - 1;

  bool active_vertex[n_vertices];
  // bool mst_set[csr_h.n_edges];
  value_t curr_edge[n_vertices];

  for (auto i = 0; i < n_vertices; i++) {
    active_vertex[i] = false;
    curr_edge[i] = INT_MAX;
  }
  curr_edge[0] = 0;

  // for (auto i = 0; i < csr_h.n_edges; i++) {
  //   mst_set[i] = false;
  // }

  // function to pick next min vertex-edge
  auto min_vertex_edge = [](auto *curr_edge, auto *active_vertex, auto n_vertices) {
    value_t min = INT_MAX;
    vertex_t min_vertex;

    for (auto v = 0; v < n_vertices; v++) {
      if (!active_vertex[v] && curr_edge[v] < min) {
        min = curr_edge[v];
        min_vertex = v;
      }
    }

    return min_vertex;
  };

  // iterate over n vertices
  for (auto v = 0; v < n_vertices - 1; v++) {
    // pick min vertex-edge
    vertex_t curr_v = min_vertex_edge(curr_edge, active_vertex, n_vertices);

    active_vertex[curr_v] = true; // set to active

    // iterate through edges of current active vertex
    auto edge_st = csr_h.offsets[curr_v];
    auto edge_end = csr_h.offsets[curr_v + 1];

    for (auto e = edge_st; e < edge_end; e++) {

      // put edges to be considered for next iteration
      auto neighbor_idx = csr_h.indices[e];
      if (!active_vertex[neighbor_idx] && csr_h.weights[e]) {
        curr_edge[neighbor_idx] = csr_h.weights[e];
      }

    }

  }

  // find sum of MST
  value_t total_weight = 0;
  for(auto v = 1; v < n_vertices; v++) {
    total_weight += curr_edge[v];
  }

  return total_weight;

}

template <typename vertex_t, typename edge_t, typename value_t>
class MSTTest : public ::testing::TestWithParam<CSRHost<vertex_t, edge_t, value_t>> {
  protected:
    void mst_sequential() {
      csr_h = ::testing::TestWithParam<CSRHost<vertex_t, edge_t, value_t>>::GetParam();

      // rmm::device_vector<vertex_t> mst_src;
      // rmm::device_vector<vertex_t> mst_dst;

      // MST_solver<vertex_t, edge_t, value_t> solver(handle, csr_d.offsets.data(), csr_d.indices.data(), csr_d.weights.data(), csr_d.offsets.size() - 1,
      //                                             csr_d.indices.size());

      // //nullptr expected to trigger exceptions
      // EXPECT_ANY_THROW(solver.solve(mst_src, mst_dst));
    }

    void SetUp() override {
      // csr_d.n_vertices = csr_h.n_vertices;
      // csr_d.n_edges = csr_h.n_edges;

      // CUDA_CHECK(cudaMalloc(&csr_d.offsets, csr_d.n_vertices * sizeof(vertex_t)));
      // CUDA_CHECK(cudaMalloc(&csr_d.indices, csr_d.n_edges * sizeof(vertex_t)));
      // CUDA_CHECK(cudaMalloc(&csr_d.weights, csr_d.n_edges * sizeof(vertex_t)));

      // raft::update_device(csr_d.offsets, csr_h.offsets, csr_h.n_vertices, handle.get_stream());
      // raft::update_device(csr_d.indices, csr_h.indices, csr_h.n_edges, handle.get_stream());
      // raft::update_device(csr_d.weights, csr_h.weights, csr_h.n_edges, handle.get_stream());
    }

    void TearDown() override { }
  
  protected:
    CSRHost<vertex_t, edge_t, value_t> csr_h;
    CSRDevice<vertex_t, edge_t, value_t> csr_d;

    raft::handle_t handle;
};

/*
Graph 1:
    2
(0) - (1)
3| 4\ 1|
(2)   (3)

*/

const std::vector<CSRHost<int, int, int>> csr_in_h = {
  // {nullptr, nullptr, nullptr, 0, 0},
  {
    {0, 3, 5, 7, 8},
    {1, 2, 3, 0, 3, 0, 0, 1},
    {2, 3, 4, 2, 1, 3, 4, 1}
  }
};

typedef MSTTest<int, int, int> MSTTestSequential;
TEST_P(MSTTestSequential, Sequential) {
  mst_sequential();

  // do assertions here
  // in this case, running sequential MST
  std::cout << prims(csr_h);
}

INSTANTIATE_TEST_SUITE_P(MSTTests, MSTTestSequential, ::testing::ValuesIn(csr_in_h));

}  // namespace mst
}  // namespace raft
