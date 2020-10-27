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
#include <rmm/device_buffer.hpp>
#include <vector>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>

#include <raft/sparse/mst/mst.cuh>

template <typename vertex_t, typename edge_t, typename weight_t>
struct CSRHost {
  std::vector<vertex_t> offsets;
  std::vector<edge_t> indices;
  std::vector<weight_t> weights;
};

template <typename vertex_t, typename edge_t, typename weight_t>
struct CSRDevice {
  rmm::device_buffer offsets;
  rmm::device_buffer indices;
  rmm::device_buffer weights;
};

namespace raft {
namespace mst {

// Sequential prims function
// Returns total weight of MST
template <typename vertex_t, typename edge_t, typename weight_t>
weight_t prims(CSRHost<vertex_t, edge_t, weight_t> &csr_h) {
  auto n_vertices = csr_h.offsets.size() - 1;

  bool active_vertex[n_vertices];
  // bool mst_set[csr_h.n_edges];
  weight_t curr_edge[n_vertices];

  for (auto i = 0; i < n_vertices; i++) {
    active_vertex[i] = false;
    curr_edge[i] = INT_MAX;
  }
  curr_edge[0] = 0;

  // for (auto i = 0; i < csr_h.n_edges; i++) {
  //   mst_set[i] = false;
  // }

  // function to pick next min vertex-edge
  auto min_vertex_edge = [](auto *curr_edge, auto *active_vertex,
                            auto n_vertices) {
    weight_t min = INT_MAX;
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
    auto curr_v = min_vertex_edge(curr_edge, active_vertex, n_vertices);

    active_vertex[curr_v] = true;  // set to active

    // iterate through edges of current active vertex
    auto edge_st = csr_h.offsets[curr_v];
    auto edge_end = csr_h.offsets[curr_v + 1];

    for (auto e = edge_st; e < edge_end; e++) {
      // put edges to be considered for next iteration
      auto neighbor_idx = csr_h.indices[e];
      if (!active_vertex[neighbor_idx] &&
          csr_h.weights[e] < curr_edge[neighbor_idx]) {
        curr_edge[neighbor_idx] = csr_h.weights[e];
      }
    }
  }

  // find sum of MST
  weight_t total_weight = 0;
  for (auto v = 1; v < n_vertices; v++) {
    total_weight += curr_edge[v];
  }

  return total_weight;
}

template <typename vertex_t, typename edge_t, typename weight_t>
class MSTTest
  : public ::testing::TestWithParam<CSRHost<vertex_t, edge_t, weight_t>> {
 protected:
  void mst_sequential() {
    rmm::device_vector<vertex_t> mst_src;
    rmm::device_vector<vertex_t> mst_dst;

    vertex_t *offsets = static_cast<vertex_t *>(csr_d.offsets.data());
    edge_t *indices = static_cast<edge_t *>(csr_d.indices.data());
    weight_t *weights = static_cast<weight_t *>(csr_d.weights.data());

    auto v =
      static_cast<vertex_t>((csr_d.offsets.size() / sizeof(weight_t)) - 1);
    auto e = static_cast<edge_t>(csr_d.indices.size() / sizeof(edge_t));

    mst<vertex_t, edge_t, weight_t>(handle, offsets, indices, weights, v, e);
  }

  void SetUp() override {
    csr_h =
      ::testing::TestWithParam<CSRHost<vertex_t, edge_t, weight_t>>::GetParam();

    csr_d.offsets = rmm::device_buffer(csr_h.offsets.data(),
                                       csr_h.offsets.size() * sizeof(vertex_t));
    csr_d.indices = rmm::device_buffer(csr_h.indices.data(),
                                       csr_h.indices.size() * sizeof(edge_t));
    csr_d.weights = rmm::device_buffer(csr_h.weights.data(),
                                       csr_h.weights.size() * sizeof(weight_t));
  }

  void TearDown() override {}

 protected:
  CSRHost<vertex_t, edge_t, weight_t> csr_h;
  CSRDevice<vertex_t, edge_t, weight_t> csr_d;

  raft::handle_t handle;
};

/*
Graph 1:
    2
(0) - (1)
3| 4\ 1|
(2)   (3)

*/

const std::vector<CSRHost<int, int, float>> csr_in_h = {
  {{0, 3, 5, 7, 8}, {1, 2, 3, 0, 3, 0, 0, 1}, {2, 3, 4, 2, 1, 3, 4, 1}}};

const std::vector<CSRHost<int, int, float>> csr_in2_h = {
  {{0, 4, 6, 9, 12, 15, 17, 20},
   {2, 4, 5, 6, 3, 6, 0, 4, 5, 1, 4, 6, 0, 2, 3, 0, 2, 0, 1, 3},
   {5.0, 9.0, 1.0, 4.0, 8.0, 7.0, 5.0, 2.0, 6.0, 8.0,
    3.0, 4.0, 9.0, 2.0, 3.0, 1.0, 6.0, 4.0, 7.0, 10.0}}};

typedef MSTTest<int, int, float> MSTTestSequential;
TEST_P(MSTTestSequential, Sequential) {
  mst_sequential();

  // do assertions here
  // in this case, running sequential MST
  // std::cout << prims(csr_h);
}

INSTANTIATE_TEST_SUITE_P(MSTTests, MSTTestSequential,
                         ::testing::ValuesIn(csr_in_h));

}  // namespace mst
}  // namespace raft
