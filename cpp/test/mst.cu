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

#include "test_utils.h"

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
  raft::Graph_COO<vertex_t, edge_t, weight_t> mst_sequential() {
    vertex_t *offsets = static_cast<vertex_t *>(csr_d.offsets.data());
    edge_t *indices = static_cast<edge_t *>(csr_d.indices.data());
    weight_t *weights = static_cast<weight_t *>(csr_d.weights.data());

    v = static_cast<vertex_t>((csr_d.offsets.size() / sizeof(weight_t)) - 1);
    e = static_cast<edge_t>(csr_d.indices.size() / sizeof(edge_t));

    rmm::device_vector<vertex_t> mst_src(2 * v - 2,
                                         std::numeric_limits<vertex_t>::max());
    rmm::device_vector<vertex_t> mst_dst(2 * v - 2,
                                         std::numeric_limits<vertex_t>::max());
    rmm::device_vector<vertex_t> color(v, 0);

    vertex_t *color_ptr = thrust::raw_pointer_cast(color.data());

    MST_solver<vertex_t, edge_t, weight_t> mst_solver(
      handle, offsets, indices, weights, v, e, color_ptr, handle.get_stream());
    auto result = mst_solver.solve();
    raft::print_device_vector("Final MST Src: ", result.src.data(),
                              result.n_edges, std::cout);
    raft::print_device_vector("Final MST Dst: ", result.dst.data(),
                              result.n_edges, std::cout);
    raft::print_device_vector("Final MST Weights: ", result.weights.data(),
                              result.n_edges, std::cout);
    raft::print_device_vector("Final MST Colors: ", color_ptr, v, std::cout);

    std::cout << "number_of_MST_edges: " << result.n_edges << std::endl;
    EXPECT_LE(result.n_edges, 2 * v - 2);

    return result;
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
  rmm::device_vector<bool> mst_edge;
  vertex_t v;
  edge_t e;

  raft::handle_t handle;
};

// connected components tests
// a full MST is produced
const std::vector<CSRHost<int, int, float>> csr_in_h = {
  // single iteration
  {{0, 3, 5, 7, 8}, {1, 2, 3, 0, 3, 0, 0, 1}, {2, 3, 4, 2, 1, 3, 4, 1}},

  //  multiple iterations and cycles
  {{0, 4, 6, 9, 12, 15, 17, 20},
   {2, 4, 5, 6, 3, 6, 0, 4, 5, 1, 4, 6, 0, 2, 3, 0, 2, 0, 1, 3},
   {5.0f, 9.0f,  1.0f, 4.0f, 8.0f, 7.0f, 5.0f, 2.0f, 6.0f, 8.0f,
    1.0f, 10.0f, 9.0f, 2.0f, 1.0f, 1.0f, 6.0f, 4.0f, 7.0f, 10.0f}},
  // negative weights
  {{0, 4, 6, 9, 12, 15, 17, 20},
   {2, 4, 5, 6, 3, 6, 0, 4, 5, 1, 4, 6, 0, 2, 3, 0, 2, 0, 1, 3},
   {-5.0f, -9.0f,  -1.0f, 4.0f,  -8.0f, -7.0f, -5.0f, -2.0f, -6.0f, -8.0f,
    -1.0f, -10.0f, -9.0f, -2.0f, -1.0f, -1.0f, -6.0f, 4.0f,  -7.0f, -10.0f}},

  // equal weights
  {{0, 4, 6, 9, 12, 15, 17, 20},
   {2, 4, 5, 6, 3, 6, 0, 4, 5, 1, 4, 6, 0, 2, 3, 0, 2, 0, 1, 3},
   {0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.2, 0.2, 0.2,
    0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1}},

  //self loop
  {{0, 4, 6, 9, 12, 15, 17, 20},
   {0, 4, 5, 6, 3, 6, 2, 4, 5, 1, 4, 6, 0, 2, 3, 0, 2, 0, 1, 3},
   {0.5f, 9.0f,  1.0f, 4.0f, 8.0f, 7.0f, 0.5f, 2.0f, 6.0f, 8.0f,
    1.0f, 10.0f, 9.0f, 2.0f, 1.0f, 1.0f, 6.0f, 4.0f, 7.0f, 10.0f}},
};

//  disconnected
const std::vector<CSRHost<int, int, float>> csr_in4_h = {
  {{0, 3, 5, 8, 10, 12, 14, 16},
   {2, 4, 5, 3, 6, 0, 4, 5, 1, 6, 0, 2, 0, 2, 1, 3},
   {5.0f, 9.0f, 1.0f, 8.0f, 7.0f, 5.0f, 2.0f, 6.0f, 8.0f, 10.0f, 9.0f, 2.0f,
    1.0f, 6.0f, 7.0f, 10.0f}}};

//  singletons
const std::vector<CSRHost<int, int, float>> csr_in5_h = {
  {{0, 3, 5, 8, 10, 10, 10, 12, 14, 16, 16},
   {2, 8, 7, 3, 8, 0, 8, 7, 1, 8, 0, 2, 0, 2, 1, 3},
   {5.0f, 9.0f, 1.0f, 8.0f, 7.0f, 5.0f, 2.0f, 6.0f, 8.0f, 10.0f, 9.0f, 2.0f,
    1.0f, 6.0f, 7.0f, 10.0f}}};

typedef MSTTest<int, int, float> MSTTestSequential;
TEST_P(MSTTestSequential, Sequential) {
  auto gpu_result = mst_sequential();

  // do assertions here
  // in this case, running sequential MST
  auto prims_result = prims(csr_h);

  auto parallel_mst_result =
    thrust::reduce(thrust::device, gpu_result.weights.data(),
                   gpu_result.weights.data() + gpu_result.n_edges);

  ASSERT_TRUE(raft::match(2 * prims_result, parallel_mst_result,
                          raft::CompareApprox<float>(0.1)));
}

INSTANTIATE_TEST_SUITE_P(MSTTests, MSTTestSequential,
                         ::testing::ValuesIn(csr_in_h));

}  // namespace mst
}  // namespace raft
