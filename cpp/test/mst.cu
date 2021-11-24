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
#include <iostream>
#include <rmm/device_buffer.hpp>
#include <vector>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>

#include <raft/sparse/mst/mst.cuh>

#include "test_utils.h"

template <typename vertex_t, typename edge_t, typename weight_t>
struct CSRHost {
  std::vector<edge_t> offsets;
  std::vector<vertex_t> indices;
  std::vector<weight_t> weights;
};

template <typename vertex_t, typename edge_t, typename weight_t>
struct MSTTestInput {
  struct CSRHost<vertex_t, edge_t, weight_t> csr_h;
  int iterations;
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
  //  bool mst_set[csr_h.n_edges];
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
  : public ::testing::TestWithParam<MSTTestInput<vertex_t, edge_t, weight_t>> {
 protected:
  std::pair<raft::Graph_COO<vertex_t, edge_t, weight_t>,
            raft::Graph_COO<vertex_t, edge_t, weight_t>>
  mst_gpu() {
    edge_t *offsets = static_cast<edge_t *>(csr_d.offsets.data());
    vertex_t *indices = static_cast<vertex_t *>(csr_d.indices.data());
    weight_t *weights = static_cast<weight_t *>(csr_d.weights.data());

    v = static_cast<vertex_t>((csr_d.offsets.size() / sizeof(vertex_t)) - 1);
    e = static_cast<edge_t>(csr_d.indices.size() / sizeof(edge_t));

    rmm::device_vector<vertex_t> mst_src(2 * v - 2,
                                         std::numeric_limits<vertex_t>::max());
    rmm::device_vector<vertex_t> mst_dst(2 * v - 2,
                                         std::numeric_limits<vertex_t>::max());
    rmm::device_vector<vertex_t> color(v, 0);

    vertex_t *color_ptr = thrust::raw_pointer_cast(color.data());

    if (iterations == 0) {
      MST_solver<vertex_t, edge_t, weight_t, float> symmetric_solver(
        handle, offsets, indices, weights, v, e, color_ptr, handle.get_stream(),
        true, true, 0);
      auto symmetric_result = symmetric_solver.solve();

      MST_solver<vertex_t, edge_t, weight_t, float> non_symmetric_solver(
        handle, offsets, indices, weights, v, e, color_ptr, handle.get_stream(),
        false, true, 0);
      auto non_symmetric_result = non_symmetric_solver.solve();

      EXPECT_LE(symmetric_result.n_edges, 2 * v - 2);
      EXPECT_LE(non_symmetric_result.n_edges, v - 1);

      return std::make_pair(std::move(symmetric_result),
                            std::move(non_symmetric_result));
    } else {
      MST_solver<vertex_t, edge_t, weight_t, float> intermediate_solver(
        handle, offsets, indices, weights, v, e, color_ptr, handle.get_stream(),
        true, true, iterations);
      auto intermediate_result = intermediate_solver.solve();

      MST_solver<vertex_t, edge_t, weight_t, float> symmetric_solver(
        handle, offsets, indices, weights, v, e, color_ptr, handle.get_stream(),
        true, false, 0);
      auto symmetric_result = symmetric_solver.solve();

      // symmetric_result.n_edges += intermediate_result.n_edges;
      auto total_edge_size =
        symmetric_result.n_edges + intermediate_result.n_edges;
      symmetric_result.src.resize(total_edge_size, handle.get_stream());
      symmetric_result.dst.resize(total_edge_size, handle.get_stream());
      symmetric_result.weights.resize(total_edge_size, handle.get_stream());

      raft::copy(symmetric_result.src.data() + symmetric_result.n_edges,
                 intermediate_result.src.data(), intermediate_result.n_edges,
                 handle.get_stream());
      raft::copy(symmetric_result.dst.data() + symmetric_result.n_edges,
                 intermediate_result.dst.data(), intermediate_result.n_edges,
                 handle.get_stream());
      raft::copy(symmetric_result.weights.data() + symmetric_result.n_edges,
                 intermediate_result.weights.data(),
                 intermediate_result.n_edges, handle.get_stream());
      symmetric_result.n_edges = total_edge_size;

      MST_solver<vertex_t, edge_t, weight_t, float> non_symmetric_solver(
        handle, offsets, indices, weights, v, e, color_ptr, handle.get_stream(),
        false, true, 0);
      auto non_symmetric_result = non_symmetric_solver.solve();

      EXPECT_LE(symmetric_result.n_edges, 2 * v - 2);
      EXPECT_LE(non_symmetric_result.n_edges, v - 1);

      return std::make_pair(std::move(symmetric_result),
                            std::move(non_symmetric_result));
    }
  }

  void SetUp() override {
    mst_input = ::testing::TestWithParam<
      MSTTestInput<vertex_t, edge_t, weight_t>>::GetParam();
    iterations = mst_input.iterations;

    csr_d.offsets = rmm::device_buffer(
      mst_input.csr_h.offsets.data(),
      mst_input.csr_h.offsets.size() * sizeof(edge_t), handle.get_stream());
    csr_d.indices = rmm::device_buffer(
      mst_input.csr_h.indices.data(),
      mst_input.csr_h.indices.size() * sizeof(vertex_t), handle.get_stream());
    csr_d.weights = rmm::device_buffer(
      mst_input.csr_h.weights.data(),
      mst_input.csr_h.weights.size() * sizeof(weight_t), handle.get_stream());
  }

  void TearDown() override {}

 protected:
  MSTTestInput<vertex_t, edge_t, weight_t> mst_input;
  CSRDevice<vertex_t, edge_t, weight_t> csr_d;
  rmm::device_vector<bool> mst_edge;
  vertex_t v;
  edge_t e;
  int iterations;

  raft::handle_t handle;
};

// connected components tests
// a full MST is produced
const std::vector<MSTTestInput<int, int, float>> csr_in_h = {
  // single iteration
  {{{0, 3, 5, 7, 8}, {1, 2, 3, 0, 3, 0, 0, 1}, {2, 3, 4, 2, 1, 3, 4, 1}}, 0},

  //  multiple iterations and cycles
  {{{0, 4, 6, 9, 12, 15, 17, 20},
    {2, 4, 5, 6, 3, 6, 0, 4, 5, 1, 4, 6, 0, 2, 3, 0, 2, 0, 1, 3},
    {5.0f, 9.0f,  1.0f, 4.0f, 8.0f, 7.0f, 5.0f, 2.0f, 6.0f, 8.0f,
     1.0f, 10.0f, 9.0f, 2.0f, 1.0f, 1.0f, 6.0f, 4.0f, 7.0f, 10.0f}},
   1},
  // negative weights
  {{{0, 4, 6, 9, 12, 15, 17, 20},
    {2, 4, 5, 6, 3, 6, 0, 4, 5, 1, 4, 6, 0, 2, 3, 0, 2, 0, 1, 3},
    {-5.0f, -9.0f,  -1.0f, 4.0f,  -8.0f, -7.0f, -5.0f, -2.0f, -6.0f, -8.0f,
     -1.0f, -10.0f, -9.0f, -2.0f, -1.0f, -1.0f, -6.0f, 4.0f,  -7.0f, -10.0f}},
   0},

  // // equal weights
  {{{0, 4, 6, 9, 12, 15, 17, 20},
    {2, 4, 5, 6, 3, 6, 0, 4, 5, 1, 4, 6, 0, 2, 3, 0, 2, 0, 1, 3},
    {0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.2, 0.2, 0.2,
     0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1}},
   0},

  // //self loop
  {{{0, 4, 6, 9, 12, 15, 17, 20},
    {0, 4, 5, 6, 3, 6, 2, 4, 5, 1, 4, 6, 0, 2, 3, 0, 2, 0, 1, 3},
    {0.5f, 9.0f,  1.0f, 4.0f, 8.0f, 7.0f, 0.5f, 2.0f, 6.0f, 8.0f,
     1.0f, 10.0f, 9.0f, 2.0f, 1.0f, 1.0f, 6.0f, 4.0f, 7.0f, 10.0f}},
   0}};

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
  auto results_pair = mst_gpu();
  auto &symmetric_result = results_pair.first;
  auto &non_symmetric_result = results_pair.second;

  // do assertions here
  // in this case, running sequential MST
  auto prims_result = prims(mst_input.csr_h);

  auto symmetric_sum =
    thrust::reduce(thrust::device, symmetric_result.weights.data(),
                   symmetric_result.weights.data() + symmetric_result.n_edges);
  auto non_symmetric_sum = thrust::reduce(
    thrust::device, non_symmetric_result.weights.data(),
    non_symmetric_result.weights.data() + non_symmetric_result.n_edges);

  ASSERT_TRUE(raft::match(2 * prims_result, symmetric_sum,
                          raft::CompareApprox<float>(0.1)));
  ASSERT_TRUE(raft::match(prims_result, non_symmetric_sum,
                          raft::CompareApprox<float>(0.1)));
}

INSTANTIATE_TEST_SUITE_P(MSTTests, MSTTestSequential,
                         ::testing::ValuesIn(csr_in_h));

}  // namespace mst
}  // namespace raft
