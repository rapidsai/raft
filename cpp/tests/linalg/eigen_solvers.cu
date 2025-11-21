/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/resource/device_id.hpp>
#include <raft/core/resources.hpp>
#include <raft/spectral/modularity_maximization.cuh>
#include <raft/spectral/partition.cuh>

#include <thrust/device_vector.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <iostream>
#include <memory>
#include <type_traits>

namespace raft {
namespace spectral {

TEST(Raft, SpectralAnalyzers)
{
  common::nvtx::range fun_scope("test::SpectralAnalyzers");
  using namespace matrix;
  using index_type = int;
  using value_type = double;
  using nnz_type   = int;

  raft::resources h;
  ASSERT_EQ(0, resource::get_device_id(h));

  index_type* clusters{nullptr};
  index_type k{5};

  sparse_matrix_t<index_type, value_type, nnz_type> sm{h, nullptr, nullptr, nullptr, 0, 0};

  value_type edgeCut{0};
  value_type cost{0};
  EXPECT_ANY_THROW(spectral::analyzePartition(h, sm, k, clusters, edgeCut, cost));
}

TEST(Raft, AnalyzePartition)
{
  // Test analyzePartition with manually created cluster assignments
  // Karate club graph (34 vertices)
  auto offsets = thrust::device_vector<int>(std::vector<int>{
    0,  16, 25, 35, 41, 44, 48, 52,  56,  61,  63,  66,  67,  69,  74,  76,  78, 80,
    82, 84, 87, 89, 91, 93, 98, 101, 104, 106, 110, 113, 117, 121, 127, 139, 156});
  auto indices = thrust::device_vector<int>(std::vector<int>{
    1,  2,  3,  4,  5,  6,  7,  8,  10, 11, 12, 13, 17, 19, 21, 31, 0,  2,  3,  7,  13, 17, 19,
    21, 30, 0,  1,  3,  7,  8,  9,  13, 27, 28, 32, 0,  1,  2,  7,  12, 13, 0,  6,  10, 0,  6,
    10, 16, 0,  4,  5,  16, 0,  1,  2,  3,  0,  2,  30, 32, 33, 2,  33, 0,  4,  5,  0,  0,  3,
    0,  1,  2,  3,  33, 32, 33, 32, 33, 5,  6,  0,  1,  32, 33, 0,  1,  33, 32, 33, 0,  1,  32,
    33, 25, 27, 29, 32, 33, 25, 27, 31, 23, 24, 31, 29, 33, 2,  23, 24, 33, 2,  31, 33, 23, 26,
    32, 33, 1,  8,  32, 33, 0,  24, 25, 28, 32, 33, 2,  8,  14, 15, 18, 20, 22, 23, 29, 30, 31,
    33, 8,  9,  13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32});
  auto values  = thrust::device_vector<float>(std::vector<float>{
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});

  auto num_verts = int(offsets.size() - 1);
  auto num_edges = indices.size();

  // Create cluster assignments: 4 clusters, vertices assigned cyclically
  auto constexpr const n_clusters = 4;
  auto clusters                   = thrust::device_vector<int>(num_verts);
  for (int i = 0; i < num_verts; ++i) {
    clusters[i] = i % n_clusters;
  }

  auto handle = raft::handle_t{};

  auto const adj_matrix = raft::spectral::matrix::sparse_matrix_t<int, float>{handle,
                                                                              offsets.data().get(),
                                                                              indices.data().get(),
                                                                              values.data().get(),
                                                                              num_verts,
                                                                              num_verts,
                                                                              num_edges};

  auto edge_cut = float{};
  auto cost     = float{};

  // Test analyzePartition - should not throw
  EXPECT_NO_THROW(raft::spectral::analyzePartition(
    handle, adj_matrix, n_clusters, clusters.data().get(), edge_cut, cost));

  // Verify outputs are reasonable (not NaN, not negative)
  EXPECT_FALSE(std::isnan(edge_cut));
  EXPECT_FALSE(std::isnan(cost));
  EXPECT_GE(edge_cut, 0.0f);
  EXPECT_GE(cost, 0.0f);
}

TEST(Raft, AnalyzeModularity)
{
  // Test analyzeModularity with manually created cluster assignments
  // Karate club graph (34 vertices)
  auto offsets = thrust::device_vector<int>(std::vector<int>{
    0,  16, 25, 35, 41, 44, 48, 52,  56,  61,  63,  66,  67,  69,  74,  76,  78, 80,
    82, 84, 87, 89, 91, 93, 98, 101, 104, 106, 110, 113, 117, 121, 127, 139, 156});
  auto indices = thrust::device_vector<int>(std::vector<int>{
    1,  2,  3,  4,  5,  6,  7,  8,  10, 11, 12, 13, 17, 19, 21, 31, 0,  2,  3,  7,  13, 17, 19,
    21, 30, 0,  1,  3,  7,  8,  9,  13, 27, 28, 32, 0,  1,  2,  7,  12, 13, 0,  6,  10, 0,  6,
    10, 16, 0,  4,  5,  16, 0,  1,  2,  3,  0,  2,  30, 32, 33, 2,  33, 0,  4,  5,  0,  0,  3,
    0,  1,  2,  3,  33, 32, 33, 32, 33, 5,  6,  0,  1,  32, 33, 0,  1,  33, 32, 33, 0,  1,  32,
    33, 25, 27, 29, 32, 33, 25, 27, 31, 23, 24, 31, 29, 33, 2,  23, 24, 33, 2,  31, 33, 23, 26,
    32, 33, 1,  8,  32, 33, 0,  24, 25, 28, 32, 33, 2,  8,  14, 15, 18, 20, 22, 23, 29, 30, 31,
    33, 8,  9,  13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32});
  auto values  = thrust::device_vector<float>(std::vector<float>{
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});

  auto num_verts = int(offsets.size() - 1);
  auto num_edges = indices.size();

  // Create cluster assignments: 4 clusters, vertices assigned cyclically
  auto constexpr const n_clusters = 4;
  auto clusters                   = thrust::device_vector<int>(num_verts);
  for (int i = 0; i < num_verts; ++i) {
    clusters[i] = i % n_clusters;
  }

  auto handle = raft::handle_t{};

  auto const adj_matrix = raft::spectral::matrix::sparse_matrix_t<int, float>{handle,
                                                                              offsets.data().get(),
                                                                              indices.data().get(),
                                                                              values.data().get(),
                                                                              num_verts,
                                                                              num_verts,
                                                                              num_edges};

  auto modularity = float{};

  // Test analyzeModularity - should not throw
  EXPECT_NO_THROW(raft::spectral::analyzeModularity(
    handle, adj_matrix, n_clusters, clusters.data().get(), modularity));

  // Verify output is reasonable (not NaN, typically in range [-1, 1])
  EXPECT_FALSE(std::isnan(modularity));
  EXPECT_GE(modularity, -1.0f);
  EXPECT_LE(modularity, 1.0f);
}

}  // namespace spectral
}  // namespace raft
