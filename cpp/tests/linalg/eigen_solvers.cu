/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/resource/device_id.hpp>
#include <raft/core/resources.hpp>
#include <raft/spectral/eigen_solvers.cuh>
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

TEST(Raft, EigenSolvers)
{
  common::nvtx::range fun_scope("test::EigenSolvers");
  using namespace matrix;
  using index_type = int;
  using value_type = double;
  using nnz_type   = int;

  raft::resources h;
  ASSERT_EQ(0, resource::get_device_id(h));

  index_type* ro{nullptr};
  index_type* ci{nullptr};
  value_type* vs{nullptr};
  index_type nnz   = 0;
  index_type nrows = 0;

  sparse_matrix_t<index_type, value_type, nnz_type> sm1{h, ro, ci, vs, nrows, nnz};
  ASSERT_EQ(nullptr, sm1.row_offsets_);

  index_type neigvs{10};
  index_type maxiter{100};
  index_type restart_iter{10};
  value_type tol{1.0e-10};
  bool reorthog{true};

  // nullptr expected to trigger exceptions:
  //
  value_type* eigvals{nullptr};
  value_type* eigvecs{nullptr};
  std::uint64_t seed{100110021003};

  eigen_solver_config_t<index_type, value_type> cfg{
    neigvs, maxiter, restart_iter, tol, reorthog, seed};

  lanczos_solver_t<index_type, value_type, nnz_type> eig_solver{cfg};

  EXPECT_ANY_THROW(eig_solver.solve_smallest_eigenvectors(h, sm1, eigvals, eigvecs));

  EXPECT_ANY_THROW(eig_solver.solve_largest_eigenvectors(h, sm1, eigvals, eigvecs));
}

TEST(Raft, SpectralSolvers)
{
  common::nvtx::range fun_scope("test::SpectralSolvers");
  using namespace matrix;
  using index_type = int;
  using value_type = double;
  using nnz_type   = int;

  raft::resources h;
  ASSERT_EQ(0, resource::get_device_id(h)

  );

  index_type neigvs{10};
  index_type maxiter{100};
  index_type restart_iter{10};
  value_type tol{1.0e-10};
  bool reorthog{true};

  // nullptr expected to trigger exceptions:
  //
  index_type* clusters{nullptr};
  value_type* eigvals{nullptr};
  value_type* eigvecs{nullptr};

  unsigned long long seed{100110021003};

  eigen_solver_config_t<index_type, value_type> eig_cfg{
    neigvs, maxiter, restart_iter, tol, reorthog, seed};
  lanczos_solver_t<index_type, value_type, nnz_type> eig_solver{eig_cfg};

  index_type k{5};

  cluster_solver_config_t<index_type, value_type> clust_cfg{k, maxiter, tol, seed};
  kmeans_solver_t<index_type, value_type> cluster_solver{clust_cfg};

  sparse_matrix_t<index_type, value_type, nnz_type> sm{h, nullptr, nullptr, nullptr, 0, 0};
  EXPECT_ANY_THROW(
    spectral::partition(h, sm, eig_solver, cluster_solver, clusters, eigvals, eigvecs));

  value_type edgeCut{0};
  value_type cost{0};
  EXPECT_ANY_THROW(spectral::analyzePartition(h, sm, k, clusters, edgeCut, cost));
}

TEST(Raft, SpectralPartition)
{
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

  auto result_v = thrust::device_vector<int>(std::vector<int>(num_verts, -1));

  auto constexpr const n_clusters  = 8;
  auto constexpr const n_eig_vects = std::uint64_t{8};

  auto constexpr const evs_tolerance   = .00001f;
  auto constexpr const kmean_tolerance = .00001f;
  auto constexpr const evs_max_iter    = 100;
  auto constexpr const kmean_max_iter  = 100;

  auto eig_vals  = thrust::device_vector<float>(n_eig_vects);
  auto eig_vects = thrust::device_vector<float>(n_eig_vects * num_verts);

  auto handle = raft::handle_t{};

  auto restartIter_lanczos = int{15 + n_eig_vects};

  auto seed_eig_solver     = std::uint64_t{1234567};
  auto seed_cluster_solver = std::uint64_t{12345678};

  auto const adj_matrix = raft::spectral::matrix::sparse_matrix_t<int, float>{handle,
                                                                              offsets.data().get(),
                                                                              indices.data().get(),
                                                                              values.data().get(),
                                                                              num_verts,
                                                                              num_verts,
                                                                              num_edges};

  auto eig_cfg = raft::spectral::eigen_solver_config_t<int, float>{
    n_eig_vects, evs_max_iter, restartIter_lanczos, evs_tolerance, false, seed_eig_solver};
  auto eigen_solver = raft::spectral::lanczos_solver_t<int, float>{eig_cfg};

  auto clust_cfg = raft::spectral::cluster_solver_config_t<int, float>{
    n_clusters, kmean_max_iter, kmean_tolerance, seed_cluster_solver};
  auto cluster_solver = raft::spectral::kmeans_solver_t<int, float>{clust_cfg};

  partition(handle,
            adj_matrix,
            eigen_solver,
            cluster_solver,
            result_v.data().get(),
            eig_vals.data().get(),
            eig_vects.data().get());

  auto edge_cut = float{};
  auto cost     = float{};

  raft::spectral::analyzePartition(
    handle, adj_matrix, n_clusters, result_v.data().get(), edge_cut, cost);

  ASSERT_LT(edge_cut, 55.0);
}

}  // namespace spectral
}  // namespace raft
