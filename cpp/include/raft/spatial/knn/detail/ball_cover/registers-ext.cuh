/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include "../../ball_cover_types.hpp"  // BallCoverIndex
#include "registers_types.cuh"         // DistFunc

#include <raft/util/raft_explicit.hpp>  //RAFT_EXPLICIT

#include <cstdint>  // uint32_t

#if defined(RAFT_EXPLICIT_INSTANTIATE_ONLY)

namespace raft::spatial::knn::detail {

template <typename value_idx,
          typename value_t,
          typename value_int    = std::int64_t,
          typename matrix_idx_t = std::int64_t,
          int dims              = 2,
          typename dist_func>
void rbc_low_dim_pass_one(raft::resources const& handle,
                          const BallCoverIndex<value_idx, value_t, value_int, matrix_idx_t>& index,
                          const value_t* query,
                          const value_int n_query_rows,
                          value_int k,
                          const value_idx* R_knn_inds,
                          const value_t* R_knn_dists,
                          dist_func& dfunc,
                          value_idx* inds,
                          value_t* dists,
                          float weight,
                          value_int* dists_counter) RAFT_EXPLICIT;

template <typename value_idx,
          typename value_t,
          typename value_int    = std::int64_t,
          typename matrix_idx_t = std::int64_t,
          int dims              = 2,
          typename dist_func>
void rbc_low_dim_pass_two(raft::resources const& handle,
                          const BallCoverIndex<value_idx, value_t, value_int, matrix_idx_t>& index,
                          const value_t* query,
                          const value_int n_query_rows,
                          value_int k,
                          const value_idx* R_knn_inds,
                          const value_t* R_knn_dists,
                          dist_func& dfunc,
                          value_idx* inds,
                          value_t* dists,
                          float weight,
                          value_int* post_dists_counter) RAFT_EXPLICIT;

template <typename value_idx,
          typename value_t,
          typename value_int    = std::int64_t,
          typename matrix_idx_t = std::int64_t,
          typename dist_func>
void rbc_eps_pass(raft::resources const& handle,
                  const BallCoverIndex<value_idx, value_t, value_int, matrix_idx_t>& index,
                  const value_t* query,
                  const value_int n_query_rows,
                  value_t eps,
                  const value_t* R_dists,
                  dist_func& dfunc,
                  bool* adj,
                  value_idx* vd) RAFT_EXPLICIT;

template <typename value_idx,
          typename value_t,
          typename value_int    = std::int64_t,
          typename matrix_idx_t = std::int64_t,
          typename dist_func>
void rbc_eps_pass(raft::resources const& handle,
                  const BallCoverIndex<value_idx, value_t, value_int, matrix_idx_t>& index,
                  const value_t* query,
                  const value_int n_query_rows,
                  value_t eps,
                  value_int* max_k,
                  const value_t* R_dists,
                  dist_func& dfunc,
                  value_idx* adj_ia,
                  value_idx* adj_ja,
                  value_idx* vd) RAFT_EXPLICIT;

};  // namespace raft::spatial::knn::detail

#endif  // RAFT_EXPLICIT_INSTANTIATE_ONLY

#define instantiate_raft_spatial_knn_detail_rbc_low_dim_pass_one(                 \
  Mvalue_idx, Mvalue_t, Mvalue_int, Mmatrix_idx, Mdims, Mdist_func)               \
  extern template void raft::spatial::knn::detail::                               \
    rbc_low_dim_pass_one<Mvalue_idx, Mvalue_t, Mvalue_int, Mmatrix_idx, Mdims>(   \
      raft::resources const& handle,                                              \
      const BallCoverIndex<Mvalue_idx, Mvalue_t, Mvalue_int, Mmatrix_idx>& index, \
      const Mvalue_t* query,                                                      \
      const Mvalue_int n_query_rows,                                              \
      Mvalue_int k,                                                               \
      const Mvalue_idx* R_knn_inds,                                               \
      const Mvalue_t* R_knn_dists,                                                \
      Mdist_func<Mvalue_t, Mvalue_int>& dfunc,                                    \
      Mvalue_idx* inds,                                                           \
      Mvalue_t* dists,                                                            \
      float weight,                                                               \
      Mvalue_int* dists_counter)

#define instantiate_raft_spatial_knn_detail_rbc_low_dim_pass_two(                 \
  Mvalue_idx, Mvalue_t, Mvalue_int, Mmatrix_idx, Mdims, Mdist_func)               \
  extern template void raft::spatial::knn::detail::                               \
    rbc_low_dim_pass_two<Mvalue_idx, Mvalue_t, Mvalue_int, Mmatrix_idx, Mdims>(   \
      raft::resources const& handle,                                              \
      const BallCoverIndex<Mvalue_idx, Mvalue_t, Mvalue_int, Mmatrix_idx>& index, \
      const Mvalue_t* query,                                                      \
      const Mvalue_int n_query_rows,                                              \
      Mvalue_int k,                                                               \
      const Mvalue_idx* R_knn_inds,                                               \
      const Mvalue_t* R_knn_dists,                                                \
      Mdist_func<Mvalue_t, Mvalue_int>& dfunc,                                    \
      Mvalue_idx* inds,                                                           \
      Mvalue_t* dists,                                                            \
      float weight,                                                               \
      Mvalue_int* dists_counter)

#define instantiate_raft_spatial_knn_detail_rbc_eps_pass(                                  \
  Mvalue_idx, Mvalue_t, Mvalue_int, Mmatrix_idx, Mdist_func)                               \
  extern template void                                                                     \
  raft::spatial::knn::detail::rbc_eps_pass<Mvalue_idx, Mvalue_t, Mvalue_int, Mmatrix_idx>( \
    raft::resources const& handle,                                                         \
    const BallCoverIndex<Mvalue_idx, Mvalue_t, Mvalue_int, Mmatrix_idx>& index,            \
    const Mvalue_t* query,                                                                 \
    const Mvalue_int n_query_rows,                                                         \
    Mvalue_t eps,                                                                          \
    const Mvalue_t* R_dists,                                                               \
    Mdist_func<Mvalue_t, Mvalue_int>& dfunc,                                               \
    bool* adj,                                                                             \
    Mvalue_idx* vd);                                                                       \
                                                                                           \
  extern template void                                                                     \
  raft::spatial::knn::detail::rbc_eps_pass<Mvalue_idx, Mvalue_t, Mvalue_int, Mmatrix_idx>( \
    raft::resources const& handle,                                                         \
    const BallCoverIndex<Mvalue_idx, Mvalue_t, Mvalue_int, Mmatrix_idx>& index,            \
    const Mvalue_t* query,                                                                 \
    const Mvalue_int n_query_rows,                                                         \
    Mvalue_t eps,                                                                          \
    Mvalue_int* max_k,                                                                     \
    const Mvalue_t* R_dists,                                                               \
    Mdist_func<Mvalue_t, Mvalue_int>& dfunc,                                               \
    Mvalue_idx* adj_ia,                                                                    \
    Mvalue_idx* adj_ja,                                                                    \
    Mvalue_idx* vd);

instantiate_raft_spatial_knn_detail_rbc_low_dim_pass_one(
  std::int64_t, float, std::int64_t, std::int64_t, 2, raft::spatial::knn::detail::HaversineFunc);
instantiate_raft_spatial_knn_detail_rbc_low_dim_pass_one(
  std::int64_t, float, std::int64_t, std::int64_t, 3, raft::spatial::knn::detail::HaversineFunc);
instantiate_raft_spatial_knn_detail_rbc_low_dim_pass_one(
  std::int64_t, float, std::int64_t, std::int64_t, 2, raft::spatial::knn::detail::EuclideanFunc);
instantiate_raft_spatial_knn_detail_rbc_low_dim_pass_one(
  std::int64_t, float, std::int64_t, std::int64_t, 3, raft::spatial::knn::detail::EuclideanFunc);
instantiate_raft_spatial_knn_detail_rbc_low_dim_pass_one(
  std::int64_t, float, std::int64_t, std::int64_t, 2, raft::spatial::knn::detail::DistFunc);
instantiate_raft_spatial_knn_detail_rbc_low_dim_pass_one(
  std::int64_t, float, std::int64_t, std::int64_t, 3, raft::spatial::knn::detail::DistFunc);

instantiate_raft_spatial_knn_detail_rbc_low_dim_pass_two(
  std::int64_t, float, std::int64_t, std::int64_t, 2, raft::spatial::knn::detail::HaversineFunc);
instantiate_raft_spatial_knn_detail_rbc_low_dim_pass_two(
  std::int64_t, float, std::int64_t, std::int64_t, 3, raft::spatial::knn::detail::HaversineFunc);
instantiate_raft_spatial_knn_detail_rbc_low_dim_pass_two(
  std::int64_t, float, std::int64_t, std::int64_t, 2, raft::spatial::knn::detail::EuclideanFunc);
instantiate_raft_spatial_knn_detail_rbc_low_dim_pass_two(
  std::int64_t, float, std::int64_t, std::int64_t, 3, raft::spatial::knn::detail::EuclideanFunc);
instantiate_raft_spatial_knn_detail_rbc_low_dim_pass_two(
  std::int64_t, float, std::int64_t, std::int64_t, 2, raft::spatial::knn::detail::DistFunc);
instantiate_raft_spatial_knn_detail_rbc_low_dim_pass_two(
  std::int64_t, float, std::int64_t, std::int64_t, 3, raft::spatial::knn::detail::DistFunc);

instantiate_raft_spatial_knn_detail_rbc_eps_pass(
  std::int64_t, float, std::int64_t, std::int64_t, raft::spatial::knn::detail::EuclideanSqFunc);

#undef instantiate_raft_spatial_knn_detail_rbc_low_dim_pass_two
#undef instantiate_raft_spatial_knn_detail_rbc_low_dim_pass_one
#undef instantiate_raft_spatial_knn_detail_rbc_eps_pass
