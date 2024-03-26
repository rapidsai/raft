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

#include <raft/distance/distance_types.hpp>     // raft::distance::DistanceType
#include <raft/neighbors/ball_cover_types.hpp>  // BallCoverIndex
#include <raft/util/raft_explicit.hpp>          // RAFT_EXPLICIT

#include <cstdint>  // uint32_t

#ifdef RAFT_EXPLICIT_INSTANTIATE_ONLY

namespace raft::neighbors::ball_cover {

template <typename idx_t, typename value_t, typename int_t, typename matrix_idx_t>
void build_index(raft::resources const& handle,
                 BallCoverIndex<idx_t, value_t, int_t, matrix_idx_t>& index) RAFT_EXPLICIT;

template <typename idx_t, typename value_t, typename int_t, typename matrix_idx_t>
void all_knn_query(raft::resources const& handle,
                   BallCoverIndex<idx_t, value_t, int_t, matrix_idx_t>& index,
                   int_t k,
                   idx_t* inds,
                   value_t* dists,
                   bool perform_post_filtering = true,
                   float weight                = 1.0) RAFT_EXPLICIT;

template <typename idx_t, typename value_t, typename int_t, typename matrix_idx_t>
void all_knn_query(raft::resources const& handle,
                   BallCoverIndex<idx_t, value_t, int_t, matrix_idx_t>& index,
                   raft::device_matrix_view<idx_t, matrix_idx_t, row_major> inds,
                   raft::device_matrix_view<value_t, matrix_idx_t, row_major> dists,
                   int_t k,
                   bool perform_post_filtering = true,
                   float weight                = 1.0) RAFT_EXPLICIT;

template <typename idx_t, typename value_t, typename int_t>
void knn_query(raft::resources const& handle,
               const BallCoverIndex<idx_t, value_t, int_t>& index,
               int_t k,
               const value_t* query,
               int_t n_query_pts,
               idx_t* inds,
               value_t* dists,
               bool perform_post_filtering = true,
               float weight                = 1.0) RAFT_EXPLICIT;

template <typename idx_t, typename value_t, typename int_t, typename matrix_idx_t>
void knn_query(raft::resources const& handle,
               const BallCoverIndex<idx_t, value_t, int_t, matrix_idx_t>& index,
               raft::device_matrix_view<const value_t, matrix_idx_t, row_major> query,
               raft::device_matrix_view<idx_t, matrix_idx_t, row_major> inds,
               raft::device_matrix_view<value_t, matrix_idx_t, row_major> dists,
               int_t k,
               bool perform_post_filtering = true,
               float weight                = 1.0) RAFT_EXPLICIT;

template <typename idx_t, typename value_t, typename int_t, typename matrix_idx_t>
void eps_nn(raft::resources const& handle,
            const BallCoverIndex<idx_t, value_t, int_t, matrix_idx_t>& index,
            raft::device_matrix_view<bool, matrix_idx_t, row_major> adj,
            raft::device_vector_view<idx_t, matrix_idx_t> vd,
            raft::device_matrix_view<const value_t, matrix_idx_t, row_major> query,
            value_t eps) RAFT_EXPLICIT;

template <typename idx_t, typename value_t, typename int_t, typename matrix_idx_t>
void eps_nn(raft::resources const& handle,
            const BallCoverIndex<idx_t, value_t, int_t, matrix_idx_t>& index,
            raft::device_vector_view<idx_t, matrix_idx_t> adj_ia,
            raft::device_vector_view<idx_t, matrix_idx_t> adj_ja,
            raft::device_vector_view<idx_t, matrix_idx_t> vd,
            raft::device_matrix_view<const value_t, matrix_idx_t, row_major> query,
            value_t eps,
            std::optional<raft::host_scalar_view<int_t, matrix_idx_t>> max_k = std::nullopt)
  RAFT_EXPLICIT;

}  // namespace raft::neighbors::ball_cover

#endif  // RAFT_EXPLICIT_INSTANTIATE_ONLY

#define instantiate_raft_neighbors_ball_cover(idx_t, value_t, int_t, matrix_idx_t)                 \
  extern template void                                                                             \
  raft::neighbors::ball_cover::build_index<idx_t, value_t, int_t, matrix_idx_t>(                   \
    raft::resources const& handle,                                                                 \
    raft::neighbors::ball_cover::BallCoverIndex<idx_t, value_t, int_t, matrix_idx_t>& index);      \
                                                                                                   \
  extern template void                                                                             \
  raft::neighbors::ball_cover::all_knn_query<idx_t, value_t, int_t, matrix_idx_t>(                 \
    raft::resources const& handle,                                                                 \
    raft::neighbors::ball_cover::BallCoverIndex<idx_t, value_t, int_t, matrix_idx_t>& index,       \
    int_t k,                                                                                       \
    idx_t* inds,                                                                                   \
    value_t* dists,                                                                                \
    bool perform_post_filtering,                                                                   \
    float weight);                                                                                 \
                                                                                                   \
  extern template void raft::neighbors::ball_cover::eps_nn<idx_t, value_t, int_t, matrix_idx_t>(   \
    raft::resources const& handle,                                                                 \
    const raft::neighbors::ball_cover::BallCoverIndex<idx_t, value_t, int_t, matrix_idx_t>& index, \
    raft::device_matrix_view<bool, matrix_idx_t, row_major> adj,                                   \
    raft::device_vector_view<idx_t, matrix_idx_t> vd,                                              \
    raft::device_matrix_view<const value_t, matrix_idx_t, row_major> query,                        \
    value_t eps);                                                                                  \
                                                                                                   \
  extern template void raft::neighbors::ball_cover::eps_nn<idx_t, value_t, int_t, matrix_idx_t>(   \
    raft::resources const& handle,                                                                 \
    const raft::neighbors::ball_cover::BallCoverIndex<idx_t, value_t, int_t, matrix_idx_t>& index, \
    raft::device_vector_view<idx_t, matrix_idx_t> adj_ia,                                          \
    raft::device_vector_view<idx_t, matrix_idx_t> adj_ja,                                          \
    raft::device_vector_view<idx_t, matrix_idx_t> vd,                                              \
    raft::device_matrix_view<const value_t, matrix_idx_t, row_major> query,                        \
    value_t eps,                                                                                   \
    std::optional<raft::host_scalar_view<int_t, matrix_idx_t>> max_k);                             \
                                                                                                   \
  extern template void                                                                             \
  raft::neighbors::ball_cover::all_knn_query<idx_t, value_t, int_t, matrix_idx_t>(                 \
    raft::resources const& handle,                                                                 \
    raft::neighbors::ball_cover::BallCoverIndex<idx_t, value_t, int_t, matrix_idx_t>& index,       \
    raft::device_matrix_view<idx_t, matrix_idx_t, row_major> inds,                                 \
    raft::device_matrix_view<value_t, matrix_idx_t, row_major> dists,                              \
    int_t k,                                                                                       \
    bool perform_post_filtering,                                                                   \
    float weight);                                                                                 \
                                                                                                   \
  extern template void raft::neighbors::ball_cover::knn_query<idx_t, value_t, int_t>(              \
    raft::resources const& handle,                                                                 \
    const raft::neighbors::ball_cover::BallCoverIndex<idx_t, value_t, int_t>& index,               \
    int_t k,                                                                                       \
    const value_t* query,                                                                          \
    int_t n_query_pts,                                                                             \
    idx_t* inds,                                                                                   \
    value_t* dists,                                                                                \
    bool perform_post_filtering,                                                                   \
    float weight);                                                                                 \
                                                                                                   \
  extern template void                                                                             \
  raft::neighbors::ball_cover::knn_query<idx_t, value_t, int_t, matrix_idx_t>(                     \
    raft::resources const& handle,                                                                 \
    const raft::neighbors::ball_cover::BallCoverIndex<idx_t, value_t, int_t, matrix_idx_t>& index, \
    raft::device_matrix_view<const value_t, matrix_idx_t, row_major> query,                        \
    raft::device_matrix_view<idx_t, matrix_idx_t, row_major> inds,                                 \
    raft::device_matrix_view<value_t, matrix_idx_t, row_major> dists,                              \
    int_t k,                                                                                       \
    bool perform_post_filtering,                                                                   \
    float weight);

instantiate_raft_neighbors_ball_cover(int64_t, float, int64_t, int64_t);

#undef instantiate_raft_neighbors_ball_cover
