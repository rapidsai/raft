/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#undef RAFT_EXPLICIT_INSTANTIATE_ONLY

#include <cstdint>
#include <raft/neighbors/ball_cover-inl.cuh>

#define instantiate_raft_neighbors_ball_cover(idx_t, value_t, int_t, matrix_idx_t)                 \
  template void raft::neighbors::ball_cover::build_index<idx_t, value_t, int_t, matrix_idx_t>(     \
    raft::resources const& handle,                                                                 \
    raft::neighbors::ball_cover::BallCoverIndex<idx_t, value_t, int_t, matrix_idx_t>& index);      \
                                                                                                   \
  template void                                                                                    \
  raft::neighbors::ball_cover::epsUnexpL2NeighborhoodRbc<idx_t, value_t, int_t, matrix_idx_t>(     \
    raft::resources const& handle,                                                                 \
    const raft::neighbors::ball_cover::BallCoverIndex<idx_t, value_t, int_t, matrix_idx_t>& index, \
    bool* adj,                                                                                     \
    idx_t* vd,                                                                                     \
    const value_t* x,                                                                              \
    int_t m,                                                                                       \
    int_t n,                                                                                       \
    value_t eps);                                                                                  \
                                                                                                   \
  template void                                                                                    \
  raft::neighbors::ball_cover::epsUnexpL2NeighborhoodRbc<idx_t, value_t, int_t, matrix_idx_t>(     \
    raft::resources const& handle,                                                                 \
    const raft::neighbors::ball_cover::BallCoverIndex<idx_t, value_t, int_t, matrix_idx_t>& index, \
    idx_t* ia,                                                                                     \
    idx_t* ja,                                                                                     \
    idx_t* vd,                                                                                     \
    const value_t* x,                                                                              \
    int_t m,                                                                                       \
    int_t n,                                                                                       \
    value_t eps,                                                                                   \
    int_t* max_k);                                                                                 \
                                                                                                   \
  template void raft::neighbors::ball_cover::all_knn_query<idx_t, value_t, int_t, matrix_idx_t>(   \
    raft::resources const& handle,                                                                 \
    raft::neighbors::ball_cover::BallCoverIndex<idx_t, value_t, int_t, matrix_idx_t>& index,       \
    int_t k,                                                                                       \
    idx_t* inds,                                                                                   \
    value_t* dists,                                                                                \
    bool perform_post_filtering,                                                                   \
    float weight);                                                                                 \
                                                                                                   \
  template void raft::neighbors::ball_cover::all_knn_query<idx_t, value_t, int_t, matrix_idx_t>(   \
    raft::resources const& handle,                                                                 \
    raft::neighbors::ball_cover::BallCoverIndex<idx_t, value_t, int_t, matrix_idx_t>& index,       \
    raft::device_matrix_view<idx_t, matrix_idx_t, row_major> inds,                                 \
    raft::device_matrix_view<value_t, matrix_idx_t, row_major> dists,                              \
    int_t k,                                                                                       \
    bool perform_post_filtering,                                                                   \
    float weight);                                                                                 \
                                                                                                   \
  template void raft::neighbors::ball_cover::knn_query<idx_t, value_t, int_t>(                     \
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
  template void raft::neighbors::ball_cover::knn_query<idx_t, value_t, int_t, matrix_idx_t>(       \
    raft::resources const& handle,                                                                 \
    const raft::neighbors::ball_cover::BallCoverIndex<idx_t, value_t, int_t, matrix_idx_t>& index, \
    raft::device_matrix_view<const value_t, matrix_idx_t, row_major> query,                        \
    raft::device_matrix_view<idx_t, matrix_idx_t, row_major> inds,                                 \
    raft::device_matrix_view<value_t, matrix_idx_t, row_major> dists,                              \
    int_t k,                                                                                       \
    bool perform_post_filtering,                                                                   \
    float weight);

instantiate_raft_neighbors_ball_cover(int64_t, float, uint32_t, uint32_t);

#undef instantiate_raft_neighbors_ball_cover

#define instantiate_raft_neighbors_ball_cover_eps(idx_t, value_t, int_t, matrix_idx_t)             \
  template void raft::neighbors::ball_cover::build_index<idx_t, value_t, int_t, matrix_idx_t>(     \
    raft::resources const& handle,                                                                 \
    raft::neighbors::ball_cover::BallCoverIndex<idx_t, value_t, int_t, matrix_idx_t>& index);      \
                                                                                                   \
  template void                                                                                    \
  raft::neighbors::ball_cover::epsUnexpL2NeighborhoodRbc<idx_t, value_t, int_t, matrix_idx_t>(     \
    raft::resources const& handle,                                                                 \
    const raft::neighbors::ball_cover::BallCoverIndex<idx_t, value_t, int_t, matrix_idx_t>& index, \
    bool* adj,                                                                                     \
    idx_t* vd,                                                                                     \
    const value_t* x,                                                                              \
    int_t m,                                                                                       \
    int_t n,                                                                                       \
    value_t eps);                                                                                  \
                                                                                                   \
  template void                                                                                    \
  raft::neighbors::ball_cover::epsUnexpL2NeighborhoodRbc<idx_t, value_t, int_t, matrix_idx_t>(     \
    raft::resources const& handle,                                                                 \
    const raft::neighbors::ball_cover::BallCoverIndex<idx_t, value_t, int_t, matrix_idx_t>& index, \
    idx_t* ia,                                                                                     \
    idx_t* ja,                                                                                     \
    idx_t* vd,                                                                                     \
    const value_t* x,                                                                              \
    int_t m,                                                                                       \
    int_t n,                                                                                       \
    value_t eps,                                                                                   \
    int_t* max_k);

instantiate_raft_neighbors_ball_cover_eps(int64_t, float, int64_t, uint32_t);
instantiate_raft_neighbors_ball_cover_eps(int64_t, double, int64_t, uint32_t);
instantiate_raft_neighbors_ball_cover_eps(int32_t, float, int32_t, uint32_t);
instantiate_raft_neighbors_ball_cover_eps(int32_t, double, int32_t, uint32_t);
instantiate_raft_neighbors_ball_cover_eps(int64_t, float, int64_t, int64_t);
instantiate_raft_neighbors_ball_cover_eps(int64_t, double, int64_t, int64_t);

#undef instantiate_raft_neighbors_ball_cover_eps