/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/neighbors/ball_cover_types.hpp>

namespace raft::runtime::neighbors::epsilon_neighborhood {

#define RAFT_INST_BFEPSN(IDX_T, DATA_T, MATRIX_IDX_T, INDEX_LAYOUT, SEARCH_LAYOUT)               \
  void eps_neighbors(raft::resources const& handle,                                              \
                     raft::device_matrix_view<const DATA_T, MATRIX_IDX_T, INDEX_LAYOUT> index,   \
                     raft::device_matrix_view<const DATA_T, MATRIX_IDX_T, SEARCH_LAYOUT> search, \
                     raft::device_matrix_view<bool, MATRIX_IDX_T, row_major> adj,                \
                     raft::device_vector_view<IDX_T, MATRIX_IDX_T> vd,                           \
                     DATA_T eps);

RAFT_INST_BFEPSN(int64_t, float, int64_t, raft::row_major, raft::row_major);

#undef RAFT_INST_BFEPSN

#define RAFT_INST_RBCEPSN(IDX_T, DATA_T, INT_T, MATRIX_IDX_T, INDEX_LAYOUT, SEARCH_LAYOUT)       \
  void eps_neighbors_rbc(                                                                        \
    raft::resources const& handle,                                                               \
    raft::device_matrix_view<const DATA_T, MATRIX_IDX_T, INDEX_LAYOUT> index,                    \
    raft::device_matrix_view<const DATA_T, MATRIX_IDX_T, SEARCH_LAYOUT> search,                  \
    raft::device_matrix_view<bool, MATRIX_IDX_T, row_major> adj,                                 \
    raft::device_vector_view<IDX_T, MATRIX_IDX_T> vd,                                            \
    DATA_T eps);                                                                                 \
  void build_rbc_index(                                                                          \
    raft::resources const& handle,                                                               \
    raft::neighbors::ball_cover::BallCoverIndex<IDX_T, DATA_T, INT_T, MATRIX_IDX_T>& rbc_index); \
  void eps_neighbors_rbc_pass1(                                                                  \
    raft::resources const& handle,                                                               \
    raft::neighbors::ball_cover::BallCoverIndex<IDX_T, DATA_T, INT_T, MATRIX_IDX_T> rbc_index,   \
    raft::device_matrix_view<const DATA_T, MATRIX_IDX_T, SEARCH_LAYOUT> search,                  \
    raft::device_vector_view<IDX_T, MATRIX_IDX_T> adj_ia,                                        \
    raft::device_vector_view<IDX_T, MATRIX_IDX_T> vd,                                            \
    DATA_T eps);                                                                                 \
  void eps_neighbors_rbc_pass2(                                                                  \
    raft::resources const& handle,                                                               \
    raft::neighbors::ball_cover::BallCoverIndex<IDX_T, DATA_T, INT_T, MATRIX_IDX_T> rbc_index,   \
    raft::device_matrix_view<const DATA_T, MATRIX_IDX_T, SEARCH_LAYOUT> search,                  \
    raft::device_vector_view<IDX_T, MATRIX_IDX_T> adj_ia,                                        \
    raft::device_vector_view<IDX_T, MATRIX_IDX_T> adj_ja,                                        \
    raft::device_vector_view<IDX_T, MATRIX_IDX_T> vd,                                            \
    DATA_T eps);

RAFT_INST_RBCEPSN(int64_t, float, int64_t, int64_t, raft::row_major, raft::row_major);

#undef RAFT_INST_RBCEPSN

}  // namespace raft::runtime::neighbors::epsilon_neighborhood
