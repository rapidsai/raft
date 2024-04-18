/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
/**
 * This file is deprecated and will be removed in release 22.06.
 * Please use the cuh version instead.
 */

/**
 * DISCLAIMER: this file is deprecated: use epsilon_neighborhood.cuh instead
 */

#pragma once

#ifndef RAFT_HIDE_DEPRECATION_WARNINGS
#pragma message(__FILE__                                                    \
                  " is deprecated and will be removed in a future release." \
                  " Please use the raft::neighbors version instead.")
#endif

#include <raft/neighbors/ball_cover.cuh>
#include <raft/spatial/knn/ball_cover_types.hpp>

namespace raft::spatial::knn {

template <typename idx_t, typename value_t, typename int_t, typename matrix_idx_t>
void rbc_build_index(raft::resources const& handle,
                     BallCoverIndex<idx_t, value_t, int_t, matrix_idx_t>& index)
{
  raft::neighbors::ball_cover::build_index(handle, index);
}

template <typename idx_t, typename value_t, typename int_t, typename matrix_idx_t>
void rbc_all_knn_query(raft::resources const& handle,
                       BallCoverIndex<idx_t, value_t, int_t, matrix_idx_t>& index,
                       int_t k,
                       idx_t* inds,
                       value_t* dists,
                       bool perform_post_filtering = true,
                       float weight                = 1.0)
{
  raft::neighbors::ball_cover::all_knn_query(
    handle, index, k, inds, dists, perform_post_filtering, weight);
}

template <typename idx_t, typename value_t, typename int_t>
void rbc_knn_query(raft::resources const& handle,
                   const BallCoverIndex<idx_t, value_t, int_t>& index,
                   int_t k,
                   const value_t* query,
                   int_t n_query_pts,
                   idx_t* inds,
                   value_t* dists,
                   bool perform_post_filtering = true,
                   float weight                = 1.0)
{
  raft::neighbors::ball_cover::knn_query(
    handle, index, k, query, n_query_pts, inds, dists, perform_post_filtering, weight);
}
}  // namespace raft::spatial::knn
