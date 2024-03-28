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

#pragma once

#include <raft/core/device_mdspan.hpp>       // raft::device_matrix_view
#include <raft/core/operators.hpp>           // raft::identity_op
#include <raft/core/resources.hpp>           // raft::resources
#include <raft/distance/distance_types.hpp>  // raft::distance::DistanceType
#include <raft/neighbors/brute_force_types.hpp>
#include <raft/util/raft_explicit.hpp>  // RAFT_EXPLICIT

#include <optional>

#ifdef RAFT_EXPLICIT_INSTANTIATE_ONLY

namespace raft::neighbors::brute_force {

template <typename value_t, typename idx_t>
inline void knn_merge_parts(
  raft::resources const& handle,
  raft::device_matrix_view<const value_t, idx_t, row_major> in_keys,
  raft::device_matrix_view<const idx_t, idx_t, row_major> in_values,
  raft::device_matrix_view<value_t, idx_t, row_major> out_keys,
  raft::device_matrix_view<idx_t, idx_t, row_major> out_values,
  size_t n_samples,
  std::optional<raft::device_vector_view<idx_t, idx_t>> translations = std::nullopt) RAFT_EXPLICIT;

template <typename T, typename Accessor>
index<T> build(raft::resources const& res,
               mdspan<const T, matrix_extent<int64_t>, row_major, Accessor> dataset,
               raft::distance::DistanceType metric = distance::DistanceType::L2Unexpanded,
               T metric_arg                        = 0.0) RAFT_EXPLICIT;

template <typename T, typename Accessor>
index<T> build(raft::resources const& res,
               index_params const& params,
               mdspan<const T, matrix_extent<int64_t>, row_major, Accessor> dataset) RAFT_EXPLICIT;

template <typename T, typename IdxT>
void search(raft::resources const& res,
            const index<T>& idx,
            raft::device_matrix_view<const T, int64_t, row_major> queries,
            raft::device_matrix_view<IdxT, int64_t, row_major> neighbors,
            raft::device_matrix_view<T, int64_t, row_major> distances) RAFT_EXPLICIT;

template <typename T, typename IdxT>
void search(raft::resources const& res,
            search_params const& params,
            const index<T>& idx,
            raft::device_matrix_view<const T, int64_t, row_major> queries,
            raft::device_matrix_view<IdxT, int64_t, row_major> neighbors,
            raft::device_matrix_view<T, int64_t, row_major> distances) RAFT_EXPLICIT;

template <typename idx_t,
          typename value_t,
          typename matrix_idx,
          typename index_layout,
          typename search_layout,
          typename epilogue_op = raft::identity_op>
void knn(raft::resources const& handle,
         std::vector<raft::device_matrix_view<const value_t, matrix_idx, index_layout>> index,
         raft::device_matrix_view<const value_t, matrix_idx, search_layout> search,
         raft::device_matrix_view<idx_t, matrix_idx, row_major> indices,
         raft::device_matrix_view<value_t, matrix_idx, row_major> distances,
         distance::DistanceType metric         = distance::DistanceType::L2Unexpanded,
         std::optional<float> metric_arg       = std::make_optional<float>(2.0f),
         std::optional<idx_t> global_id_offset = std::nullopt,
         epilogue_op distance_epilogue         = raft::identity_op()) RAFT_EXPLICIT;

template <typename value_t, typename idx_t, typename idx_layout, typename query_layout>
void fused_l2_knn(raft::resources const& handle,
                  raft::device_matrix_view<const value_t, idx_t, idx_layout> index,
                  raft::device_matrix_view<const value_t, idx_t, query_layout> query,
                  raft::device_matrix_view<idx_t, idx_t, row_major> out_inds,
                  raft::device_matrix_view<value_t, idx_t, row_major> out_dists,
                  raft::distance::DistanceType metric) RAFT_EXPLICIT;

}  // namespace raft::neighbors::brute_force

#endif  // RAFT_EXPLICIT_INSTANTIATE_ONLY

// No extern template for raft::neighbors::brute_force::knn_merge_parts

#define instantiate_raft_neighbors_brute_force_knn(                                         \
  idx_t, value_t, matrix_idx, index_layout, search_layout, epilogue_op)                     \
  extern template void raft::neighbors::brute_force::                                       \
    knn<idx_t, value_t, matrix_idx, index_layout, search_layout, epilogue_op>(              \
      raft::resources const& handle,                                                        \
      std::vector<raft::device_matrix_view<const value_t, matrix_idx, index_layout>> index, \
      raft::device_matrix_view<const value_t, matrix_idx, search_layout> search,            \
      raft::device_matrix_view<idx_t, matrix_idx, row_major> indices,                       \
      raft::device_matrix_view<value_t, matrix_idx, row_major> distances,                   \
      raft::distance::DistanceType metric,                                                  \
      std::optional<float> metric_arg,                                                      \
      std::optional<idx_t> global_id_offset,                                                \
      epilogue_op distance_epilogue);

instantiate_raft_neighbors_brute_force_knn(
  int64_t, float, uint32_t, raft::row_major, raft::row_major, raft::identity_op);
instantiate_raft_neighbors_brute_force_knn(
  int64_t, float, int64_t, raft::row_major, raft::row_major, raft::identity_op);
instantiate_raft_neighbors_brute_force_knn(
  int, float, int, raft::row_major, raft::row_major, raft::identity_op);
instantiate_raft_neighbors_brute_force_knn(
  uint32_t, float, uint32_t, raft::row_major, raft::row_major, raft::identity_op);

#undef instantiate_raft_neighbors_brute_force_knn

namespace raft::neighbors::brute_force {

extern template void search<float, int>(
  raft::resources const& res,
  const raft::neighbors::brute_force::index<float>& idx,
  raft::device_matrix_view<const float, int64_t, row_major> queries,
  raft::device_matrix_view<int, int64_t, row_major> neighbors,
  raft::device_matrix_view<float, int64_t, row_major> distances);

extern template void search<float, int>(
  raft::resources const& res,
  search_params const& params,
  const raft::neighbors::brute_force::index<float>& idx,
  raft::device_matrix_view<const float, int64_t, row_major> queries,
  raft::device_matrix_view<int, int64_t, row_major> neighbors,
  raft::device_matrix_view<float, int64_t, row_major> distances);

extern template void search<float, int64_t>(
  raft::resources const& res,
  const raft::neighbors::brute_force::index<float>& idx,
  raft::device_matrix_view<const float, int64_t, row_major> queries,
  raft::device_matrix_view<int64_t, int64_t, row_major> neighbors,
  raft::device_matrix_view<float, int64_t, row_major> distances);

extern template void search<float, int64_t>(
  raft::resources const& res,
  search_params const& params,
  const raft::neighbors::brute_force::index<float>& idx,
  raft::device_matrix_view<const float, int64_t, row_major> queries,
  raft::device_matrix_view<int64_t, int64_t, row_major> neighbors,
  raft::device_matrix_view<float, int64_t, row_major> distances);

extern template raft::neighbors::brute_force::index<float> build<float>(
  raft::resources const& res,
  raft::device_matrix_view<const float, int64_t, row_major> dataset,
  raft::distance::DistanceType metric,
  float metric_arg);

extern template raft::neighbors::brute_force::index<float> build<float>(
  raft::resources const& res,
  index_params const& params,
  raft::device_matrix_view<const float, int64_t, row_major> dataset);

extern template raft::neighbors::brute_force::index<float> build<float>(
  raft::resources const& res,
  raft::host_matrix_view<const float, int64_t, row_major> dataset,
  raft::distance::DistanceType metric,
  float metric_arg);

extern template raft::neighbors::brute_force::index<float> build<float>(
  raft::resources const& res,
  index_params const& params,
  raft::host_matrix_view<const float, int64_t, row_major> dataset);
}  // namespace raft::neighbors::brute_force

#define instantiate_raft_neighbors_brute_force_fused_l2_knn(            \
  value_t, idx_t, idx_layout, query_layout)                             \
  extern template void raft::neighbors::brute_force::fused_l2_knn(      \
    raft::resources const& handle,                                      \
    raft::device_matrix_view<const value_t, idx_t, idx_layout> index,   \
    raft::device_matrix_view<const value_t, idx_t, query_layout> query, \
    raft::device_matrix_view<idx_t, idx_t, row_major> out_inds,         \
    raft::device_matrix_view<value_t, idx_t, row_major> out_dists,      \
    raft::distance::DistanceType metric);

instantiate_raft_neighbors_brute_force_fused_l2_knn(float,
                                                    int64_t,
                                                    raft::row_major,
                                                    raft::row_major)

#undef instantiate_raft_neighbors_brute_force_fused_l2_knn
