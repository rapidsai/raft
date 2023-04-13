/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
#include <raft/core/operators.hpp>  // raft::identity_op
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/detail/knn_brute_force.cuh>
#include <raft/spatial/knn/detail/fused_l2_knn.cuh>
#include <raft/util/raft_explicit.hpp>

#ifdef RAFT_EXPLICIT_INSTANTIATE

namespace raft::neighbors::brute_force {

/**
 * @defgroup brute_force_knn Brute-force K-Nearest Neighbors
 * @{
 */

/**
 * @brief Performs a k-select across several (contiguous) row-partitioned index/distance
 * matrices formatted like the following:
 *
 * part1row1: k0, k1, k2, k3
 * part1row2: k0, k1, k2, k3
 * part1row3: k0, k1, k2, k3
 * part2row1: k0, k1, k2, k3
 * part2row2: k0, k1, k2, k3
 * part2row3: k0, k1, k2, k3
 * etc...
 *
 * The example above shows what an aggregated index/distance matrix
 * would look like with two partitions when n_samples=3 and k=4.
 *
 * When working with extremely large data sets that have been broken
 * over multiple indexes, such as when computing over multiple GPUs,
 * the ids will often start at 0 for each local knn index but the
 * global ids need to be used when merging them together. An optional
 * translations vector can be supplied to map the starting id of
 * each partition to its global id so that the final merged knn
 * is based on the global ids.
 *
 * Usage example:
 * @code{.cpp}
 *  #include <raft/core/device_resources.hpp>
 *  #include <raft/neighbors/brute_force.cuh>
 *  using namespace raft::neighbors;
 *
 *  raft::raft::device_resources handle;
 *  ...
 *  compute multiple knn graphs and aggregate row-wise
 *  (see detailed description above)
 *  ...
 *  brute_force::knn_merge_parts(handle, in_keys, in_values, out_keys, out_values, n_samples);
 * @endcode
 *
 * @tparam idx_t
 * @tparam value_t
 *
 * @param[in] handle
 * @param[in] in_keys matrix of input keys (size n_samples * n_parts * k)
 * @param[in] in_values matrix of input values (size n_samples * n_parts * k)
 * @param[out] out_keys matrix of output keys (size n_samples * k)
 * @param[out] out_values matrix of output values (size n_samples * k)
 * @param[in] n_samples number of rows in each partition
 * @param[in] translations optional vector of starting global id mappings for each local partition
 */
template <typename value_t, typename idx_t>
inline void knn_merge_parts(
  raft::device_resources const& handle,
  raft::device_matrix_view<const value_t, idx_t, row_major> in_keys,
  raft::device_matrix_view<const idx_t, idx_t, row_major> in_values,
  raft::device_matrix_view<value_t, idx_t, row_major> out_keys,
  raft::device_matrix_view<idx_t, idx_t, row_major> out_values,
  size_t n_samples,
  std::optional<raft::device_vector_view<idx_t, idx_t>> translations = std::nullopt) RAFT_EXPLICIT;

/**
 * @brief Flat C++ API function to perform a brute force knn on
 * a series of input arrays and combine the results into a single
 * output array for indexes and distances. Inputs can be either
 * row- or column-major but the output matrices will always be in
 * row-major format.
 *
 * Usage example:
 * @code{.cpp}
 *  #include <raft/core/device_resources.hpp>
 *  #include <raft/neighbors/brute_force.cuh>
 *  #include <raft/distance/distance_types.hpp>
 *  using namespace raft::neighbors;
 *
 *  raft::raft::device_resources handle;
 *  ...
 *  auto metric = raft::distance::DistanceType::L2SqrtExpanded;
 *  brute_force::knn(handle, index, search, indices, distances, metric);
 * @endcode
 *
 * @param[in] handle: the cuml handle to use
 * @param[in] index: vector of device matrices (each size m_i*d) to be used as the knn index
 * @param[in] search: matrix (size n*d) to be used for searching the index
 * @param[out] indices: matrix (size n*k) to store output knn indices
 * @param[out] distances: matrix (size n*k) to store the output knn distance
 * @param[in] metric: distance metric to use. Euclidean (L2) is used by default
 * @param[in] metric_arg: the value of `p` for Minkowski (l-p) distances. This
 * 					 is ignored if the metric_type is not Minkowski.
 * @param[in] global_id_offset: optional starting global id mapping for the local partition
 *                              (assumes the index contains contiguous ids in the global id space)
 * @param[in] distance_epilogue: optional epilogue function to run after computing distances. This
                                 function takes a triple of the (value, rowid, colid) for each
                                 element in the pairwise distances and returns a transformed value
                                 back.
 */
template <typename idx_t,
          typename value_t,
          typename matrix_idx,
          typename index_layout,
          typename search_layout,
          typename epilogue_op = raft::identity_op>
void knn(raft::device_resources const& handle,
         std::vector<raft::device_matrix_view<const value_t, matrix_idx, index_layout>> index,
         raft::device_matrix_view<const value_t, matrix_idx, search_layout> search,
         raft::device_matrix_view<idx_t, matrix_idx, row_major> indices,
         raft::device_matrix_view<value_t, matrix_idx, row_major> distances,
         distance::DistanceType metric         = distance::DistanceType::L2Unexpanded,
         std::optional<float> metric_arg       = std::make_optional<float>(2.0f),
         std::optional<idx_t> global_id_offset = std::nullopt,
         epilogue_op distance_epilogue         = raft::identity_op()) RAFT_EXPLICIT;

/**
 * @brief Compute the k-nearest neighbors using L2 expanded/unexpanded distance.
 *
 * This is a specialized function for fusing the k-selection with the distance
 * computation when k < 64. The value of k will be inferred from the number
 * of columns in the output matrices.
 *
 * Usage example:
 * @code{.cpp}
 *  #include <raft/core/device_resources.hpp>
 *  #include <raft/neighbors/brute_force.cuh>
 *  #include <raft/distance/distance_types.hpp>
 *  using namespace raft::neighbors;
 *
 *  raft::raft::device_resources handle;
 *  ...
 *  auto metric = raft::distance::DistanceType::L2SqrtExpanded;
 *  brute_force::fused_l2_knn(handle, index, search, indices, distances, metric);
 * @endcode

 * @tparam value_t type of values
 * @tparam idx_t type of indices
 * @tparam idx_layout layout type of index matrix
 * @tparam query_layout layout type of query matrix
 * @param[in] handle raft handle for sharing expensive resources
 * @param[in] index input index array on device (size m * d)
 * @param[in] query input query array on device (size n * d)
 * @param[out] out_inds output indices array on device (size n * k)
 * @param[out] out_dists output dists array on device (size n * k)
 * @param[in] metric type of distance computation to perform (must be a variant of L2)
 */
template <typename value_t, typename idx_t, typename idx_layout, typename query_layout>
void fused_l2_knn(raft::device_resources const& handle,
                  raft::device_matrix_view<const value_t, idx_t, idx_layout> index,
                  raft::device_matrix_view<const value_t, idx_t, query_layout> query,
                  raft::device_matrix_view<idx_t, idx_t, row_major> out_inds,
                  raft::device_matrix_view<value_t, idx_t, row_major> out_dists,
                  raft::distance::DistanceType metric) RAFT_EXPLICIT;

/** @} */  // end group brute_force_knn

}  // namespace raft::neighbors::brute_force

#endif  // RAFT_EXPLICIT_INSTANTIATE

// No extern template for raft::neighbors::brute_force::knn_merge_parts

#define instantiate_raft_neighbors_brute_force_knn(                                         \
  idx_t, value_t, matrix_idx, index_layout, search_layout, epilogue_op)                     \
  extern template void raft::neighbors::brute_force::                                       \
    knn<idx_t, value_t, matrix_idx, index_layout, search_layout, epilogue_op>(              \
      raft::device_resources const& handle,                                                 \
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

#define instantiate_raft_neighbors_brute_force_fused_l2_knn(            \
  value_t, idx_t, idx_layout, query_layout)                             \
  extern template void raft::neighbors::brute_force::fused_l2_knn(      \
    raft::device_resources const& handle,                               \
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
