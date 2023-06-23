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
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/detail/knn_brute_force.cuh>
#include <raft/spatial/knn/detail/fused_l2_knn.cuh>

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
 *  #include <raft/core/resources.hpp>
 *  #include <raft/neighbors/brute_force.cuh>
 *  using namespace raft::neighbors;
 *
 *  raft::raft::resources handle;
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
  raft::resources const& handle,
  raft::device_matrix_view<const value_t, idx_t, row_major> in_keys,
  raft::device_matrix_view<const idx_t, idx_t, row_major> in_values,
  raft::device_matrix_view<value_t, idx_t, row_major> out_keys,
  raft::device_matrix_view<idx_t, idx_t, row_major> out_values,
  size_t n_samples,
  std::optional<raft::device_vector_view<idx_t, idx_t>> translations = std::nullopt)
{
  RAFT_EXPECTS(in_keys.extent(1) == in_values.extent(1) && in_keys.extent(0) == in_values.extent(0),
               "in_keys and in_values must have the same shape.");
  RAFT_EXPECTS(
    out_keys.extent(0) == out_values.extent(0) && out_keys.extent(0) == n_samples,
    "Number of rows in output keys and val matrices must equal number of rows in search matrix.");
  RAFT_EXPECTS(
    out_keys.extent(1) == out_values.extent(1) && out_keys.extent(1) == in_keys.extent(1),
    "Number of columns in output indices and distances matrices must be equal to k");

  idx_t* translations_ptr = nullptr;
  if (translations.has_value()) { translations_ptr = translations.value().data_handle(); }

  auto n_parts = in_keys.extent(0) / n_samples;
  detail::knn_merge_parts(in_keys.data_handle(),
                          in_values.data_handle(),
                          out_keys.data_handle(),
                          out_values.data_handle(),
                          n_samples,
                          n_parts,
                          in_keys.extent(1),
                          resource::get_cuda_stream(handle),
                          translations_ptr);
}

/**
 * @brief Flat C++ API function to perform a brute force knn on
 * a series of input arrays and combine the results into a single
 * output array for indexes and distances. Inputs can be either
 * row- or column-major but the output matrices will always be in
 * row-major format.
 *
 * Usage example:
 * @code{.cpp}
 *  #include <raft/core/resources.hpp>
 *  #include <raft/neighbors/brute_force.cuh>
 *  #include <raft/distance/distance_types.hpp>
 *  using namespace raft::neighbors;
 *
 *  raft::raft::resources handle;
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
void knn(raft::resources const& handle,
         std::vector<raft::device_matrix_view<const value_t, matrix_idx, index_layout>> index,
         raft::device_matrix_view<const value_t, matrix_idx, search_layout> search,
         raft::device_matrix_view<idx_t, matrix_idx, row_major> indices,
         raft::device_matrix_view<value_t, matrix_idx, row_major> distances,
         distance::DistanceType metric         = distance::DistanceType::L2Unexpanded,
         std::optional<float> metric_arg       = std::make_optional<float>(2.0f),
         std::optional<idx_t> global_id_offset = std::nullopt,
         epilogue_op distance_epilogue         = raft::identity_op())
{
  RAFT_EXPECTS(index[0].extent(1) == search.extent(1),
               "Number of dimensions for both index and search matrices must be equal");

  RAFT_EXPECTS(indices.extent(0) == distances.extent(0) && distances.extent(0) == search.extent(0),
               "Number of rows in output indices and distances matrices must equal number of rows "
               "in search matrix.");
  RAFT_EXPECTS(indices.extent(1) == distances.extent(1) && distances.extent(1),
               "Number of columns in output indices and distances matrices must the same");

  bool rowMajorIndex = std::is_same_v<index_layout, layout_c_contiguous>;
  bool rowMajorQuery = std::is_same_v<search_layout, layout_c_contiguous>;

  std::vector<value_t*> inputs;
  std::vector<matrix_idx> sizes;
  for (std::size_t i = 0; i < index.size(); ++i) {
    inputs.push_back(const_cast<value_t*>(index[i].data_handle()));
    sizes.push_back(index[i].extent(0));
  }

  std::vector<idx_t> trans;
  if (global_id_offset.has_value()) { trans.push_back(global_id_offset.value()); }

  std::vector<idx_t>* trans_arg = global_id_offset.has_value() ? &trans : nullptr;

  raft::neighbors::detail::brute_force_knn_impl(handle,
                                                inputs,
                                                sizes,
                                                index[0].extent(1),
                                                // TODO: This is unfortunate. Need to fix.
                                                const_cast<value_t*>(search.data_handle()),
                                                search.extent(0),
                                                indices.data_handle(),
                                                distances.data_handle(),
                                                indices.extent(1),
                                                rowMajorIndex,
                                                rowMajorQuery,
                                                trans_arg,
                                                metric,
                                                metric_arg.value_or(2.0f),
                                                distance_epilogue);
}

/**
 * @brief Compute the k-nearest neighbors using L2 expanded/unexpanded distance.
 *
 * This is a specialized function for fusing the k-selection with the distance
 * computation when k < 64. The value of k will be inferred from the number
 * of columns in the output matrices.
 *
 * Usage example:
 * @code{.cpp}
 *  #include <raft/core/resources.hpp>
 *  #include <raft/neighbors/brute_force.cuh>
 *  #include <raft/distance/distance_types.hpp>
 *  using namespace raft::neighbors;
 *
 *  raft::raft::resources handle;
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
void fused_l2_knn(raft::resources const& handle,
                  raft::device_matrix_view<const value_t, idx_t, idx_layout> index,
                  raft::device_matrix_view<const value_t, idx_t, query_layout> query,
                  raft::device_matrix_view<idx_t, idx_t, row_major> out_inds,
                  raft::device_matrix_view<value_t, idx_t, row_major> out_dists,
                  raft::distance::DistanceType metric)
{
  int k = static_cast<int>(out_inds.extent(1));

  RAFT_EXPECTS(k <= 64, "For fused k-selection, k must be < 64");
  RAFT_EXPECTS(out_inds.extent(1) == out_dists.extent(1), "Value of k must match for outputs");
  RAFT_EXPECTS(index.extent(1) == query.extent(1),
               "Number of columns in input matrices must be the same.");

  RAFT_EXPECTS(metric == distance::DistanceType::L2Expanded ||
                 metric == distance::DistanceType::L2Unexpanded ||
                 metric == distance::DistanceType::L2SqrtUnexpanded ||
                 metric == distance::DistanceType::L2SqrtExpanded,
               "Distance metric must be L2");

  size_t n_index_rows = index.extent(0);
  size_t n_query_rows = query.extent(0);
  size_t D            = index.extent(1);

  RAFT_EXPECTS(raft::is_row_or_column_major(index), "Index must be row or column major layout");
  RAFT_EXPECTS(raft::is_row_or_column_major(query), "Query must be row or column major layout");

  const bool rowMajorIndex = raft::is_row_major(index);
  const bool rowMajorQuery = raft::is_row_major(query);

  raft::spatial::knn::detail::fusedL2Knn(D,
                                         out_inds.data_handle(),
                                         out_dists.data_handle(),
                                         index.data_handle(),
                                         query.data_handle(),
                                         n_index_rows,
                                         n_query_rows,
                                         k,
                                         rowMajorIndex,
                                         rowMajorQuery,
                                         resource::get_cuda_stream(handle),
                                         metric);
}

/** @} */  // end group brute_force_knn

}  // namespace raft::neighbors::brute_force
