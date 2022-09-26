/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include "detail/knn_brute_force_faiss.cuh"
#include "detail/selection_faiss.cuh"
#include <raft/core/device_mdspan.hpp>

namespace raft::spatial::knn {

/**
 * Performs a k-select across row partitioned index/distance
 * matrices formatted like the following:
 * row1: k0, k1, k2
 * row2: k0, k1, k2
 * row3: k0, k1, k2
 * row1: k0, k1, k2
 * row2: k0, k1, k2
 * row3: k0, k1, k2
 *
 * etc...
 *
 * @tparam idx_t
 * @tparam value_t
 * @param handle
 * @param in_keys
 * @param in_values
 * @param out_keys
 * @param out_values
 * @param n_samples
 * @param k
 * @param translations
 */
template <typename idx_t = int64_t, typename value_t = float>
inline void knn_merge_parts(
  const raft::handle_t& handle,
  raft::device_matrix_view<const value_t, idx_t, row_major> in_keys,
  raft::device_matrix_view<const idx_t, idx_t, row_major> in_values,
  raft::device_matrix_view<value_t, idx_t, row_major> out_keys,
  raft::device_matrix_view<idx_t, idx_t, row_major> out_values,
  size_t n_samples,
  int k,
  std::optional<raft::device_vector_view<idx_t, idx_t>> translations = std::nullopt)
{
  RAFT_EXPECTS(in_keys.extent(1) == in_values.extent(1) && in_keys.extent(0) == in_values.extent(0),
               "in_keys and in_values must have the same shape.");
  RAFT_EXPECTS(
    out_keys.extent(0) == out_values.extent(0) == n_samples,
    "Number of rows in output keys and val matrices must equal number of rows in search matrix.");
  RAFT_EXPECTS(out_keys.extent(1) == out_values.extent(1) == k,
               "Number of columns in output indices and distances matrices must be equal to k");

  auto n_parts = in_keys.extent(0) / n_samples;
  detail::knn_merge_parts(in_keys.data_handle(),
                          in_values.data_handle(),
                          out_keys.data_handle(),
                          out_values.data_handle(),
                          n_samples,
                          n_parts,
                          k,
                          handle.get_stream(),
                          translations.value_or(nullptr));
}

/**
 * @brief Flat C++ API function to perform a brute force knn on
 * a series of input arrays and combine the results into a single
 * output array for indexes and distances. Inputs can be either
 * row- or column-major but the output matrices will always be in
 * row-major format.
 *
 * @example
 *
 *
 *
 * @param[in] handle the cuml handle to use
 * @param[in] index vector of device matrices (each size m_i*d) to be used as the knn index
 * @param[in] search matrix (size n*d) to be used for searching the index
 * @param[out] indices matrix (size n*k) to store output knn indices
 * @param[out] distances matrix (size n*k) to store the output knn distance
 * @param[in] k the number of nearest neighbors to return
 * @param[in] metric_arg the value of `p` for Minkowski (l-p) distances. This
 * 					 is ignored if the metric_type is not Minkowski.
 * @param[in] metric distance metric to use. Euclidean (L2) is used by default
 * @param[in] translations starting offsets for partitions. should be the same size
 *            as input vector.
 */
template <typename idx_t      = std::int64_t,
          typename value_t    = float,
          typename value_int  = int,
          typename matrix_idx = int,
          typename index_layout,
          typename search_layout>
void brute_force_knn(
  raft::handle_t const& handle,
  std::vector<raft::device_matrix_view<const value_t, matrix_idx, index_layout>> index,
  raft::device_matrix_view<const value_t, matrix_idx, search_layout> search,
  raft::device_matrix_view<idx_t, matrix_idx, row_major> indices,
  raft::device_matrix_view<value_t, matrix_idx, row_major> distances,
  value_int k,
  distance::DistanceType metric                  = distance::DistanceType::L2Unexpanded,
  std::optional<float> metric_arg                = std::make_optional<float>(2.0f),
  std::optional<std::vector<idx_t>> translations = std::nullopt)
{
  RAFT_EXPECTS(index[0].extent(1) == search.extent(1),
               "Number of dimensions for both index and search matrices must be equal");

  RAFT_EXPECTS(indices.extent(0) == distances.extent(0) && distances.extent(0) == search.extent(0),
               "Number of rows in output indices and distances matrices must equal number of rows "
               "in search matrix.");
  RAFT_EXPECTS(
    indices.extent(1) == distances.extent(1) && distances.extent(1) == static_cast<matrix_idx>(k),
    "Number of columns in output indices and distances matrices must be equal to k");

  bool rowMajorIndex = std::is_same_v<index_layout, layout_c_contiguous>;
  bool rowMajorQuery = std::is_same_v<search_layout, layout_c_contiguous>;

  std::vector<value_t*> inputs;
  std::vector<value_int> sizes;
  for (std::size_t i = 0; i < index.size(); ++i) {
    inputs.push_back(const_cast<value_t*>(index[i].data_handle()));
    sizes.push_back(index[i].extent(0));
  }

  std::vector<idx_t>* trans = translations.has_value() ? &(*translations) : nullptr;

  detail::brute_force_knn_impl(handle,
                               inputs,
                               sizes,
                               static_cast<value_int>(index[0].extent(1)),
                               // TODO: This is unfortunate. Need to fix.
                               const_cast<value_t*>(search.data_handle()),
                               static_cast<value_int>(search.extent(0)),
                               indices.data_handle(),
                               distances.data_handle(),
                               k,
                               rowMajorIndex,
                               rowMajorQuery,
                               trans,
                               metric,
                               metric_arg.value_or(2.0f));
}

}  // namespace raft::spatial::knn
