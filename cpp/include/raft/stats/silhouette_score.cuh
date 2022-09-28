/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#ifndef __SILHOUETTE_SCORE_H
#define __SILHOUETTE_SCORE_H

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/stats/detail/batched/silhouette_score.cuh>
#include <raft/stats/detail/silhouette_score.cuh>

namespace raft {
namespace stats {

/**
 * @brief main function that returns the average silhouette score for a given set of data and its
 * clusterings
 * @tparam DataT: type of the data samples
 * @tparam LabelT: type of the labels
 * @param handle: raft handle for managing expensive resources
 * @param X_in: pointer to the input Data samples array (nRows x nCols)
 * @param nRows: number of data samples
 * @param nCols: number of features
 * @param labels: the pointer to the array containing labels for every data sample (1 x nRows)
 * @param nLabels: number of Labels
 * @param silhouette_scorePerSample: pointer to the array that is optionally taken in as input and
 * is populated with the silhouette score for every sample (1 x nRows)
 * @param stream: the cuda stream where to launch this kernel
 * @param metric: the numerical value that maps to the type of distance metric to be used in the
 * calculations
 */
template <typename DataT, typename LabelT>
DataT silhouette_score(
  const raft::handle_t& handle,
  DataT* X_in,
  int nRows,
  int nCols,
  LabelT* labels,
  int nLabels,
  DataT* silhouette_scorePerSample,
  cudaStream_t stream,
  raft::distance::DistanceType metric = raft::distance::DistanceType::L2Unexpanded)
{
  return detail::silhouette_score(
    handle, X_in, nRows, nCols, labels, nLabels, silhouette_scorePerSample, stream, metric);
}

template <typename value_t, typename value_idx, typename label_idx>
value_t silhouette_score_batched(
  const raft::handle_t& handle,
  value_t* X,
  value_idx n_rows,
  value_idx n_cols,
  label_idx* y,
  label_idx n_labels,
  value_t* scores,
  value_idx chunk,
  raft::distance::DistanceType metric = raft::distance::DistanceType::L2Unexpanded)
{
  return batched::detail::silhouette_score(
    handle, X, n_rows, n_cols, y, n_labels, scores, chunk, metric);
}

/**
 * @brief main function that returns the average silhouette score for a given set of data and its
 * clusterings
 * @tparam value_t: type of the data samples
 * @tparam label_t: type of the labels
 * @tparam idx_t index type
 * @param[in]  handle: raft handle for managing expensive resources
 * @param[in]  X_in: input matrix Data in row-major format (nRows x nCols)
 * @param[in]  labels: the pointer to the array containing labels for every data sample (length:
 * nRows)
 * @param[out] silhouette_score_per_sample: optional array populated with the silhouette score
 * for every sample (length: nRows)
 * @param[in]  n_unique_labels: number of unique labels in the labels array
 * @param[in]  metric: the numerical value that maps to the type of distance metric to be used in
 * the calculations
 * @return: The silhouette score.
 */
template <typename value_t, typename label_t, typename idx_t>
value_t silhouette_score(
  const raft::handle_t& handle,
  raft::device_matrix_view<value_t, idx_t, raft::row_major> X_in,
  raft::device_vector_view<label_t, idx_t> labels,
  std::optional<raft::device_vector_view<value_t, idx_t>> silhouette_score_per_sample,
  idx_t n_unique_labels,
  raft::distance::DistanceType metric = raft::distance::DistanceType::L2Unexpanded)
{
  RAFT_EXPECTS(labels.extent(0) == X_in.extent(0), "Size mismatch betwen labels and data");

  value_t* silhouette_score_per_sample_ptr = nullptr;
  if (silhouette_score_per_sample.has_value()) {
    silhouette_score_per_sample_ptr = silhouette_score_per_sample.value().data_handle();
    RAFT_EXPECTS(silhouette_score_per_sample.value().extent(0) == X_in.extent(0),
                 "Size mismatch betwen silhouette_score_per_sample and data");
  }
  return detail::silhouette_score(handle,
                                  X_in.data_handle(),
                                  X_in.extent(0),
                                  X_in.extent(1),
                                  labels.data_handle(),
                                  n_unique_labels,
                                  silhouette_score_per_sample_ptr,
                                  handle.get_stream(),
                                  metric);
}

template <typename value_t, typename label_t, typename idx_t>
value_t silhouette_score_batched(
  const raft::handle_t& handle,
  raft::device_matrix_view<value_t, idx_t, raft::row_major> X,
  raft::device_vector_view<label_t, idx_t> y,
  std::optional<raft::device_vector_view<value_t, idx_t>> scores,
  idx_t n_unique_labels,
  raft::distance::DistanceType metric = raft::distance::DistanceType::L2Unexpanded)
{
  static_assert(std::is_integral_v<idx_t>,
                "silhouette_score_batched: The index type "
                "of each mdspan argument must be an integral type.");
  static_assert(std::is_integral_v<label_t>,
                "silhouette_score_batched: The label type must be an integral type.");
  RAFT_EXPECTS(y.extent(0) == X.extent(0), "Size mismatch betwen y and X");

  value_t* scores_ptr = nullptr;
  idx_t nscores       = 0;
  if (scores.has_value()) {
    scores_ptr = scores.value().data_handle();
    nscores    = scores.value().extent(0);
  }
  return batched::detail::silhouette_score(handle,
                                           X.data_handle(),
                                           X.extent(0),
                                           X.extent(1),
                                           y.data_handle(),
                                           n_unique_labels,
                                           scores_ptr,
                                           nscores,
                                           metric);
}

};  // namespace stats
};  // namespace raft

#endif