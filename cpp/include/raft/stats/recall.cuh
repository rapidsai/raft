/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "detail/recall.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <optional>

namespace raft::stats {

/**
 * @defgroup stats_recall Recall Score
 * @{
 */

/**
 * @brief Calculate Recall score on the device for indices, distances computed by any Nearest
 * Neighbors Algorithm against reference indices, distances. Recall score is calculated by comparing
 * the total number of matching indices and dividing that value by the total size of the indices
 * matrix of dimensions (D, k). If distance matrices are provided, then non-matching indices could
 * be considered a match if abs(dist, ref_dist) < threshold.
 *
 * @tparam IndicesValueType data-type of the indices
 * @tparam IndexType data-type to index all matrices
 * @tparam ScalarType data-type to store recall score
 * @tparam DistanceValueType data-type of the distances
 * @param res raft::resources object to manage resources
 * @param[in] indices raft::device_matrix_view indices of neighbors
 * @param[in] ref_indices raft::device_matrix_view reference indices of neighbors
 * @param[out] recall_score raft::device_scalar_view output recall score
 * @param[in] distances (optional) raft::device_matrix_view distances of neighbors
 * @param[in] ref_distances (optional) raft::device_matrix_view reference distances of neighbors
 * @param[in] threshold (optional, default = 0.001) value for distance comparison
 */
template <typename IndicesValueType,
          typename IndexType,
          typename ScalarType,
          typename DistanceValueType = float>
void recall(
  raft::resources const& res,
  raft::device_matrix_view<const IndicesValueType, IndexType, raft::row_major> indices,
  raft::device_matrix_view<const IndicesValueType, IndexType, raft::row_major> ref_indices,
  raft::device_scalar_view<ScalarType> recall_score,
  std::optional<raft::device_matrix_view<const DistanceValueType, IndexType, raft::row_major>>
    distances = std::nullopt,
  std::optional<raft::device_matrix_view<const DistanceValueType, IndexType, raft::row_major>>
    ref_distances                                                          = std::nullopt,
  std::optional<raft::host_scalar_view<const DistanceValueType>> threshold = std::nullopt)
{
  RAFT_EXPECTS(indices.extent(0) == ref_indices.extent(0),
               "The number of rows in indices and reference indices should be equal");
  RAFT_EXPECTS(indices.extent(1) == ref_indices.extent(1),
               "The number of columns in indices and reference indices should be equal");

  if (distances.has_value() or ref_distances.has_value()) {
    RAFT_EXPECTS(distances.has_value() and ref_distances.has_value(),
                 "Both distances and reference distances should have values");

    RAFT_EXPECTS(distances.value().extent(0) == ref_distances.value().extent(0),
                 "The number of rows in distances and reference distances should be equal");
    RAFT_EXPECTS(distances.value().extent(1) == ref_distances.value().extent(1),
                 "The number of columns in indices and reference indices should be equal");

    RAFT_EXPECTS(indices.extent(0) == distances.value().extent(0),
                 "The number of rows in indices and distances should be equal");
    RAFT_EXPECTS(indices.extent(1) == distances.value().extent(1),
                 "The number of columns in indices and distances should be equal");
  }

  DistanceValueType threshold_val = 0.001;
  if (threshold.has_value()) { threshold_val = *threshold.value().data_handle(); }

  detail::recall(res, indices, ref_indices, distances, ref_distances, recall_score, threshold_val);
}

/**
 * @brief Calculate Recall score on the host for indices, distances computed by any Nearest
 * Neighbors Algorithm against reference indices, distances. Recall score is calculated by comparing
 * the total number of matching indices and dividing that value by the total size of the indices
 * matrix of dimensions (D, k). If distance matrices are provided, then non-matching indices could
 * be considered a match if abs(dist, ref_dist) < threshold.
 *
 * @tparam IndicesValueType data-type of the indices
 * @tparam IndexType data-type to index all matrices
 * @tparam ScalarType data-type to store recall score
 * @tparam DistanceValueType data-type of the distances
 * @param res raft::resources object to manage resources
 * @param[in] indices raft::device_matrix_view indices of neighbors
 * @param[in] ref_indices raft::device_matrix_view reference indices of neighbors
 * @param[out] recall_score raft::host_scalar_view output recall score
 * @param[in] distances (optional) raft::device_matrix_view distances of neighbors
 * @param[in] ref_distances (optional) raft::device_matrix_view reference distances of neighbors
 * @param[in] threshold (optional, default = 0.001) value for distance comparison
 */
template <typename IndicesValueType,
          typename IndexType,
          typename ScalarType,
          typename DistanceValueType = float>
void recall(
  raft::resources const& res,
  raft::device_matrix_view<const IndicesValueType, IndexType, raft::row_major> indices,
  raft::device_matrix_view<const IndicesValueType, IndexType, raft::row_major> ref_indices,
  raft::host_scalar_view<ScalarType> recall_score,
  std::optional<raft::device_matrix_view<const DistanceValueType, IndexType, raft::row_major>>
    distances = std::nullopt,
  std::optional<raft::device_matrix_view<const DistanceValueType, IndexType, raft::row_major>>
    ref_distances                                                          = std::nullopt,
  std::optional<raft::host_scalar_view<const DistanceValueType>> threshold = std::nullopt)
{
  auto recall_score_d = raft::make_device_scalar(res, *recall_score.data_handle());
  recall(res, indices, ref_indices, recall_score_d.view(), distances, ref_distances, threshold);
  raft::update_host(recall_score.data_handle(),
                    recall_score_d.data_handle(),
                    1,
                    raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
}

/** @} */  // end group stats_recall

}  // end namespace raft::stats
