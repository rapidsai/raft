/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#ifndef __STATS_ACCURACY_H
#define __STATS_ACCURACY_H

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/stats/detail/scores.cuh>

namespace raft {
namespace stats {

/**
 * @brief Compute accuracy of predictions. Useful for classification.
 * @tparam math_t: data type for predictions (e.g., int for classification)
 * @param[in] predictions: array of predictions (GPU pointer).
 * @param[in] ref_predictions: array of reference (ground-truth) predictions (GPU pointer).
 * @param[in] n: number of elements in each of predictions, ref_predictions.
 * @param[in] stream: cuda stream.
 * @return: Accuracy score in [0, 1]; higher is better.
 */
template <typename math_t>
float accuracy(const math_t* predictions, const math_t* ref_predictions, int n, cudaStream_t stream)
{
  return detail::accuracy_score(predictions, ref_predictions, n, stream);
}

/**
 * @defgroup stats_accuracy Accuracy Score
 * @{
 */

/**
 * @brief Compute accuracy of predictions. Useful for classification.
 * @tparam value_t: data type for predictions (e.g., int for classification)
 * @tparam idx_t Index type of matrix extent.
 * @param[in] handle: the raft handle.
 * @param[in] predictions: array of predictions (GPU pointer).
 * @param[in] ref_predictions: array of reference (ground-truth) predictions (GPU pointer).
 * @return: Accuracy score in [0, 1]; higher is better.
 */
template <typename value_t, typename idx_t>
float accuracy(raft::resources const& handle,
               raft::device_vector_view<const value_t, idx_t> predictions,
               raft::device_vector_view<const value_t, idx_t> ref_predictions)
{
  RAFT_EXPECTS(predictions.size() == ref_predictions.size(), "Size mismatch");
  RAFT_EXPECTS(predictions.is_exhaustive(), "predictions must be contiguous");
  RAFT_EXPECTS(ref_predictions.is_exhaustive(), "ref_predictions must be contiguous");

  return detail::accuracy_score(predictions.data_handle(),
                                ref_predictions.data_handle(),
                                predictions.extent(0),
                                resource::get_cuda_stream(handle));
}

/** @} */  // end group stats_accuracy

}  // namespace stats
}  // namespace raft

#endif