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
/**
 * @warning This file is deprecated and will be removed in release 22.06.
 * Please use the cuh version instead.
 */

#ifndef __REGRESSION_METRICS_H
#define __REGRESSION_METRICS_H

#pragma once

#include <raft/stats/detail/scores.cuh>

namespace raft {
namespace stats {

/**
 * @brief Compute regression metrics mean absolute error, mean squared error, median absolute error
 * @tparam T: data type for predictions (e.g., float or double for regression).
 * @param[in] predictions: array of predictions (GPU pointer).
 * @param[in] ref_predictions: array of reference (ground-truth) predictions (GPU pointer).
 * @param[in] n: number of elements in each of predictions, ref_predictions. Should be > 0.
 * @param[in] stream: cuda stream.
 * @param[out] mean_abs_error: Mean Absolute Error. Sum over n of (|predictions[i] -
 * ref_predictions[i]|) / n.
 * @param[out] mean_squared_error: Mean Squared Error. Sum over n of ((predictions[i] -
 * ref_predictions[i])^2) / n.
 * @param[out] median_abs_error: Median Absolute Error. Median of |predictions[i] -
 * ref_predictions[i]| for i in [0, n).
 */
template <typename T>
void regression_metrics(const T* predictions,
                        const T* ref_predictions,
                        int n,
                        cudaStream_t stream,
                        double& mean_abs_error,
                        double& mean_squared_error,
                        double& median_abs_error)
{
  detail::regression_metrics(
    predictions, ref_predictions, n, stream, mean_abs_error, mean_squared_error, median_abs_error);
}
}  // namespace stats
}  // namespace raft

#endif