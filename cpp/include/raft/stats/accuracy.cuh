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

#ifndef __STATS_ACCURACY_H
#define __STATS_ACCURACY_H

#pragma once

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

}  // namespace stats
}  // namespace raft

#endif