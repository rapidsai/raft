/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <raft/stats/detail/trustworthiness_score.cuh>

namespace raft {
namespace stats {

/**
 * @brief Compute the trustworthiness score
 * @param h Raft handle
 * @param X[in]: Data in original dimension
 * @param X_embedded[in]: Data in target dimension (embedding)
 * @param n: Number of samples
 * @param m: Number of features in high/original dimension
 * @param d: Number of features in low/embedded dimension
 * @param n_neighbors Number of neighbors considered by trustworthiness score
 * @param batchSize Batch size
 * @return Trustworthiness score
 */
template <typename math_t, raft::distance::DistanceType distance_type>
double trustworthiness_score(const raft::handle_t& h,
                             const math_t* X,
                             math_t* X_embedded,
                             int n,
                             int m,
                             int d,
                             int n_neighbors,
                             int batchSize = 512) {
    return detail::trustworthiness_score<math_t, distance_type>(h, X, X_embedded, n, m, d, n_neighbors, batchSize);
}
}  // namespace stats
}  // namespace raft
