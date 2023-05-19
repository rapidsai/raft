/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#ifndef __TRUSTWORTHINESS_SCORE_H
#define __TRUSTWORTHINESS_SCORE_H

#pragma once
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/stats/detail/trustworthiness_score.cuh>

namespace raft {
namespace stats {

/**
 * @brief Compute the trustworthiness score
 * @param[in] h: raft handle
 * @param[in] X: Data in original dimension
 * @param[in] X_embedded: Data in target dimension (embedding)
 * @param[in] n: Number of samples
 * @param[in] m: Number of features in high/original dimension
 * @param[in] d: Number of features in low/embedded dimension
 * @param[in] n_neighbors Number of neighbors considered by trustworthiness score
 * @param[in] batchSize Batch size
 * @return[out] Trustworthiness score
 */
template <typename math_t, raft::distance::DistanceType distance_type>
double trustworthiness_score(const raft::resources& h,
                             const math_t* X,
                             math_t* X_embedded,
                             int n,
                             int m,
                             int d,
                             int n_neighbors,
                             int batchSize = 512)
{
  return detail::trustworthiness_score<math_t, distance_type>(
    h, X, X_embedded, n, m, d, n_neighbors, batchSize);
}

/**
 * @defgroup stats_trustworthiness Trustworthiness
 * @{
 */

/**
 * @brief Compute the trustworthiness score
 * @tparam value_t the data type
 * @tparam idx_t Integer type used to for addressing
 * @param[in] handle the raft handle
 * @param[in] X: Data in original dimension
 * @param[in] X_embedded: Data in target dimension (embedding)
 * @param[in] n_neighbors Number of neighbors considered by trustworthiness score
 * @param[in] batch_size Batch size
 * @return Trustworthiness score
 * @note The constness of the data in X_embedded is currently casted away and the data is slightly
 * modified.
 */
template <raft::distance::DistanceType distance_type, typename value_t, typename idx_t>
double trustworthiness_score(
  raft::resources const& handle,
  raft::device_matrix_view<const value_t, idx_t, raft::row_major> X,
  raft::device_matrix_view<const value_t, idx_t, raft::row_major> X_embedded,
  int n_neighbors,
  int batch_size = 512)
{
  RAFT_EXPECTS(X.extent(0) == X_embedded.extent(0), "Size mismatch between X and X_embedded");
  RAFT_EXPECTS(std::is_integral_v<idx_t> && X.extent(0) <= std::numeric_limits<int>::max(),
               "Index type not supported");

  // TODO: Change the underlying implementation to remove the need to const_cast X_embedded.
  return detail::trustworthiness_score<value_t, distance_type>(
    handle,
    X.data_handle(),
    const_cast<value_t*>(X_embedded.data_handle()),
    X.extent(0),
    X.extent(1),
    X_embedded.extent(1),
    n_neighbors,
    batch_size);
}

/** @} */  // end group stats_trustworthiness

}  // namespace stats
}  // namespace raft

#endif