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

#ifndef __R2_SCORE_H
#define __R2_SCORE_H

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/stats/detail/scores.cuh>

namespace raft {
namespace stats {

/**
 * Calculates the "Coefficient of Determination" (R-Squared) score
 * normalizing the sum of squared errors by the total sum of squares.
 *
 * This score indicates the proportionate amount of variation in an
 * expected response variable is explained by the independent variables
 * in a linear regression model. The larger the R-squared value, the
 * more variability is explained by the linear regression model.
 *
 * @param y: Array of ground-truth response variables
 * @param y_hat: Array of predicted response variables
 * @param n: Number of elements in y and y_hat
 * @param stream: cuda stream
 * @return: The R-squared value.
 */
template <typename math_t>
math_t r2_score(math_t* y, math_t* y_hat, int n, cudaStream_t stream)
{
  return detail::r2_score(y, y_hat, n, stream);
}

/**
 * @defgroup stats_r2_score Regression R2 Score
 * @{
 */

/**
 * Calculates the "Coefficient of Determination" (R-Squared) score
 * normalizing the sum of squared errors by the total sum of squares.
 *
 * This score indicates the proportionate amount of variation in an
 * expected response variable is explained by the independent variables
 * in a linear regression model. The larger the R-squared value, the
 * more variability is explained by the linear regression model.
 *
 * @tparam value_t the data type
 * @tparam idx_t index type
 * @param[in] handle the raft handle
 * @param[in] y: Array of ground-truth response variables
 * @param[in] y_hat: Array of predicted response variables
 * @return: The R-squared value.
 * @note The constness of y and y_hat is currently casted away.
 */
template <typename value_t, typename idx_t>
value_t r2_score(raft::resources const& handle,
                 raft::device_vector_view<const value_t, idx_t> y,
                 raft::device_vector_view<const value_t, idx_t> y_hat)
{
  RAFT_EXPECTS(y.extent(0) == y_hat.extent(0), "Size mismatch between y and y_hat");
  RAFT_EXPECTS(y.is_exhaustive(), "y must be contiguous");
  RAFT_EXPECTS(y_hat.is_exhaustive(), "y_hat must be contiguous");

  // TODO: Change the underlying implementation to remove the need to const_cast
  return detail::r2_score(const_cast<value_t*>(y.data_handle()),
                          const_cast<value_t*>(y_hat.data_handle()),
                          y.extent(0),
                          resource::get_cuda_stream(handle));
}

/** @} */  // end group stats_r2_score

}  // namespace stats
}  // namespace raft

#endif