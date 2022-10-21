/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <raft/core/handle.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/solver/detail/lars.cuh>
#include <raft/solver/solver_types.hpp>

namespace raft::solver::least_angle_regression {

/**
 * @brief Train a regression model using Least Angle Regression (LARS).
 *
 * Least Angle Regression (LAR or LARS) is a model selection algorithm. It
 * builds up the model using the following algorithm:
 *
 * 1. We start with all the coefficients equal to zero.
 * 2. At each step we select the predictor that has the largest absolute
 *      correlation with the residual.
 * 3. We take the largest step possible in the direction which is equiangular
 *    with all the predictors selected so far. The largest step is determined
 *    such that using this step a new predictor will have as much correlation
 *    with the residual as any of the currently active predictors.
 * 4. Stop if max_iter reached or all the predictors are used, or if the
 *    correlation between any unused predictor and the residual is lower than
 *    a tolerance.
 *
 * The solver is based on [1]. The equations referred in the comments correspond
 * to the equations in the paper.
 *
 * Note: this algorithm assumes that the offset is removed from X and y, and
 * each feature is normalized:
 * - sum_i y_i = 0,
 * - sum_i x_{i,j} = 0, sum_i x_{i,j}^2=1 for j=0..n_col-1
 *
 * References:
 * [1] B. Efron, T. Hastie, I. Johnstone, R Tibshirani, Least Angle Regression
 * The Annals of Statistics (2004) Vol 32, No 2, 407-499
 * http://statweb.stanford.edu/~tibs/ftp/lars.pdf
 *
 * @param handle RAFT handle
 * @param[in] A device array of training vectors in column major format,
 *     size [n_rows * n_cols]. Note that the columns of X will be permuted if
 *     the Gram matrix is not specified. It is expected that X is normalized so
 *     that each column has zero mean and unit variance.
 * @param[in] b device array of the regression targets, size [n_rows]. y should
 *     be normalized to have zero mean.
 * @param[in] Gram device array containing Gram matrix containing X.T * X. Can be
 *     nullptr.
 * @param[out] x: device array of regression coefficients, has to be allocated on
 *     entry, size [max_iter]
 * @param[in] active_idx device vector containing the indices of active variables.
 *     Must be allocated on entry. Size [max_iter]
 * @param[out] alphas device array to return the maximum correlation along the
 *     regularization path. Must be allocated on entry, size [max_iter+1].
 * @param[out] n_active host pointer to return the number of active elements (scalar)
 * @param[out] coef_path coefficients along the regularization path are returned
 *    here. Must be nullptr, or a device array already allocated on entry.
 *    Size [max_iter * (max_iter+1)].
 * @param[in] params: lars hyper-parameters
 * @param[in] ld_X leading dimension of A (stride of columns)
 * @param[in] ld_G leading dimesion of G
 */
template <typename math_t, typename idx_t>
void minimize(const raft::handle_t& handle,
              raft::device_matrix_view<const math_t, idx_t, col_major> A,
              raft::device_vector_view<const math_t, idx_t> b,
              std::optional<raft::device_matrix_view<const math_t, idx_t>> Gram,
              raft::device_vector_view<math_t, idx_t> x,
              raft::device_vector_view<idx_t, idx_t> active_idx,
              raft::device_vector_view<math_t, idx_t> alphas,
              raft::host_scalar_view<idx_t> n_active,
              std::optional<raft::device_vector_view<math_t, idx_t>> coef_path,
              lars_params<math_t>& params,
              idx_t ld_X = 0,
              idx_t ld_G = 0)
{
}
}  // namespace raft::solver::least_angle_regression