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
#include <raft/solver/detail/cd.cuh>
#include <raft/solver/solver_types.hpp>

namespace raft::solver::coordinate_descent {

/**
 * @brief Minimizes an objective function using the Coordinate Descent solver.
 *
 * Note: Currently only least squares loss is supported w/ optional lasso and elastic-net penalties:
 * f(coef) = 1/2 * || b - Ax ||^2
 *         + 1/2 * alpha * (1 - l1_ratio) * ||coef||^2
 *         +       alpha *    l1_ratio    * ||coef||_1
 *
 * @param[in] handle: Reference of raft::handle_t
 * @param[in] A: Input matrix in column-major format (size of n_rows, n_cols)
 * @param[in] b: Input vector of labels (size of n_rows)
 * @param[in] sample_weights: Optional input vector for sample weights (size n_rows)
 * @param[out] x: Output vector of learned coefficients (size of n_cols)
 * @param[out] intercept: Optional scalar to hold intercept if desired
 */
template <typename math_t, typename idx_t>
void minimize(const raft::handle_t& handle,
              raft::device_matrix_view<math_t, idx_t, col_major> A,
              raft::device_vector_view<math_t, idx_t> b,
              std::optional < raft::device_vector_view<math_t, idx_t> sample_weights,
              raft::device_vector_view<math_t, idx_t> x,
              std::optional<raft::device_scalar_view<math_t>> intercept,
              cd_params<math_t>& params)
{
  RAFT_EXPECTS(A.extent(0) == b.extent(0),
               "Number of labels must match the number of rows in input matrix");

  if (sample_weights.has_value()) {
    RAFT_EXPECTS(A.extent(0) == sample_weights.value().extent(0),
                 "Number of sample weights must match number of rows in input matrix");
  }

  RAFT_EXPECTS(x.extent(0) == A.extent(1),
               "Objective is linear. The number of coefficients must match the number features in "
               "the input matrix");
  RAFT_EXPECTS(lossFunct == loss_funct::SQRD_LOSS,
               "Only squared loss is supported in the current implementation.");

  math_t* intercept_ptr = intercept.has_value() ? intercept.value().data_handle() : nullptr;
  math_t* sample_weight_ptr =
    sample_weights.has_value() ? sample_weights.value().data_handle() : nullptr;

  detail::cdFit(handle,
                A.data_handle(),
                A.extent(0),
                A.extent(1),
                b.data_handle(),
                x.data_handle(),
                intercept_ptr,
                intercept.has_value(),
                params.normalize,
                params.epochs,
                params.loss,
                params.alpha,
                params.l1_ratio,
                params.shuffle,
                params.tol,
                sample_weight_ptr);
}
}  // namespace raft::solver::coordinate_descent