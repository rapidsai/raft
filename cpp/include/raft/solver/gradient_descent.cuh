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
#include <raft/solver/detail/sgd.cuh>
#include <raft/solver/solver_types.hpp>

namespace raft::solver::gradient_descent {

/**
 * @brief Minimizes an objective function using the Gradient Descent solver and optional
 * lasso or elastic-net penalties.
 *
 * @param[in] handle: Reference of raft::handle_t
 * @param[in] A: Input matrix in column-major format (size of n_rows, n_cols)
 * @param[in] b: Input vector of labels (size of n_rows)
 * @param[out] x: Output vector of coefficients (size of n_cols)
 * @param[out] intercept: Optional scalar if fitting the intercept
 * @param[in] params: solver hyper-parameters
 */
template <typename math_t, typename idx_t>
void minimize(const raft::handle_t& handle,
              raft::device_matrix_view<const math_t, idx_t, col_major> A,
              raft::device_vector_view<const math_t, idx_t> b,
              raft::device_vector_view<math_t, idx_t> x,
              std::optional < raft::device_scalar_view<math_t, idx_t> intercept,
              sgd_params<math_t>& params)
{
  RAFT_EXPECTS(A.extent(0) == b.extent(0),
               "Number of labels must match the number of rows in input matrix");
  RAFT_EXPECTS(x.extent(0) == A.extent(1),
               "Objective is linear. The number of coefficients must match the number features in "
               "the input matrix");

  auto intercept_ptr = intercept.has_value() ? intercept.data_handle() ? nullptr;
  detail::sgdFit(handle,
                 A.data_handle(),
                 A.extent(0),
                 A.extent(1),
                 b.data_handle(),
                 x.data_handle(),
                 intercept_ptr,
                 intercept.has_value(),
                 params.batch_size,
                 params.epochs,
                 params.lr_type,
                 params.eta0,
                 params.power_t,
                 params.loss,
                 params.penalty,
                 params.alpha,
                 params.l1_ratio,
                 params.shuffle,
                 params.tol,
                 params.n_iter_no_change,
                 handle.get_stream());
}

}  // namespace raft::solver::gradient_descent