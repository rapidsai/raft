/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/detail/math.cuh>

namespace raft::matrix {

/**
 * @defgroup matrix_reciprocal Matrix Reciprocal Operations
 * @{
 */

/**
 * @brief Reciprocal of every element in the input matrix
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam idx_t integer type used for indexing
 * @param handle: raft handle
 * @param in: input matrix and also the result is stored
 * @param out: output matrix. The result is stored in the out matrix
 * @param scalar: every element is multiplied with scalar
 * @param setzero round down to zero if the input is less the threshold
 * @param thres the threshold used to forcibly set inputs to zero
 * @{
 */
template <typename math_t, typename idx_t, typename layout>
void reciprocal(raft::resources const& handle,
                raft::device_matrix_view<const math_t, idx_t, layout> in,
                raft::device_matrix_view<math_t, idx_t, layout> out,
                raft::host_scalar_view<math_t> scalar,
                bool setzero = false,
                math_t thres = 1e-15)
{
  RAFT_EXPECTS(in.size() == out.size(), "Input and output matrices must have the same size.");
  detail::reciprocal<math_t>(in.data_handle(),
                             out.data_handle(),
                             *(scalar.data_handle()),
                             in.size(),
                             resource::get_cuda_stream(handle),
                             setzero,
                             thres);
}

/**
 * @brief Reciprocal of every element in the input matrix (in place)
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam idx_t integer type used for indexing
 * @tparam layout layout of the matrix data (must be row or col major)
 * @param[in] handle: raft handle to manage resources
 * @param[inout] inout: input matrix with in-place results
 * @param[in] scalar: every element is multiplied with scalar
 * @param[in] setzero round down to zero if the input is less the threshold
 * @param[in] thres the threshold used to forcibly set inputs to zero
 * @{
 */
template <typename math_t, typename idx_t, typename layout>
void reciprocal(raft::resources const& handle,
                raft::device_matrix_view<math_t, idx_t, layout> inout,
                raft::host_scalar_view<math_t> scalar,
                bool setzero = false,
                math_t thres = 1e-15)
{
  detail::reciprocal<math_t>(inout.data_handle(),
                             *(scalar.data_handle()),
                             inout.size(),
                             resource::get_cuda_stream(handle),
                             setzero,
                             thres);
}

/** @} */  // end group matrix_reciprocal

}  // namespace raft::matrix
