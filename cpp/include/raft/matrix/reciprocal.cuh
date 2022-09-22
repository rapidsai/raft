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
#include <raft/matrix/detail/matrix.cuh>
#include <raft/matrix/matrix.cuh>

namespace raft::matrix {

/**
 * @brief Reciprocal of every element in the input matrix
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param handle: raft handle
 * @param in: input matrix and also the result is stored
 * @param out: output matrix. The result is stored in the out matrix
 * @param scalar: every element is multiplied with scalar
 * @param setzero round down to zero if the input is less the threshold
 * @param thres the threshold used to forcibly set inputs to zero
 * @{
 */
template <typename math_t>
void reciprocal(raft::device_matrix_view<math_t> in,
                raft::device_matrix_view<math_t> out,
                math_t scalar,
                bool setzero = false,
                math_t thres = 1e-15)
{
  RAFT_EXPECTS(in.size() == out.size(), "Input and output matrices must have the same size.");
  detail::reciprocal(
    in.data_handle(), out.data_handle(), scalar, in.size(), handle.get_stream(), setzero, thres);
}

/**
 * @brief Reciprocal of every element in the input matrix (in place)
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param inout: input matrix with in-place results
 * @param scalar: every element is multiplied with scalar
 * @param len: number elements of input matrix
 * @param stream cuda stream
 * @param setzero round down to zero if the input is less the threshold
 * @param thres the threshold used to forcibly set inputs to zero
 * @{
 */
template <typename math_t>
void reciprocal(const raft::handle_t& handle,
                raft::device_matrix_view<math_t> inout,
                math_t scalar,
                bool setzero = false,
                math_t thres = 1e-15)
{
  detail::reciprocal(
    inout.data_handle(), scalar, inout.size(), handle.get_stream(), setzero, thres);
}
}  // namespace raft::matrix
