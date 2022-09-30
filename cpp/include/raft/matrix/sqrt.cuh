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
 * @brief Square root of every element in the input matrix
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam idx_t integer type used for indexing
 * @tparam layout layout of the matrix data (must be row or col major)
 * @param[in] handle: raft handle
 * @param[in] in: input matrix and also the result is stored
 * @param[out] out: output matrix. The result is stored in the out matrix
 */
template <typename math_t, typename idx_t, typename layout>
void sqrt(const raft::handle_t& handle,
          raft::device_matrix_view<const math_t, idx_t, layout> in,
          raft::device_matrix_view<math_t, idx_t, layout> out)
{
  RAFT_EXPECTS(in.size() == out.size(), "Input and output matrices must have same size.");
  detail::seqRoot(in.data_handle(), out.data_handle(), in.size(), handle.get_stream());
}

/**
 * @brief Square root of every element in the input matrix (in place)
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam idx_t integer type used for indexing
 * @tparam layout layout of the matrix data (must be row or col major)
 * @param[in] handle: raft handle
 * @param[inout] inout: input matrix with in-place results
 */
template <typename math_t, typename idx_t, typename layout>
void sqrt(const raft::handle_t& handle, raft::device_matrix_view<math_t, idx_t, layout> inout)
{
  detail::seqRoot(inout.data_handle(), inout.size(), handle.get_stream());
}

/**
 * @brief Square root of every element in the input matrix
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam idx_t integer type used for indexing
 * @tparam layout layout of the matrix data (must be row or col major)
 * @param[in] handle: raft handle
 * @param[in] in: input matrix and also the result is stored
 * @param[out] out: output matrix. The result is stored in the out matrix
 * @param[in] scalar: every element is multiplied with scalar
 * @param[in] set_neg_zero whether to set negative numbers to zero
 */
template <typename math_t, typename idx_t, typename layout>
void weighted_sqrt(const raft::handle_t& handle,
                   raft::device_matrix_view<const math_t, idx_t, layout> in,
                   raft::device_matrix_view<math_t, idx_t, layout> out,
                   math_t scalar,
                   bool set_neg_zero = false)
{
  RAFT_EXPECTS(in.size() == out.size(), "Input and output matrices must have same size.");
  detail::seqRoot(
    in.data_handle(), out.data_handle(), scalar, in.size(), handle.get_stream(), set_neg_zero);
}

/**
 * @brief Square root of every element in the input matrix (in place)
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam idx_t integer type used for indexing
 * @tparam layout layout of the matrix data (must be row or col major)
 * @param[in] handle: raft handle
 * @param[inout] inout: input matrix and also the result is stored
 * @param[in] scalar: every element is multiplied with scalar
 * @param[in] set_neg_zero whether to set negative numbers to zero
 */
template <typename math_t, typename idx_t, typename layout>
void weighted_sqrt(const raft::handle_t& handle,
                   raft::device_matrix_view<math_t, idx_t, layout> inout,
                   math_t scalar,
                   bool set_neg_zero = false)
{
  detail::seqRoot(inout.data_handle(), scalar, inout.size(), handle.get_stream(), set_neg_zero);
}

}  // namespace raft::matrix
