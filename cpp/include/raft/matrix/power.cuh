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
#include <raft/matrix/detail/math.cuh>

namespace raft::matrix {

/**
 * @brief Power of every element in the input matrix
 * @param[in] handle: raft handle
 * @param[in] in: input matrix
 * @param[out] out: output matrix. The result is stored in the out matrix
 * @param[in] scalar: every element is multiplied with scalar.
 */
template <typename math_t>
void weighted_power(const raft::handle_t& handle,
                    raft::device_matrix_view<math_t> in,
                    raft::device_matrix_view<math_t> out,
                    math_t scalar)
{
  RAFT_EXPECTS(in.size() == out.size(), "Size of input and output matrices must be equal");
  detail::power(in.data_handle(), out.data_handle(), scalar, in.size(), handle.get_stream());
}

/**
 * @brief Power of every element in the input matrix (inplace)
 * @param[inout] inout: input matrix and also the result is stored
 * @param[in] scalar: every element is multiplied with scalar.
 */
template <typename math_t>
void weighted_power(const raft::handle_t& handle,
                    raft::device_matrix_view<math_t> inout,
                    math_t scalar)
{
  detail::power(inout.data_handle(), scalar, inout.size(), handle.get_stream());
}

/**
 * @brief Power of every element in the input matrix (inplace)
 * @param[inout] inout: input matrix and also the result is stored
 */
template <typename math_t>
void power(const raft::handle_t& handle, raft::device_matrix_view<math_t> inout)
{
  detail::power<math_t>(inout.data_handle(), inout.size(), handle.get_stream());
}

/**
 * @brief Power of every element in the input matrix
 * @param[in] handle: raft handle
 * @param[in] in: input matrix
 * @param[out] out: output matrix. The result is stored in the out matrix
 * @{
 */
template <typename math_t>
void power(const raft::handle_t& handle,
           raft::device_matrix_view<math_t> in,
           raft::device_matrix_view<math_t> out)
{
  RAFT_EXPECTS(in.size() == out.size(), "Input and output matrices must be same size.");
  detail::power<math_t>(in.data_handle(), out.data_handle(), in.size(), handle.get_stream());
}

}  // namespace raft::matrix
