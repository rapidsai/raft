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
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/detail/math.cuh>

namespace raft::matrix {

/**
 * @defgroup matrix_power Matrix Power Operations
 * @{
 */

/**
 * @brief Power of every element in the input matrix
 * @tparam math_t type of matrix elements
 * @tparam idx_t integer type used for indexing
 * @tparam layout layout of the matrix data (must be row or col major)
 * @param[in] handle: raft handle
 * @param[in] in: input matrix
 * @param[out] out: output matrix. The result is stored in the out matrix
 * @param[in] scalar: every element is multiplied with scalar.
 */
template <typename math_t, typename idx_t, typename layout>
void weighted_power(raft::resources const& handle,
                    raft::device_matrix_view<const math_t, idx_t, layout> in,
                    raft::device_matrix_view<math_t, idx_t, layout> out,
                    math_t scalar)
{
  RAFT_EXPECTS(in.size() == out.size(), "Size of input and output matrices must be equal");
  detail::power(
    in.data_handle(), out.data_handle(), scalar, in.size(), resource::get_cuda_stream(handle));
}

/**
 * @brief Power of every element in the input matrix (inplace)
 * @tparam math_t matrix element type
 * @tparam idx_t integer type used for indexing
 * @tparam layout layout of the matrix data (must be row or col major)
 * @param[in] handle: raft handle
 * @param[inout] inout: input matrix and also the result is stored
 * @param[in] scalar: every element is multiplied with scalar.
 */
template <typename math_t, typename idx_t, typename layout>
void weighted_power(raft::resources const& handle,
                    raft::device_matrix_view<math_t, idx_t, layout> inout,
                    math_t scalar)
{
  detail::power(inout.data_handle(), scalar, inout.size(), resource::get_cuda_stream(handle));
}

/**
 * @brief Power of every element in the input matrix (inplace)
 * @tparam math_t matrix element type
 * @tparam idx_t integer type used for indexing
 * @tparam layout layout of the matrix data (must be row or col major)
 * @param[in] handle: raft handle
 * @param[inout] inout: input matrix and also the result is stored
 */
template <typename math_t, typename idx_t, typename layout>
void power(raft::resources const& handle, raft::device_matrix_view<math_t, idx_t, layout> inout)
{
  detail::power<math_t>(inout.data_handle(), inout.size(), resource::get_cuda_stream(handle));
}

/**
 * @brief Power of every element in the input matrix
 * @tparam math_t type used for matrix elements
 * @tparam idx_t integer type used for indexing
 * @tparam layout layout of the matrix (row or column major)
 * @param[in] handle: raft handle
 * @param[in] in: input matrix
 * @param[out] out: output matrix. The result is stored in the out matrix
 * @{
 */
template <typename math_t, typename idx_t, typename layout>
void power(raft::resources const& handle,
           raft::device_matrix_view<const math_t, idx_t, layout> in,
           raft::device_matrix_view<math_t, idx_t, layout> out)
{
  RAFT_EXPECTS(in.size() == out.size(), "Input and output matrices must be same size.");
  detail::power<math_t>(
    in.data_handle(), out.data_handle(), in.size(), resource::get_cuda_stream(handle));
}

/** @} */  // end group matrix_power

}  // namespace raft::matrix
