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

namespace raft::matrix {

/**
 * @brief ratio of every element over sum of input vector is calculated
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param handle
 * @param src: input matrix
 * @param dest: output matrix. The result is stored in the dest matrix
 * @param len: number elements of input matrix
 * @param stream cuda stream
 */
template <typename math_t>
void ratio(const raft::handle_t& handle,
           raft::device_matrix_view<math_t> src,
           raft::device_matrix_view<math_t> dest)
{
  RAFT_EXPECTS(src.size() == dst.size(), "Input and output matrices must be the same size.");
  detail::ratio(handle, src.data_handle(), dest.data_handle(), src.size(), handle.get_stream());
}

/**
 * @brief ratio of every element over sum of input vector is calculated
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param[in] handle
 * @param[inout] inout: input matrix
 */
    template <typename math_t>
    void ratio(const raft::handle_t& handle,
               raft::device_matrix_view<math_t> inout)
    {
        detail::ratio(handle, inout.data_handle(), inout.data_handle(), inout.size(), handle.get_stream());
    }
}  // namespace raft::matrix
