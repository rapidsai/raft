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
#include <raft/core/host_mdspan.hpp>
#include <raft/matrix/detail/matrix.cuh>
#include <raft/matrix/matrix.cuh>

namespace raft::matrix {

/**
 * @brief Prints the data stored in GPU memory
 * @tparam m_t type of matrix elements
 * @tparam idx_t integer type used for indexing
 * @param[in] handle: raft handle
 * @param[in] in: input matrix
 * @param[in] h_separator: horizontal separator character
 * @param[in] v_separator: vertical separator character
 */
template <typename m_t, typename idx_t>
void print(const raft::handle_t& handle,
           raft::device_matrix_view<const m_t, idx_t, col_major> in,
           char h_separator = ' ',
           char v_separator = '\n')
{
  detail::print(
    in.data_handle(), in.extent(0), in.extent(1), h_separator, v_separator, handle.get_stream());
}

/**
 * @brief Prints the host data stored in CPU memory
 * @tparam m_t type of matrix elements
 * @tparam idx_t integer type used for indexing
 * @param[in] handle raft handle for managing resources
 * @param[in] in input matrix with column-major layout
 * @param[in] h_separator: horizontal separator character
 * @param[in] v_separator: vertical separator character
 */
template <typename m_t, typename idx_t>
void print(const raft::handle_t& handle,
           raft::host_matrix_view<const m_t, idx_t, col_major> in,
           char h_separator = ' ',
           char v_separator = '\n')
{
  detail::printHost(in.data_handle(), in.extent(0), in.extent(1), h_separator, v_separator);
}
}  // namespace raft::matrix
