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
 * @brief Columns of a column major matrix are reversed in place (i.e. first column and
 * last column are swapped)
 * @param[in] handle: raft handle
 * @param[inout] inout: input and output matrix
 */
template <typename m_t, typename idx_t>
void col_reverse(const raft::handle_t& handle,
                 raft::device_matrix_view<m_t, idx_t, col_major> inout)
{
  detail::colReverse(inout.data_handle(), inout.extent(0), inout.extent(1), handle.get_stream());
}

/**
 * @brief Columns of a row major matrix are reversed in place (i.e. first column and
 * last column are swapped)
 * @param[in] handle: raft handle
 * @param[inout] inout: input and output matrix
 */
template <typename m_t, typename idx_t>
void col_reverse(const raft::handle_t& handle,
                 raft::device_matrix_view<m_t, idx_t, row_major> inout)
{
  detail::rowReverse(inout.data_handle(), inout.extent(1), inout.extent(0), handle.get_stream());
}

/**
 * @brief Rows of a column major matrix are reversed in place (i.e. first row and last
 * row are swapped)
 * @param[in] handle: raft handle
 * @param[inout] inout: input and output matrix
 */
template <typename m_t, typename idx_t>
void row_reverse(const raft::handle_t& handle,
                 raft::device_matrix_view<m_t, idx_t, col_major> inout)
{
  detail::rowReverse(inout.data_handle(), inout.extent(0), inout.extent(1), handle.get_stream());
}

/**
 * @brief Rows of a row major matrix are reversed in place (i.e. first row and last
 * row are swapped)
 * @param[in] handle: raft handle
 * @param[inout] inout: input and output matrix
 */
template <typename m_t, typename idx_t>
void row_reverse(const raft::handle_t& handle,
                 raft::device_matrix_view<m_t, idx_t, row_major> inout)
{
  detail::colReverse(inout.data_handle(), inout.extent(1), inout.extent(0), handle.get_stream());
}

}  // namespace raft::matrix