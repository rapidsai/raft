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

#include "detail/normalize.cuh"

namespace raft {
namespace linalg {

/**
 * @brief Divide rows by their L2 norm.
 *
 * Note that the implementation is efficient for matrices with a large number of rows, not "thick"
 * matrices with few long rows.
 *
 * @tparam ElementType Input/Output data type
 * @tparam IndexType Integer type used to for addressing
 * @param[in] handle raft::handle_t
 * @param[in] in the input raft::device_matrix_view
 * @param[out] out the output raft::device_vector_view
 * @param out the output matrix (row-major)
 * @param in the input matrix (row-major)
 */
template <typename ElementType, typename IndexType>
void rowNormalize(const raft::handle_t& handle,
                  raft::device_matrix_view<const ElementType, IndexType, row_major> in,
                  raft::device_matrix_view<ElementType, IndexType, row_major> out)
{
  RAFT_EXPECTS(raft::is_row_or_column_major(in), "Input must be contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(out), "Output must be contiguous");
  RAFT_EXPECTS(in.extent(0) == out.extent(0),
               "The number of rows of the input and output should be equal");
  RAFT_EXPECTS(in.extent(1) == out.extent(1),
               "The number of columns of the input and output should be equal");

  detail::coalescedNormalize(
    out.data_handle(), in.data_handle(), in.extent(1), in.extent(0), handle.get_stream());
}

}  // namespace linalg
}  // namespace raft
