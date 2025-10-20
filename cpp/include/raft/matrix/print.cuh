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
#include <raft/matrix/detail/matrix.cuh>
#include <raft/matrix/matrix_types.hpp>

namespace raft::matrix {

/**
 * @defgroup matrix_print Matrix print operations
 * @{
 */

/**
 * @brief Prints the data stored in GPU memory
 * @tparam m_t type of matrix elements
 * @tparam idx_t integer type used for indexing
 * @param[in] handle: raft handle
 * @param[in] in: input matrix
 * @param[in] separators: horizontal and vertical separator characters
 */
template <typename m_t, typename idx_t>
void print(raft::resources const& handle,
           raft::device_matrix_view<const m_t, idx_t, col_major> in,
           print_separators& separators)
{
  detail::print(in.data_handle(),
                in.extent(0),
                in.extent(1),
                separators.horizontal,
                separators.vertical,
                resource::get_cuda_stream(handle));
}

/** @} */  // end group matrix_print
}  // namespace raft::matrix
