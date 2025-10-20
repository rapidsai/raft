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
 * @defgroup argmax Argmax operation
 * @{
 */

/**
 * @brief Argmax: find the col idx with maximum value for each row
 * @param[in] handle: raft handle
 * @param[in] in: input matrix of size (n_rows, n_cols)
 * @param[out] out: output vector of size n_rows
 */
template <typename math_t, typename idx_t, typename matrix_idx_t>
void argmax(raft::resources const& handle,
            raft::device_matrix_view<const math_t, matrix_idx_t, row_major> in,
            raft::device_vector_view<idx_t, matrix_idx_t> out)
{
  RAFT_EXPECTS(out.extent(0) == in.extent(0),
               "Size of output vector must equal number of rows in input matrix.");
  detail::argmax(in.data_handle(),
                 in.extent(1),
                 in.extent(0),
                 out.data_handle(),
                 resource::get_cuda_stream(handle));
}

/** @} */  // end of group argmax

}  // namespace raft::matrix
