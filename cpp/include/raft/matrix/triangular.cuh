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
#include <raft/matrix/detail/matrix.cuh>

namespace raft::matrix {

/**
 * @defgroup matrix_triangular Extract Matrix Triangles
 * @{
 */

/**
 * @brief Copy the upper triangular part of a matrix to another
 * @param[in] handle: raft handle
 * @param[in] src: input matrix with a size of n_rows x n_cols
 * @param[out] dst: output matrix with a size of kxk, k = min(n_rows, n_cols)
 */
template <typename m_t, typename idx_t>
void upper_triangular(raft::resources const& handle,
                      raft::device_matrix_view<const m_t, idx_t, col_major> src,
                      raft::device_matrix_view<m_t, idx_t, col_major> dst)
{
  auto k = std::min(src.extent(0), src.extent(1));
  RAFT_EXPECTS(k == dst.extent(0) && k == dst.extent(1),
               "dst should be of size kxk, k = min(n_rows, n_cols)");
  detail::copyUpperTriangular(src.data_handle(),
                              dst.data_handle(),
                              src.extent(0),
                              src.extent(1),
                              resource::get_cuda_stream(handle));
}
/** @} */  // end group matrix_triangular

}  // namespace raft::matrix
