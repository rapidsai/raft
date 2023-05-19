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
#include <raft/linalg/map.cuh>
#include <raft/matrix/detail/math.cuh>

namespace raft::matrix {

/**
 * @defgroup matrix_init Matrix initialization operations
 * @{
 */

/**
 * @brief set values to scalar in matrix
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam extents dimension and indexing type used for the input
 * @tparam layout layout of the matrix data (must be row or col major)
 * @param[in] handle: raft handle
 * @param[in] in input matrix
 * @param[out] out output matrix. The result is stored in the out matrix
 * @param[in] scalar scalar value to fill matrix elements
 */
template <typename math_t, typename extents, typename layout>
void fill(raft::resources const& handle,
          raft::device_mdspan<const math_t, extents, layout> in,
          raft::device_mdspan<math_t, extents, layout> out,
          raft::host_scalar_view<math_t> scalar)
{
  RAFT_EXPECTS(raft::is_row_or_column_major(out), "Data layout not supported");
  RAFT_EXPECTS(in.size() == out.size(), "Input and output matrices must be the same size.");
  RAFT_EXPECTS(scalar.data_handle() != nullptr, "Empty scalar");
  detail::setValue(out.data_handle(),
                   in.data_handle(),
                   *(scalar.data_handle()),
                   in.size(),
                   resource::get_cuda_stream(handle));
}

/**
 * @brief set values to scalar in matrix
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam extents dimension and indexing type used for the input
 * @tparam layout_t layout of the matrix data (must be row or col major)
 * @param[in] handle: raft handle
 * @param[inout] inout input matrix
 * @param[in] scalar scalar value to fill matrix elements
 */
template <typename math_t, typename extents, typename layout>
void fill(raft::resources const& handle,
          raft::device_mdspan<math_t, extents, layout> inout,
          math_t scalar)
{
  linalg::map(handle, inout, raft::const_op{scalar});
}

/** @} */  // end of group matrix_init

}  // namespace raft::matrix
