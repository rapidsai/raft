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
 * @defgroup matrix_sign_flip Matrix sign flip operations
 * @{
 */

/**
 * @brief sign flip stabilizes the sign of col major eigen vectors.
 * The sign is flipped if the column has negative |max|.
 * @tparam math_t floating point type used for matrix elements
 * @tparam idx_t integer type used for indexing
 * @param[in] handle: raft handle
 * @param[inout] inout: input matrix. Result also stored in this parameter
 */
template <typename math_t, typename idx_t>
void sign_flip(raft::resources const& handle,
               raft::device_matrix_view<math_t, idx_t, col_major> inout)
{
  detail::signFlip(
    inout.data_handle(), inout.extent(0), inout.extent(1), resource::get_cuda_stream(handle));
}

/** @} */  // end group matrix_sign_flip
}  // namespace raft::matrix
