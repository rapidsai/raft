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
 * @defgroup matrix_norm Matrix Norm Operations
 * @{
 */

/**
 * @brief Get the L2/F-norm of a matrix
 * @param[in] handle: raft handle
 * @param[in] in: input matrix/vector with totally size elements
 * @returns matrix l2 norm
 */
template <typename m_t, typename idx_t>
m_t l2_norm(raft::resources const& handle, raft::device_mdspan<const m_t, idx_t> in)
{
  return detail::getL2Norm(handle, in.data_handle(), in.size(), resource::get_cuda_stream(handle));
}

/** @} */  // end of group matrix_norm

}  // namespace raft::matrix