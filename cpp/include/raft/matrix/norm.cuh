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
#include <raft/matrix/matrix.cuh>

namespace raft::matrix {

/**
 * @brief Get the L2/F-norm of a matrix
 * @param handle
 * @param in: input matrix/vector with totally size elements
 * @param size: size of the matrix/vector
 * @param stream: cuda stream
 */
template <typename m_t, typename idx_t>
m_t l2_norm(const raft::handle_t& handle, raft::device_mdspan<m_t, idx_t> in)
{
  return detail::getL2Norm(handle, in.data_handle(), in.size(), handle.get_stream());
}
}  // namespace raft::matrix