/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/neighbors/detail/ivf_flat_build.cuh>
#include <raft/neighbors/ivf_flat_types.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace raft::neighbors::ivf_flat::helpers {
/**
 * @defgroup ivf_flat_helpers Helper functions for manipulationg IVF Flat Index
 * @{
 */

namespace codepacker {

template <typename T>
inline void pack_full_list(
  raft::resources const& res,
  device_matrix_view<const T, uint32_t, row_major> codes,
  uint32_t veclen,
  device_mdspan<T, typename list_spec<uint32_t, T, uint32_t>::list_extents, row_major> list_data)
{
  raft::neighbors::ivf_flat::detail::pack_list_data(res, codes, veclen, list_data);
}

template <typename T>
inline void unpack_full_list(
  raft::resources const& res,
  device_mdspan<const T, typename list_spec<uint32_t, T, uint32_t>::list_extents, row_major>
    list_data,
  uint32_t veclen,
  device_matrix_view<T, uint32_t, row_major> codes)
{
  raft::neighbors::ivf_flat::detail::unpack_list_data(res, list_data, veclen, codes);
}
}  // namespace codepacker
/** @} */
}  // namespace raft::neighbors::ivf_flat::helpers
