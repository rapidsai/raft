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
#include <algorithm>
#include <cstring>
#include <raft/core/device_support.hpp>
#include <raft/core/device_type.hpp>
#include <raft/core/resources.hpp>

namespace raft {
namespace detail {

template <device_type dst_type, device_type src_type, typename T>
std::enable_if_t<std::conjunction_v<std::bool_constant<dst_type == device_type::cpu>,
                                    std::bool_constant<src_type == device_type::cpu>>,
                 void>
copy(raft::resources const& handle, T* dst, T const* src, uint32_t size)
{
  std::copy(src, src + size, dst);
}

template <device_type dst_type, device_type src_type, typename T>
std::enable_if_t<
  std::conjunction_v<std::disjunction<std::bool_constant<dst_type == device_type::gpu>,
                                      std::bool_constant<src_type == device_type::gpu>>,
                     std::bool_constant<!CUDA_ENABLED>>,
  void>
copy(raft::resources const& handle, T* dst, T const* src, uint32_t size)
{
  throw raft::cuda_unsupported("Copying from or to device in non-GPU build");
}

}  // namespace detail
}  // namespace raft