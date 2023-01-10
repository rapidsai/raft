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

namespace raft {
enum class memory_type { host, device, managed, pinned };

auto constexpr is_device_accessible(memory_type mem_type)
{
  return (mem_type == memory_type::device || mem_type == memory_type::managed);
}
auto constexpr is_host_accessible(memory_type mem_type)
{
  return (mem_type == memory_type::host || mem_type == memory_type::managed ||
          mem_type == memory_type::pinned);
}
auto constexpr is_host_device_accessible(memory_type mem_type)
{
  return is_device_accessible(mem_type) && is_host_accessible(mem_type);
}

namespace detail {

template <bool is_host_accessible, bool is_device_accessible>
auto constexpr memory_type_from_access()
{
  if constexpr (is_host_accessible && is_device_accessible) {
    return memory_type::managed;
  } else if constexpr (is_host_accessible) {
    return memory_type::host;
  } else if constexpr (is_device_accessible) {
    return memory_type::device;
  }
  static_assert(is_host_accessible || is_device_accessible,
                "Must be either host or device accessible to return a valid memory type");
}

}  // end namespace detail
}  // end namespace raft
