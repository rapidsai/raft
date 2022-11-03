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
  return is_device_accessible() && is_host_accessible();
}

}  // end namespace raft
