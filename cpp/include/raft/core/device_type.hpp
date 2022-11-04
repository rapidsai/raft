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
#include <raft/core/error.hpp>
#include <raft/core/memory_type.hpp>

namespace raft {
enum class device_type { cpu, gpu };

auto constexpr is_compatible(device_type dev_type, memory_type mem_type)
{
  return (dev_type == device_type::gpu && is_device_accessible(mem_type)) ||
         (dev_type == device_type::cpu && is_host_accessible(mem_type));
}

struct bad_device_type : raft::exception {
  bad_device_type() : bad_device_type("Incorrect device type for this operation") {}
};

}  // end namespace raft
