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

#include <raft/core/detail/execution_device_id_base.hpp>
#include <raft/core/detail/execution_device_id_cpu.hpp>
#ifndef RAFT_DISABLE_GPU
#include <raft/core/detail/execution_device_id_gpu.hpp>
#endif
#include <raft/core/device_type.hpp>
#include <variant>

namespace raft {
template <device_type D>
using execution_device_id = detail::execution_device_id<D>;

using execution_device_id_variant =
  std::variant<execution_device_id<device_type::cpu>, execution_device_id<device_type::gpu>>;
}  // namespace raft
