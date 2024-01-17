/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <rmm/mr/device/device_memory_resource.hpp>  // rmm::mr::device_memory_resource

#include <cstddef>  // size_t
#include <memory>   // std::unique_ptr

namespace raft {

std::unique_ptr<rmm::mr::device_memory_resource> get_pool_memory_resource(
  rmm::mr::device_memory_resource*& mr, size_t initial_size);

}  // namespace raft
