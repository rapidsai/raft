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
#include <raft/core/device_support.hpp>
#include <raft/core/device_type.hpp>

namespace raft {
namespace detail {
template<device_type D>
struct device_id {
  using value_type = int;

  device_id(value_type device_index=value_type{}) noexcept {}
  auto value() const noexcept(false) {
    throw cuda_unsupported{"Attempting to use a GPU device in a non-CUDA build"};
  }
  auto rmm_id() const noexcept(false) {
    throw cuda_unsupported{"Attempting to use a GPU device in a non-CUDA build"};
  }
};
}  // namespace detail
}  // namespace raft
