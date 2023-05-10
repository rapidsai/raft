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
#include "raft/core/host_container_policy.hpp"
#include "raft/core/logger.hpp"
// #include "raft/core/mdspan.hpp"
#include <memory>
#include <raft/core/device_type.hpp>
// #include <raft/core/host_mdspan.hpp>
// #include <raft/core/device_mdspan.hpp>

namespace raft {
namespace detail {
template <typename ElementType,
          device_type D,
          typename Extents>
struct non_owning_buffer {

  non_owning_buffer() : data_{nullptr} {}

  non_owning_buffer(ElementType* ptr) : data_{ptr} {
  }

  auto* get() const { return data_; }

 private:
  // TODO(wphicks): Back this with RMM-allocated host memory
  ElementType* data_;
};

}  // namespace detail
}  // namespace raft