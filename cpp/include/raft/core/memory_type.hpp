/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#include <cstdint>
#include <optional>
#ifndef RAFT_DISABLE_CUDA
#include <raft/util/cuda_rt_essentials.hpp>

#include <cuda_runtime.h>

#include <type_traits>
#else
#include <raft/core/logger.hpp>
#endif

namespace raft {
enum class memory_type : std::uint8_t {
  host    = std::uint8_t{0},
  pinned  = std::uint8_t{1},
  device  = std::uint8_t{2},
  managed = std::uint8_t{3}
};

auto constexpr is_device_accessible(memory_type mem_type)
{
  return (mem_type == memory_type::device || mem_type == memory_type::managed ||
          mem_type == memory_type::pinned);
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

auto constexpr has_compatible_accessibility(memory_type old_mem_type, memory_type new_mem_type)
{
  return ((!is_device_accessible(new_mem_type) || is_device_accessible(old_mem_type)) &&
          (!is_host_accessible(new_mem_type) || is_host_accessible(old_mem_type)));
}

template <memory_type... mem_types>
struct memory_type_constant {
  static_assert(sizeof...(mem_types) < 2, "At most one memory type can be specified");
  auto static constexpr value = []() {
    auto result = std::optional<memory_type>{};
    if constexpr (sizeof...(mem_types) == 1) { result = std::make_optional(mem_types...); }
    return result;
  }();
};

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

template <typename T>
auto memory_type_from_pointer(T* ptr)
{
  auto result = memory_type::host;
#ifndef RAFT_DISABLE_CUDA
  auto attrs = cudaPointerAttributes{};
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&attrs, ptr));
  switch (attrs.type) {
    case cudaMemoryTypeDevice: result = memory_type::device; break;
    case cudaMemoryTypeHost: result = memory_type::host; break;
    case cudaMemoryTypeManaged: result = memory_type::managed; break;
    default: result = memory_type::host;
  }
#else
  RAFT_LOG_DEBUG("RAFT compiled without CUDA support, assuming pointer is host pointer");
#endif
  return result;
}
}  // end namespace raft
