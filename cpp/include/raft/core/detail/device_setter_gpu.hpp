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
#include <cuda_runtime_api.h>
#include <raft/core/detail/device_setter_base.hpp>
#include <raft/core/device_type.hpp>
#include <raft/core/execution_device_id.hpp>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cudart_utils.hpp>

namespace raft {
namespace detail {

/** Class for setting current device within a code block */
template <>
struct device_setter<device_type::gpu> {
  device_setter(raft::execution_device_id<device_type::gpu> device) noexcept(false)
    : prev_device_{[]() {
        auto result = int{};
        RAFT_CUDA_TRY(cudaGetDevice(&result));
        return result;
      }()}
  {
    RAFT_CUDA_TRY(cudaSetDevice(device.value()));
  }

  ~device_setter() { RAFT_CUDA_TRY_NO_THROW(cudaSetDevice(prev_device_.value())); }

 private:
  execution_device_id<device_type::gpu> prev_device_;
};

}  // namespace detail
}  // namespace raft
