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
#include <raft/core/logger.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#include <cuda_runtime_api.h>
namespace raft {

/**
 * @brief A scoped setter for the active CUDA device
 *
 * On construction, the device_setter will set the active CUDA device to the
 * indicated value. On deletion, the active CUDA device will be set back to
 * its previous value. If the call to set the new active device fails, an
 * exception will be thrown. If the call to set the device back to its
 * previously selected value throws, an error will be logged, but no
 * exception will be thrown.
 *
 * @param int device_id The ID of the CUDA device to make active
 *
 */
struct device_setter {
  /**
   * Return the id of the current device as an integer
   */
  static auto get_current_device()
  {
    auto result = int{};
    RAFT_CUDA_TRY(cudaGetDevice(&result));
    return result;
  }
  /**
   * Return the count of currently available CUDA devices
   */
  static auto get_device_count()
  {
    auto result = int{};
    RAFT_CUDA_TRY(cudaGetDeviceCount(&result));
    return result;
  }

  explicit device_setter(int new_device) : prev_device_{get_current_device()}
  {
    RAFT_CUDA_TRY(cudaSetDevice(new_device));
  }
  ~device_setter() { RAFT_CUDA_TRY_NO_THROW(cudaSetDevice(prev_device_)); }

 private:
  int prev_device_;
};

}  // namespace raft
