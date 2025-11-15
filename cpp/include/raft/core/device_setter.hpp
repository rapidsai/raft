/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
