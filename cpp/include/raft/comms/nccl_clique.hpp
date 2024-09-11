/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <nccl.h>

namespace raft {
struct device_resources;
}

namespace raft::comms {

struct nccl_clique {
  using pool_mr = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;

  /**
   * Instantiates a NCCL clique with all available GPUs
   *
   * @param[in] percent_of_free_memory percentage of device memory to pre-allocate as memory pool
   *
   */
  nccl_clique(int percent_of_free_memory = 80);

  /**
   * Instantiates a NCCL clique
   *
   * Usage example:
   * @code{.cpp}
   * int n_devices;
   * cudaGetDeviceCount(&n_devices);
   * std::vector<int> device_ids(n_devices);
   * std::iota(device_ids.begin(), device_ids.end(), 0);
   * cuvs::neighbors::mg::nccl_clique& clique(device_ids); // first device is the root rank
   * @endcode
   *
   * @param[in] device_ids list of device IDs to be used to initiate the clique
   * @param[in] percent_of_free_memory percentage of device memory to pre-allocate as memory pool
   *
   */
  nccl_clique(const std::vector<int>& device_ids, int percent_of_free_memory = 80);

  void nccl_clique_init();
  const raft::device_resources& set_current_device_to_root_rank() const;
  ~nccl_clique();

  int root_rank_;
  int num_ranks_;
  int percent_of_free_memory_;
  std::vector<int> device_ids_;
  std::vector<ncclComm_t> nccl_comms_;
  std::vector<std::shared_ptr<pool_mr>> per_device_pools_;
  std::vector<raft::device_resources> device_resources_;
};

}  // namespace raft::comms