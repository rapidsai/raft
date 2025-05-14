/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <raft/core/device_resources.hpp>
#include <raft/core/resource/multi_gpu.hpp>
#include <raft/core/resource/resource_types.hpp>

#include <unordered_set>
#include <vector>

namespace raft {

const std::unordered_set<resource::resource_type> snmg_related_resources = {
  raft::resource::MULTI_GPU, raft::resource::NCCL_COMM, raft::resource::ROOT_RANK};

/**
 * @brief SNMG (single-node multi-GPU) resource container object that tracks resources for each GPU.
 * Note the `device_resources_snmg` object can also be used as a classic
 * `device_resources` object. The associated resources will be the ones of the GPU used during
 * object instantiation and a GPU switch operation will be ordered during the retrieval of said
 * resources.
 *
 * The `device_resources_snmg` class is intended to be used in a single process to manage several
 * GPUs.
 */
class device_resources_snmg : public device_resources {
 public:
  /**
   * @brief Construct a SNMG resources instance with all available GPUs
   */
  device_resources_snmg() : device_resources()
  {
    RAFT_CUDA_TRY(cudaGetDevice(&main_gpu_id_));
    raft::resource::set_root_rank(*this, 0);

    // initialize all resources
    std::vector<raft::resources>& world_resources = raft::resource::get_multi_gpu_resource(*this);
    _init_world(world_resources);

    RAFT_CUDA_TRY(cudaSetDevice(main_gpu_id_));
  }

  /**
   * @brief Construct a SNMG resources instance with a subset of available GPUs
   *
   * @param[in] device_ids List of device IDs to be used
   */
  device_resources_snmg(const std::vector<int>& device_ids) : device_resources()
  {
    RAFT_CUDA_TRY(cudaGetDevice(&main_gpu_id_));
    raft::resource::set_root_rank(*this, 0);

    // initialize resources for the given device ids
    std::vector<raft::resources>& world_resources = raft::resource::get_multi_gpu_resource(*this);
    _init_world(world_resources, device_ids);
    RAFT_CUDA_TRY(cudaSetDevice(main_gpu_id_));
  }

  /**
   * @brief SNMG resources instance copy constructor
   *
   * @param[in] world A SNMG resources instance
   */
  device_resources_snmg(const device_resources_snmg& world)
    : device_resources(world), main_gpu_id_(world.main_gpu_id_)
  {
  }

  device_resources_snmg(device_resources_snmg&&)            = delete;
  device_resources_snmg& operator=(device_resources_snmg&&) = delete;
  ~device_resources_snmg() {};

  /**
   * @brief Set a memory pool on all GPUs of the multi-gpu world
   */
  void set_memory_pool(int percent_of_free_memory) const
  {
    int world_size = raft::resource::get_num_ranks(*this);
    for (int gpu_id = 0; gpu_id < world_size; gpu_id++) {
      const raft::resources& dev_res = raft::resource::set_current_device_to_rank(*this, gpu_id);
      // check limit for each device
      size_t limit = rmm::percent_of_free_device_memory(percent_of_free_memory);
      raft::resource::set_workspace_to_pool_resource(dev_res, limit);
    }
    RAFT_CUDA_TRY(cudaSetDevice(main_gpu_id_));
  }

  bool has_resource_factory(resource::resource_type resource_type) const override
  {
    if (snmg_related_resources.find(resource_type) == snmg_related_resources.end()) {
      // for resources unrelated to SNMG switch current GPU to main GPU ID
      RAFT_CUDA_TRY(cudaSetDevice(main_gpu_id_));
    }
    return raft::resources::has_resource_factory(resource_type);
  }

 private:
  inline void _init_world(std::vector<raft::resources>& world_resources)
  {
    int world_size;
    RAFT_CUDA_TRY(cudaGetDeviceCount(&world_size));

    for (int rank = 0; rank < world_size; rank++) {
      RAFT_CUDA_TRY(cudaSetDevice(rank));
      world_resources.emplace_back();

      // initialize the device ID
      raft::resource::get_device_id(world_resources.back());
    }
  }

  inline void _init_world(std::vector<raft::resources>& world_resources,
                          const std::vector<int>& device_ids)
  {
    for (size_t rank = 0; rank < device_ids.size(); rank++) {
      RAFT_CUDA_TRY(cudaSetDevice(device_ids[rank]));
      world_resources.emplace_back();

      // initialize the device ID
      raft::resource::get_device_id(world_resources.back());
    }
  }
  int main_gpu_id_;
};  // class device_resources_snmg

}  // namespace raft
