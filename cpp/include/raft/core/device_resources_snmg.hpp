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

#include <vector>

namespace raft {

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
   * @brief Construct a SNMG base resources instance with all available GPUs
   */
  device_resources_snmg() : device_resources()
  {
    RAFT_CUDA_TRY(cudaGetDevice(&main_gpu_id_));
    int root_rank = 0;
    raft::resource::set_root_rank(*this, root_rank);

    // initialize all resources
    std::vector<raft::resources>& world_resources = raft::resource::get_multi_gpu_resource(*this);
    _init_world(world_resources);
    RAFT_CUDA_TRY(cudaSetDevice(main_gpu_id_));
  }

  /**
   * @brief Construct a SNMG base resources instance with a subset of available GPUs
   *
   * @param[in] device_ids List of device IDs to be used
   */
  device_resources_snmg(const std::vector<int>& device_ids) : device_resources()
  {
    RAFT_CUDA_TRY(cudaGetDevice(&main_gpu_id_));
    int root_rank = 0;
    raft::resource::set_root_rank(*this, root_rank);

    // initialize resources for the given device ids
    std::vector<raft::resources>& clique = raft::resource::get_multi_gpu_resource(*this);
    _init_world(clique, device_ids);
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

  ~device_resources_snmg(){};

  /**
   * @brief Set a memory pool on all GPUs of the clique
   */
  void set_memory_pool(int percent_of_free_memory) const
  {
    int num_ranks = raft::resource::get_num_ranks(*this);
    for (int rank = 0; rank < num_ranks; rank++) {
      const raft::resources& dev_res = raft::resource::set_current_device_to_rank(*this, rank);
      // check limit for each device
      size_t limit = rmm::percent_of_free_device_memory(percent_of_free_memory);
      raft::resource::set_workspace_to_pool_resource(dev_res, limit);
    }
    cudaSetDevice(this->main_gpu_id_);
  }

  bool has_resource_factory(resource::resource_type resource_type) const override
  {
    if (resource_type != raft::resource::MULTI_GPU && resource_type != raft::resource::NCCL_COMM &&
        resource_type != raft::resource::CLIQUE_ROOT_RANK) {
      // for resources unrelated to SNMG switch current GPU to main GPU ID
      cudaSetDevice(this->main_gpu_id_);
    }
    return raft::resources::has_resource_factory(resource_type);
  }

  int main_gpu_id_;

 private:
  inline void _init_world(std::vector<raft::resources>& world_resources)
  {
    int num_ranks;
    RAFT_CUDA_TRY(cudaGetDeviceCount(&num_ranks));

    for (int rank = 0; rank < num_ranks; rank++) {
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
};  // class device_resources_snmg

}  // namespace raft
