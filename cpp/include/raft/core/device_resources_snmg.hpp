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

#include <raft/core/device_resources.hpp>
#include <raft/core/resource/nccl_clique.hpp>

#include <nccl.h>

#include <memory>
#include <numeric>
#include <vector>

namespace raft {

/**
 * @brief SNMG (single-node multi-GPU) resource container object that stores a NCCL clique and all
 * necessary resources used for calling device functions, cuda kernels, libraries and/or NCCL
 * communications on each GPU. Note the `device_resources_snmg` object can also be used as a classic
 * `device_resources` object. The associated resources will be the ones of the GPU used during
 * object instantiation and a GPU switch operation will be ordered during the retrieval of said
 * resources.
 *
 * The `device_resources_snmg` class is intended to be used in a single process to manage several
 * GPUs. Please note that NCCL communications are the responsibility of the user. Blocking NCCL
 * calls will sometimes require the use of several threads to avoid hangs.
 */
class device_resources_snmg : public device_resources {
 public:
  /**
   * @brief Construct a SNMG resources instance with all available GPUs
   */
  device_resources_snmg() : device_resources()
  {
    RAFT_CUDA_TRY(cudaGetDevice(&main_gpu_id_));

    // set root rank
    int root_rank = 0;
    raft::resource::set_clique_root_rank(*this, root_rank);

    // initialize the NCCL clique
    raft::resource::get_nccl_clique(*this);
  }

  /**
   * @brief Construct a SNMG resources instance with a subset of available GPUs
   *
   * @param[in] device_ids List of device IDs to be used by the NCCL clique
   */
  device_resources_snmg(const std::vector<int>& device_ids) : device_resources()
  {
    RAFT_CUDA_TRY(cudaGetDevice(&main_gpu_id_));

    // set root rank
    int root_rank = 0;
    raft::resource::set_clique_root_rank(*this, root_rank);

    // initialize the NCCL clique
    raft::resource::get_nccl_clique(*this, device_ids);
  }

  /**
   * @brief SNMG resources instance copy constructor
   *
   * @param[in] clique A SNMG resources instance
   */
  device_resources_snmg(const device_resources_snmg& clique)
    : device_resources(clique), main_gpu_id_(clique.main_gpu_id_)
  {
  }

  device_resources_snmg(device_resources_snmg&&)            = delete;
  device_resources_snmg& operator=(device_resources_snmg&&) = delete;

  /**
   * @brief Set root rank of NCCL clique
   */
  inline void set_root_rank(int rank) { raft::resource::set_clique_root_rank(*this, rank); }

  /**
   * @brief Get root rank of NCCL clique
   */
  inline int get_root_rank() const { return raft::resource::get_clique_root_rank(*this); }

  /**
   * @brief Get number of ranks in NCCL clique
   */
  inline int get_num_ranks() const { return raft::resource::get_num_ranks(*this); }

  /**
   * @brief Get device ID of rank in NCCL clique
   */
  inline int get_device_id_of_rank(int rank) const
  {
    const raft::resources& dev_res = raft::resource::get_device_resources(*this, rank);
    return raft::resource::get_device_id(dev_res);
  }

  /**
   * @brief Get NCCL comm object of rank in NCCL clique
   */
  inline ncclComm_t& get_nccl_comm(int rank) const
  {
    return raft::resource::get_nccl_comm(*this, rank);
  }

  /**
   * @brief Get raft::resources object of rank in NCCL clique
   */
  inline const raft::resources& get_device_resources(int rank) const
  {
    return raft::resource::get_device_resources(*this, rank);
  }

  /**
   * @brief Set current device ID to rank and return its raft::resources object
   */
  inline const raft::resources& set_current_device_to_rank(int rank) const
  {
    return raft::resource::set_current_device_to_rank(*this, rank);
  }

  /**
   * @brief Set current device ID to root rank and return its raft::resources object
   */
  inline const raft::resources& set_current_device_to_root_rank() const
  {
    return raft::resource::set_current_device_to_root_rank(*this);
  }

  /**
   * @brief Set a memory pool on all GPUs of the clique
   */
  void set_memory_pool(int percent_of_free_memory) const
  {
    for (int rank = 0; rank < get_num_ranks(); rank++) {
      const raft::resources& dev_res = set_current_device_to_rank(rank);
      // check limit for each device
      size_t limit = rmm::percent_of_free_device_memory(percent_of_free_memory);
      raft::resource::set_workspace_to_pool_resource(dev_res, limit);
    }
    cudaSetDevice(this->main_gpu_id_);
  }

  bool has_resource_factory(resource::resource_type resource_type) const override
  {
    if (resource_type != raft::resource::NCCL_CLIQUE &&
        resource_type != raft::resource::NCCL_COMM &&
        resource_type != raft::resource::CLIQUE_ROOT_RANK)
      // for resources unrelated to SNMG switch current GPU to main GPU ID
      cudaSetDevice(this->main_gpu_id_);
    return raft::resources::has_resource_factory(resource_type);
  }

 private:
  int main_gpu_id_;
};  // class device_resources_snmg

}  // namespace raft
