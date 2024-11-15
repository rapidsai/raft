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

#include <nccl.h>
#include <omp.h>

#include <memory>
#include <vector>

/**
 * @brief Error checking macro for NCCL runtime API functions.
 *
 * Invokes a NCCL runtime API function call, if the call does not return ncclSuccess, throws an
 * exception detailing the NCCL error that occurred
 */
#define RAFT_NCCL_TRY(call)                        \
  do {                                             \
    ncclResult_t const status = (call);            \
    if (ncclSuccess != status) {                   \
      std::string msg{};                           \
      SET_ERROR_MSG(msg,                           \
                    "NCCL error encountered at: ", \
                    "call='%s', Reason=%d:%s",     \
                    #call,                         \
                    status,                        \
                    ncclGetErrorString(status));   \
      throw raft::logic_error(msg);                \
    }                                              \
  } while (0);

namespace raft {

class device_resources_snmg : public resources {
 public:
  device_resources_snmg() : resources{}, root_rank_(0)
  {
    int num_ranks;
    RAFT_CUDA_TRY(cudaGetDeviceCount(&num_ranks));
    device_ids_.resize(num_ranks);
    std::iota(device_ids_.begin(), device_ids_.end(), 0);
    nccl_comms_.resize(num_ranks);
    initialize();
  }

  device_resources_snmg(const std::vector<int>& device_ids)
    : resources{}, root_rank_(0), device_ids_(device_ids), nccl_comms_(device_ids.size())
  {
    initialize();
  }

  device_resources_snmg(const device_resources_snmg& clique)
    : resources(clique),
      root_rank_(clique.root_rank_),
      device_ids_(clique.device_ids_),
      nccl_comms_(clique.nccl_comms_),
      device_resources_(clique.device_resources_)
  {
  }

  device_resources_snmg(device_resources_snmg&&)            = delete;
  device_resources_snmg& operator=(device_resources_snmg&&) = delete;

  inline int set_root_rank(int rank) { this->root_rank_ = rank; }

  inline int get_root_rank() const { return this->root_rank_; }

  inline int get_num_ranks() const { return this->device_ids_.size(); }

  inline int get_device_id(int rank) const { return this->device_ids_[rank]; }

  inline ncclComm_t get_nccl_comm(int rank) const { return this->nccl_comms_[rank]; }

  inline const raft::device_resources& get_device_resources(int rank) const
  {
    return this->device_resources_[rank];
  }

  inline const raft::device_resources& set_current_device_to_root_rank() const
  {
    int root_device_id = get_device_id(get_root_rank());
    RAFT_CUDA_TRY(cudaSetDevice(root_device_id));
    return get_device_resources(root_rank_);
  }

  inline const raft::device_resources& set_current_device_to_rank(int rank) const
  {
    RAFT_CUDA_TRY(cudaSetDevice(get_device_id(rank)));
    return get_device_resources(rank);
  }

  void set_memory_pool(int percent_of_free_memory) const
  {
    for (int rank = 0; rank < get_num_ranks(); rank++) {
      RAFT_CUDA_TRY(cudaSetDevice(get_device_id(rank)));
      size_t limit =
        rmm::percent_of_free_device_memory(percent_of_free_memory);  // check limit for each device
      raft::resource::set_workspace_to_pool_resource(get_device_resources(rank), limit);
    }
  }

  ~device_resources_snmg()
  {
#pragma omp parallel for  // necessary to avoid hangs
    for (int rank = 0; rank < get_num_ranks(); rank++) {
      RAFT_CUDA_TRY(cudaSetDevice(get_device_id(rank)));
      RAFT_NCCL_TRY(ncclCommDestroy(get_nccl_comm(rank)));
    }
  }

 private:
  void initialize()
  {
    RAFT_NCCL_TRY(ncclCommInitAll(nccl_comms_.data(), get_num_ranks(), device_ids_.data()));

    for (int rank = 0; rank < get_num_ranks(); rank++) {
      RAFT_CUDA_TRY(cudaSetDevice(get_device_id(rank)));
      device_resources_.emplace_back();

      // ideally add the ncclComm_t to the device_resources object with
      // raft::comms::build_comms_nccl_only
    }
  }

  int root_rank_;
  std::vector<int> device_ids_;
  std::vector<ncclComm_t> nccl_comms_;
  std::vector<raft::device_resources> device_resources_;

};  // class device_resources_snmg

}  // namespace raft
