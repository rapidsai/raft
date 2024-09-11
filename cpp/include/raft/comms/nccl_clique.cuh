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

#include <raft/comms/nccl_clique.hpp>
#include <raft/core/device_resources.hpp>

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

namespace raft::comms {
void build_comms_nccl_only(raft::resources* handle, ncclComm_t nccl_comm, int num_ranks, int rank);
}

namespace raft::comms {

nccl_clique::nccl_clique(int percent_of_free_memory)
  : root_rank_(0),
    percent_of_free_memory_(percent_of_free_memory),
    per_device_pools_(0),
    device_resources_(0)
{
  cudaGetDeviceCount(&num_ranks_);
  device_ids_.resize(num_ranks_);
  std::iota(device_ids_.begin(), device_ids_.end(), 0);
  nccl_comms_.resize(num_ranks_);
  nccl_clique_init();
}

nccl_clique::nccl_clique(const std::vector<int>& device_ids, int percent_of_free_memory)
  : root_rank_(0),
    num_ranks_(device_ids.size()),
    percent_of_free_memory_(percent_of_free_memory),
    device_ids_(device_ids),
    nccl_comms_(device_ids.size()),
    per_device_pools_(0),
    device_resources_(0)
{
  nccl_clique_init();
}

void nccl_clique::nccl_clique_init()
{
  RAFT_NCCL_TRY(ncclCommInitAll(nccl_comms_.data(), num_ranks_, device_ids_.data()));

  for (int rank = 0; rank < num_ranks_; rank++) {
    RAFT_CUDA_TRY(cudaSetDevice(device_ids_[rank]));

    // create a pool memory resource for each device
    auto old_mr = rmm::mr::get_current_device_resource();
    per_device_pools_.push_back(std::make_unique<pool_mr>(
      old_mr, rmm::percent_of_free_device_memory(percent_of_free_memory_)));
    rmm::cuda_device_id id(device_ids_[rank]);
    rmm::mr::set_per_device_resource(id, per_device_pools_.back().get());

    // create a device resource handle for each device
    device_resources_.emplace_back();

    // add NCCL communications to the device resource handle
    raft::comms::build_comms_nccl_only(
      &device_resources_[rank], nccl_comms_[rank], num_ranks_, rank);
  }

  for (int rank = 0; rank < num_ranks_; rank++) {
    RAFT_CUDA_TRY(cudaSetDevice(device_ids_[rank]));
    raft::resource::sync_stream(device_resources_[rank]);
  }
}

const raft::device_resources& nccl_clique::set_current_device_to_root_rank() const
{
  int root_device_id = device_ids_[root_rank_];
  RAFT_CUDA_TRY(cudaSetDevice(root_device_id));
  return device_resources_[root_rank_];
}

nccl_clique::~nccl_clique()
{
#pragma omp parallel for  // necessary to avoid hangs
  for (int rank = 0; rank < num_ranks_; rank++) {
    cudaSetDevice(device_ids_[rank]);
    ncclCommDestroy(nccl_comms_[rank]);
    rmm::cuda_device_id id(device_ids_[rank]);
    rmm::mr::set_per_device_resource(id, nullptr);
  }
}

}  // namespace raft::comms
