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

#include <vector>
#include <nccl.h>
#include <raft/core/resources.hpp>
#include <raft/comms/detail/util.hpp>

namespace raft::comms {
  void build_comms_nccl_only(resources* handle, ncclComm_t nccl_comm, int num_ranks, int rank);
}

namespace raft::neighbors::mg {

struct nccl_clique {

  nccl_clique(const std::vector<int>& device_ids)
    : root_rank_(0),
      num_ranks_(device_ids.size()),
      device_ids_(device_ids),
      nccl_comms_(device_ids.size()),
      device_resources_(0)
  {
    RAFT_NCCL_TRY(ncclCommInitAll(nccl_comms_.data(), num_ranks_, device_ids_.data()));

    for (int rank = 0; rank < num_ranks_; rank++) {
      RAFT_CUDA_TRY(cudaSetDevice(device_ids[rank]));
      device_resources_.emplace_back();
      raft::comms::build_comms_nccl_only(&device_resources_[rank], nccl_comms_[rank], num_ranks_, rank);
    }
  }

  const raft::device_resources& set_current_device_to_root_rank() const
  {
    int root_device_id = device_ids_[root_rank_];
    RAFT_CUDA_TRY(cudaSetDevice(root_device_id));
    return device_resources_[root_rank_];
  }

  ~nccl_clique()
  {
    for (int rank = 0; rank < num_ranks_; rank++) {
      cudaSetDevice(device_ids_[rank]);
      ncclCommDestroy(nccl_comms_[rank]);
    }
  }

  int root_rank_;
  int num_ranks_;
  std::vector<int> device_ids_;
  std::vector<ncclComm_t> nccl_comms_;
  std::vector<raft::device_resources> device_resources_;
};

}  // namespace raft::neighbors::mg