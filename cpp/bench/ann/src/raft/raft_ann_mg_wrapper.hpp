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

#include "../common/ann_types.hpp"
#include "raft_ann_bench_utils.h"
#include <raft/neighbors/ann_mg_helpers.cuh>

namespace raft::bench::ann {

template <typename T>
class RaftAnnMG : public ANN<T>, public AnnGPU {

  public:
    RaftAnnMG(Metric metric, int dim)
      : ANN<T>(metric, dim), dimension_(dim)
    {
      this->init_nccl_clique();
    }

    AlgoProperty get_preference() const override
    {
      AlgoProperty property;
      property.dataset_memory_type = MemoryType::HostMmap;
      property.query_memory_type   = MemoryType::HostMmap;
      return property;
    }

  private:
    void init_nccl_clique() {
      int n_devices;
      cudaGetDeviceCount(&n_devices);
      std::cout << n_devices << " GPUs detected" << std::endl;

      std::vector<int> device_ids(n_devices);
      std::iota(device_ids.begin(), device_ids.end(), 0);
      clique_ = std::make_shared<raft::neighbors::mg::nccl_clique>(device_ids);
    }

    [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
    {
      const auto& handle = clique_->set_current_device_to_root_rank();
      return resource::get_cuda_stream(handle);
    }

  protected:
    std::shared_ptr<raft::neighbors::mg::nccl_clique> clique_;
    int dimension_;
};

}
