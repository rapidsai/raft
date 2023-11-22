/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace raft::bench::ann {

inline raft::distance::DistanceType parse_metric_type(raft::bench::ann::Metric metric)
{
  if (metric == raft::bench::ann::Metric::kInnerProduct) {
    return raft::distance::DistanceType::InnerProduct;
  } else if (metric == raft::bench::ann::Metric::kEuclidean) {
    // Even for L2 expanded RAFT IVF Flat uses unexpanded formula
    return raft::distance::DistanceType::L2Expanded;
  } else {
    throw std::runtime_error("raft supports only metric type of inner product and L2");
  }
}

class configured_raft_resources {
 public:
  configured_raft_resources()
    : mr_{rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull},
      res_{cudaStreamPerThread},
      sync_{nullptr}
  {
    rmm::mr::set_current_device_resource(&mr_);
    RAFT_CUDA_TRY(cudaEventCreate(&sync_, cudaEventDisableTiming));
  }

  ~configured_raft_resources() noexcept
  {
    RAFT_CUDA_TRY_NO_THROW(cudaEventDestroy(sync_));
    if (rmm::mr::get_current_device_resource()->is_equal(mr_)) {
      rmm::mr::set_current_device_resource(mr_.get_upstream());
    }
  }

  operator raft::resources&() noexcept { return res_; }
  operator const raft::resources&() const noexcept { return res_; }

  /** Make the given stream wait on all work submitted to the resource. */
  void stream_wait(cudaStream_t stream) const
  {
    RAFT_CUDA_TRY(cudaEventRecord(sync_, resource::get_cuda_stream(res_)));
    RAFT_CUDA_TRY(cudaStreamWaitEvent(stream, sync_));
  }

  /** Get the internal sync event (which otherwise used only in `stream_wait`). */
  cudaEvent_t get_sync_event() const { return sync_; }

 private:
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> mr_;
  raft::device_resources res_;
  cudaEvent_t sync_;
};

}  // namespace raft::bench::ann
