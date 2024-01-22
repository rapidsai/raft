/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/logger.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
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

/**
 * This struct is used by multiple raft benchmark wrappers. It serves as a thread-safe keeper of
 * shared and private GPU resources (see below).
 *
 * - Accessing the same `configured_raft_resources` from concurrent threads is not safe.
 * - Accessing the copies of `configured_raft_resources` from concurrent threads is safe.
 * - There must be at most one "original" `configured_raft_resources` at any time, but as many
 *   copies of it as needed (modifies the program static state).
 */
class configured_raft_resources {
 public:
  using device_mr_t = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
  /**
   * This constructor has the shared state passed unmodified but creates the local state anew.
   * It's used by the copy constructor.
   */
  explicit configured_raft_resources(const std::shared_ptr<device_mr_t>& mr)
    : mr_{mr},
      sync_{[]() {
              auto* ev = new cudaEvent_t;
              RAFT_CUDA_TRY(cudaEventCreate(ev, cudaEventDisableTiming));
              return ev;
            }(),
            [](cudaEvent_t* ev) {
              RAFT_CUDA_TRY_NO_THROW(cudaEventDestroy(*ev));
              delete ev;
            }},
      res_{cudaStreamPerThread}
  {
  }

  /** Default constructor creates all resources anew. */
  configured_raft_resources()
    : configured_raft_resources{
        {[]() {
           auto* mr =
             new device_mr_t{rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull};
           rmm::mr::set_current_device_resource(mr);
           return mr;
         }(),
         [](device_mr_t* mr) {
           if (mr == nullptr) { return; }
           auto* cur_mr = dynamic_cast<device_mr_t*>(rmm::mr::get_current_device_resource());
           if (cur_mr != nullptr && (*cur_mr) == (*mr)) {
             // Normally, we'd always want to set the rmm resource back to the upstream of the pool
             // here. However, we expect some implementations may be buggy and mess up the rmm
             // resource, especially during development. This extra check here adds a little bit of
             // resilience: let the program crash/fail somewhere else rather than in the destructor
             // of the shared pointer.
             rmm::mr::set_current_device_resource(mr->get_upstream());
           }
           delete mr;
         }}}
  {
  }

  configured_raft_resources(configured_raft_resources&&)            = default;
  configured_raft_resources& operator=(configured_raft_resources&&) = default;
  ~configured_raft_resources()                                      = default;
  configured_raft_resources(const configured_raft_resources& res)
    : configured_raft_resources{res.mr_}
  {
  }
  configured_raft_resources& operator=(const configured_raft_resources& other)
  {
    this->mr_ = other.mr_;
    return *this;
  }

  operator raft::resources&() noexcept { return res_; }
  operator const raft::resources&() const noexcept { return res_; }

  /** Make the given stream wait on all work submitted to the resource. */
  void stream_wait(cudaStream_t stream) const
  {
    RAFT_CUDA_TRY(cudaEventRecord(*sync_, resource::get_cuda_stream(res_)));
    RAFT_CUDA_TRY(cudaStreamWaitEvent(stream, *sync_));
  }

  /** Get the internal sync event (which otherwise used only in `stream_wait`). */
  cudaEvent_t get_sync_event() const { return *sync_; }

 private:
  /**
   * This pool is set as the RMM current device, hence its shared among all users of RMM resources.
   * Its lifetime must be longer than that of any other cuda resources. It's not exposed and not
   * used by anyone directly.
   */
  std::shared_ptr<device_mr_t> mr_;
  /** Each benchmark wrapper must have its own copy of the synchronization event. */
  std::unique_ptr<cudaEvent_t, std::function<void(cudaEvent_t*)>> sync_;
  /**
   * Until we make the use of copies of raft::resources thread-safe, each benchmark wrapper must
   * have its own copy of it.
   */
  raft::device_resources res_;
};

}  // namespace raft::bench::ann
