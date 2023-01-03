/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cuda_runtime.h>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

namespace raft::resource {

class cuda_event_resource : public resource {
 public:
  cuda_event_resource()
  {
    RAFT_CUDA_TRY_NO_THROW(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
  }
  void* get_resource() override { return &event_; }

  ~cuda_event_resource() override { RAFT_CUDA_TRY_NO_THROW(cudaEventDestroy(event_)); }

 private:
  cudaEvent_t event_;
};

/**
 * Factory that knows how to construct a
 * specific raft::resource to populate
 * the res_t.
 */
class cuda_stream_sync_event_resource_factory : public resource_factory {
 public:
  resource_type get_resource_type() override { return resource_type::CUDA_STREAM_SYNC_EVENT; }
  resource* make_resource() override { return new cuda_event_resource(); }
};

/**
 * Load a cudaStream_t from a res (and populate it on the res
 * if needed).
 * @param res raft res object for managing resources
 * @return
 */
inline cudaEvent_t& get_cuda_stream_sync_event(resources const& res)
{
  if (!res.has_resource_factory(resource_type::CUDA_STREAM_SYNC_EVENT)) {
    res.add_resource_factory(std::make_shared<cuda_stream_sync_event_resource_factory>());
  }
  return *res.get_resource<cudaEvent_t>(resource_type::CUDA_STREAM_SYNC_EVENT);
};

}  // namespace raft::resource
