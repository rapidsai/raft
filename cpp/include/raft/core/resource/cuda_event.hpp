/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuda_runtime.h>

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
}  // namespace raft::resource
