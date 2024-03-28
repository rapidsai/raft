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

#include <raft/core/resource/cuda_event.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuda_runtime.h>

namespace raft::resource::detail {

/**
 * Factory that knows how to construct a specific raft::resource to populate
 * the res_t.
 */
class cuda_stream_sync_event_resource_factory : public resource_factory {
 public:
  resource_type get_resource_type() override { return resource_type::CUDA_STREAM_SYNC_EVENT; }
  resource* make_resource() override { return new cuda_event_resource(); }
};

/**
 * Load a cudaEvent from a resources instance (and populate it on the resources instance)
 * if needed) for syncing the main cuda stream.
 * @param res raft resources instance for managing resources
 * @return
 */
inline cudaEvent_t& get_cuda_stream_sync_event(resources const& res)
{
  if (!res.has_resource_factory(resource_type::CUDA_STREAM_SYNC_EVENT)) {
    res.add_resource_factory(std::make_shared<cuda_stream_sync_event_resource_factory>());
  }
  return *res.get_resource<cudaEvent_t>(resource_type::CUDA_STREAM_SYNC_EVENT);
};

}  // namespace raft::resource::detail
