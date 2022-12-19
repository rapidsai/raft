/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cuda_runtime.h>
#include <raft/core/interruptible.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/cuda_stream_view.hpp>

class cuda_stream_resource_t : public resource_t {
 public:
  cuda_stream_resource_t() { RAFT_CUDA_TRY_NO_THROW(cudaStreamCreate(&stream)); }
  void* get_resource() { return &stream; }

  ~cuda_stream_resource_t() { RAFT_CUDA_TRY_NO_THROW(cudaStreamDestroy(stream)); }

 private:
  rmm::cuda_stream_view stream;
};

/**
 * Factory that knows how to construct a
 * specific raft::resource_t to populate
 * the handle_t.
 */
class cuda_stream_resource_factory_t : public resource_factory_t {
  resource_type_t resource_type() { return resource_type_t::CUDA_STREAM_VIEW; }
  resource_t* make_resource() { return new cuda_stream_resource_t(); }
};

/**
 * Load a cudaStream_t from a handle (and populate it on the handle
 * if needed).
 * @param handle raft handle object for managing resources
 * @return
 */
rmm::cuda_stream_view get_cuda_stream(raft::base_handle const& handle)
{
    if(!handle.has_resource_factory(resource_type_t::CUDA_STREAM_VIEW) {
    handle.add_resource_factory(std::make_shared<cuda_stream_resource_factory_t>());
    }
    return *handle.get_resource<cudaStream_t>(resource_type_t::CUDA_STREAM_VIEW);
};

/**
 * @brief synchronize a specific stream
 */
void sync_stream(const raft::base_handle_t& handle, rmm::cuda_stream_view stream) const
{
  interruptible::synchronize(stream);
}

/**
 * @brief synchronize main stream on the handle
 */
void sync_stream(const raft::base_handle_t& handle) const
{
  sync_stream(handle, get_cuda_stream(handle));
}