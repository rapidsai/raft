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
#pragma once

#include <cuda_runtime.h>
#include <raft/core/resource/cuda_event.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <rmm/cuda_stream_pool.hpp>

namespace raft::core {

class cuda_stream_pool_resource_t : public resource_t {
 public:
  cuda_stream_pool_resource_t(std::shared_ptr<rmm::cuda_stream_pool> stream_pool)
    : stream_pool_(stream_pool)
  {
  }

  ~cuda_stream_pool_resource_t() override {}
  void* get_resource() override { return &stream_pool_; }

 private:
  std::shared_ptr<rmm::cuda_stream_pool> stream_pool_{nullptr};
};

/**
 * Factory that knows how to construct a
 * specific raft::resource_t to populate
 * the handle_t.
 */
class cuda_stream_pool_resource_factory_t : public resource_factory_t {
 public:
  cuda_stream_pool_resource_factory_t(
    std::shared_ptr<rmm::cuda_stream_pool> stream_pool = {nullptr})
    : stream_pool_(stream_pool)
  {
  }

  resource_type_t resource_type() override { return resource_type_t::CUDA_STREAM_POOL; }
  resource_t* make_resource() override { return new cuda_stream_pool_resource_t(stream_pool_); }

 private:
  std::shared_ptr<rmm::cuda_stream_pool> stream_pool_{nullptr};
};

inline bool is_stream_pool_initialized(const base_handle_t& handle)
{
  return *handle.get_resource<std::shared_ptr<rmm::cuda_stream_pool>>(
           resource_type_t::CUDA_STREAM_POOL) != nullptr;
}

/**
 * Load a cuda_stream_pool, and create a new one if it doesn't already exist
 * @param handle raft handle object for managing resources
 * @return
 */
inline const rmm::cuda_stream_pool& get_cuda_stream_pool(const base_handle_t& handle)
{
  if (!handle.has_resource_factory(resource_type_t::CUDA_STREAM_POOL)) {
    handle.add_resource_factory(std::make_shared<cuda_stream_pool_resource_factory_t>());
  }
  return *(*handle.get_resource<std::shared_ptr<rmm::cuda_stream_pool>>(
    resource_type_t::CUDA_STREAM_POOL));
};

/**
 * Explicitly set a stream pool on the current handle. Note that this will overwrite
 * an existing stream pool on the handle.
 * @param handle
 * @param stream_pool
 */
inline void set_cuda_stream_pool(const base_handle_t& handle,
                                 std::shared_ptr<rmm::cuda_stream_pool> stream_pool)
{
  handle.add_resource_factory(std::make_shared<cuda_stream_pool_resource_factory_t>(stream_pool));
};

inline std::size_t get_stream_pool_size(const base_handle_t& handle)
{
  return is_stream_pool_initialized(handle) ? get_cuda_stream_pool(handle).get_pool_size() : 0;
}

/**
 * @brief return stream from pool
 */
inline rmm::cuda_stream_view get_stream_from_stream_pool(const base_handle_t& handle)
{
  RAFT_EXPECTS(is_stream_pool_initialized(handle),
               "ERROR: rmm::cuda_stream_pool was not initialized");
  return get_cuda_stream_pool(handle).get_stream();
}

/**
 * @brief return stream from pool at index
 */
inline rmm::cuda_stream_view get_stream_from_stream_pool(const base_handle_t& handle,
                                                         std::size_t stream_idx)
{
  RAFT_EXPECTS(is_stream_pool_initialized(handle),
               "ERROR: rmm::cuda_stream_pool was not initialized");
  return get_cuda_stream_pool(handle).get_stream(stream_idx);
}

/**
 * @brief return stream from pool if size > 0, else main stream on handle
 */
inline rmm::cuda_stream_view get_next_usable_stream(const base_handle_t& handle)
{
  return is_stream_pool_initialized(handle) ? get_stream_from_stream_pool(handle)
                                            : get_cuda_stream(handle);
}

/**
 * @brief return stream from pool at index if size > 0, else main stream on handle
 *
 * @param[in] stream_idx the required index of the stream in the stream pool if available
 */
inline rmm::cuda_stream_view get_next_usable_stream(const base_handle_t& handle,
                                                    std::size_t stream_idx)
{
  return is_stream_pool_initialized(handle) ? get_stream_from_stream_pool(handle, stream_idx)
                                            : get_cuda_stream(handle);
}

/**
 * @brief synchronize the stream pool on the handle
 */
inline void sync_stream_pool(const base_handle_t& handle)
{
  for (std::size_t i = 0; i < get_stream_pool_size(handle); i++) {
    sync_stream(handle, get_cuda_stream_pool(handle).get_stream(i));
  }
}

/**
 * @brief synchronize subset of stream pool
 *
 * @param[in] stream_indices the indices of the streams in the stream pool to synchronize
 */
inline void sync_stream_pool(const base_handle_t& handle,
                             const std::vector<std::size_t> stream_indices)
{
  RAFT_EXPECTS(is_stream_pool_initialized(handle),
               "ERROR: rmm::cuda_stream_pool was not initialized");
  for (const auto& stream_index : stream_indices) {
    sync_stream(handle, get_cuda_stream_pool(handle).get_stream(stream_index));
  }
}

/**
 * @brief ask stream pool to wait on last event in main stream
 */
inline void wait_stream_pool_on_stream(const base_handle_t& handle)
{
  printf("waiting on stream pooll\n");
  printf("stream pool size: %ld\n", get_stream_pool_size(handle));
  cudaEvent_t event = get_cuda_stream_sync_event(handle);
  RAFT_CUDA_TRY(cudaEventRecord(event, get_cuda_stream(handle)));
  for (std::size_t i = 0; i < get_stream_pool_size(handle); i++) {
    RAFT_CUDA_TRY(cudaStreamWaitEvent(get_cuda_stream_pool(handle).get_stream(i), event, 0));
  }
}
}  // namespace raft::core
