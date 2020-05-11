/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <raft/mr/allocator.hpp>
#include <cuda_runtime.h>
#include <raft/cudart_utils.h>
#include <atomic>

namespace raft {
namespace mr {
namespace host {

/**
 * @brief An explicit interface for an asynchronous host allocations.
 *
 * This is mostly done in order to reduce work needed in cuML codebase.
 * An implementation of this interface can make the following assumptions,
 * further to the ones listed in `Allocator`:
 * - Allocations don't need to be zero copy accessible form a device.
 */
class allocator : public base_allocator {};

/** Default cudaMallocHost/cudaFreeHost based host allocator */
class default_allocator : public allocator {
 public:
  void* allocate(std::size_t n, cudaStream_t stream) override {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr, n));
    return ptr;
  }

  void deallocate(void* p, std::size_t n, cudaStream_t stream) override {
    ///@todo: enable this once logging is enabled in raft
    //CUDA_CHECK_NO_THROW(cudaFreeHost(p));
    CUDA_CHECK(cudaFreeHost(p));
  }
};  // class default_allocator

namespace {

allocator* get_default_impl() {
  static default_allocator obj;
  return &obj;
}

std::atomic<allocator*>& get_default() {
  static std::atomic<allocator*> alloc{get_default_impl()};
  return alloc;
}

}  // namespace

/**
 * @brief Gets the default host allocator
 *
 * This is thread-safe
 *
 * @return the allocator object
 */
allocator* get_default_allocator() {
  return get_default().load();
}

/**
 * @brief Sets the new default host allocator
 *
 * This is thread-safe
 *
 * @param[in] new_allocator the new host allocator that will be the default
 *                          If a nullptr is passed, the default allocator will
 *                          be reset to the one based on `default_allocator`
 * @return the old allocator
 */
allocator* set_default_allocator(allocator* new_allocator) {
  if (new_allocator == nullptr) {
    new_allocator = get_default();
  }
  return get_default().exchange(new_allocator);
}

};  // namespace host
};  // namespace mr
};  // namespace raft
