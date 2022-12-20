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

#include "cuda_stream.hpp"
#include <cusolverDn.h>
#include <raft/core/cusolver_macros.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <rmm/cuda_stream_view.hpp>

namespace raft::core {

/**
 *
 */
class cusolver_dn_resource_t : public resource_t {
 public:
  cusolver_dn_resource_t(rmm::cuda_stream_view stream)
  {
    RAFT_CUSOLVER_TRY_NO_THROW(cusolverDnCreate(&cusolver_handle));
    RAFT_CUSOLVER_TRY_NO_THROW(cusolverDnSetStream(cusolver_handle, stream));
  }

  void* get_resource() override { return &cusolver_handle; }

  ~cusolver_dn_resource_t() override
  {
    RAFT_CUSOLVER_TRY_NO_THROW(cusolverDnDestroy(cusolver_handle));
  }

 private:
  cusolverDnHandle_t cusolver_handle;
};

/**
 * Factory that knows how to construct a
 * specific raft::resource_t to populate
 * the handle_t.
 */
class cusolver_dn_resource_factory_t : public resource_factory_t {
 public:
  cusolver_dn_resource_factory_t(rmm::cuda_stream_view stream) : stream_(stream) {}
  resource_type_t resource_type() override { return resource_type_t::CUSOLVER_DN_HANDLE; }
  resource_t* make_resource() override { return new cusolver_dn_resource_t(stream_); }

 private:
  rmm::cuda_stream_view stream_;
};

/**
 * Load a cusolverSpHandle_t from raft handle if it exists, otherwise
 * add it and return it.
 * @param handle
 * @return
 */
cusolverDnHandle_t get_cusolver_dn_handle(base_handle_t const& handle)
{
  if (!handle.has_resource_factory(resource_type_t::CUSOLVER_DN_HANDLE)) {
    cudaStream_t stream = get_cuda_stream(handle);
    handle.add_resource_factory(std::make_shared<cusolver_dn_resource_factory_t>(stream));
  }
  return *handle.get_resource<cusolverDnHandle_t>(resource_type_t::CUSOLVER_DN_HANDLE);
};
}  // end NAMESPACE raft::core
