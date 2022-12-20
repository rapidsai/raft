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

#include <cublas_v2.h>
#include <raft/core/cublas_macros.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/resource_types.hpp>

namespace raft::core {
class cublas_resource_t : public resource_t {
 public:
  cublas_resource_t(rmm::cuda_stream_view stream)
  {
    RAFT_CUBLAS_TRY_NO_THROW(cublasCreate(&cublas_handle));
    RAFT_CUBLAS_TRY_NO_THROW(cublasSetStream(cublas_handle, stream));
  }

  ~cublas_resource_t() override { RAFT_CUBLAS_TRY_NO_THROW(cublasDestroy(cublas_handle)); }

  void* get_resource() override { return &cublas_handle; }

 private:
  cublasHandle_t cublas_handle;
};

/**
 * Factory that knows how to construct a
 * specific raft::resource_t to populate
 * the handle_t.
 */
class cublas_resource_factory_t : public resource_factory_t {
 public:
  cublas_resource_factory_t(rmm::cuda_stream_view stream) : stream_(stream) {}
  resource_type_t resource_type() override { return resource_type_t::CUBLAS_HANDLE; }
  resource_t* make_resource() override { return new cublas_resource_t(stream_); }

 private:
  rmm::cuda_stream_view stream_;
};

/**
 * Load a cublasHandle_t from raft handle if it exists, otherwise
 * add it and return it.
 * @param handle
 * @return
 */
inline cublasHandle_t get_cublas_handle(base_handle_t const& handle)
{
  if (!handle.has_resource_factory(resource_type_t::CUBLAS_HANDLE)) {
    cudaStream_t stream = get_cuda_stream(handle);
    handle.add_resource_factory(std::make_shared<cublas_resource_factory_t>(stream));
  }
  return *handle.get_resource<cublasHandle_t>(resource_type_t::CUBLAS_HANDLE);
};
}  // end NAMESPACE raft::core
