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

#include <raft/core/cublas_macros.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

#include <cublas_v2.h>

namespace raft::resource {

class cublas_resource : public resource {
 public:
  cublas_resource(rmm::cuda_stream_view stream)
  {
    RAFT_CUBLAS_TRY_NO_THROW(cublasCreate(&cublas_res));
    RAFT_CUBLAS_TRY_NO_THROW(cublasSetStream(cublas_res, stream));
  }

  ~cublas_resource() override { RAFT_CUBLAS_TRY_NO_THROW(cublasDestroy(cublas_res)); }

  void* get_resource() override { return &cublas_res; }

 private:
  cublasHandle_t cublas_res;
};

/**
 * Factory that knows how to construct a
 * specific raft::resource to populate
 * the res_t.
 */
class cublas_resource_factory : public resource_factory {
 public:
  cublas_resource_factory(rmm::cuda_stream_view stream) : stream_(stream) {}
  resource_type get_resource_type() override { return resource_type::CUBLAS_HANDLE; }
  resource* make_resource() override { return new cublas_resource(stream_); }

 private:
  rmm::cuda_stream_view stream_;
};

/**
 * @defgroup resource_cublas cuBLAS handle resource functions
 * @{
 */

/**
 * Load a `cublasHandle_t` from raft res if it exists, otherwise add it and return it.
 *
 * @param[in] res the raft resources object
 * @return cublas handle
 */
inline cublasHandle_t get_cublas_handle(resources const& res)
{
  if (!res.has_resource_factory(resource_type::CUBLAS_HANDLE)) {
    cudaStream_t stream = get_cuda_stream(res);
    res.add_resource_factory(std::make_shared<cublas_resource_factory>(stream));
  }
  auto ret = *res.get_resource<cublasHandle_t>(resource_type::CUBLAS_HANDLE);
  RAFT_CUBLAS_TRY(cublasSetStream(ret, get_cuda_stream(res)));
  return ret;
};

/**
 * @}
 */

}  // namespace raft::resource
