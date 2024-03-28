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

#include <raft/core/cusolver_macros.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

#include <cusolverSp.h>

namespace raft::resource {

/**
 *
 */
class cusolver_sp_resource : public resource {
 public:
  cusolver_sp_resource(rmm::cuda_stream_view stream)
  {
    RAFT_CUSOLVER_TRY_NO_THROW(cusolverSpCreate(&cusolver_res));
    RAFT_CUSOLVER_TRY_NO_THROW(cusolverSpSetStream(cusolver_res, stream));
  }

  void* get_resource() override { return &cusolver_res; }

  ~cusolver_sp_resource() override { RAFT_CUSOLVER_TRY_NO_THROW(cusolverSpDestroy(cusolver_res)); }

 private:
  cusolverSpHandle_t cusolver_res;
};

/**
 * Factory that knows how to construct a
 * specific raft::resource to populate
 * the res_t.
 */
class cusolver_sp_resource_factory : public resource_factory {
 public:
  cusolver_sp_resource_factory(rmm::cuda_stream_view stream) : stream_(stream) {}
  resource_type get_resource_type() override { return resource_type::CUSOLVER_SP_HANDLE; }
  resource* make_resource() override { return new cusolver_sp_resource(stream_); }

 private:
  rmm::cuda_stream_view stream_;
};

/**
 * @defgroup resource_cusolver_sp cuSolver SP handle resource functions
 * @{
 */

/**
 * Load a cusolverSpres_t from raft res if it exists, otherwise
 * add it and return it.
 * @param[in] res the raft resources object
 * @return cusolver sp handle
 */
inline cusolverSpHandle_t get_cusolver_sp_handle(resources const& res)
{
  if (!res.has_resource_factory(resource_type::CUSOLVER_SP_HANDLE)) {
    cudaStream_t stream = get_cuda_stream(res);
    res.add_resource_factory(std::make_shared<cusolver_sp_resource_factory>(stream));
  }
  return *res.get_resource<cusolverSpHandle_t>(resource_type::CUSOLVER_SP_HANDLE);
};

/**
 * @}
 */

}  // namespace raft::resource
