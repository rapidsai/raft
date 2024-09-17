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

#include "cuda_stream.hpp"

#include <raft/core/cusolver_macros.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cusolverDn.h>

namespace raft::resource {

/**
 *
 */
class cusolver_dn_resource : public resource {
 public:
  cusolver_dn_resource(rmm::cuda_stream_view stream)
  {
    RAFT_CUSOLVER_TRY_NO_THROW(cusolverDnCreate(&cusolver_res));
    RAFT_CUSOLVER_TRY_NO_THROW(cusolverDnSetStream(cusolver_res, stream));
  }

  void* get_resource() override { return &cusolver_res; }

  ~cusolver_dn_resource() override { RAFT_CUSOLVER_TRY_NO_THROW(cusolverDnDestroy(cusolver_res)); }

 private:
  cusolverDnHandle_t cusolver_res;
};

/**
 * @defgroup resource_cusolver_dn cuSolver DN handle resource functions
 * @{
 */

/**
 * Factory that knows how to construct a
 * specific raft::resource to populate
 * the res_t.
 */
class cusolver_dn_resource_factory : public resource_factory {
 public:
  cusolver_dn_resource_factory(rmm::cuda_stream_view stream) : stream_(stream) {}
  resource_type get_resource_type() override { return resource_type::CUSOLVER_DN_HANDLE; }
  resource* make_resource() override { return new cusolver_dn_resource(stream_); }

 private:
  rmm::cuda_stream_view stream_;
};

/**
 * Load a cusolverSpres_t from raft res if it exists, otherwise
 * add it and return it.
 * @param[in] res the raft resources object
 * @return cusolver dn handle
 */
inline cusolverDnHandle_t get_cusolver_dn_handle(resources const& res)
{
  if (!res.has_resource_factory(resource_type::CUSOLVER_DN_HANDLE)) {
    cudaStream_t stream = get_cuda_stream(res);
    res.add_resource_factory(std::make_shared<cusolver_dn_resource_factory>(stream));
  }
  return *res.get_resource<cusolverDnHandle_t>(resource_type::CUSOLVER_DN_HANDLE);
};

/**
 * @}
 */

}  // namespace raft::resource
