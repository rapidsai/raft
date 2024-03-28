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

#include <raft/core/cusparse_macros.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

#include <cusparse_v2.h>

namespace raft::resource {
class cusparse_resource : public resource {
 public:
  cusparse_resource(rmm::cuda_stream_view stream)
  {
    RAFT_CUSPARSE_TRY_NO_THROW(cusparseCreate(&cusparse_res));
    RAFT_CUSPARSE_TRY_NO_THROW(cusparseSetStream(cusparse_res, stream));
  }

  ~cusparse_resource() { RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroy(cusparse_res)); }
  void* get_resource() override { return &cusparse_res; }

 private:
  cusparseHandle_t cusparse_res;
};

/**
 * Factory that knows how to construct a
 * specific raft::resource to populate
 * the res_t.
 */
class cusparse_resource_factory : public resource_factory {
 public:
  cusparse_resource_factory(rmm::cuda_stream_view stream) : stream_(stream) {}
  resource_type get_resource_type() override { return resource_type::CUSPARSE_HANDLE; }
  resource* make_resource() override { return new cusparse_resource(stream_); }

 private:
  rmm::cuda_stream_view stream_;
};

/**
 * @defgroup resource_cusparse cuSparse handle resource functions
 * @{
 */

/**
 * Load a cusparseres_t from raft res if it exists, otherwise
 * add it and return it.
 * @param[in] res the raft resources object
 * @return cusparse handle
 */
inline cusparseHandle_t get_cusparse_handle(resources const& res)
{
  if (!res.has_resource_factory(resource_type::CUSPARSE_HANDLE)) {
    rmm::cuda_stream_view stream = get_cuda_stream(res);
    res.add_resource_factory(std::make_shared<cusparse_resource_factory>(stream));
  }
  return *res.get_resource<cusparseHandle_t>(resource_type::CUSPARSE_HANDLE);
};

/**
 * @}
 */

}  // namespace raft::resource
