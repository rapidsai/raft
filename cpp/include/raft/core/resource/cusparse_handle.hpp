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

#include <cusparse_v2.h>
#include <raft/core/cusparse_macros.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/resource_types.hpp>

namespace raft::core {
class cusparse_resource_t : public resource_t {
 public:
  cusparse_resource_t(rmm::cuda_stream_view stream)
  {
    RAFT_CUSPARSE_TRY_NO_THROW(cusparseCreate(&cusparse_handle));
    RAFT_CUSPARSE_TRY_NO_THROW(cusparseSetStream(cusparse_handle, stream));
  }

  ~cusparse_resource_t() { RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroy(cusparse_handle)); }
  void* get_resource() override { return &cusparse_handle; }

 private:
  cusparseHandle_t cusparse_handle;
};

/**
 * Factory that knows how to construct a
 * specific raft::resource_t to populate
 * the handle_t.
 */
class cusparse_resource_factory_t : public resource_factory_t {
 public:
  cusparse_resource_factory_t(rmm::cuda_stream_view stream) : stream_(stream) {}
  resource_type_t resource_type() override { return resource_type_t::CUSPARSE_HANDLE; }
  resource_t* make_resource() override { return new cusparse_resource_t(stream_); }

 private:
  rmm::cuda_stream_view stream_;
};

/**
 * Load a cusparseHandle_t from raft handle if it exists, otherwise
 * add it and return it.
 * @param handle
 * @return
 */
cusparseHandle_t get_cusparse_handle(base_handle_t const& handle)
{
  if (!handle.has_resource_factory(resource_type_t::CUSPARSE_HANDLE)) {
    rmm::cuda_stream_view stream = get_cuda_stream(handle);
    handle.add_resource_factory(std::make_shared<cusparse_resource_factory_t>(stream));
  }
  return *handle.get_resource<cusparseHandle_t>(resource_type_t::CUSPARSE_HANDLE);
};
}  // end NAMESPACE raft::core
