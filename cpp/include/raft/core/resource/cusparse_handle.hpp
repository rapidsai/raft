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

#include <cusparse_v2.h>
#include <raft/core/cusparse_macros.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/resource_types.hpp>

namespace raft::core {
class cusparse_resource_t : public resource_t {
 public:
  cusparse_resource_t(cudaStream_t stream)
  {
    RAFT_CUSPARSE_TRY_NO_THROW(cusparseCreate_v2(&cusparse_handle));
    RAFT_CUSPARSE_TRY_NO_THROW(cusparseSetStream_v2(cusparse_handle, stream));
  }

  void* get_resource() { return &cusparse_handle; }

 private:
  cusparseHandle_t cusparse_handle;
};

/**
 * Factory that knows how to construct a
 * specific raft::resource_t to populate
 * the handle_t.
 */
class cusparse_resource_factory_t : public resource_factory_t {
  resource_type_t resource_type() { return resource_type_t::CUSPARSE_HANDLE; }
  resource_t* make_resource() { return new cusparse_resource_t(); }
};

/**
 * Load a cusparseHandle_t from raft handle if it exists, otherwise
 * add it and return it.
 * @param handle
 * @return
 */
cusparseeHandle_t get_cusparse_handle(raft::base_handle const& handle)
{
  if (!handle.has_resource_factory(resource_type_t::CUSPARSE_HANDLE)) {
    cudaStream_t stream = get_cuda_stream(handle);
    handle.add_resource_factory(std::make_shared<cusparse_resource_factory_t>(stream));
  }
  return *handle.get_resource<cusparseHandle_t>(resource_type_t::CUSPARSE_HANDLE);
};
}  // end NAMESPACE raft::core
