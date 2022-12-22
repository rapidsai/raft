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

namespace raft::resource {

enum resource_type {
  // device-specific resource types
  CUBLAS_res = 0,
  CUSOLVER_DN_res,
  CUSOLVER_SP_res,
  CUSPARSE_res,
  CUDA_STREAM_VIEW,
  CUDA_STREAM_POOL,
  CUDA_STREAM_SYNC_EVENT,
  COMMUNICATOR,
  SUB_COMMUNICATOR,
  DEVICE_PROPERTIES,
  DEVICE_ID,
  THRUST_POLICY
};

/**
 * A resource understands how to instantiate a specific
 * resource.
 */

class resource {
 public:
  virtual void* get_resource() = 0;

  virtual ~resource() {}
};

/**
 * Factory that knows how to construct a
 * specific raft::resource to populate
 * the res_t.
 */
class resource_factory {
 public:
  virtual resource_type get_resource_type() = 0;

  virtual resource* make_resource() = 0;
};

}  // namespace raft::resource
