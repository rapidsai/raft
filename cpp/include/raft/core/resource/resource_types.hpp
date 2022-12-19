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

namespace raft::core {

/**
 * A resource_t understands how to instantiate a specific
 * resource.
 */

class resource_t {
  virtual void* get_resource();

  virtual ~resource_t();
};

/**
 * Factory that knows how to construct a
 * specific raft::resource_t to populate
 * the handle_t.
 */
class resource_factory_t {
  virtual resource_type_t resource_type();

  virtual resource_t* make_resource();
};

enum resource_type_t {
  CUBLAS_HANDLE,
  CUSOLVER_DN_HANDLE,
  CUSOLVER_SP_HANDLE,
  CUSPARSE_HANDLE,
  CUDA_STREAM_VIEW,
  CUDA_STREAM_POOL,
  CUDA_STREAM_SYNC_EVENT,
  COMMUNICATOR
};

}  // end NAMESPACE raft::core
