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

namespace raft::resource {

/**
 * @defgroup resource_types Core resource vocabulary types
 * @{
 */

/**
 * @brief Resource types can apply to any resource and don't have to be host- or device-specific.
 */
enum resource_type {
  // device-specific resource types
  CUBLAS_HANDLE = 0,         // cublas handle
  CUSOLVER_DN_HANDLE,        // cusolver dn handle
  CUSOLVER_SP_HANDLE,        // cusolver sp handle
  CUSPARSE_HANDLE,           // cusparse handle
  CUDA_STREAM_VIEW,          // view of a cuda stream
  CUDA_STREAM_POOL,          // cuda stream pool
  CUDA_STREAM_SYNC_EVENT,    // cuda event for syncing streams
  COMMUNICATOR,              // raft communicator
  SUB_COMMUNICATOR,          // raft sub communicator
  DEVICE_PROPERTIES,         // cuda device properties
  DEVICE_ID,                 // cuda device id
  STREAM_VIEW,               // view of a cuda stream or a placeholder in
                             // CUDA-free builds
  THRUST_POLICY,             // thrust execution policy
  WORKSPACE_RESOURCE,        // rmm device memory resource for small temporary allocations
  CUBLASLT_HANDLE,           // cublasLt handle
  CUSTOM,                    // runtime-shared default-constructible resource
  LARGE_WORKSPACE_RESOURCE,  // rmm device memory resource for somewhat large temporary allocations

  LAST_KEY  // reserved for the last key
};

/**
 * @brief A resource constructs and contains an instance of
 * some pre-determined object type and facades that object
 * behind a common API.
 */
class resource {
 public:
  virtual void* get_resource() = 0;

  virtual ~resource() {}
};

class empty_resource : public resource {
 public:
  empty_resource() : resource() {}

  void* get_resource() override { return nullptr; }

  ~empty_resource() override {}
};

/**
 * @brief A resource factory knows how to construct an instance of
 * a specific raft::resource::resource.
 */
class resource_factory {
 public:
  /**
   * @brief Return the resource_type associated with the current factory
   * @return resource_type corresponding to the current factory
   */
  virtual resource_type get_resource_type() = 0;

  /**
   * @brief Construct an instance of the factory's underlying resource.
   * @return resource instance
   */
  virtual resource* make_resource() = 0;

  virtual ~resource_factory() {}
};

/**
 * @brief A resource factory knows how to construct an instance of
 * a specific raft::resource::resource.
 */
class empty_resource_factory : public resource_factory {
 public:
  empty_resource_factory() : resource_factory() {}
  /**
   * @brief Return the resource_type associated with the current factory
   * @return resource_type corresponding to the current factory
   */
  resource_type get_resource_type() override { return resource_type::LAST_KEY; }

  /**
   * @brief Construct an instance of the factory's underlying resource.
   * @return resource instance
   */
  resource* make_resource() override { return &res; }

 private:
  empty_resource res;
};

/**
 * @}
 */

}  // namespace raft::resource
