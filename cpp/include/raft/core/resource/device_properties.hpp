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

#include <cuda_runtime.h>
#include <raft/core/resource/device_id.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/util/cudart_utils.hpp>

namespace raft::core {
class device_properties_resource_t : public resource_t {
 public:
  device_properties_resource_t(int dev_id)
  {
    RAFT_CUDA_TRY_NO_THROW(cudaGetDeviceProperties(&prop_, dev_id));
  }
  void* get_resource() override { return &prop_; }

  ~device_properties_resource_t() override {}

 private:
  cudaDeviceProp prop_;
};

/**
 * Factory that knows how to construct a
 * specific raft::resource_t to populate
 * the handle_t.
 */
class device_properties_resource_factory_t : public resource_factory_t {
 public:
  device_properties_resource_factory_t(int dev_id) : dev_id_(dev_id) {}
  resource_type_t resource_type() override { return resource_type_t::DEVICE_PROPERTIES; }
  resource_t* make_resource() override { return new device_properties_resource_t(dev_id_); }

 private:
  int dev_id_;
};

/**
 * Load a cudaDeviceProp from a handle (and populate it on the handle if needed).
 * @param handle raft handle object for managing resources
 * @return
 */
cudaDeviceProp& get_device_properties(base_handle_t const& handle)
{
  if (!handle.has_resource_factory(resource_type_t::DEVICE_PROPERTIES)) {
    int dev_id = get_device_id(handle);
    handle.add_resource_factory(std::make_shared<device_properties_resource_factory_t>(dev_id));
  }
  return *handle.get_resource<cudaDeviceProp>(resource_type_t::DEVICE_PROPERTIES);
};
}  // namespace raft::core