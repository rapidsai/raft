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

#include <raft/core/resource/device_id.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuda_runtime.h>

namespace raft::resource {

class device_properties_resource : public resource {
 public:
  device_properties_resource(int dev_id)
  {
    RAFT_CUDA_TRY_NO_THROW(cudaGetDeviceProperties(&prop_, dev_id));
  }
  void* get_resource() override { return &prop_; }

  ~device_properties_resource() override {}

 private:
  cudaDeviceProp prop_;
};

/**
 * @defgroup resource_device_props Device properties resource functions
 * @{
 */

/**
 * Factory that knows how to construct a
 * specific raft::resource to populate
 * the res_t.
 */
class device_properties_resource_factory : public resource_factory {
 public:
  device_properties_resource_factory(int dev_id) : dev_id_(dev_id) {}
  resource_type get_resource_type() override { return resource_type::DEVICE_PROPERTIES; }
  resource* make_resource() override { return new device_properties_resource(dev_id_); }

 private:
  int dev_id_;
};

/**
 * Load a cudaDeviceProp from a res (and populate it on the res if needed).
 * @param res raft res object for managing resources
 * @return populated cuda device properties instance
 */
inline cudaDeviceProp& get_device_properties(resources const& res)
{
  if (!res.has_resource_factory(resource_type::DEVICE_PROPERTIES)) {
    int dev_id = get_device_id(res);
    res.add_resource_factory(std::make_shared<device_properties_resource_factory>(dev_id));
  }
  return *res.get_resource<cudaDeviceProp>(resource_type::DEVICE_PROPERTIES);
};

/**
 * @}
 */
}  // namespace raft::resource