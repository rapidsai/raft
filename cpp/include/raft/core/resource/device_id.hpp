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

#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuda_runtime.h>

namespace raft::resource {

class device_id_resource : public resource {
 public:
  device_id_resource()
    : dev_id_([]() -> int {
        int cur_dev = -1;
        RAFT_CUDA_TRY_NO_THROW(cudaGetDevice(&cur_dev));
        return cur_dev;
      }())
  {
  }
  void* get_resource() override { return &dev_id_; }

  ~device_id_resource() override {}

 private:
  int dev_id_;
};

/**
 * Factory that knows how to construct a
 * specific raft::resource to populate
 * the res_t.
 */
class device_id_resource_factory : public resource_factory {
 public:
  resource_type get_resource_type() override { return resource_type::DEVICE_ID; }
  resource* make_resource() override { return new device_id_resource(); }
};

/**
 * @defgroup resource_device_id Device ID resource functions
 * @{
 */

/**
 * Load a device id from a res (and populate it on the res if needed).
 * @param res raft res object for managing resources
 * @return device id
 */
inline int get_device_id(resources const& res)
{
  if (!res.has_resource_factory(resource_type::DEVICE_ID)) {
    res.add_resource_factory(std::make_shared<device_id_resource_factory>());
  }
  return *res.get_resource<int>(resource_type::DEVICE_ID);
};

/**
 * @}
 */
}  // namespace raft::resource