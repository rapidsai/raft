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
#include <raft/core/resource/resource_types.hpp>
#include <raft/util/cudart_utils.hpp>

namespace raft::core {

class device_id_resource_t : public resource_t {
 public:
  device_id_resource_t()
    : dev_id_([]() -> int {
        int cur_dev = -1;
        RAFT_CUDA_TRY_NO_THROW(cudaGetDevice(&cur_dev));
        return cur_dev;
      }())
  {
  }
  void* get_resource() override { return &dev_id_; }

  ~device_id_resource_t() override {}

 private:
  int dev_id_;
};

/**
 * Factory that knows how to construct a
 * specific raft::resource_t to populate
 * the handle_t.
 */
class device_id_resource_factory_t : public resource_factory_t {
 public:
  resource_type_t resource_type() override { return resource_type_t::DEVICE_ID; }
  resource_t* make_resource() override { return new device_id_resource_t(); }
};

/**
 * Load a device id from a handle (and populate it on the handle if needed).
 * @param handle raft handle object for managing resources
 * @return
 */
inline int get_device_id(base_handle_t const& handle)
{
  if (!handle.has_resource_factory(resource_type_t::DEVICE_ID)) {
    handle.add_resource_factory(std::make_shared<device_id_resource_factory_t>());
  }
  return *handle.get_resource<int>(resource_type_t::DEVICE_ID);
};
}  // namespace raft::core