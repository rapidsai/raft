/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <rmm/mr/device/device_memory_resource.hpp>

namespace raft::resource {
class device_memory_resource : public resource {
 public:
  device_memory_resource(rmm::mr::device_memory_resource* mr_ = nullptr) : mr(mr_)
  {
    if (mr_ == nullptr) { mr = rmm::mr::get_current_device_resource(); }
  }
  void* get_resource() override { return mr; }

  ~device_memory_resource() override {}

 private:
  rmm::mr::device_memory_resource* mr;
};

/**
 * Factory that knows how to construct a specific raft::resource to populate
 * the resources instance.
 */
class workspace_resource_factory : public resource_factory {
 public:
  workspace_resource_factory(rmm::mr::device_memory_resource* mr_ = nullptr) : mr(mr_) {}
  resource_type get_resource_type() override { return resource_type::WORKSPACE_RESOURCE; }
  resource* make_resource() override { return new device_memory_resource(mr); }

 private:
  rmm::mr::device_memory_resource* mr;
};

/**
 * Load a temp workspace resource from a resources instance (and populate it on the res
 * if needed).
 * @param res raft resources object for managing resources
 * @return
 */
inline rmm::mr::device_memory_resource* get_workspace_resource(resources const& res)
{
  if (!res.has_resource_factory(resource_type::WORKSPACE_RESOURCE)) {
    res.add_resource_factory(std::make_shared<workspace_resource_factory>());
  }
  return res.get_resource<rmm::mr::device_memory_resource>(resource_type::WORKSPACE_RESOURCE);
};

/**
 * Set a temp workspace resource on a resources instance.
 *
 * @param res raft resources object for managing resources
 * @return
 */
inline void set_workspace_resource(resources const& res, rmm::mr::device_memory_resource* mr)
{
  res.add_resource_factory(std::make_shared<workspace_resource_factory>(mr));
};
}  // namespace raft::resource