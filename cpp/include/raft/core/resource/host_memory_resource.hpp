/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <raft/core/operators.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <memory_resource>

namespace raft::resource {

/**
 * \defgroup host_memory_resource Device memory resources
 * @{
 */

class host_memory_resource : public resource {
 public:
  explicit host_memory_resource(std::shared_ptr<std::pmr::memory_resource> mr) : mr_(mr) {}
  ~host_memory_resource() override = default;
  auto get_resource() -> void* override { return mr_.get(); }

 private:
  std::shared_ptr<std::pmr::memory_resource> mr_;
};

/**
 * Factory that knows how to construct a specific raft::resource to populate
 * the resources instance.
 */
class host_memory_resource_factory : public resource_factory {
 public:
  explicit host_memory_resource_factory(std::shared_ptr<std::pmr::memory_resource> mr = {nullptr})
    : mr_(mr ? mr : default_plain_resource())
  {
  }

  auto get_resource_type() -> resource_type override { return resource_type::HOST_MEMORY_RESOURCE; }
  auto make_resource() -> resource* override { return new host_memory_resource(mr_); }

  static inline auto default_plain_resource() -> std::shared_ptr<std::pmr::memory_resource>
  {
    return std::shared_ptr<std::pmr::memory_resource>{std::pmr::get_default_resource(), void_op{}};
  }

 private:
  std::shared_ptr<std::pmr::memory_resource> mr_;
};

/**
 * Load a host memory resource from a resources instance (and populate it on the res if needed).
 *
 * @param res raft resources object for managing resources
 * @return host memory resource object
 */
inline auto get_host_memory_resource(resources const& res) -> std::pmr::memory_resource*
{
  if (!res.has_resource_factory(resource_type::HOST_MEMORY_RESOURCE)) {
    res.add_resource_factory(std::make_shared<host_memory_resource_factory>());
  }
  return res.get_resource<std::pmr::memory_resource>(resource_type::HOST_MEMORY_RESOURCE);
};

/**
 * Set a host memory resource on a resources instance.
 *
 * @param res raft resources object for managing resources
 * @param mr an optional std::pmr::memory_resource
 */
inline void set_host_memory_resource(resources const& res,
                                     std::shared_ptr<std::pmr::memory_resource> mr = {nullptr})
{
  res.add_resource_factory(std::make_shared<host_memory_resource_factory>(mr));
};

/** @} */

}  // namespace raft::resource
