/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>

#include <memory>

namespace raft::resource {

/**
 * @defgroup managed_memory_resource Managed memory resource
 * @{
 */

/**
 * @brief Factory that creates a device_memory_resource for managed (unified) memory.
 *
 * Defaults to a lazily initialized static rmm::mr::managed_memory_resource.
 */
class managed_memory_resource_factory : public resource_factory {
 public:
  explicit managed_memory_resource_factory(
    std::shared_ptr<rmm::mr::device_memory_resource> mr = {nullptr})
    : mr_{mr ? std::move(mr) : default_resource()}
  {
  }

  auto get_resource_type() -> resource_type override
  {
    return resource_type::MANAGED_MEMORY_RESOURCE;
  }
  auto make_resource() -> resource* override { return new device_memory_resource(mr_); }

 private:
  std::shared_ptr<rmm::mr::device_memory_resource> mr_;

  static auto default_resource() -> std::shared_ptr<rmm::mr::device_memory_resource>
  {
    static auto result = std::make_shared<rmm::mr::managed_memory_resource>();
    return result;
  }
};

/**
 * @brief Get the managed memory resource from a resources handle.
 *
 * The default is a static rmm::mr::managed_memory_resource.
 *
 * @param res raft resources object
 * @return pointer to the managed rmm::mr::device_memory_resource
 */
inline auto get_managed_memory_resource(resources const& res) -> rmm::mr::device_memory_resource*
{
  if (!res.has_resource_factory(resource_type::MANAGED_MEMORY_RESOURCE)) {
    res.add_resource_factory(std::make_shared<managed_memory_resource_factory>());
  }
  return res.get_resource<rmm::mr::device_memory_resource>(resource_type::MANAGED_MEMORY_RESOURCE);
}

/**
 * @brief Set the managed memory resource on a resources handle.
 *
 * @param res raft resources object
 * @param mr the managed memory resource to use
 */
inline void set_managed_memory_resource(resources const& res,
                                        std::shared_ptr<rmm::mr::device_memory_resource> mr)
{
  res.add_resource_factory(std::make_shared<managed_memory_resource_factory>(std::move(mr)));
}

/** @} */

}  // namespace raft::resource
