/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/mr/host_device_resource.hpp>
#include <raft/mr/pinned_memory_resource.hpp>

#include <rmm/resource_ref.hpp>

#include <memory>

namespace raft::resource {

/**
 * @defgroup pinned_memory_resource Pinned memory resource
 * @{
 */

class pinned_memory_resource : public resource {
 public:
  explicit pinned_memory_resource(std::shared_ptr<raft::mr::host_device_resource> mr)
    : mr_(std::move(mr))
  {
  }
  ~pinned_memory_resource() override = default;
  auto get_resource() -> void* override { return mr_.get(); }

 private:
  std::shared_ptr<raft::mr::host_device_resource> mr_;
};

class pinned_memory_resource_factory : public resource_factory {
 public:
  explicit pinned_memory_resource_factory(std::shared_ptr<raft::mr::host_device_resource> mr = {})
    : mr_(mr ? std::move(mr) : default_resource())
  {
  }

  auto get_resource_type() -> resource_type override
  {
    return resource_type::PINNED_MEMORY_RESOURCE;
  }
  auto make_resource() -> resource* override { return new pinned_memory_resource(mr_); }

 private:
  std::shared_ptr<raft::mr::host_device_resource> mr_;

  static auto default_resource() -> std::shared_ptr<raft::mr::host_device_resource>
  {
    return std::make_shared<raft::mr::host_device_resource>(raft::mr::pinned_memory_resource{});
  }
};

/** @brief Get the pinned memory resource. Default: raft::mr::pinned_memory_resource. */
inline auto get_pinned_memory_resource(resources const& res) -> rmm::host_device_resource_ref
{
  if (!res.has_resource_factory(resource_type::PINNED_MEMORY_RESOURCE)) {
    res.add_resource_factory(std::make_shared<pinned_memory_resource_factory>());
  }
  return rmm::host_device_resource_ref{
    *res.get_resource<raft::mr::host_device_resource>(resource_type::PINNED_MEMORY_RESOURCE)};
}

/**
 * @brief Set the pinned memory resource.
 *
 * @param res raft resources object for managing resources
 * @param mr  shared pointer to a host+device accessible resource
 */
inline void set_pinned_memory_resource(resources const& res,
                                       std::shared_ptr<raft::mr::host_device_resource> mr)
{
  res.add_resource_factory(std::make_shared<pinned_memory_resource_factory>(std::move(mr)));
}

/** @} */

}  // namespace raft::resource
