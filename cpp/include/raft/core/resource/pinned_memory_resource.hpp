/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/pmr/pinned_memory_resource.hpp>

#include <memory>
#include <memory_resource>

namespace raft::resource {

/**
 * @defgroup pinned_memory_resource Pinned memory resource
 * @{
 */

/**
 * @brief Resource that holds a std::pmr::memory_resource for host allocations.
 */
class host_memory_resource : public resource {
 public:
  explicit host_memory_resource(std::shared_ptr<std::pmr::memory_resource> mr) : mr_(std::move(mr))
  {
  }
  ~host_memory_resource() override = default;

  auto get_resource() -> void* override { return mr_.get(); }

 private:
  std::shared_ptr<std::pmr::memory_resource> mr_;
};

/**
 * @brief Factory that creates a host_memory_resource.
 *
 * Defaults to a lazily initialized static pinned memory resource
 * (cudaMallocHost/cudaFreeHost).
 */
class pinned_memory_resource_factory : public resource_factory {
 public:
  explicit pinned_memory_resource_factory(std::shared_ptr<std::pmr::memory_resource> mr = {nullptr})
    : mr_{mr ? std::move(mr) : default_resource()}
  {
  }

  auto get_resource_type() -> resource_type override
  {
    return resource_type::PINNED_MEMORY_RESOURCE;
  }
  auto make_resource() -> resource* override { return new host_memory_resource(mr_); }

 private:
  std::shared_ptr<std::pmr::memory_resource> mr_;

  static auto default_resource() -> std::shared_ptr<std::pmr::memory_resource>
  {
    static auto result = std::make_shared<raft::pmr::pinned_memory_resource>();
    return result;
  }
};

/**
 * @brief Get the pinned memory resource from a resources handle.
 *
 * The default is a static pinned_memory_resource backed by cudaMallocHost/cudaFreeHost.
 *
 * @param res raft resources object
 * @return pointer to the pinned std::pmr::memory_resource
 */
inline auto get_pinned_memory_resource(resources const& res) -> std::pmr::memory_resource*
{
  if (!res.has_resource_factory(resource_type::PINNED_MEMORY_RESOURCE)) {
    res.add_resource_factory(std::make_shared<pinned_memory_resource_factory>());
  }
  return res.get_resource<std::pmr::memory_resource>(resource_type::PINNED_MEMORY_RESOURCE);
}

/**
 * @brief Set the pinned memory resource on a resources handle.
 *
 * @param res raft resources object
 * @param mr the pinned memory resource to use
 */
inline void set_pinned_memory_resource(resources const& res,
                                       std::shared_ptr<std::pmr::memory_resource> mr)
{
  res.add_resource_factory(std::make_shared<pinned_memory_resource_factory>(std::move(mr)));
}

/** @} */

}  // namespace raft::resource
