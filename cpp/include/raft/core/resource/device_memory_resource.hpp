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
#include <raft/util/cudart_utils.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/limiting_resource_adaptor.hpp>

#include <cstddef>
#include <optional>

namespace raft::resource {
class limited_memory_resource : public resource {
 public:
  limited_memory_resource(rmm::mr::device_memory_resource* mr,
                          std::optional<std::size_t> allocation_limit,
                          std::optional<std::size_t> alignment)
    : mr_(make_adaptor(mr, allocation_limit, alignment))
  {
  }

  auto get_resource() -> void* override { return &mr_; }

  ~limited_memory_resource() override = default;

 private:
  rmm::mr::limiting_resource_adaptor<rmm::mr::device_memory_resource> mr_;

  static inline auto get_device_mem()
  {
    std::size_t free_size{};
    std::size_t total_size{};
    RAFT_CUDA_TRY(cudaMemGetInfo(&free_size, &total_size));
    return total_size;
  }

  static inline auto make_adaptor(rmm::mr::device_memory_resource* mr,
                                  std::optional<std::size_t> allocation_limit,
                                  std::optional<std::size_t> alignment)
    -> rmm::mr::limiting_resource_adaptor<rmm::mr::device_memory_resource>
  {
    if (mr == nullptr) { mr = rmm::mr::get_current_device_resource(); }
    auto limit = allocation_limit.value_or(get_device_mem());
    if (alignment.has_value()) {
      return rmm::mr::limiting_resource_adaptor(mr, limit, alignment.value());
    } else {
      return rmm::mr::limiting_resource_adaptor(mr, limit);
    }
  }
};

/**
 * Factory that knows how to construct a specific raft::resource to populate
 * the resources instance.
 */
class workspace_resource_factory : public resource_factory {
 public:
  workspace_resource_factory(rmm::mr::device_memory_resource* mr,
                             std::optional<std::size_t> allocation_limit,
                             std::optional<std::size_t> alignment)
    : mr_(mr), allocation_limit_(allocation_limit), alignment_(alignment)
  {
  }
  auto get_resource_type() -> resource_type override { return resource_type::WORKSPACE_RESOURCE; }
  auto make_resource() -> resource* override
  {
    return new limited_memory_resource(mr_, allocation_limit_, alignment_);
  }

 private:
  rmm::mr::device_memory_resource* mr_;
  std::optional<std::size_t> allocation_limit_;
  std::optional<std::size_t> alignment_;
};

/**
 * Load a temp workspace resource from a resources instance (and populate it on the res
 * if needed).
 * @param res raft resources object for managing resources
 * @return device memory resource object
 */
inline auto get_workspace_resource(resources const& res)
  -> rmm::mr::limiting_resource_adaptor<rmm::mr::device_memory_resource>*
{
  if (!res.has_resource_factory(resource_type::WORKSPACE_RESOURCE)) {
    res.add_resource_factory(
      std::make_shared<workspace_resource_factory>(nullptr, std::nullopt, std::nullopt));
  }
  return res.get_resource<rmm::mr::limiting_resource_adaptor<rmm::mr::device_memory_resource>>(
    resource_type::WORKSPACE_RESOURCE);
};

/**
 * Set a temp workspace resource on a resources instance.
 *
 * @param res raft resources object for managing resources
 * @param mr a valid rmm device_memory_resource
 */
inline void set_workspace_resource(resources const& res,
                                   rmm::mr::device_memory_resource* mr         = nullptr,
                                   std::optional<std::size_t> allocation_limit = std::nullopt,
                                   std::optional<std::size_t> alignment        = std::nullopt)
{
  res.add_resource_factory(
    std::make_shared<workspace_resource_factory>(mr, allocation_limit, alignment));
};
}  // namespace raft::resource
