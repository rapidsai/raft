/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/mr/host_device_resource.hpp>
#include <raft/pmr/resource_adaptor.hpp>

#include <memory_resource>

namespace raft::mr {

/**
 * @brief Get a reference to a stateless new/delete host memory resource.
 *
 * Analogous to std::pmr::new_delete_resource(), but returns rmm::host_resource_ref.
 */
inline auto new_delete_resource() -> rmm::host_resource_ref
{
  static raft::pmr::resource_adaptor instance{std::pmr::new_delete_resource()};
  return rmm::host_resource_ref{instance};
}

namespace detail {

inline auto& default_host_resource_ref()
{
  static rmm::host_resource_ref ref = new_delete_resource();
  return ref;
}

}  // namespace detail

/**
 * @brief Get the current default host memory resource.
 *
 * Returns rmm::host_resource_ref pointing to the resource installed
 * via set_default_host_resource(), or new_delete_resource() if none was set.
 */
inline auto get_default_host_resource() -> rmm::host_resource_ref
{
  return detail::default_host_resource_ref();
}

/**
 * @brief Set the default host memory resource.
 *
 * The caller must keep the underlying resource alive while it is set as the default
 * (same contract as rmm::mr::set_current_device_resource).
 *
 * @param ref Non-owning reference to the resource to install.
 * @return The previous default host resource ref.
 */
inline auto set_default_host_resource(rmm::host_resource_ref ref) -> rmm::host_resource_ref
{
  auto& current = detail::default_host_resource_ref();
  auto prev     = current;
  current       = ref;
  return prev;
}

}  // namespace raft::mr
