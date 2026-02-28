/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

namespace raft::mr {

/**
 * @brief Type-erased owning synchronous resource accessible from both host and device.
 *
 * Mirrors the role of rmm::mr::device_memory_resource for the host+device case,
 * but uses concept-based type erasure (no inheritance required from stored resources).
 * Any type satisfying cuda::mr::synchronous_resource with host_accessible and
 * device_accessible properties can be stored.
 */
using host_device_resource =
  cuda::mr::any_synchronous_resource<cuda::mr::host_accessible, cuda::mr::device_accessible>;

/**
 * @brief Type-erased owning synchronous resource accessible from host only.
 *
 * Mirrors the role of rmm::mr::device_memory_resource for the host-only case.
 */
using host_resource = cuda::mr::any_synchronous_resource<cuda::mr::host_accessible>;

}  // namespace raft::mr
