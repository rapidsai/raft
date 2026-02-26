/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#ifndef RAFT_DISABLE_CUDA

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

/** @brief Non-owning reference to a synchronous host-accessible resource. */
using host_resource_ref = rmm::host_resource_ref;

/** @brief Non-owning reference to a synchronous host+device-accessible resource. */
using host_device_resource_ref = rmm::host_device_resource_ref;

}  // namespace raft::mr

#else  // RAFT_DISABLE_CUDA

#include <raft/pmr/std_pmr_sync_adapter.hpp>

namespace raft::mr {

using host_resource     = raft::pmr::std_pmr_sync_adapter;
using host_resource_ref = raft::pmr::std_pmr_sync_adapter;

}  // namespace raft::mr

#endif
