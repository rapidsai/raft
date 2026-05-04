/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/span.hpp>

namespace RAFT_EXPORT raft {

/**
 * @defgroup device_span one-dimensional device span type
 * @{
 */

/**
 * @brief A span class for device pointer.
 */
template <typename T, size_t extent = cuda::std::dynamic_extent>
using device_span = span<T, true, extent>;

/**
 * @}
 */
}  // namespace RAFT_EXPORT raft
