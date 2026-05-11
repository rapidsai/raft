/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/span.hpp>

namespace RAFT_EXPORT raft {

/**
 * @defgroup host_span one-dimensional device span type
 * @{
 */

/**
 * @brief A span class for host pointer.
 */
template <typename T, size_t extent = cuda::std::dynamic_extent>
using host_span = span<T, false, extent>;

/**
 * @}
 */

}  // namespace RAFT_EXPORT raft
