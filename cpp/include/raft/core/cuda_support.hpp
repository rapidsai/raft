/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <raft/core/detail/macros.hpp>
namespace RAFT_EXPORT raft {
#ifndef RAFT_DISABLE_CUDA
auto constexpr static const CUDA_ENABLED = true;
#else
auto constexpr static const CUDA_ENABLED = false;
#endif
}  // namespace RAFT_EXPORT raft
