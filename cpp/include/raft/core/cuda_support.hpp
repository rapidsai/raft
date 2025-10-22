/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
namespace raft {
#ifndef RAFT_DISABLE_CUDA
auto constexpr static const CUDA_ENABLED = true;
#else
auto constexpr static const CUDA_ENABLED = false;
#endif
}  // namespace raft
