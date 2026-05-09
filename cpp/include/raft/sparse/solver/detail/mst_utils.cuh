
/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>

#include <rmm/device_uvector.hpp>

#include <iostream>

namespace RAFT_EXPORT raft {
namespace sparse::solver::detail {

template <typename idx_t>
__device__ idx_t get_1D_idx()
{
  return blockIdx.x * blockDim.x + threadIdx.x;
}

}  // namespace sparse::solver::detail
}  // namespace RAFT_EXPORT raft
