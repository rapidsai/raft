
/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/device_uvector.hpp>

#include <iostream>

namespace raft::sparse::solver::detail {

template <typename idx_t>
__device__ idx_t get_1D_idx()
{
  return blockIdx.x * blockDim.x + threadIdx.x;
}

}  // namespace raft::sparse::solver::detail
