/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "utils.hpp"

#include <raft/core/detail/macros.hpp>

#include <cuda_fp16.h>

#include <cfloat>
#include <cstdint>

namespace raft::neighbors::cagra::detail {
namespace device {

// warpSize for compile time calculation
constexpr unsigned warp_size = 32;

/** Xorshift rondem number generator.
 *
 * See https://en.wikipedia.org/wiki/Xorshift#xorshift for reference.
 */
_RAFT_HOST_DEVICE inline uint64_t xorshift64(uint64_t u)
{
  u ^= u >> 12;
  u ^= u << 25;
  u ^= u >> 27;
  return u * 0x2545F4914F6CDD1DULL;
}

template <class T, unsigned X_MAX = 1024>
_RAFT_DEVICE inline T swizzling(T x)
{
  // Address swizzling reduces bank conflicts in shared memory, but increases
  // the amount of operation instead.
  // return x;
  if constexpr (X_MAX <= 1024) {
    return (x) ^ ((x) >> 5);
  } else {
    return (x) ^ (((x) >> 5) & 0x1f);
  }
}

}  // namespace device
}  // namespace raft::neighbors::cagra::detail
