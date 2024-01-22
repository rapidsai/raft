/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "utils.hpp"
#include <cfloat>
#include <cstdint>
#include <cuda_fp16.h>
#include <raft/core/detail/macros.hpp>

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

template <class T>
_RAFT_DEVICE inline T swizzling(T x)
{
  // Address swizzling reduces bank conflicts in shared memory, but increases
  // the amount of operation instead.
  // return x;
  return x ^ (x >> 5);  // "x" must be less than 1024
}

}  // namespace device
}  // namespace raft::neighbors::cagra::detail
