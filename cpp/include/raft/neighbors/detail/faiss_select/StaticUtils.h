/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file thirdparty/LICENSES/LICENSE.faiss
 */

#pragma once

#include <cuda.h>

// allow usage for non-CUDA files
#ifndef __host__
#define __host__
#define __device__
#endif

namespace raft::neighbors::detail::faiss_select::utils {

template <typename T>
constexpr __host__ __device__ bool isPowerOf2(T v)
{
  return (v && !(v & (v - 1)));
}

static_assert(isPowerOf2(2048), "isPowerOf2");
static_assert(!isPowerOf2(3333), "isPowerOf2");

template <typename T>
constexpr __host__ __device__ T nextHighestPowerOf2(T v)
{
  return (isPowerOf2(v) ? (T)2 * v : ((T)1 << (log2(v) + 1)));
}

static_assert(nextHighestPowerOf2(1) == 2, "nextHighestPowerOf2");
static_assert(nextHighestPowerOf2(2) == 4, "nextHighestPowerOf2");
static_assert(nextHighestPowerOf2(3) == 4, "nextHighestPowerOf2");
static_assert(nextHighestPowerOf2(4) == 8, "nextHighestPowerOf2");

static_assert(nextHighestPowerOf2(15) == 16, "nextHighestPowerOf2");
static_assert(nextHighestPowerOf2(16) == 32, "nextHighestPowerOf2");
static_assert(nextHighestPowerOf2(17) == 32, "nextHighestPowerOf2");

static_assert(nextHighestPowerOf2(1536000000u) == 2147483648u, "nextHighestPowerOf2");
static_assert(nextHighestPowerOf2((size_t)2147483648ULL) == (size_t)4294967296ULL,
              "nextHighestPowerOf2");

}  // namespace raft::neighbors::detail::faiss_select::utils
