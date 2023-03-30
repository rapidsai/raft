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

// This file provides a few essential functions for use in __device__ code. The
// scope is necessarily limited to ensure that compilation times are minimized.
// Please make sure not to include large / expensive files from here.

namespace raft {

/** helper macro for device inlined functions */
#define DI  inline __device__
#define HDI inline __host__ __device__
#define HD  __host__ __device__

/**
 * @brief Provide a ceiling division operation ie. ceil(a / b)
 * @tparam IntType supposed to be only integers for now!
 */
template <typename IntType>
constexpr HDI IntType ceildiv(IntType a, IntType b)
{
  return (a + b - 1) / b;
}

/**
 * @brief Provide an alignment function ie. ceil(a / b) * b
 * @tparam IntType supposed to be only integers for now!
 */
template <typename IntType>
constexpr HDI IntType alignTo(IntType a, IntType b)
{
  return ceildiv(a, b) * b;
}

/**
 * @brief Provide an alignment function ie. (a / b) * b
 * @tparam IntType supposed to be only integers for now!
 */
template <typename IntType>
constexpr HDI IntType alignDown(IntType a, IntType b)
{
  return (a / b) * b;
}

/**
 * @brief Check if the input is a power of 2
 * @tparam IntType data type (checked only for integers)
 */
template <typename IntType>
constexpr HDI bool isPo2(IntType num)
{
  return (num && !(num & (num - 1)));
}

/**
 * @brief Give logarithm of the number to base-2
 * @tparam IntType data type (checked only for integers)
 */
template <typename IntType>
constexpr HDI IntType log2(IntType num, IntType ret = IntType(0))
{
  return num <= IntType(1) ? ret : log2(num >> IntType(1), ++ret);
}

/** number of threads per warp */
static const int WarpSize = 32;

/** get the laneId of the current thread */
DI int laneId()
{
  int id;
  asm("mov.s32 %0, %%laneid;" : "=r"(id));
  return id;
}

/** Device function to apply the input lambda across threads in the grid */
template <int ItemsPerThread, typename L>
DI void forEach(int num, L lambda)
{
  int idx              = (blockDim.x * blockIdx.x) + threadIdx.x;
  const int numThreads = blockDim.x * gridDim.x;
#pragma unroll
  for (int itr = 0; itr < ItemsPerThread; ++itr, idx += numThreads) {
    if (idx < num) lambda(idx, itr);
  }
}

/**
 * @brief Swap two values
 * @tparam T the datatype of the values
 * @param a first input
 * @param b second input
 */
template <typename T>
HDI void swapVals(T& a, T& b)
{
  T tmp = a;
  a     = b;
  b     = tmp;
}

}  // namespace raft
