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
#include <cstdint>
#include <raft/core/detail/macros.hpp>

// #pragma GCC diagnostic push
// #pragma GCC diagnostic ignored
// #pragma GCC diagnostic pop
namespace raft::neighbors::experimental::cagra::detail {
namespace hashmap {

_RAFT_HOST_DEVICE inline uint32_t get_size(const uint32_t bitlen) { return 1U << bitlen; }

template <unsigned FIRST_TID = 0>
_RAFT_DEVICE inline void init(uint32_t* table, const uint32_t bitlen)
{
  if (threadIdx.x < FIRST_TID) return;
  for (unsigned i = threadIdx.x - FIRST_TID; i < get_size(bitlen); i += blockDim.x - FIRST_TID) {
    table[i] = utils::get_max_value<uint32_t>();
  }
}

template <unsigned FIRST_TID, unsigned LAST_TID>
_RAFT_DEVICE inline void init(uint32_t* table, const uint32_t bitlen)
{
  if ((FIRST_TID > 0 && threadIdx.x < FIRST_TID) || threadIdx.x >= LAST_TID) return;
  for (unsigned i = threadIdx.x - FIRST_TID; i < get_size(bitlen); i += LAST_TID - FIRST_TID) {
    table[i] = utils::get_max_value<uint32_t>();
  }
}

_RAFT_DEVICE inline uint32_t insert(uint32_t* table, const uint32_t bitlen, const uint32_t key)
{
  // Open addressing is used for collision resolution
  const uint32_t size     = get_size(bitlen);
  const uint32_t bit_mask = size - 1;
#if 1
  // Linear probing
  uint32_t index            = (key ^ (key >> bitlen)) & bit_mask;
  constexpr uint32_t stride = 1;
#else
  // Double hashing
  uint32_t index        = key & bit_mask;
  const uint32_t stride = (key >> bitlen) * 2 + 1;
#endif
  for (unsigned i = 0; i < size; i++) {
    const uint32_t old = atomicCAS(&table[index], ~0u, key);
    if (old == ~0u) {
      return 1;
    } else if (old == key) {
      return 0;
    }
    index = (index + stride) & bit_mask;
  }
  return 0;
}

template <unsigned TEAM_SIZE>
_RAFT_DEVICE inline uint32_t insert(uint32_t* table, const uint32_t bitlen, const uint32_t key)
{
  uint32_t ret = 0;
  if (threadIdx.x % TEAM_SIZE == 0) { ret = insert(table, bitlen, key); }
  for (unsigned offset = 1; offset < TEAM_SIZE; offset *= 2) {
    ret |= __shfl_xor_sync(0xffffffff, ret, offset);
  }
  return ret;
}

}  // namespace hashmap
}  // namespace raft::neighbors::experimental::cagra::detail
