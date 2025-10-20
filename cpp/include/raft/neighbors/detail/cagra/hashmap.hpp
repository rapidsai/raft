/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <raft/core/detail/macros.hpp>
#include <raft/util/device_atomics.cuh>

#include <cstdint>

// #pragma GCC diagnostic push
// #pragma GCC diagnostic ignored
// #pragma GCC diagnostic pop
namespace raft::neighbors::cagra::detail {
namespace hashmap {

_RAFT_HOST_DEVICE inline uint32_t get_size(const uint32_t bitlen) { return 1U << bitlen; }

template <class IdxT>
_RAFT_DEVICE inline void init(IdxT* const table, const unsigned bitlen, unsigned FIRST_TID = 0)
{
  if (threadIdx.x < FIRST_TID) return;
  for (unsigned i = threadIdx.x - FIRST_TID; i < get_size(bitlen); i += blockDim.x - FIRST_TID) {
    table[i] = utils::get_max_value<IdxT>();
  }
}

template <class IdxT>
_RAFT_DEVICE inline uint32_t insert(IdxT* const table, const uint32_t bitlen, const IdxT key)
{
  // Open addressing is used for collision resolution
  const uint32_t size     = get_size(bitlen);
  const uint32_t bit_mask = size - 1;
#if 1
  // Linear probing
  IdxT index                = (key ^ (key >> bitlen)) & bit_mask;
  constexpr uint32_t stride = 1;
#else
  // Double hashing
  uint32_t index        = key & bit_mask;
  const uint32_t stride = (key >> bitlen) * 2 + 1;
#endif
  for (unsigned i = 0; i < size; i++) {
    const IdxT old = atomicCAS(&table[index], ~static_cast<IdxT>(0), key);
    if (old == ~static_cast<IdxT>(0)) {
      return 1;
    } else if (old == key) {
      return 0;
    }
    index = (index + stride) & bit_mask;
  }
  return 0;
}

template <unsigned TEAM_SIZE, class IdxT>
_RAFT_DEVICE inline uint32_t insert(IdxT* const table, const uint32_t bitlen, const IdxT key)
{
  IdxT ret = 0;
  if (threadIdx.x % TEAM_SIZE == 0) { ret = insert(table, bitlen, key); }
  for (unsigned offset = 1; offset < TEAM_SIZE; offset *= 2) {
    ret |= __shfl_xor_sync(0xffffffff, ret, offset);
  }
  return ret;
}

}  // namespace hashmap
}  // namespace raft::neighbors::cagra::detail
