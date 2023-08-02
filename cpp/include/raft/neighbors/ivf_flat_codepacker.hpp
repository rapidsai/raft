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

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/neighbors/ivf_flat_types.hpp>

#ifdef _RAFT_HAS_CUDA
#include <raft/util/pow2_utils.cuh>
#else
#include <raft/util/integer_utils.hpp>
#endif

namespace raft::neighbors::ivf_flat::codepacker {

template <typename T>
_RAFT_HOST_DEVICE inline auto roundDown(T x)
{
#if defined(_RAFT_HAS_CUDA)
  return Pow2<kIndexGroupSize>::roundDown(x);
#else
  return raft::round_down_safe(x, kIndexGroupSize);
#endif
}

template <typename T>
_RAFT_HOST_DEVICE inline auto mod(T x)
{
#if defined(_RAFT_HAS_CUDA)
  return Pow2<kIndexGroupSize>::mod(x);
#else
  return x % kIndexGroupSize;
#endif
}

/**
 * Write one flat code into a block by the given offset. The offset indicates the id of the record
 * in the list. This function interleaves the code and is intended to later copy the interleaved
 * codes over to the IVF list on device. NB: no memory allocation happens here; the block must fit
 * the record (offset + 1).
 *
 * @tparam T
 *
 * @param[in] flat_code input flat code
 * @param[out] block block of memory to write interleaved codes to
 * @param[in] dim dimension of the flat code
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset how many records to skip before writing the data into the list
 */
template <typename T>
_RAFT_HOST_DEVICE void pack_1(
  const T* flat_code, T* block, uint32_t dim, uint32_t veclen, uint32_t offset)
{
  // The data is written in interleaved groups of `index::kGroupSize` vectors
  // using interleaved_group = Pow2<kIndexGroupSize>;

  // Interleave dimensions of the source vector while recording it.
  // NB: such `veclen` is selected, that `dim % veclen == 0`
  auto group_offset = roundDown(offset);
  auto ingroup_id   = mod(offset) * veclen;

  for (uint32_t l = 0; l < dim; l += veclen) {
    for (uint32_t j = 0; j < veclen; j++) {
      block[group_offset * dim + l * kIndexGroupSize + ingroup_id + j] = flat_code[l + j];
    }
  }
}

/**
 * Unpack 1 record of a single list (cluster) in the index to fetch the flat code. The offset
 * indicates the id of the record. This function fetches one flat code from an interleaved code.
 *
 * @tparam T
 *
 * @param[in] block interleaved block. The block can be thought of as the whole inverted list in
 * interleaved format.
 * @param[out] flat_code output flat code
 * @param[in] dim dimension of the flat code
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset fetch the flat code by the given offset
 */
template <typename T>
_RAFT_HOST_DEVICE void unpack_1(
  const T* block, T* flat_code, uint32_t dim, uint32_t veclen, uint32_t offset)
{
  // The data is written in interleaved groups of `index::kGroupSize` vectors
  // using interleaved_group = Pow2<kIndexGroupSize>;

  // NB: such `veclen` is selected, that `dim % veclen == 0`
  auto group_offset = roundDown(offset);
  auto ingroup_id   = mod(offset) * veclen;

  for (uint32_t l = 0; l < dim; l += veclen) {
    for (uint32_t j = 0; j < veclen; j++) {
      flat_code[l + j] = block[group_offset * dim + l * kIndexGroupSize + ingroup_id + j];
    }
  }
}
}  // namespace raft::neighbors::ivf_flat::codepacker