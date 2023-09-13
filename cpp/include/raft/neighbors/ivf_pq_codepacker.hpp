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
#include <raft/neighbors/ivf_pq_types.hpp>
#include <raft/neighbors/ivf_flat_codepacker.hpp>

#ifdef _RAFT_HAS_CUDA
#include <raft/util/pow2_utils.cuh>
#else
#include <raft/util/integer_utils.hpp>
#endif

namespace raft::neighbors::ivf_pq::codepacker {

_RAFT_HOST_DEVICE inline uint32_t div(uint32_t x)
{
#if defined(_RAFT_HAS_CUDA)
  return Pow2<kIndexGroupSize>::div(x);
#else
  return x / kIndexGroupSize;
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
_RAFT_HOST_DEVICE void pack_1(
  const uint8_t* flat_code, uint8_t* block, uint32_t dim, uint32_t veclen, uint32_t offset)
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


template <uint32_t PqBits, uint32_t SubWarpSize, typename IdxT, typename Action>
_RAFT_HOST_DEVICE void pack_1(
  const uint8_t* flat_code,
  uint8_t* block,
  // device_mdspan<uint8_t, list_spec<uint32_t, uint32_t>::list_extents, row_major> out_list_data,
  uint32_t pq_dim,
  uint32_t offset)
{
  const uint32_t group_ix = div(offset);
  const uint32_t ingroup_ix = ivf_flat::codepacker::mod(offset);

  pq_vec_t code_chunk;
  bitfield_view_t<PqBits> code_view{reinterpret_cast<uint8_t*>(&code_chunk)};
  constexpr uint32_t kChunkSize = (kIndexGroupVecLen * 8u) / PqBits;
  for (uint32_t j = 0, i = 0; j < pq_dim; i++) {
    // clear the chunk
    code_chunk = pq_vec_t{};
    // write the codes, one/pq_dim at a time
#pragma unroll
    for (uint32_t k = 0; k < kChunkSize && j < pq_dim; k++, j++) {
      // write a single code
      uint8_t code = action(in_ix, j);
      code_view[k] = code;
    }
    // write the chunk to the list
      *reinterpret_cast<pq_vec_t*>(&out_list_data(group_ix, i, ingroup_ix, 0)) = code_chunk;
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
_RAFT_HOST_DEVICE void unpack_1(
  const uint8_t* block, uint8_t* flat_code, uint32_t dim, uint32_t veclen, uint32_t offset)
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