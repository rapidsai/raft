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

#include "raft/core/error.hpp"
#include <cstring>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/neighbors/ivf_pq_types.hpp>
#include <raft/neighbors/detail/ivf_pq_build.cuh>

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

_RAFT_HOST_DEVICE void unpack_1(
  const uint8_t* block, uint8_t* flat_code, uint32_t dim, uint32_t veclen, uint32_t offset)
{
  auto group_offset = ivf_flat::codepacker::roundDown(offset);
  auto ingroup_id   = ivf_flat::codepacker::mod(offset) * kIndexGroupVecLen;

  for (uint32_t l = 0; l < dim; l += veclen) {
    for (uint32_t j = 0; j < veclen; j++) {
      flat_code[l + j] = block[group_offset * dim + l * kIndexGroupSize + ingroup_id + j];
    }
  }
}

/**
 * Process a single vector in a list.
 *
 * @tparam PqBits
 * @tparam SubWarpSize how many threads work on the same ix (only the first thread writes data).
 * @tparam IdxT type of the index passed to the action
 * @tparam Action tells how to process a single vector (e.g. encode or just pack)
 *
 * @param[in] out_list_data the encoded cluster data.
 * @param[in] out_ix in-cluster index of the vector to be processed (one-per-SubWarpSize threads).
 * @param[in] in_ix the input index passed to the action (one-per-SubWarpSize threads).
 * @param[in] pq_dim
 * @param action a callable action to be invoked on each PQ code (component of the encoding)
 *    type: (uint32_t in_ix, uint32_t j) -> uint8_t, where j = [0..pq_dim).
 */
template <uint32_t PqBits, uint32_t SubWarpSize, typename IdxT, typename Action>
void pack_1(
  const uint8_t* flat_code,
  uint8_t* block,
  uint32_t pq_dim,
  uint32_t offset)
{
  RAFT_EXPECTS(PqBits == 8, "host codepacker supports only PqBits == 8");
  using group_align         = Pow2<kIndexGroupSize>;
  const uint32_t group_ix   = group_align::div(offset);
  const uint32_t ingroup_ix = group_align::mod(offset);

  for (uint32_t j = 0; j < pq_dim; j += kIndexGroupVecLen) {
    size_t bytes = min(pq_dim - j, kIndexGroupVecLen);
      std::memcpy(block, flat_code + j, bytes);
    }
}
}  // namespace raft::neighbors::ivf_flat::codepacker