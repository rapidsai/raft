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

#include <cstring>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/neighbors/detail/ivf_pq_codepacking.cuh>
#include <raft/neighbors/ivf_pq_types.hpp>

namespace raft::neighbors::ivf_pq::codepacker {

/**
 * A producer for the `write_vector` reads the codes byte-by-byte. That is,
 * independent of the code width (pq_bits), one code uses the whole byte, hence
 * one vectors uses pq_dim bytes.
 */
struct pass_1_action {
  const uint8_t* flat_code;

  /**
   * Create a callable to be passed to `write_vector`.
   *
   * @param[in] flat_code flat PQ codes (one byte per code) of a single vector.
   */
  __host__ __device__ inline pass_1_action(const uint8_t* flat_code) : flat_code{flat_code} {}

  /** Read j-th component (code) of the i-th vector from the source. */
  __host__ __device__ inline auto operator()(uint32_t i, uint32_t j) const -> uint8_t
  {
    return flat_code[j];
  }
};

/**
 * A consumer for the `run_on_vector` that just flattens PQ codes
 * one-per-byte. That is, independent of the code width (pq_bits), one code uses
 * the whole byte, hence one vectors uses pq_dim bytes.
 */
struct unpack_1_action {
  uint8_t* out_flat_code;

  /**
   * Create a callable to be passed to `run_on_vector`.
   *
   * @param[out] out_flat_code the destination for the read PQ codes of a single vector.
   */
  __host__ __device__ inline unpack_1_action(uint8_t* out_flat_code) : out_flat_code{out_flat_code}
  {
  }

  /**  Write j-th component (code) of the i-th vector into the output array. */
  __host__ __device__ inline void operator()(uint8_t code, uint32_t i, uint32_t j)
  {
    out_flat_code[j] = code;
  }
};

template <uint32_t PqBits>
void unpack_1(const uint8_t* block, uint8_t* flat_code, uint32_t pq_dim, uint32_t offset)
{
  ivf_pq::detail::run_on_vector<PqBits>(block, offset, 0, pq_dim, unpack_1_action{flat_code});
}

template <uint32_t PqBits>
void pack_1(const uint8_t* flat_code, uint8_t* block, uint32_t pq_dim, uint32_t offset)
{
  ivf_pq::detail::write_vector<PqBits>(block, offset, 0, pq_dim, pass_1_action{flat_code});
}
}  // namespace raft::neighbors::ivf_pq::codepacker