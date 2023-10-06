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
#include <raft/core/temporary_device_buffer.hpp>
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




// /**
//  * Unpack 1 record of a single list (cluster) in the index to fetch the flat (compressed) PQ code.
//  * The offset indicates the id of the record. This function fetches one flat code from an
//  * interleaved code.
//  *
//  * @param[in] block interleaved block. Pointer to the inverted list data in interleaved format
//  * @param[out] flat_code output flat PQ code
//  * @param[in] pq_dim
//  * @param[in] offset fetch the flat PQ code by the given offset
//  */
// template <uint32_t PqBits>
// void unpack_1_pq(const uint8_t* block, uint8_t* flat_code, uint32_t pq_dim, uint32_t offset)
// {
//   ivf_pq::detail::run_on_vector<PqBits>(block, offset, 0, pq_dim, unpack_1_action{flat_code});
// }

// /**
//  * Write one flat (compressed) PQ code into a block by the given offset. The offset indicates the id
//  * of the record in the list. This function interleaves the code and writes the interleaved result
//  * to the output. NB: no memory allocation happens here; the block must fit the record (offset + 1).
//  *
//  * @param[in] block interleaved block. Pointer to the inverted list data in interleaved format
//  * @param[out] flat_code output flat PQ code
//  * @param[in] pq_dim
//  * @param[in] offset fetch the flat PQ code by the given offset
//  */
// template <uint32_t PqBits>
// void pack_1_pq(const uint8_t* flat_code, uint8_t* block, uint32_t pq_dim, uint32_t offset)
// {
//   ivf_pq::detail::write_vector<PqBits>(block, offset, 0, pq_dim, pass_1_action{flat_code});
// }

// /**
//  * Unpack 1 record of a single list (cluster) in the index to fetch the flat (compressed) PQ code.
//  * The offset indicates the id of the record. This function fetches one flat code from an
//  * interleaved code.
//  *
//  * @param[in] block interleaved block. Pointer to the inverted list data in interleaved format
//  * @param[out] flat_code output flat PQ code
//  * @param[in] pq_dim
//  * @param[in] offset fetch the flat PQ code by the given offset
//  */
// template <uint32_t PqBits>
// void unpack_1_pq(const uint8_t* block, uint8_t* flat_code, uint32_t pq_dim, uint32_t offset)
// {
//   ivf_pq::detail::run_on_vector<PqBits>(block, offset, 0, pq_dim, unpack_1_action{flat_code});
// }

// /**
//  * Write one flat (compressed) PQ code into a block by the given offset. The offset indicates the id
//  * of the record in the list. This function interleaves the code and writes the interleaved result
//  * to the output. NB: no memory allocation happens here; the block must fit the record (offset + 1).
//  *
//  * @param[in] block interleaved block. Pointer to the inverted list data in interleaved format
//  * @param[out] flat_code output flat PQ code
//  * @param[in] pq_dim
//  * @param[in] offset fetch the flat PQ code by the given offset
//  */
// template <uint32_t PqBits>
// void pack_pq(raft::resources handle, const uint8_t* flat_codes, uint8_t* block, uint32_t pq_dim, uint32_t num_codes, uint32_t offset, uint32_t batch_size=0)
// {
//   auto spec = list_spec(PqBits, pq_dim, false);
//   auto extents = spec.make_list_extents(num_codes);
//   auto list_view = make_readonly_temporary_device_buffer<const uint8_t>(handle, flat_codes, extents).view();
//   ivf_pq::detail::pack_list_data(block, offset, 0, pq_dim, ivf_pq::detail::pass_codes{flat_code});
// }

// inline void unpack(
//   raft::resources const& res, uint8_t* list_data, uint32_t pq_bits, uint32_t offset, uint8_t* codes)
// {
//   auto extents = raft::make_extents<>();
//   auto make_writeback_temporary_device_buffer(
//     raft::resources const& handle, ElementType* data, raft::extents<IndexType, Extents...> extents)
//     make_writeback_temporary_device_buffer<typename ElementType>(res, codes, )
//       ivf_pq::detail::unpack_list_data(
//         codes, list_data, offset, pq_bits, resource::get_cuda_stream(res));
// }

// inline void pack(
//   raft::resources const& res,
//   device_matrix_view<const uint8_t, uint32_t, row_major> codes,
//   uint32_t pq_bits,
//   uint32_t offset,
//   uint32_t batch_size,
//   device_mdspan<uint8_t, list_spec<uint32_t, uint32_t>::list_extents, row_major> list_data)
// {
//   ivf_pq::detail::pack_list_data(list_data, codes, offset, pq_bits, resource::get_cuda_stream(res));
// }
}  // namespace raft::neighbors::ivf_pq::codepacker