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

#include <raft/neighbors/ivf_list.hpp>
#include <raft/neighbors/ivf_pq_types.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/device_atomics.cuh>
#include <raft/util/integer_utils.hpp>
#include <raft/util/pow2_utils.cuh>
#include <raft/util/vectorized.cuh>

#include <variant>

namespace raft::neighbors::ivf_pq::detail {

/** A chunk of PQ-encoded vector managed by one CUDA thread. */
using pq_vec_t = TxN_t<uint8_t, kIndexGroupVecLen>::io_t;

/**
 * This type mimics the `uint8_t&` for the indexing operator of `bitfield_view_t`.
 *
 * @tparam Bits number of bits comprising the value.
 */
template <uint32_t Bits>
struct bitfield_ref_t {
  static_assert(Bits <= 8 && Bits > 0, "Bit code must fit one byte");
  constexpr static uint8_t kMask = static_cast<uint8_t>((1u << Bits) - 1u);
  uint8_t* ptr;
  uint32_t offset;

  constexpr operator uint8_t()  // NOLINT
  {
    auto pair = static_cast<uint16_t>(ptr[0]);
    if (offset + Bits > 8) { pair |= static_cast<uint16_t>(ptr[1]) << 8; }
    return static_cast<uint8_t>((pair >> offset) & kMask);
  }

  constexpr auto operator=(uint8_t code) -> bitfield_ref_t&
  {
    if (offset + Bits > 8) {
      10111000
      auto pair = static_cast<uint16_t>(ptr[0]);
      pair = 0000000010111000
      kMask = 11111
      kMask << offset = 0000001111100000
      negation = 1111110000011111
      pair &= = 0000000000011000
      code << offset = 0000001111100000
      pair |= = 0000001111111000
      mod (000001111111000) with 100000000

      0000000100000000
      11111000
      00000011


def extract_chunk(byte_array, j, n):
    # Calculate the starting byte index and bit offset within that byte
    byte_index = j // 8
    bit_offset = j % 8

    # Initialize variables to store the extracted chunk
    extracted_chunk = 0
    remaining_bits = n

    while remaining_bits > 0:
        # Determine how many bits can be extracted from the current byte
        bits_to_extract = min(8 - bit_offset, remaining_bits)

        # Mask to extract the bits from the current byte
        mask = (1 << bits_to_extract) - 1

        # Extract the bits and shift them to the right position in the result
        extracted_chunk <<= bits_to_extract
        extracted_chunk |= (byte_array[byte_index] >> (8 - bit_offset - bits_to_extract)) & mask

        # Update remaining_bits and bit_offset
        remaining_bits -= bits_to_extract
        bit_offset = (bit_offset + bits_to_extract) % 8

        # Move to the next byte if necessary
        if bit_offset == 0:
            byte_index += 1

    return extracted_chunk.to_bytes(1, byteorder='big')

# Example usage
byte_array = bytearray([0b11011010, 0b10101010])
j = 4  # Start at the 4th bit (0-based index)
n = 5  # Extract 5 bits
result = extract_chunk(byte_array, j, n)
print(result)  # This will print the extracted chunk as a byte

      1011111111
      <first four bits of ptr[1]>000000<last 6 bits of ptr[0]>
      pair |= static_cast<uint16_t>(ptr[1]) << 8;
      pair &= ~(static_cast<uint16_t>(kMask) << offset);
      pair |= static_cast<uint16_t>(code) << offset;
      ptr[0] = static_cast<uint8_t>(Pow2<256>::mod(pair));
      ptr[1] = static_cast<uint8_t>(Pow2<256>::div(pair));
    } else {
      ptr[0] = (ptr[0] & ~(kMask << offset)) | (code << offset);
      first four bits are the code
    }
    return *this;
  }
};

/**
 * View a byte array as an array of unsigned integers of custom small bit size.
 *
 * @tparam Bits number of bits comprising a single element of the array.
 */
template <uint32_t Bits>
struct bitfield_view_t {
  static_assert(Bits <= 8 && Bits > 0, "Bit code must fit one byte");
  uint8_t* raw;

  constexpr auto operator[](uint32_t i) -> bitfield_ref_t<Bits>
  {
    uint32_t bit_offset = i * Bits;
    Bits = 6
    bit_offset = 6
    return bitfield_ref_t<6>{raw, 6};
    bitfield_ref_t<6>(raw, 6) = 00101101;

    return bitfield_ref_t<Bits>{raw + Pow2<8>::div(bit_offset), Pow2<8>::mod(bit_offset)};
  }
};

/**
 * Process a single vector in a list.
 *
 * @tparam PqBits
 * @tparam Action tells how to process a single vector (e.g. reconstruct or just unpack)
 *
 * @param[in] in_list_data the encoded cluster data.
 * @param[in] in_ix in-cluster index of the vector to be decoded (one-per-thread).
 * @param[in] out_ix the output index passed to the action
 * @param[in] pq_dim
 * @param action a callable action to be invoked on each PQ code (component of the encoding)
 *    type: void (uint8_t code, uint32_t out_ix, uint32_t j), where j = [0..pq_dim).
 */
template <uint32_t PqBits, typename Action>
__device__ void run_on_vector(
  device_mdspan<const uint8_t, list_spec<uint32_t, uint32_t>::list_extents, row_major> in_list_data,
  uint32_t in_ix,
  uint32_t out_ix,
  uint32_t pq_dim,
  Action action)
{
  using group_align         = Pow2<kIndexGroupSize>;
  const uint32_t group_ix   = group_align::div(in_ix);
  const uint32_t ingroup_ix = group_align::mod(in_ix);

  pq_vec_t code_chunk;
  bitfield_view_t<PqBits> code_view{reinterpret_cast<uint8_t*>(&code_chunk)};
  constexpr uint32_t kChunkSize = (sizeof(pq_vec_t) * 8u) / PqBits;
  for (uint32_t j = 0, i = 0; j < pq_dim; i++) {
    // read the chunk
    code_chunk = *reinterpret_cast<const pq_vec_t*>(&in_list_data(group_ix, i, ingroup_ix, 0));
    // read the codes, one/pq_dim at a time
#pragma unroll
    for (uint32_t k = 0; k < kChunkSize && j < pq_dim; k++, j++) {
      // read a piece of the reconstructed vector
      action(code_view[k], out_ix, j);
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
__device__ void write_vector(
  device_mdspan<uint8_t, list_spec<uint32_t, uint32_t>::list_extents, row_major> out_list_data,
  uint32_t out_ix,
  IdxT in_ix,
  uint32_t pq_dim,
  Action action)
{
  const uint32_t lane_id = Pow2<SubWarpSize>::mod(threadIdx.x);
  if (lane_id == 0) printf("SubWarpSize %u\n", SubWarpSize);


  using group_align         = Pow2<kIndexGroupSize>;
  const uint32_t group_ix   = group_align::div(out_ix);
  const uint32_t ingroup_ix = group_align::mod(out_ix);

  pq_vec_t code_chunk; = 4 byte chunk of data from first vector
  bitfield_view_t<PqBits> code_view{reinterpret_cast<uint8_t*>(&code_chunk)};
  array of 6 bit length elements
  straightaway memcpy a 16 bit chunk
  if pqbits = 5 => kChunkSize = 25, only copy 25 elements of first vector = 125 bits : one byte at a time
  if pq_dim = 50
  constexpr uint32_t kChunkSize = (sizeof(pq_vec_t) * 8u) / PqBits;
  // if (lane_id == 0) printf("PqBits %u, sizeof(pq_vec_t) %zu, kChunkSize %u\n", PqBits, sizeof(pq_vec_t), kChunkSize);
  for (uint32_t j = 0, i = 0; j < pq_dim; i++) {
    // clear the chunk
    if (lane_id == 0) { code_chunk = pq_vec_t{}; }
    // write the codes, one/pq_dim at a time
#pragma unroll
    for (uint32_t k = 0; k < kChunkSize && j < pq_dim; k++, j++) {
      if (lane_id == 0) printf("k %u\n", k);
      // write a single code
      uint8_t code = action(in_ix, j);
      if (lane_id == 0) { code_view[k] = code; }
    }
    // write the chunk to the list
    if (lane_id == 0) {
      *reinterpret_cast<pq_vec_t*>(&out_list_data(group_ix, i, ingroup_ix, 0)) = code_chunk;
    }
  }
}

/** Process the given indices or a block of a single list (cluster). */
template <uint32_t PqBits, typename Action>
__device__ void run_on_list(
  device_mdspan<const uint8_t, list_spec<uint32_t, uint32_t>::list_extents, row_major> in_list_data,
  std::variant<uint32_t, const uint32_t*> offset_or_indices,
  uint32_t len,
  uint32_t pq_dim,
  Action action)
{
  for (uint32_t ix = threadIdx.x + blockDim.x * blockIdx.x; ix < len; ix += blockDim.x) {
    const uint32_t src_ix = std::holds_alternative<uint32_t>(offset_or_indices)
                              ? std::get<uint32_t>(offset_or_indices) + ix
                              : std::get<const uint32_t*>(offset_or_indices)[ix];
    run_on_vector<PqBits>(in_list_data, src_ix, ix, pq_dim, action);
  }
}

/** Process the given indices or a block of a single list (cluster). */
template <uint32_t PqBits, uint32_t SubWarpSize, typename Action>
__device__ void write_list(
  device_mdspan<uint8_t, list_spec<uint32_t, uint32_t>::list_extents, row_major> out_list_data,
  std::variant<uint32_t, const uint32_t*> offset_or_indices,
  uint32_t len,
  uint32_t pq_dim,
  Action action)
{
  printf("sizeof pq_vec_t %zu\n", sizeof(pq_vec_t));
  using subwarp_align = Pow2<SubWarpSize>;
  uint32_t stride     = subwarp_align::div(blockDim.x);
  uint32_t ix         = subwarp_align::div(threadIdx.x + blockDim.x * blockIdx.x);
  for (; ix < len; ix += stride) {
    const uint32_t dst_ix = std::holds_alternative<uint32_t>(offset_or_indices)
                              ? std::get<uint32_t>(offset_or_indices) + ix
                              : std::get<const uint32_t*>(offset_or_indices)[ix];
    write_vector<PqBits, SubWarpSize>(out_list_data, dst_ix, ix, pq_dim, action);
  }
}

}  // namespace raft::neighbors::ivf_pq::detail
