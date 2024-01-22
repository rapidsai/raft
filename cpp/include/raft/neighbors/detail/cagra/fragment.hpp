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

#include "device_common.hpp"
#include "utils.hpp"
#include <raft/core/logger.hpp>
#include <type_traits>

namespace raft::neighbors::cagra::detail {
namespace device {

namespace detail {
template <unsigned SIZE>
struct load_unit_t {
  using type = uint4;
};
template <>
struct load_unit_t<8> {
  using type = std::uint64_t;
};
template <>
struct load_unit_t<4> {
  using type = std::uint32_t;
};
template <>
struct load_unit_t<2> {
  using type = std::uint16_t;
};
template <>
struct load_unit_t<1> {
  using type = std::uint8_t;
};
}  // namespace detail

// One dataset or query vector is distributed within a warp and stored as `fragment`.
template <int DIM, class T, unsigned TEAM_SIZE, class ENABLED>
struct fragment_base {};
template <int DIM, class T, unsigned TEAM_SIZE = warp_size>
struct fragment
  : fragment_base<DIM,
                  T,
                  TEAM_SIZE,
                  typename std::enable_if<DIM % (TEAM_SIZE * utils::size_of<T>()) == 0>::type> {
  static constexpr unsigned num_elements = DIM / TEAM_SIZE;
  using block_t = typename detail::load_unit_t<num_elements * utils::size_of<T>()>::type;
  static constexpr unsigned num_load_blocks =
    num_elements * utils::size_of<T>() / utils::size_of<block_t>();

  union {
    T x[num_elements];
    block_t load_block[num_load_blocks];
  };
};

// Load a vector from device/shared memory
template <int DIM, class T, unsigned TEAM_SIZE, class INPUT_T>
_RAFT_DEVICE void load_vector_sync(device::fragment<DIM, T, TEAM_SIZE>& frag,
                                   const INPUT_T* const input_vector_ptr,
                                   const unsigned input_vector_length,
                                   const bool sync = true)
{
  const auto lane_id = threadIdx.x % TEAM_SIZE;
  if (DIM == input_vector_length) {
    for (unsigned i = 0; i < frag.num_load_blocks; i++) {
      const auto vector_index = i * TEAM_SIZE + lane_id;
      frag.load_block[i] =
        reinterpret_cast<const typename device::fragment<DIM, T, TEAM_SIZE>::block_t*>(
          input_vector_ptr)[vector_index];
    }
  } else {
    for (unsigned i = 0; i < frag.num_elements; i++) {
      const auto vector_index = i * TEAM_SIZE + lane_id;

      INPUT_T v;
      if (vector_index < input_vector_length) {
        v = static_cast<INPUT_T>(input_vector_ptr[vector_index]);
      } else {
        v = static_cast<INPUT_T>(0);
      }

      frag.x[i] = v;
    }
  }
  if (sync) { __syncwarp(); }
}

// Compute the square of the L2 norm of two vectors
template <class COMPUTE_T, int DIM, class T, unsigned TEAM_SIZE>
_RAFT_DEVICE COMPUTE_T norm2(const device::fragment<DIM, T, TEAM_SIZE>& a,
                             const device::fragment<DIM, T, TEAM_SIZE>& b)
{
  COMPUTE_T sum = 0;

  // Compute the thread-local norm2
  for (unsigned i = 0; i < a.num_elements; i++) {
    const auto diff = static_cast<COMPUTE_T>(a.x[i]) - static_cast<COMPUTE_T>(b.x[i]);
    sum += diff * diff;
  }

  // Compute the result norm2 summing up the thread-local norm2s.
  for (unsigned offset = TEAM_SIZE / 2; offset > 0; offset >>= 1)
    sum += __shfl_xor_sync(0xffffffff, sum, offset);

  return sum;
}

template <class COMPUTE_T, int DIM, class T, unsigned TEAM_SIZE>
_RAFT_DEVICE COMPUTE_T norm2(const device::fragment<DIM, T, TEAM_SIZE>& a,
                             const device::fragment<DIM, T, TEAM_SIZE>& b,
                             const float scale)
{
  COMPUTE_T sum = 0;

  // Compute the thread-local norm2
  for (unsigned i = 0; i < a.num_elements; i++) {
    const auto diff =
      static_cast<COMPUTE_T>((static_cast<float>(a.x[i]) - static_cast<float>(b.x[i])) * scale);
    sum += diff * diff;
  }

  // Compute the result norm2 summing up the thread-local norm2s.
  for (unsigned offset = TEAM_SIZE / 2; offset > 0; offset >>= 1)
    sum += __shfl_xor_sync(0xffffffff, sum, offset);

  return sum;
}

template <class COMPUTE_T, int DIM, class T, unsigned TEAM_SIZE>
_RAFT_DEVICE COMPUTE_T norm2(const device::fragment<DIM, T, TEAM_SIZE>& a,
                             const T* b,  // [DIM]
                             const float scale)
{
  COMPUTE_T sum = 0;

  // Compute the thread-local norm2
  const unsigned chunk_size = a.num_elements / a.num_load_blocks;
  const unsigned lane_id    = threadIdx.x % TEAM_SIZE;
  for (unsigned i = 0; i < a.num_elements; i++) {
    unsigned j      = (i % chunk_size) + chunk_size * (lane_id + TEAM_SIZE * (i / chunk_size));
    const auto diff = static_cast<COMPUTE_T>(a.x[i] * scale) - static_cast<COMPUTE_T>(b[j] * scale);
    sum += diff * diff;
  }

  // Compute the result norm2 summing up the thread-local norm2s.
  for (unsigned offset = TEAM_SIZE / 2; offset > 0; offset >>= 1)
    sum += __shfl_xor_sync(0xffffffff, sum, offset);

  return sum;
}

template <class COMPUTE_T, int DIM, class T, unsigned TEAM_SIZE>
_RAFT_DEVICE inline COMPUTE_T norm2x(const device::fragment<DIM, T, TEAM_SIZE>& a,
                                     const COMPUTE_T* b,  // [dim]
                                     const uint32_t dim,
                                     const float scale)
{
  // Compute the thread-local norm2
  COMPUTE_T sum          = 0;
  const unsigned lane_id = threadIdx.x % TEAM_SIZE;
  if (dim == DIM) {
    const unsigned chunk_size = a.num_elements / a.num_load_blocks;
    for (unsigned i = 0; i < a.num_elements; i++) {
      unsigned j      = (i % chunk_size) + chunk_size * (lane_id + TEAM_SIZE * (i / chunk_size));
      const auto diff = static_cast<COMPUTE_T>(a.x[i] * scale) - b[j];
      sum += diff * diff;
    }
  } else {
    for (unsigned i = 0; i < a.num_elements; i++) {
      unsigned j = lane_id + (TEAM_SIZE * i);
      if (j >= dim) break;
      const auto diff = static_cast<COMPUTE_T>(a.x[i] * scale) - b[j];
      sum += diff * diff;
    }
  }

  // Compute the result norm2 summing up the thread-local norm2s.
  for (unsigned offset = TEAM_SIZE / 2; offset > 0; offset >>= 1)
    sum += __shfl_xor_sync(0xffffffff, sum, offset);

  return sum;
}

template <int DIM, class T, unsigned TEAM_SIZE>
_RAFT_DEVICE void print_fragment(const device::fragment<DIM, T, TEAM_SIZE>& a)
{
  for (unsigned i = 0; i < TEAM_SIZE; i++) {
    if ((threadIdx.x % TEAM_SIZE) == i) {
      for (unsigned j = 0; j < a.num_elements; j++) {
        RAFT_LOG_DEBUG("%+e ", static_cast<float>(a.x[j]));
      }
    }
    __syncwarp();
  }
}

}  // namespace device
}  // namespace raft::neighbors::cagra::detail
