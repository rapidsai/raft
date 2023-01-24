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

#include <raft/util/cuda_utils.cuh>
#include <raft/util/device_atomics.cuh>

namespace raft {
namespace distance {
namespace detail {

template <typename T = uint64_t, typename = std::enable_if_t<std::is_integral<T>::value>>
__global__ void compress_to_bits_naive(const bool* in, int in_rows, int in_cols, T* out)
{
  constexpr int bits_per_element = 8 * sizeof(T);

  const size_t i = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t j = threadIdx.x + blockIdx.x * blockDim.x;

  if (in_rows <= i || in_cols <= j) { return; }

  bool bit   = in[i * in_cols + j];
  int bitpos = j % bits_per_element;

  T bitfield = bit ? T(1) << bitpos : 0;

  const size_t out_rows = raft::ceildiv(in_cols, bits_per_element);
  const size_t out_cols = in_rows;
  const size_t out_j    = i;
  const size_t out_i    = j / bits_per_element;
  if (out_i < out_rows && out_j < out_cols) { atomicOr(&out[out_i * out_cols + out_j], bitfield); }
}

};  // namespace detail
};  // namespace distance
};  // namespace raft
