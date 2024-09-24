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

#include <raft/core/bitset.hpp>
#include <raft/core/device_container_policy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/util/device_atomics.cuh>
#include <raft/util/popc.cuh>

#include <rmm/device_scalar.hpp>

#include <thrust/for_each.h>

namespace raft::core {

template <typename bitset_t, typename index_t>
_RAFT_HOST_DEVICE inline bool bitset_view<bitset_t, index_t>::test(const index_t sample_index) const
{
  const bitset_t bit_element = bitset_ptr_[sample_index / bitset_element_size];
  const index_t bit_index    = sample_index % bitset_element_size;
  const bool is_bit_set      = (bit_element & (bitset_t{1} << bit_index)) != 0;
  return is_bit_set;
}

template <typename bitset_t, typename index_t>
_RAFT_HOST_DEVICE bool bitset_view<bitset_t, index_t>::operator[](const index_t sample_index) const
{
  return test(sample_index);
}

template <typename bitset_t, typename index_t>
_RAFT_DEVICE void bitset_view<bitset_t, index_t>::set(const index_t sample_index,
                                                      bool set_value) const
{
  const index_t bit_element = sample_index / bitset_element_size;
  const index_t bit_index   = sample_index % bitset_element_size;
  const bitset_t bitmask    = bitset_t{1} << bit_index;
  if (set_value) {
    atomicOr(bitset_ptr_ + bit_element, bitmask);
  } else {
    const bitset_t bitmask2 = ~bitmask;
    atomicAnd(bitset_ptr_ + bit_element, bitmask2);
  }
}

template <typename bitset_t, typename index_t>
void bitset_view<bitset_t, index_t>::count(const raft::resources& res,
                                           raft::device_scalar_view<index_t> count_gpu_scalar) const
{
  auto max_len = raft::make_host_scalar_view<const index_t, index_t>(&bitset_len_);
  auto values  = raft::make_device_vector_view<const bitset_t, index_t>(bitset_ptr_, n_elements());
  raft::popc(res, values, max_len, count_gpu_scalar);
}

template <typename bitset_t, typename index_t>
RAFT_KERNEL bitset_repeat_kernel(const bitset_t* src,
                                 bitset_t* output,
                                 index_t src_bit_len,
                                 index_t repeat_times)
{
  constexpr index_t bits_per_element = sizeof(bitset_t) * 8;
  int output_idx                     = blockIdx.x * blockDim.x + threadIdx.x;

  index_t total_bits  = src_bit_len * repeat_times;
  index_t output_size = (total_bits + bits_per_element - 1) / bits_per_element;
  index_t src_size    = (src_bit_len + bits_per_element - 1) / bits_per_element;

  if (output_idx < output_size) {
    bitset_t result     = 0;
    index_t bit_written = 0;

    index_t start_bit = output_idx * bits_per_element;

    while (bit_written < bits_per_element && start_bit + bit_written < total_bits) {
      index_t bit_idx      = (start_bit + bit_written) % src_bit_len;
      index_t src_word_idx = bit_idx / bits_per_element;
      index_t src_offset   = bit_idx % bits_per_element;

      index_t remaining_bits = min(bits_per_element - bit_written, src_bit_len - bit_idx);

      bitset_t src_value = (src[src_word_idx] >> src_offset);

      if (src_offset + remaining_bits > bits_per_element) {
        bitset_t next_value = src[(src_word_idx + 1) % src_size];
        src_value |= (next_value << (bits_per_element - src_offset));
      }
      src_value &= ((bitset_t{1} << remaining_bits) - 1);
      result |= (src_value << bit_written);
      bit_written += remaining_bits;
    }
    output[output_idx] = result;
  }
}

template <typename bitset_t, typename index_t>
void bitset_repeat(raft::resources const& handle,
                   const bitset_t* d_src,
                   bitset_t* d_output,
                   index_t src_bit_len,
                   index_t repeat_times)
{
  if (src_bit_len == 0 || repeat_times == 0) return;
  auto stream = resource::get_cuda_stream(handle);

  constexpr index_t bits_per_element = sizeof(bitset_t) * 8;
  const index_t total_bits           = src_bit_len * repeat_times;
  const index_t output_size          = (total_bits + bits_per_element - 1) / bits_per_element;

  int threadsPerBlock = 128;
  int blocksPerGrid   = (output_size + threadsPerBlock - 1) / threadsPerBlock;
  bitset_repeat_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
    d_src, d_output, src_bit_len, repeat_times);

  return;
}

template <typename bitset_t, typename index_t>
void bitset_view<bitset_t, index_t>::repeat(const raft::resources& res,
                                            index_t times,
                                            bitset_t* output_device_ptr) const
{
  auto thrust_policy                 = raft::resource::get_thrust_policy(res);
  constexpr index_t bits_per_element = sizeof(bitset_t) * 8;

  if (bitset_len_ % bits_per_element == 0) {
    index_t num_elements_to_copy = bitset_len_ / bits_per_element;

    for (index_t i = 0; i < times; ++i) {
      raft::copy(output_device_ptr + i * num_elements_to_copy,
                 bitset_ptr_,
                 num_elements_to_copy,
                 raft::resource::get_cuda_stream(res));
    }
  } else {
    bitset_repeat(res, bitset_ptr_, output_device_ptr, bitset_len_, times);
  }
}

template <typename bitset_t, typename index_t>
double bitset_view<bitset_t, index_t>::sparsity(const raft::resources& res) const
{
  index_t size_h = this->size();
  if (0 == size_h) { return static_cast<double>(1.0); }
  index_t count_h = this->count(res);

  return static_cast<double>((1.0 * (size_h - count_h)) / (1.0 * size_h));
}

template <typename bitset_t, typename index_t>
bitset<bitset_t, index_t>::bitset(const raft::resources& res,
                                  raft::device_vector_view<const index_t, index_t> mask_index,
                                  index_t bitset_len,
                                  bool default_value)
  : bitset_{std::size_t(raft::div_rounding_up_safe(bitset_len, bitset_element_size)),
            raft::resource::get_cuda_stream(res)},
    bitset_len_{bitset_len}
{
  reset(res, default_value);
  set(res, mask_index, !default_value);
}

template <typename bitset_t, typename index_t>
bitset<bitset_t, index_t>::bitset(const raft::resources& res,
                                  index_t bitset_len,
                                  bool default_value)
  : bitset_{std::size_t(raft::div_rounding_up_safe(bitset_len, bitset_element_size)),
            raft::resource::get_cuda_stream(res)},
    bitset_len_{bitset_len}
{
  reset(res, default_value);
}

template <typename bitset_t, typename index_t>
void bitset<bitset_t, index_t>::resize(const raft::resources& res,
                                       index_t new_bitset_len,
                                       bool default_value)
{
  auto old_size = raft::div_rounding_up_safe(bitset_len_, bitset_element_size);
  auto new_size = raft::div_rounding_up_safe(new_bitset_len, bitset_element_size);
  bitset_.resize(new_size);
  bitset_len_ = new_bitset_len;
  if (old_size < new_size) {
    // If the new size is larger, set the new bits to the default value
    thrust::fill_n(raft::resource::get_thrust_policy(res),
                   bitset_.data() + old_size,
                   new_size - old_size,
                   default_value ? ~bitset_t{0} : bitset_t{0});
  }
}

template <typename bitset_t, typename index_t>
template <typename output_t>
void bitset<bitset_t, index_t>::test(const raft::resources& res,
                                     raft::device_vector_view<const index_t, index_t> queries,
                                     raft::device_vector_view<output_t, index_t> output) const
{
  RAFT_EXPECTS(output.extent(0) == queries.extent(0), "Output and queries must be same size");
  auto bitset_view = view();
  raft::linalg::map(
    res,
    output,
    [bitset_view] __device__(index_t query) { return bitset_view.test(query); },
    queries);
}

template <typename bitset_t, typename index_t>
void bitset<bitset_t, index_t>::set(const raft::resources& res,
                                    raft::device_vector_view<const index_t, index_t> mask_index,
                                    bool set_value)
{
  auto this_bitset_view = view();
  thrust::for_each_n(raft::resource::get_thrust_policy(res),
                     mask_index.data_handle(),
                     mask_index.extent(0),
                     [this_bitset_view, set_value] __device__(const index_t sample_index) {
                       this_bitset_view.set(sample_index, set_value);
                     });
}

template <typename bitset_t, typename index_t>
void bitset<bitset_t, index_t>::flip(const raft::resources& res)
{
  auto bitset_span = this->to_mdspan();
  raft::linalg::map(
    res,
    bitset_span,
    [] __device__(bitset_t element) { return bitset_t(~element); },
    raft::make_const_mdspan(bitset_span));
}

template <typename bitset_t, typename index_t>
void bitset<bitset_t, index_t>::reset(const raft::resources& res, bool default_value)
{
  thrust::fill_n(raft::resource::get_thrust_policy(res),
                 bitset_.data(),
                 n_elements(),
                 default_value ? ~bitset_t{0} : bitset_t{0});
}

template <typename bitset_t, typename index_t>
void bitset<bitset_t, index_t>::count(const raft::resources& res,
                                      raft::device_scalar_view<index_t> count_gpu_scalar)
{
  auto max_len = raft::make_host_scalar_view<const index_t, index_t>(&bitset_len_);
  auto values =
    raft::make_device_vector_view<const bitset_t, index_t>(bitset_.data(), n_elements());
  raft::popc(res, values, max_len, count_gpu_scalar);
}

}  // end namespace raft::core
