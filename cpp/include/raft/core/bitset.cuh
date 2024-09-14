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
struct bitset_copy_functor {
  const bitset_t* bitset_ptr;
  bitset_t* output_device_ptr;
  index_t valid_bits;
  index_t bits_per_element;
  index_t total_bits;

  bitset_copy_functor(const bitset_t* _bitset_ptr,
                      bitset_t* _output_device_ptr,
                      index_t _valid_bits,
                      index_t _bits_per_element,
                      index_t _total_bits)
    : bitset_ptr(_bitset_ptr),
      output_device_ptr(_output_device_ptr),
      valid_bits(_valid_bits),
      bits_per_element(_bits_per_element),
      total_bits(_total_bits)
  {
  }

  __device__ void operator()(index_t i)
  {
    if (i < total_bits) {
      index_t src_bit_index = i % valid_bits;
      index_t dst_bit_index = i;

      index_t src_element_index = src_bit_index / bits_per_element;
      index_t src_bit_offset    = src_bit_index % bits_per_element;

      index_t dst_element_index = dst_bit_index / bits_per_element;
      index_t dst_bit_offset    = dst_bit_index % bits_per_element;

      bitset_t src_element = bitset_ptr[src_element_index];
      bitset_t src_bit     = (src_element >> src_bit_offset) & 1;

      if (src_bit) {
        atomicOr(output_device_ptr + dst_element_index, bitset_t(1) << dst_bit_offset);
      } else {
        atomicAnd(output_device_ptr + dst_element_index, ~(bitset_t(1) << dst_bit_offset));
      }
    }
  }
};

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
    index_t valid_bits          = bitset_len_;
    index_t total_bits          = valid_bits * times;
    index_t output_row_elements = (total_bits + bits_per_element - 1) / bits_per_element;
    thrust::for_each_n(thrust_policy,
                       thrust::counting_iterator<index_t>(0),
                       total_bits,
                       bitset_copy_functor<bitset_t, index_t>(
                         bitset_ptr_, output_device_ptr, valid_bits, bits_per_element, total_bits));
  }
}

template <typename bitset_t, typename index_t>
double bitset_view<bitset_t, index_t>::sparsity(const raft::resources& res) const
{
  index_t nnz_h  = 0;
  index_t size_h = this->size();
  auto stream    = raft::resource::get_cuda_stream(res);

  if (0 == size_h) { return static_cast<double>(1.0); }

  rmm::device_scalar<index_t> nnz(0, stream);

  auto vector_view = raft::make_device_vector_view<const bitset_t, index_t>(data(), n_elements());
  auto nnz_view    = raft::make_device_scalar_view<index_t>(nnz.data());
  auto size_view   = raft::make_host_scalar_view<index_t>(&size_h);

  raft::popc(res, vector_view, size_view, nnz_view);
  raft::copy(&nnz_h, nnz.data(), 1, stream);

  raft::resource::sync_stream(res, stream);
  return static_cast<double>((1.0 * (size_h - nnz_h)) / (1.0 * size_h));
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
  auto max_len = raft::make_host_scalar_view<index_t>(&bitset_len_);
  auto values =
    raft::make_device_vector_view<const bitset_t, index_t>(bitset_.data(), n_elements());
  raft::popc(res, values, max_len, count_gpu_scalar);
}

}  // end namespace raft::core
