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
_RAFT_HOST_DEVICE void bitset_view<bitset_t, index_t>::set(const index_t sample_index,
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
_RAFT_HOST_DEVICE inline index_t bitset_view<bitset_t, index_t>::n_elements() const
{
  return raft::ceildiv(bitset_len_, bitset_element_size);
}

template <typename bitset_t, typename index_t>
bitset<bitset_t, index_t>::bitset(const raft::resources& res,
                                  raft::device_vector_view<const index_t, index_t> mask_index,
                                  index_t bitset_len,
                                  bool default_value)
  : bitset_{std::size_t(raft::ceildiv(bitset_len, bitset_element_size)),
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
  : bitset_{std::size_t(raft::ceildiv(bitset_len, bitset_element_size)),
            raft::resource::get_cuda_stream(res)},
    bitset_len_{bitset_len}
{
  reset(res, default_value);
}

template <typename bitset_t, typename index_t>
index_t bitset<bitset_t, index_t>::n_elements() const
{
  return raft::ceildiv(bitset_len_, bitset_element_size);
}

template <typename bitset_t, typename index_t>
void bitset<bitset_t, index_t>::resize(const raft::resources& res,
                                       index_t new_bitset_len,
                                       bool default_value)
{
  auto old_size = raft::ceildiv(bitset_len_, bitset_element_size);
  auto new_size = raft::ceildiv(new_bitset_len, bitset_element_size);
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
