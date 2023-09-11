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

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/util/device_atomics.cuh>
#include <thrust/for_each.h>

namespace raft::utils {
/**
 * @defgroup bitset Bitset
 * @{
 */
/**
 * @brief View of a RAFT Bitset.
 *
 * This lightweight structure stores a pointer to a bitset in device memory with it's length.
 * It provides a test() device function to check if a given index is set in the bitset.
 *
 * @tparam bitset_t Underlying type of the bitset array. Default is uint32_t.
 * @tparam index_t Indexing type used. Default is uint32_t.
 */
template <typename bitset_t = uint32_t, typename index_t = uint32_t>
struct bitset_view {
  index_t static constexpr const bitset_element_size = sizeof(bitset_t) * 8;

  _RAFT_HOST_DEVICE bitset_view(bitset_t* bitset_ptr, index_t bitset_len)
    : bitset_ptr_{bitset_ptr}, bitset_len_{bitset_len}
  {
  }
  /**
   * @brief Create a bitset view from a device vector view of the bitset.
   *
   * @param bitset_span Device vector view of the bitset
   */
  _RAFT_HOST_DEVICE bitset_view(raft::device_vector_view<bitset_t, index_t> bitset_span)
    : bitset_ptr_{bitset_span.data_handle()}, bitset_len_{bitset_span.extent(0)}
  {
  }
  /**
   * @brief Device function to test if a given index is set in the bitset.
   *
   * @param sample_index Single index to test
   * @return bool True if index has not been unset in the bitset
   */
  inline _RAFT_DEVICE auto test(const index_t sample_index) const -> bool
  {
    const bitset_t bit_element = bitset_ptr_[sample_index / bitset_element_size];
    const index_t bit_index    = sample_index % bitset_element_size;
    const bool is_bit_set      = (bit_element & (bitset_t{1} << bit_index)) != 0;
    return is_bit_set;
  }
  /**
   * @brief Get the device pointer to the bitset.
   */
  inline _RAFT_HOST_DEVICE auto data_handle() -> bitset_t* { return bitset_ptr_; }
  inline _RAFT_HOST_DEVICE auto data_handle() const -> const bitset_t* { return bitset_ptr_; }
  /**
   * @brief Get the number of bits of the bitset representation.
   */
  inline _RAFT_HOST_DEVICE auto size() const -> index_t
  {
    return bitset_len_ * bitset_element_size;
  }

  inline auto to_mdspan() -> raft::device_vector_view<bitset_t, index_t>
  {
    return raft::make_device_vector_view<bitset_t, index_t>(bitset_ptr_, bitset_len_);
  }
  inline auto to_mdspan() const -> raft::device_vector_view<const bitset_t, index_t>
  {
    return raft::make_device_vector_view<const bitset_t, index_t>(bitset_ptr_, bitset_len_);
  }

 private:
  bitset_t* bitset_ptr_;
  index_t bitset_len_;
};

/**
 * @brief RAFT Bitset.
 *
 * This structure encapsulates a bitset in device memory. It provides a view() method to get a
 * device-usable lightweight view of the bitset.
 * Each index is represented by a single bit in the bitset. The total number of bytes used is
 * ceil(bitset_len / 4).
 * @tparam bitset_t Underlying type of the bitset array. Default is uint32_t.
 * @tparam index_t Indexing type used. Default is uint32_t.
 */
template <typename bitset_t = uint32_t, typename index_t = uint32_t>
struct bitset {
  index_t static constexpr const bitset_element_size = sizeof(bitset_t) * 8;

  /**
   * @brief Construct a new bitset object with a list of indices to unset.
   *
   * @param res RAFT resources
   * @param mask_index List of indices to unset in the bitset
   * @param bitset_len Length of the bitset
   * @param default_value Default value to set the bits to. Default is true.
   */
  bitset(const raft::resources& res,
         raft::device_vector_view<const index_t, index_t> mask_index,
         index_t bitset_len,
         bool default_value = true)
    : bitset_{raft::make_device_vector<bitset_t, index_t>(
        res, raft::ceildiv(bitset_len, bitset_element_size))}
  {
    cudaMemsetAsync(bitset_.data_handle(),
                    default_value ? 0xff : 0x00,
                    bitset_.size() * sizeof(bitset_t),
                    resource::get_cuda_stream(res));
    bitset_set(res, view(), mask_index, !default_value);
  }

  /**
   * @brief Construct a new bitset object
   *
   * @param res RAFT resources
   * @param bitset_len Length of the bitset
   * @param default_value Default value to set the bits to. Default is true.
   */
  bitset(const raft::resources& res, index_t bitset_len, bool default_value = true)
    : bitset_{raft::make_device_vector<bitset_t, index_t>(
        res, raft::ceildiv(bitset_len, bitset_element_size))}
  {
    cudaMemsetAsync(bitset_.data_handle(),
                    default_value ? 0xff : 0x00,
                    bitset_.size() * sizeof(bitset_t),
                    resource::get_cuda_stream(res));
  }
  // Disable copy constructor
  bitset(const bitset&)            = delete;
  bitset(bitset&&)                 = default;
  bitset& operator=(const bitset&) = delete;
  bitset& operator=(bitset&&)      = default;

  /**
   * @brief Create a device-usable view of the bitset.
   *
   * @return bitset_view<bitset_t, index_t>
   */
  inline auto view() -> raft::utils::bitset_view<bitset_t, index_t>
  {
    return bitset_view<bitset_t, index_t>(bitset_.view());
  }
  [[nodiscard]] inline auto view() const -> raft::utils::bitset_view<const bitset_t, index_t>
  {
    return bitset_view<const bitset_t, index_t>(bitset_.view());
  }

  /**
   * @brief Get the device pointer to the bitset.
   */
  inline auto data_handle() -> bitset_t* { return bitset_.data_handle(); }
  inline auto data_handle() const -> const bitset_t* { return bitset_.data_handle(); }
  /**
   * @brief Get the number of bits of the bitset representation.
   */
  inline auto size() const -> index_t { return bitset_.size() * bitset_element_size; }

  inline auto view_mdspan() -> raft::device_vector_view<bitset_t, index_t>
  {
    return bitset_.view();
  }
  [[nodiscard]] inline auto view_mdspan() const -> raft::device_vector_view<const bitset_t, index_t>
  {
    return bitset_.view();
  }

 private:
  raft::device_vector<bitset_t, index_t> bitset_;
};

/**
 * @brief Set a list of indices in a bitset to set_value.
 *
 * @tparam bitset_t Underlying type of the bitset array
 * @tparam index_t Indexing type used.
 * @param res RAFT resources
 * @param bitset_view_ View of the bitset
 * @param mask_index indices to remove from the bitset
 * @param set_value Value to set the bits to (true or false)
 */
template <typename bitset_t, typename index_t>
void bitset_set(const raft::resources& res,
                raft::utils::bitset_view<bitset_t, index_t> bitset_view_,
                raft::device_vector_view<const index_t, index_t> mask_index,
                bool set_value = false)
{
  auto* bitset_ptr = bitset_view_.data_handle();
  constexpr auto bitset_element_size =
    raft::utils::bitset_view<bitset_t, index_t>::bitset_element_size;
  thrust::for_each_n(
    resource::get_thrust_policy(res),
    mask_index.data_handle(),
    mask_index.extent(0),
    [bitset_ptr, set_value, bitset_element_size] __device__(const index_t sample_index) {
      const index_t bit_element = sample_index / bitset_element_size;
      const index_t bit_index   = sample_index % bitset_element_size;
      const bitset_t bitmask    = bitset_t{1} << bit_index;
      if (set_value) {
        atomicOr(bitset_ptr + bit_element, bitmask);
      } else {
        const bitset_t bitmask2 = ~bitmask;
        atomicAnd(bitset_ptr + bit_element, bitmask2);
      }
    });
}

/**
 * @brief Test a list of indices in a bitset.
 *
 * @tparam bitset_t Underlying type of the bitset array
 * @tparam index_t Indexing type
 * @tparam output_t Output type of the test. Default is bool.
 * @param res RAFT resources
 * @param bitset_view_ View of the bitset
 * @param queries List of indices to test
 * @param output List of outputs
 */
template <typename bitset_t, typename index_t, typename output_t = bool>
void bitset_test(const raft::resources& res,
                 const raft::utils::bitset_view<bitset_t, index_t> bitset_view_,
                 raft::device_vector_view<const index_t, index_t> queries,
                 raft::device_vector_view<output_t, index_t> output)
{
  RAFT_EXPECTS(output.extent(0) == queries.extent(0), "Output and queries must be same size");
  raft::linalg::map(
    res,
    output,
    [=] __device__(index_t query) { return output_t(bitset_view_.test(query)); },
    queries);
}

/**
 * @brief Flip all the bit in a bitset.
 *
 * @tparam bitset_t Underlying type of the bitset array
 * @tparam index_t Indexing type
 * @param res RAFT resources
 * @param bitset_view_ View of the bitset
 */
template <typename bitset_t, typename index_t>
void bitset_flip(const raft::resources& res,
                 raft::utils::bitset_view<bitset_t, index_t> bitset_view_)
{
  auto bitset_span = bitset_view_.to_mdspan();
  raft::linalg::map(
    res,
    bitset_span,
    [] __device__(bitset_t element) { return bitset_t(~element); },
    raft::make_const_mdspan(bitset_span));
}
/** @} */
}  // end namespace raft::utils
