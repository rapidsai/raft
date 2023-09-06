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
 * @tparam IdxT Indexing type used. Default is uint32_t.
 */
template <typename IdxT = uint32_t>
struct bitset_view {
  using BitsetT            = uint32_t;
  IdxT bitset_element_size = sizeof(BitsetT) * 8;

  _RAFT_HOST_DEVICE bitset_view(BitsetT* bitset_ptr, IdxT bitset_len)
    : bitset_ptr_{bitset_ptr}, bitset_len_{bitset_len}
  {
  }
  /**
   * @brief Create a bitset view from a device vector view of the bitset.
   *
   * @param bitset_span Device vector view of the bitset
   */
  _RAFT_HOST_DEVICE bitset_view(raft::device_vector_view<BitsetT, IdxT> bitset_span)
    : bitset_ptr_{bitset_span.data_handle()}, bitset_len_{bitset_span.extent(0)}
  {
  }
  /**
   * @brief Device function to test if a given index is set in the bitset.
   *
   * @param sample_index Single index to test
   * @return bool True if index has not been unset in the bitset
   */
  inline _RAFT_DEVICE auto test(const IdxT sample_index) const -> bool
  {
    const IdxT bit_element = bitset_ptr_[sample_index / bitset_element_size];
    const IdxT bit_index   = sample_index % bitset_element_size;
    const bool is_bit_set  = (bit_element & (1ULL << bit_index)) != 0;
    return is_bit_set;
  }
  /**
   * @brief Get the device pointer to the bitset.
   */
  inline _RAFT_HOST_DEVICE auto get_bitset_ptr() -> BitsetT* { return bitset_ptr_; }
  inline _RAFT_HOST_DEVICE auto get_bitset_ptr() const -> const BitsetT* { return bitset_ptr_; }
  /**
   * @brief Get the length of the bitset representation.
   */
  inline _RAFT_HOST_DEVICE auto get_bitset_len() const -> IdxT { return bitset_len_; }

 private:
  BitsetT* bitset_ptr_;
  IdxT bitset_len_;
};

/**
 * @brief RAFT Bitset.
 *
 * This structure encapsulates a bitset in device memory. It provides a view() method to get a
 * device-usable lightweight view of the bitset.
 * Each index is represented by a single bit in the bitset. The total number of bytes used is
 * ceil(bitset_len / 4).
 * The underlying type of the bitset array is uint32_t.
 * @tparam IdxT Indexing type used. Default is uint32_t.
 */
template <typename IdxT = uint32_t>
struct bitset {
  using BitsetT            = uint32_t;
  IdxT bitset_element_size = sizeof(BitsetT) * 8;

  /**
   * @brief Construct a new bitset object with a list of indices to unset.
   *
   * @param res RAFT resources
   * @param mask_index List of indices to unset in the bitset
   * @param bitset_len Length of the bitset
   */
  bitset(const raft::resources& res,
         raft::device_vector_view<const IdxT, IdxT> mask_index,
         IdxT bitset_len)
    : bitset_{raft::make_device_vector<BitsetT, IdxT>(
        res, raft::ceildiv(bitset_len, bitset_element_size))}
  {
    RAFT_EXPECTS(mask_index.extent(0) <= bitset_len, "Mask index cannot be larger than bitset len");
    cudaMemsetAsync(bitset_.data_handle(),
                    0xff,
                    bitset_.size() * sizeof(BitsetT),
                    resource::get_cuda_stream(res));
    bitset_unset(res, view(), mask_index);
  }

  /**
   * @brief Construct a new bitset object
   *
   * @param res RAFT resources
   * @param bitset_len Length of the bitset
   */
  bitset(const raft::resources& res, IdxT bitset_len)
    : bitset_{raft::make_device_vector<BitsetT, IdxT>(
        res, raft::ceildiv(bitset_len, bitset_element_size))}
  {
    cudaMemsetAsync(bitset_.data_handle(),
                    0xff,
                    bitset_.size() * sizeof(BitsetT),
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
   * @return bitset_view<IdxT>
   */
  inline auto view() -> raft::utils::bitset_view<IdxT> { return bitset_view<IdxT>(bitset_.view()); }
  [[nodiscard]] inline auto view() const -> raft::utils::bitset_view<IdxT>
  {
    return bitset_view<IdxT>(bitset_.view());
  }

 private:
  raft::device_vector<BitsetT, IdxT> bitset_;
};

/**
 * @brief Function to unset a list of indices in a bitset.
 *
 * @tparam IdxT Indexing type used. Default is uint32_t.
 * @param res RAFT resources
 * @param bitset_view_ View of the bitset
 * @param mask_index indices to remove from the bitset
 */
template <typename IdxT>
void bitset_unset(const raft::resources& res,
                  raft::utils::bitset_view<IdxT> bitset_view_,
                  raft::device_vector_view<const IdxT, IdxT> mask_index)
{
  auto* bitset_ptr = bitset_view_.get_bitset_ptr();
  thrust::for_each_n(resource::get_thrust_policy(res),
                     mask_index.data_handle(),
                     mask_index.extent(0),
                     [bitset_ptr] __device__(const IdxT sample_index) {
                       const IdxT bit_element = sample_index / 32;
                       const IdxT bit_index   = sample_index % 32;
                       const uint32_t bitmask = ~(1 << bit_index);
                       atomicAnd(bitset_ptr + bit_element, bitmask);
                     });
}

/**
 * @brief Function to test a list of indices in a bitset.
 *
 * @tparam IdxT Indexing type
 * @tparam OutputT Output type of the test. Default is bool.
 * @param res RAFT resources
 * @param bitset_view_ View of the bitset
 * @param queries List of indices to test
 * @param output List of outputs
 */
template <typename IdxT, typename OutputT = bool>
void bitset_test(const raft::resources& res,
                 const raft::utils::bitset_view<IdxT> bitset_view_,
                 raft::device_vector_view<const IdxT, IdxT> queries,
                 raft::device_vector_view<OutputT, IdxT> output)
{
  RAFT_EXPECTS(output.extent(0) == queries.extent(0), "Output and queries must be same size");
  raft::linalg::map(
    res, output, [=] __device__(IdxT query) { return OutputT(bitset_view_.test(query)); }, queries);
}
/** @} */
}  // end namespace raft::utils
