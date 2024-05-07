/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <raft/core/device_container_policy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>

namespace raft::core {
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
  static constexpr index_t bitset_element_size = sizeof(bitset_t) * 8;

  _RAFT_HOST_DEVICE bitset_view(bitset_t* bitset_ptr, index_t bitset_len)
    : bitset_ptr_{bitset_ptr}, bitset_len_{bitset_len}
  {
  }
  /**
   * @brief Create a bitset view from a device vector view of the bitset.
   *
   * @param bitset_span Device vector view of the bitset
   * @param bitset_len Number of bits in the bitset
   */
  _RAFT_HOST_DEVICE bitset_view(raft::device_vector_view<bitset_t, index_t> bitset_span,
                                index_t bitset_len)
    : bitset_ptr_{bitset_span.data_handle()}, bitset_len_{bitset_len}
  {
  }
  /**
   * @brief Device function to test if a given index is set in the bitset.
   *
   * @param sample_index Single index to test
   * @return bool True if index has not been unset in the bitset
   */
  inline _RAFT_HOST_DEVICE auto test(const index_t sample_index) const -> bool;
  /**
   * @brief Device function to test if a given index is set in the bitset.
   *
   * @param sample_index Single index to test
   * @return bool True if index has not been unset in the bitset
   */
  inline _RAFT_HOST_DEVICE auto operator[](const index_t sample_index) const -> bool;
  /**
   * @brief Device function to set a given index to set_value in the bitset.
   *
   * @param sample_index index to set
   * @param set_value Value to set the bit to (true or false)
   */
  inline _RAFT_HOST_DEVICE void set(const index_t sample_index, bool set_value) const;

  /**
   * @brief Get the device pointer to the bitset.
   */
  inline _RAFT_HOST_DEVICE auto data() -> bitset_t* { return bitset_ptr_; }
  inline _RAFT_HOST_DEVICE auto data() const -> const bitset_t* { return bitset_ptr_; }
  /**
   * @brief Get the number of bits of the bitset representation.
   */
  inline _RAFT_HOST_DEVICE auto size() const -> index_t { return bitset_len_; }

  /**
   * @brief Get the number of elements used by the bitset representation.
   */
  inline _RAFT_HOST_DEVICE auto n_elements() const -> index_t;

  inline auto to_mdspan() -> raft::device_vector_view<bitset_t, index_t>
  {
    return raft::make_device_vector_view<bitset_t, index_t>(bitset_ptr_, n_elements());
  }
  inline auto to_mdspan() const -> raft::device_vector_view<const bitset_t, index_t>
  {
    return raft::make_device_vector_view<const bitset_t, index_t>(bitset_ptr_, n_elements());
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
 * ceil(bitset_len / 8).
 * @tparam bitset_t Underlying type of the bitset array. Default is uint32_t.
 * @tparam index_t Indexing type used. Default is uint32_t.
 */
template <typename bitset_t = uint32_t, typename index_t = uint32_t>
struct bitset {
  static constexpr index_t bitset_element_size = sizeof(bitset_t) * 8;

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
         bool default_value = true);

  /**
   * @brief Construct a new bitset object
   *
   * @param res RAFT resources
   * @param bitset_len Length of the bitset
   * @param default_value Default value to set the bits to. Default is true.
   */
  bitset(const raft::resources& res, index_t bitset_len, bool default_value = true);
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
  inline auto view() -> raft::core::bitset_view<bitset_t, index_t>
  {
    return bitset_view<bitset_t, index_t>(to_mdspan(), bitset_len_);
  }
  [[nodiscard]] inline auto view() const -> raft::core::bitset_view<const bitset_t, index_t>
  {
    return bitset_view<const bitset_t, index_t>(to_mdspan(), bitset_len_);
  }

  /**
   * @brief Get the device pointer to the bitset.
   */
  inline auto data() -> bitset_t* { return bitset_.data(); }
  inline auto data() const -> const bitset_t* { return bitset_.data(); }
  /**
   * @brief Get the number of bits of the bitset representation.
   */
  inline auto size() const -> index_t { return bitset_len_; }

  /**
   * @brief Get the number of elements used by the bitset representation.
   */
  inline auto n_elements() const -> index_t;

  /** @brief Get an mdspan view of the current bitset */
  inline auto to_mdspan() -> raft::device_vector_view<bitset_t, index_t>
  {
    return raft::make_device_vector_view<bitset_t, index_t>(bitset_.data(), n_elements());
  }
  [[nodiscard]] inline auto to_mdspan() const -> raft::device_vector_view<const bitset_t, index_t>
  {
    return raft::make_device_vector_view<const bitset_t, index_t>(bitset_.data(), n_elements());
  }

  /** @brief Resize the bitset. If the requested size is larger, new memory is allocated and set to
   * the default value.
   * @param res RAFT resources
   * @param new_bitset_len new size of the bitset
   * @param default_value default value to initialize the new bits to
   */
  void resize(const raft::resources& res, index_t new_bitset_len, bool default_value = true);

  /**
   * @brief Test a list of indices in a bitset.
   *
   * @tparam output_t Output type of the test. Default is bool.
   * @param res RAFT resources
   * @param queries List of indices to test
   * @param output List of outputs
   */
  template <typename output_t = bool>
  void test(const raft::resources& res,
            raft::device_vector_view<const index_t, index_t> queries,
            raft::device_vector_view<output_t, index_t> output) const;
  /**
   * @brief Set a list of indices in a bitset to set_value.
   *
   * @param res RAFT resources
   * @param mask_index indices to remove from the bitset
   * @param set_value Value to set the bits to (true or false)
   */
  void set(const raft::resources& res,
           raft::device_vector_view<const index_t, index_t> mask_index,
           bool set_value = false);
  /**
   * @brief Flip all the bits in a bitset.
   * @param res RAFT resources
   */
  void flip(const raft::resources& res);
  /**
   * @brief Reset the bits in a bitset.
   *
   * @param res RAFT resources
   * @param default_value Value to set the bits to (true or false)
   */
  void reset(const raft::resources& res, bool default_value = true);
  /**
   * @brief Returns the number of bits set to true in count_gpu_scalar.
   *
   * @param[in] res RAFT resources
   * @param[out] count_gpu_scalar Device scalar to store the count
   */
  void count(const raft::resources& res, raft::device_scalar_view<index_t> count_gpu_scalar);
  /**
   * @brief Returns the number of bits set to true.
   *
   * @param res RAFT resources
   * @return index_t Number of bits set to true
   */
  auto count(const raft::resources& res) -> index_t
  {
    auto count_gpu_scalar = raft::make_device_scalar<index_t>(res, 0.0);
    count(res, count_gpu_scalar.view());
    index_t count_cpu = 0;
    raft::update_host(
      &count_cpu, count_gpu_scalar.data_handle(), 1, resource::get_cuda_stream(res));
    resource::sync_stream(res);
    return count_cpu;
  }
  /**
   * @brief Checks if any of the bits are set to true in the bitset.
   * @param res RAFT resources
   */
  bool any(const raft::resources& res) { return count(res) > 0; }
  /**
   * @brief Checks if all of the bits are set to true in the bitset.
   * @param res RAFT resources
   */
  bool all(const raft::resources& res) { return count(res) == bitset_len_; }
  /**
   * @brief Checks if none of the bits are set to true in the bitset.
   * @param res RAFT resources
   */
  bool none(const raft::resources& res) { return count(res) == 0; }

 private:
  raft::device_uvector<bitset_t> bitset_;
  index_t bitset_len_;
};

/** @} */
}  // end namespace raft::core
