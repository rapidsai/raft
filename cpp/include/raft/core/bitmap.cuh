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

#include <raft/core/bitset.cuh>
#include <raft/core/detail/mdspan_util.cuh>
#include <raft/core/device_container_policy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>

namespace raft::core {
/**
 * @defgroup bitmap Bitmap
 * @{
 */
/**
 * @brief View of a RAFT Bitmap.
 *
 * This lightweight structure which represents and manipulates a two-dimensional bitmap matrix view
 * with row major order. This class provides functionality for handling a matrix where each element
 * is represented as a bit in a bitmap.
 *
 * @tparam bitmap_t Underlying type of the bitmap array. Default is uint32_t.
 * @tparam index_t Indexing type used. Default is uint32_t.
 */
template <typename bitmap_t = uint32_t, typename index_t = uint32_t>
struct bitmap_view : public bitset_view<bitmap_t, index_t> {
  static constexpr index_t bitmap_element_size = sizeof(bitmap_t) * 8;
  static_assert((std::is_same<bitmap_t, uint32_t>::value ||
                 std::is_same<bitmap_t, uint64_t>::value),
                "The bitmap_t must be uint32_t or uint64_t.");
  /**
   * @brief Create a bitmap view from a device raw pointer.
   *
   * @param bitmap_ptr Device raw pointer
   * @param rows Number of row in the matrix.
   * @param cols Number of col in the matrix.
   */
  _RAFT_HOST_DEVICE bitmap_view(bitmap_t* bitmap_ptr, index_t rows, index_t cols)
    : bitset_view<bitmap_t, index_t>(bitmap_ptr, rows * cols),
      bitmap_ptr_{bitmap_ptr},
      rows_(rows),
      cols_(cols)
  {
  }

  /**
   * @brief Create a bitmap view from a device vector view of the bitset.
   *
   * @param bitmap_span Device vector view of the bitmap
   * @param rows Number of row in the matrix.
   * @param cols Number of col in the matrix.
   */
  _RAFT_HOST_DEVICE bitmap_view(raft::device_vector_view<bitmap_t, index_t> bitmap_span,
                                index_t rows,
                                index_t cols)
    : bitset_view<bitmap_t, index_t>(bitmap_span, rows * cols),
      bitmap_ptr_{bitmap_span.data_handle()},
      rows_(rows),
      cols_(cols)
  {
  }

 private:
  // Hide the constructors of bitset_view.
  _RAFT_HOST_DEVICE bitmap_view(bitmap_t* bitmap_ptr, index_t bitmap_len)
    : bitset_view<bitmap_t, index_t>(bitmap_ptr, bitmap_len)
  {
  }

  _RAFT_HOST_DEVICE bitmap_view(raft::device_vector_view<bitmap_t, index_t> bitmap_span,
                                index_t bitmap_len)
    : bitset_view<bitmap_t, index_t>(bitmap_span, bitmap_len)
  {
  }

 public:
  /**
   * @brief Device function to test if a given row and col are set in the bitmap.
   *
   * @param row Row index of the bit to test
   * @param col Col index of the bit to test
   * @return bool True if index has not been unset in the bitset
   */
  inline _RAFT_DEVICE auto test(const index_t row, const index_t col) const -> bool
  {
    return test(row * cols_ + col);
  }

  /**
   * @brief Device function to set a given row and col to set_value in the bitset.
   *
   * @param row Row index of the bit to set
   * @param col Col index of the bit to set
   * @param new_value Value to set the bit to (true or false)
   */
  inline _RAFT_DEVICE void set(const index_t row, const index_t col, bool new_value) const
  {
    set(row * cols_ + col, &new_value);
  }

  /**
   * @brief Get the total number of rows
   * @return index_t The total number of rows
   */
  inline _RAFT_HOST_DEVICE index_t get_n_rows() const { return rows_; }

  /**
   * @brief Get the total number of columns
   * @return index_t The total number of columns
   */
  inline _RAFT_HOST_DEVICE index_t get_n_cols() const { return cols_; }

  /**
   * @brief Returns the number of non-zero bits in nnz_gpu_scalar.
   *
   * @param[in] res RAFT resources
   * @param[out] nnz_gpu_scalar Device scalar to store the nnz
   */
  void get_nnz(const raft::resources& res, raft::device_scalar_view<index_t> nnz_gpu_scalar)
  {
    auto n_elements_ = raft::ceildiv(rows_ * cols_, bitmap_element_size);
    auto nnz_gpu = raft::make_device_vector_view<index_t, index_t>(nnz_gpu_scalar.data_handle(), 1);
    auto bitmap_matrix_view = raft::make_device_matrix_view<const bitmap_t, index_t, col_major>(
      bitmap_ptr_, n_elements_, 1);

    bitmap_t n_last_element = ((rows_ * cols_) % bitmap_element_size);
    bitmap_t last_element_mask =
      n_last_element ? (bitmap_t)((bitmap_t{1} << n_last_element) - bitmap_t{1}) : ~bitmap_t{0};
    raft::linalg::coalesced_reduction(
      res,
      bitmap_matrix_view,
      nnz_gpu,
      index_t{0},
      false,
      [last_element_mask, n_elements_] __device__(bitmap_t element, index_t index) {
        index_t result = 0;
        if constexpr (bitmap_element_size == 64) {
          if (index == n_elements_ - 1)
            result = index_t(raft::detail::popc(element & last_element_mask));
          else
            result = index_t(raft::detail::popc(element));
        } else {  // Needed because popc is not overloaded for 16 and 8 bit elements
          if (index == n_elements_ - 1)
            result = index_t(raft::detail::popc(uint32_t{element} & last_element_mask));
          else
            result = index_t(raft::detail::popc(uint32_t{element}));
        }

        return result;
      });
  }

  /**
   * @brief Returns the number of non-zero bits.
   *
   * @param res RAFT resources
   * @return index_t Number of non-zero bits
   */
  auto get_nnz(const raft::resources& res) -> index_t
  {
    auto nnz_gpu_scalar = raft::make_device_scalar<index_t>(res, 0.0);
    get_nnz(res, nnz_gpu_scalar.view());
    index_t nnz_gpu = 0;
    raft::update_host(&nnz_gpu, nnz_gpu_scalar.data_handle(), 1, resource::get_cuda_stream(res));
    resource::sync_stream(res);
    return nnz_gpu;
  }

 private:
  index_t rows_;
  index_t cols_;

  bitmap_t* bitmap_ptr_;
};

/** @} */
}  // end namespace raft::core
