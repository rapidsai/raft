/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace raft::matrix::detail {

template <typename T>
RAFT_KERNEL col_right_shift(T* in_out, size_t n_rows, size_t n_cols, size_t k, T val)
{
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n_rows) {
    size_t base_idx      = row * n_cols;
    size_t cols_to_shift = n_cols - k;
    for (size_t i = 1; i <= cols_to_shift; i++) {
      in_out[base_idx + (n_cols - i)] = in_out[base_idx + (n_cols - k - i)];
    }
    for (size_t i = 0; i < k; i++) {
      in_out[base_idx + i] = val;
    }
  }
}

template <typename math_t, typename matrix_idx_t>
void col_right_shift(raft::resources const& handle,
                     raft::device_matrix_view<math_t, matrix_idx_t, row_major> in_out,
                     math_t val,
                     size_t k)
{
  size_t n_rows     = in_out.extent(0);
  size_t n_cols     = in_out.extent(1);
  size_t TPB        = 256;
  size_t num_blocks = static_cast<size_t>((n_rows + TPB) / TPB);

  col_right_shift<math_t><<<num_blocks, TPB, 0, raft::resource::get_cuda_stream(handle)>>>(
    in_out.data_handle(), n_rows, n_cols, k, val);
}

template <typename T>
RAFT_KERNEL col_right_shift(T* in_out, size_t n_rows, size_t n_cols, size_t k, const T* values)
{
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n_rows) {
    size_t base_idx      = row * n_cols;
    size_t cols_to_shift = n_cols - k;
    for (size_t i = 1; i <= cols_to_shift; i++) {
      in_out[base_idx + (n_cols - i)] = in_out[base_idx + (n_cols - k - i)];
    }
    for (size_t i = 0; i < k; i++) {
      in_out[base_idx + i] = values[row * k + i];
    }
  }
}

template <typename math_t, typename matrix_idx_t>
void col_right_shift(raft::resources const& handle,
                     raft::device_matrix_view<math_t, matrix_idx_t, row_major> in_out,
                     raft::device_matrix_view<const math_t, matrix_idx_t> values)
{
  size_t n_rows     = in_out.extent(0);
  size_t n_cols     = in_out.extent(1);
  size_t TPB        = 256;
  size_t num_blocks = static_cast<size_t>((n_rows + TPB) / TPB);

  size_t k = values.extent(1);

  col_right_shift<math_t><<<num_blocks, TPB, 0, raft::resource::get_cuda_stream(handle)>>>(
    in_out.data_handle(), n_rows, n_cols, k, values.data_handle());
  return;
}

template <typename T>
RAFT_KERNEL col_right_shift_self(T* in_out, size_t n_rows, size_t n_cols, size_t k)
{
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n_rows) {
    size_t base_idx      = row * n_cols;
    size_t cols_to_shift = n_cols - k;
    for (size_t i = 1; i <= cols_to_shift; i++) {
      in_out[base_idx + (n_cols - i)] = in_out[base_idx + (n_cols - k - i)];
    }
    for (size_t i = 0; i < k; i++) {
      in_out[base_idx + i] = row;
    }
  }
}

template <typename math_t, typename matrix_idx_t>
void col_right_shift_self(raft::resources const& handle,
                          raft::device_matrix_view<math_t, matrix_idx_t, row_major> in_out,
                          size_t k)
{
  size_t n_rows     = in_out.extent(0);
  size_t n_cols     = in_out.extent(1);
  size_t TPB        = 256;
  size_t num_blocks = static_cast<size_t>((n_rows + TPB) / TPB);

  col_right_shift_self<math_t><<<num_blocks, TPB, 0, raft::resource::get_cuda_stream(handle)>>>(
    in_out.data_handle(), n_rows, n_cols, k);
  return;
}

}  // namespace raft::matrix::detail
