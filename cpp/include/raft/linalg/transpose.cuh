/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#ifndef __TRANSPOSE_H
#define __TRANSPOSE_H

#pragma once

#include "detail/transpose.cuh"
#include <raft/core/mdarray.hpp>

namespace raft {
namespace linalg {

/**
 * @brief transpose on the column major input matrix using Jacobi method
 * @param handle: raft handle
 * @param in: input matrix
 * @param out: output. Transposed input matrix
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param stream: cuda stream
 */
template <typename math_t>
void transpose(const raft::handle_t& handle,
               math_t* in,
               math_t* out,
               int n_rows,
               int n_cols,
               cudaStream_t stream)
{
  detail::transpose(handle, in, out, n_rows, n_cols, stream);
}

/**
 * @brief transpose on the column major input matrix using Jacobi method
 * @param inout: input and output matrix
 * @param n: number of rows and columns of input matrix
 * @param stream: cuda stream
 */
template <typename math_t>
void transpose(math_t* inout, int n, cudaStream_t stream)
{
  detail::transpose(inout, n, stream);
}

/**
 * @brief Transpose a matrix. The output has same layout policy as the input.
 *
 * @tparam T Data type of input matrix element.
 * @tparam LayoutPolicy Layout type of the input matrix. When layout is strided, it can
 *                      be a submatrix of a larger matrix. Arbitrary stride is not supported.
 *
 * @param[in]  handle raft handle for managing expensive cuda resources.
 * @param[in]  in     Input matrix.
 * @param[out] out    Output matirx, storage is pre-allocated by caller.
 */
template <typename T, typename LayoutPolicy>
auto transpose(handle_t const& handle,
               device_matrix_view<T, LayoutPolicy> in,
               device_matrix_view<T, LayoutPolicy> out)
  -> std::enable_if_t<std::is_floating_point_v<T>, void>
{
  RAFT_EXPECTS(out.extent(0) == in.extent(1), "Invalid shape for transpose.");
  RAFT_EXPECTS(out.extent(1) == in.extent(0), "Invalid shape for transpose.");

  if constexpr (std::is_same_v<typename decltype(in)::layout_type, layout_c_contiguous>) {
    detail::transpose_row_major_impl(handle, in, out);
  } else if (std::is_same_v<typename decltype(in)::layout_type, layout_f_contiguous>) {
    detail::transpose_col_major_impl(handle, in, out);
  } else {
    RAFT_EXPECTS(in.stride(0) == 1 || in.stride(1) == 1, "Unsupported matrix layout.");
    if (in.stride(1) == 1) {
      // row-major submatrix
      detail::transpose_row_major_impl(handle, in, out);
    } else {
      // col-major submatrix
      detail::transpose_col_major_impl(handle, in, out);
    }
  }
}

/**
 * @brief Transpose a matrix. The output has same layout policy as the input.
 *
 * @tparam T Data type of input matrix elements.
 * @tparam LayoutPolicy Layout type of the input matrix. When layout is strided, it can
 *                      be a submatrix of a larger matrix. Arbitrary stride is not supported.
 *
 * @param[in] handle raft handle for managing expensive cuda resources.
 * @param[in] in     Input matrix.
 *
 * @return The transposed matrix.
 */
template <typename T, typename LayoutPolicy>
[[nodiscard]] auto transpose(handle_t const& handle, device_matrix_view<T, LayoutPolicy> in)
  -> std::enable_if_t<std::is_floating_point_v<T> &&
                        (std::is_same_v<LayoutPolicy, layout_c_contiguous> ||
                         std::is_same_v<LayoutPolicy, layout_f_contiguous>),
                      device_matrix<T, LayoutPolicy>>
{
  auto out = make_device_matrix<T, LayoutPolicy>(handle, in.extent(1), in.extent(0));
  transpose(handle, in, out.view());
  return out;
}

/**
 * @brief Transpose a matrix. The output has same layout policy as the input.
 *
 * @tparam T Data type of input matrix elements.
 * @tparam LayoutPolicy Layout type of the input matrix. When layout is strided, it can
 *                      be a submatrix of a larger matrix. Arbitrary stride is not supported.
 *
 * @param[in] handle raft handle for managing expensive cuda resources.
 * @param[in] in     Input matrix.
 *
 * @return The transposed matrix.
 */
template <typename T>
[[nodiscard]] auto transpose(handle_t const& handle,
                             device_matrix_view<T, raft::detail::stdex::layout_stride> in)
  -> std::enable_if_t<std::is_floating_point_v<T>,
                      device_matrix<T, raft::detail::stdex::layout_stride>>
{
  using extent_type = raft::extents<raft::dynamic_extent, raft::dynamic_extent>;
  extent_type exts{in.extent(1), in.extent(0)};
  using policy_type = typename raft::device_matrix<T, layout_stride>::container_policy_type;
  policy_type policy(handle.get_stream());

  RAFT_EXPECTS(in.stride(0) == 1 || in.stride(1) == 1, "Unsupported matrix layout.");
  if (in.stride(1) == 1) {
    // row-major submatrix
    std::array<size_t, 2> strides{in.extent(0), 1};
    auto layout = layout_stride::mapping<extent_type>{exts, strides};
    raft::device_matrix<T, layout_stride> out{layout, policy};
    transpose(handle, in, out.view());
    return out;
  } else {
    RAFT_EXPECTS(in.stride(0) == 1, "Unsupported layout type.");
    // col-major submatrix
    std::array<size_t, 2> strides{1, in.extent(1)};
    auto layout = layout_stride::mapping<extent_type>{exts, strides};
    raft::device_matrix<T, layout_stride> out{layout, policy};
    transpose(handle, in, out.view());
    return out;
  }
}
};  // end namespace linalg
};  // end namespace raft

#endif
