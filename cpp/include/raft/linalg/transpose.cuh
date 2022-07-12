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
 * @brief Transpose a contiguous matrix. The output have same layout policy as input.
 *
 * @param in Input matrix, the storage should be contiguous.
 * @param out Output matirx, storage is pre-allocated by caller and should be contiguous.
 */
template <typename T, typename LayoutPolicy>
auto transpose(handle_t const& handle,
               device_matrix_view<T, LayoutPolicy> in,
               device_matrix_view<T, LayoutPolicy> out)
  -> std::enable_if_t<std::is_floating_point_v<T> &&
                        (std::is_same_v<LayoutPolicy, layout_c_contiguous> ||
                         std::is_same_v<LayoutPolicy, layout_f_contiguous>),
                      void>
{
  RAFT_EXPECTS(out.extent(0) == in.extent(1), "Invalid shape for transpose.");
  RAFT_EXPECTS(out.extent(1) == in.extent(0), "Invalid shape for transpose.");
  RAFT_EXPECTS(in.is_contiguous(), "Invalid format for transpose input.");
  RAFT_EXPECTS(out.is_contiguous(), "Invalid format for transpose output.");

  auto out_n_rows = in.extent(1);
  auto out_n_cols = in.extent(0);

  T constexpr kOne  = 1;
  T constexpr kZero = 0;
  if constexpr (std::is_same_v<typename decltype(in)::layout_type, layout_c_contiguous>) {
    CUBLAS_TRY(detail::cublasgeam(handle.get_cublas_handle(),
                                  CUBLAS_OP_T,
                                  CUBLAS_OP_N,
                                  out_n_cols,
                                  out_n_rows,
                                  &kOne,
                                  in.data(),
                                  in.stride(0),
                                  &kZero,
                                  static_cast<T*>(nullptr),
                                  out.stride(0),
                                  out.data(),
                                  out.stride(0),
                                  handle.get_stream()));
  } else {
    static_assert(std::is_same_v<typename decltype(in)::layout_type, layout_f_contiguous>);
    CUBLAS_TRY(detail::cublasgeam(handle.get_cublas_handle(),
                                  CUBLAS_OP_T,
                                  CUBLAS_OP_N,
                                  out_n_rows,
                                  out_n_cols,
                                  &kOne,
                                  in.data(),
                                  in.stride(1),
                                  &kZero,
                                  static_cast<T*>(nullptr),
                                  out.stride(1),
                                  out.data(),
                                  out.stride(1),
                                  handle.get_stream()));
  }
}

/**
 * @brief Transpose a contiguous matrix. The output have same layout policy as input.
 *
 * @param in Input matrix, the storage should be contiguous.
 *
 * @return The transposed matrix
 */
template <typename T, typename LayoutPolicy>
auto transpose(handle_t const& handle, device_matrix_view<T, LayoutPolicy> in)
  -> std::enable_if_t<std::is_floating_point_v<T> &&
                        (std::is_same_v<LayoutPolicy, layout_c_contiguous> ||
                         std::is_same_v<LayoutPolicy, layout_f_contiguous>),
                      device_matrix<T, LayoutPolicy>>
{
  auto out = make_device_matrix<T, LayoutPolicy>(handle, in.extent(1), in.extent(0));
  transpose(handle, in, out.view());
  return out;
}
};  // end namespace linalg
};  // end namespace raft

#endif
