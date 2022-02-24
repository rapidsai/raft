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
/**
 * @warning This file is deprecated and will be removed in release 22.06.
 * Please use the cuh version instead.
 */

#ifndef __TRANSPOSE_H
#define __TRANSPOSE_H

#pragma once

#include "detail/transpose.cuh"

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

};  // end namespace linalg
};  // end namespace raft

#endif