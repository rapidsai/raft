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

#pragma once

#include <raft/core/mdarray.hpp>
#include <raft/cuda_utils.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/vectorized.cuh>

namespace raft {
namespace stats {
namespace detail {

/**
 * @brief Center the input matrix wrt its mean
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @tparam TPB threads per block of the cuda kernel launched
 * @param out the output mean-centered matrix
 * @param data input matrix
 * @param mu the mean vector
 * @param D number of columns of data
 * @param N number of rows of data
 * @param rowMajor whether input is row or col major
 * @param bcastAlongRows whether to broadcast vector along rows or columns
 * @param stream cuda stream where to launch work
 */
template <typename Type, typename IdxType = int, int TPB = 256>
void meanCenter(Type* out,
                const Type* data,
                const Type* mu,
                IdxType D,
                IdxType N,
                bool rowMajor,
                bool bcastAlongRows,
                cudaStream_t stream)
{
  raft::linalg::matrixVectorOp(
    out,
    data,
    mu,
    D,
    N,
    rowMajor,
    bcastAlongRows,
    [] __device__(Type a, Type b) { return a - b; },
    stream);
}

/**
 * @brief Center the input matrix wrt its mean
 * @tparam Type the data type
 * @tparam LayoutPolicy Layout type of the input matrix.
 * @tparam TPB threads per block of the cuda kernel launched
 * @param handle the raft handle
 * @param data input matrix
 * @param mu the mean vector
 * @param out the output mean-centered matrix
 * @param bcastAlongRows whether to broadcast vector along rows or columns
 */
template <typename Type, typename LayoutPolicy, int TPB = 256>
void meanCenter(const raft::handle_t& handle,
                const raft::device_matrix_view<const Type, LayoutPolicy>& data,
                const raft::device_vector_view<const Type, LayoutPolicy>& mu,
                const raft::device_matrix_view<Type, LayoutPolicy>& out,
                bool bcastAlongRows)
{
  cudaStream_t stream = handle.get_stream();
  auto N = data.extent(0);
  auto D = data.extent(1);
  raft::linalg::matrixVectorOp(
    out.data(),
    data.data(),
    mu.data(),
    D,
    N,
    std::is_same_v<LayoutPolicy, raft::row_major>,
    bcastAlongRows,
    [] __device__(Type a, Type b) { return a - b; },
    stream);
}

/**
 * @brief Add the input matrix wrt its mean
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @tparam TPB threads per block of the cuda kernel launched
 * @param out the output mean-added matrix
 * @param data input matrix
 * @param mu the mean vector
 * @param D number of columns of data
 * @param N number of rows of data
 * @param rowMajor whether input is row or col major
 * @param bcastAlongRows whether to broadcast vector along rows or columns
 * @param stream cuda stream where to launch work
 */
template <typename Type, typename IdxType = int, int TPB = 256>
void meanAdd(Type* out,
             const Type* data,
             const Type* mu,
             IdxType D,
             IdxType N,
             bool rowMajor,
             bool bcastAlongRows,
             cudaStream_t stream)
{
  raft::linalg::matrixVectorOp(
    out,
    data,
    mu,
    D,
    N,
    rowMajor,
    bcastAlongRows,
    [] __device__(Type a, Type b) { return a + b; },
    stream);
}

/**
 * @brief Add the input matrix wrt its mean
 * @tparam Type the data type
 * @tparam LayoutPolicy Layout type of the input matrix.
 * @tparam TPB threads per block of the cuda kernel launched
 * @param handle the raft handle
 * @param data input matrix
 * @param mu the mean vector
 * @param out the output mean-centered matrix
 * @param bcastAlongRows whether to broadcast vector along rows or columns
 */
template <typename Type, typename LayoutPolicy, int TPB = 256>
void meanAdd(const raft::handle_t& handle,
  const raft::device_matrix_view<const Type, LayoutPolicy>& data,
  const raft::device_vector_view<const Type, LayoutPolicy>& mu,
  const raft::device_matrix_view<Type, LayoutPolicy>& out,
  bool bcastAlongRows)
{
  cudaStream_t stream = handle.get_stream();
  auto N = data.extent(0);
  auto D = data.extent(1);
  raft::linalg::matrixVectorOp(
    out.data(),
    data.data(),
    mu.data(),
    D,
    N,
    std::is_same_v<LayoutPolicy, raft::row_major>,
    bcastAlongRows,
    [] __device__(Type a, Type b) { return a + b; },
    stream);
}
};  // end namespace detail
};  // end namespace stats
};  // end namespace raft
