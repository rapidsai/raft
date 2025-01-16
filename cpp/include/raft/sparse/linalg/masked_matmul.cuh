/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain A copy of the License at
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

#include <raft/sparse/linalg/detail/masked_matmul.cuh>

namespace raft {
namespace sparse {
namespace linalg {

/**
 * @defgroup masked_matmul Masked Matrix Multiplication
 * @{
 */

/**
 * @brief Performs a masked multiplication of dense matrices A and B, followed by an element-wise
 * multiplication with the sparsity pattern defined by the mask, resulting in the computation
 * C = alpha * ((A * B) ∘ spy(mask)) + beta * C.
 *
 * This function multiplies two dense matrices A and B, and then applies an element-wise
 * multiplication using the sparsity pattern provided by the mask. The result is scaled by alpha
 * and added to beta times the original matrix C.
 *
 * @tparam value_t Data type of elements in the input matrices (e.g., half, float, double)
 * @tparam output_t Data type of elements in the output matrices (e.g., float, double)
 * @tparam index_t Type used for matrix indices
 * @tparam nnz_t Type used for the number of non-zero entries in CSR format
 * @tparam bitmap_t Type of the bitmap used for the mask
 *
 * @param[in] handle RAFT handle for resource management
 * @param[in] A Input dense matrix (device_matrix_view) with shape [m, k]
 * @param[in] B Input dense matrix (device_matrix_view) with shape [n, k]
 * @param[in] mask Bitmap view representing the sparsity pattern (bitmap_view) with logical shape
 * [m, n]. Each bit in the mask indicates whether the corresponding element pair in A and B is
 * included (1) or masked out (0).
 * @param[inout] C Output sparse matrix in CSR format (device_csr_matrix_view) with dense shape [m,
 * n]
 * @param[in] alpha Optional scalar multiplier for the product of A and B (default: 1.0 if
 * std::nullopt)
 * @param[in] beta Optional scalar multiplier for the original matrix C (default: 0 if std::nullopt)
 */
template <typename value_t, typename output_t, typename index_t, typename nnz_t, typename bitmap_t>
void masked_matmul(raft::resources const& handle,
                   raft::device_matrix_view<const value_t, index_t, raft::row_major> A,
                   raft::device_matrix_view<const value_t, index_t, raft::row_major> B,
                   raft::core::bitmap_view<bitmap_t, index_t> mask,
                   raft::device_csr_matrix_view<output_t, index_t, index_t, nnz_t> C,
                   std::optional<raft::host_scalar_view<output_t>> alpha = std::nullopt,
                   std::optional<raft::host_scalar_view<output_t>> beta  = std::nullopt)
{
  detail::masked_matmul(handle, A, B, mask, C, alpha, beta);
}

/**
 * @brief Computes a sparse matrix product with a masked sparsity pattern and scaling.
 *
 * This function computes the result of:
 * C = alpha * ((A * B) ∘ spy(mask)) + beta * C
 * where:
 * - A and B are dense input matrices.
 * - "mask" defines the sparsity pattern for element-wise multiplication.
 * - The result is scaled by alpha and added to beta times the original C.
 *
 * **Special behavior of the mask**:
 * - The `bitset` mask represents a single row of data, with its bits indicating whether
 *   each corresponding element in (A * B) is included (1) or masked out (0).
 * - If the output CSR matrix `C` has multiple rows, the `bitset` is logically repeated
 *   across all rows of `C`. For example, if `C` has `n_rows` rows, the same `bitset`
 *   pattern is applied to all rows.
 *
 * @tparam value_t    Data type of input matrix elements (e.g., half, float, double).
 * @tparam output_t   Data type of output matrix elements (e.g., float, double).
 * @tparam index_t    Type for matrix indices.
 * @tparam nnz_t      Type for non-zero entries in CSR format.
 * @tparam bitset_t   Type for the bitset mask.
 *
 * @param[in] handle  RAFT handle for managing resources.
 * @param[in] A       Dense input matrix [m, k] (row-major).
 * @param[in] B       Dense input matrix [n, k] (row-major).
 * @param[in] mask    Bitmap view representing a single row [1, n], where each bit
 *                    indicates if the corresponding element in (A * B) is included (1)
 *                    or masked out (0). The pattern is repeated for all rows of `C`.
 * @param[inout] C    Output sparse matrix in CSR format [m, n].
 * @param[in] alpha   Scalar multiplier for (A * B) (default: 1.0 if std::nullopt).
 * @param[in] beta    Scalar multiplier for the initial C (default: 0 if std::nullopt).
 */
template <typename value_t, typename output_t, typename index_t, typename nnz_t, typename bitset_t>
void masked_matmul(raft::resources const& handle,
                   raft::device_matrix_view<const value_t, index_t, raft::row_major> A,
                   raft::device_matrix_view<const value_t, index_t, raft::row_major> B,
                   raft::core::bitset_view<bitset_t, index_t> mask,
                   raft::device_csr_matrix_view<output_t, index_t, index_t, nnz_t> C,
                   std::optional<raft::host_scalar_view<output_t>> alpha = std::nullopt,
                   std::optional<raft::host_scalar_view<output_t>> beta  = std::nullopt)
{
  detail::masked_matmul(handle, A, B, mask, C, alpha, beta);
}

/** @} */  // end of masked_matmul

}  // end namespace linalg
}  // end namespace sparse
}  // end namespace raft
