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

#include <raft/core/handle.hpp>
namespace raft::sparse::solver::detail {

template <typename value_t, typename index_t>
void spmm(raft::spectral::matrix::sparse_matrix_t<index_t, value_t> A,
          raft::device_matrix_view<value_t, index_t, raft::col_major> B,
          raft::device_matrix_view<value_t, index_t, raft::col_major> C,
          bool transpose_a,
          bool transpose_b)
{}

template <typename value_t, typename index_t>
void b_orthonormalize(
    const raft::handle_t& handle,
    raft::spectral::matrix::sparse_matrix_t<index_t, value_t> A,
    raft::device_matrix_view<const value_t, index_t, raft::col_major> V,
    raft::device_matrix_view<value_t, index_t, raft::col_major> BV,
    bool retInvR=false)
{
    auto V_max = raft::make_device_vector_view<value_t, index_t>(handle, V.extent(1));
    normalization = raft::linalg::reduce(V, axis=0);
}
    


)    

template <typename value_t, typename index_t>
void lobpcg(const raft::handle_t& handle,
    // IN
    const raft::spectral::matrix::sparse_matrix_t<index_t, value_t> A, // shape=(n,n)
    raft::device_matrix_view<value_t, index_t, raft::col_major> X, // shape=(n,k) IN OUT Eigvectors
    raft::device_vector_view<value_t, index_t> W, // shape=(k) OUT Eigvals
    std::optional<raft::spectral::matrix::sparse_matrix_t<index_t, value_t>> B, // shape=(n,n)
    std::optional<raft::spectral::matrix::sparse_matrix_t<index_t, value_t>> M, // shape=(n,n)
    std::optional<raft::device_matrix_view<const value_t, index_t, raft::col_major>> Y, // Constraint matrix shape=(n,Y)
    value_t tol=0,
    std::uint32_t max_iter=20,
    bool largest=true)
{
    cudaStream_t stream = handle.get_stream();
    auto size_y = 0;
    if (Y.has_value()) size_y = Y.extent(1);
    auto n = X.nrows_;
    auto size_x = X.ncols_;

    if ((n - size_y) < (5 * size_x))
    {
        // DENSE SOLUTION
        return;
    }
    if (tol == 0)
    {
        tol = raft::mySqrt(1e-15) * n;
    }
    // Apply constraints to X
    if (Y.has_value())
    {
        cusparseDnMatDescr_t denseY;
        RAFT_CUSPARSE_TRY(cusparsecreatednmat(&denseY, n, size_y, n, Y.value().data_handle(), CUSPARSE_ORDER_COL)));
        auto* ptr_BY = Y.value().data_handle();
        if (B.has_value())
        {
            cusparseSpMatDescr_t sparseB;
            cusparseDnMatDescr_t dense_BY;
            RAFT_CUSPARSE_TRY(cusparsecreatecsr(&sparseB, n, n, B.nnz_, B.row_offsets, B.col_indices_, B.values_));
            auto matrix_BY = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, n, size_y);
            RAFT_CUSPARSE_TRY(cusparsecreatednmat(&dense_BY, n, size_y, n, matrix_BY.data_handle(), CUSPARSE_ORDER_COL)));
            // B * Y
            value_t alpha = 1;
            value_t beta = 0;
            size_t buff_size = 0;
            cusparsespmm_bufferSize(handle.get_cusparse_handle(),
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, sparseB, denseY, &beta, dense_BY, CUSPARSE_SPMM_ALG_DEFAULT, &buff_size, stream);
            rmm::device_uvector<char> dev_buffer(buff_size, stream);
            cusparsespmm_bufferSize(handle.get_cusparse_handle(),
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, sparseB, denseY, &beta, dense_BY, CUSPARSE_SPMM_ALG_DEFAULT, stream);
            cusparseDestroyDnMat(dense_B);
            cusparseDestroyDnMat(dense_BY);
            cusparseDestroySpMat(sparseB);
            // CONTINUE
        }

        // GramYBY
        // ApplyConstraints
    }
    
}
} // raft::sparse::solver::detail