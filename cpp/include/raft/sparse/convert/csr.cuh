/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#ifndef __CSR_H
#define __CSR_H

#pragma once

#include <raft/core/bitmap.cuh>
#include <raft/core/device_csr_matrix.hpp>
#include <raft/sparse/convert/detail/adj_to_csr.cuh>
#include <raft/sparse/convert/detail/bitmap_to_csr.cuh>
#include <raft/sparse/convert/detail/csr.cuh>
#include <raft/sparse/csr.hpp>

namespace raft {
namespace sparse {
namespace convert {

template <typename value_t>
void coo_to_csr(raft::resources const& handle,
                const int* srcRows,
                const int* srcCols,
                const value_t* srcVals,
                int nnz,
                int m,
                int* dst_offsets,
                int* dstCols,
                value_t* dstVals)
{
  detail::coo_to_csr(handle, srcRows, srcCols, srcVals, nnz, m, dst_offsets, dstCols, dstVals);
}

/**
 * @brief Generate the row indices array for a sorted COO matrix
 *
 * @param rows: COO rows array
 * @param nnz: size of COO rows array
 * @param row_ind: output row indices array
 * @param m: number of rows in dense matrix
 * @param stream: cuda stream to use
 */
template <typename T>
void sorted_coo_to_csr(const T* rows, int nnz, T* row_ind, int m, cudaStream_t stream)
{
  detail::sorted_coo_to_csr(rows, nnz, row_ind, m, stream);
}

/**
 * @brief Generate the row indices array for a sorted COO matrix
 *
 * @param coo: Input COO matrix
 * @param row_ind: output row indices array
 * @param stream: cuda stream to use
 */
template <typename T>
void sorted_coo_to_csr(COO<T>* coo, int* row_ind, cudaStream_t stream)
{
  detail::sorted_coo_to_csr(coo->rows(), coo->nnz, row_ind, coo->n_rows, stream);
}

/**
 * @brief Converts a boolean adjacency matrix into unsorted CSR format.
 *
 * The conversion supports non-square matrices.
 *
 * @tparam     index_t     Indexing arithmetic type
 *
 * @param[in]  handle      RAFT handle
 * @param[in]  adj         A num_rows x num_cols boolean matrix in contiguous row-major
 *                         format.
 * @param[in]  row_ind     An array of length num_rows that indicates at which index
 *                         a row starts in out_col_ind. Equivalently, it is the
 *                         exclusive scan of the number of non-zeros in each row of
 *                         adj.
 * @param[in]  num_rows    Number of rows of adj.
 * @param[in]  num_cols    Number of columns of adj.
 * @param      tmp         A pre-allocated array of size num_rows.
 * @param[out] out_col_ind An array containing the column indices of the
 *                         non-zero values in adj. Size should be at least the
 *                         number of non-zeros in adj.
 */
template <typename index_t = int>
void adj_to_csr(raft::resources const& handle,
                const bool* adj,         // Row-major adjacency matrix
                const index_t* row_ind,  // Precomputed row indices
                index_t num_rows,        // # rows of adj
                index_t num_cols,        // # cols of adj
                index_t* tmp,  // Pre-allocated atomic counters. Minimum size: num_rows elements.
                index_t* out_col_ind  // Output column indices
)
{
  detail::adj_to_csr(handle, adj, row_ind, num_rows, num_cols, tmp, out_col_ind);
}

/**
 * @brief  Converts a bitmap matrix to a Compressed Sparse Row (CSR) format matrix.
 *
 * @tparam       bitmap_t       The data type of the elements in the bitmap matrix.
 * @tparam       index_t        The data type used for indexing the elements in the matrices.
 * @tparam       csr_matrix_t   Specifies the CSR matrix type, constrained to
 * raft::device_csr_matrix.
 *
 * @param[in]    handle         The RAFT handle containing the CUDA stream for operations.
 * @param[in]    bitmap         The bitmap matrix view, to be converted to CSR format.
 * @param[out]   csr            Output parameter where the resulting CSR matrix is stored. In the
 * bitmap, each '1' bit corresponds to a non-zero element in the CSR matrix.
 */
template <typename bitmap_t,
          typename index_t,
          typename csr_matrix_t,
          typename = std::enable_if_t<raft::is_device_csr_matrix_v<csr_matrix_t>>>
void bitmap_to_csr(raft::resources const& handle,
                   raft::core::bitmap_view<bitmap_t, index_t> bitmap,
                   csr_matrix_t& csr)
{
  detail::bitmap_to_csr(handle, bitmap, csr);
}

};  // end NAMESPACE convert
};  // end NAMESPACE sparse
};  // end NAMESPACE raft

#endif
