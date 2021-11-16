/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <raft/sparse/convert/detail/csr.cuh>
#include <raft/sparse/csr.cuh>

namespace raft {
namespace sparse {
namespace convert {

template <typename value_t>
void coo_to_csr(const raft::handle_t &handle, const int *srcRows,
                const int *srcCols, const value_t *srcVals, int nnz, int m,
                int *dst_offsets, int *dstCols, value_t *dstVals) {
  detail::coo_to_csr(handle, srcRows, srcCols, srcVals, nnz, m, dst_offsets,
                     dstCols, dstVals);
}

/**
 * @brief Constructs an adjacency graph CSR row_ind_ptr array from
 * a row_ind array and adjacency array.
 * @tparam T the numeric type of the index arrays
 * @tparam TPB_X the number of threads to use per block for kernels
 * @tparam Lambda function for fused operation in the adj_graph construction
 * @param row_ind the input CSR row_ind array
 * @param total_rows number of vertices in graph
 * @param nnz number of non-zeros
 * @param batchSize number of vertices in current batch
 * @param adj an adjacency array (size batchSize x total_rows)
 * @param row_ind_ptr output CSR row_ind_ptr for adjacency graph
 * @param stream cuda stream to use
 * @param fused_op: the fused operation
 */
template <typename Index_, typename Lambda = auto(Index_, Index_, Index_)->void>
void csr_adj_graph_batched(const Index_ *row_ind, Index_ total_rows, Index_ nnz,
                           Index_ batchSize, const bool *adj,
                           Index_ *row_ind_ptr, cudaStream_t stream,
                           Lambda fused_op) {
  detail::csr_adj_graph_batched<Index_, 32, Lambda>(
    row_ind, total_rows, nnz, batchSize, adj, row_ind_ptr, stream, fused_op);
}

template <typename Index_, typename Lambda = auto(Index_, Index_, Index_)->void>
void csr_adj_graph_batched(const Index_ *row_ind, Index_ total_rows, Index_ nnz,
                           Index_ batchSize, const bool *adj,
                           Index_ *row_ind_ptr, cudaStream_t stream) {
  detail::csr_adj_graph_batched<Index_, 32, Lambda>(
    row_ind, total_rows, nnz, batchSize, adj, row_ind_ptr, stream);
}

/**
 * @brief Constructs an adjacency graph CSR row_ind_ptr array from a
 * a row_ind array and adjacency array.
 * @tparam T the numeric type of the index arrays
 * @tparam TPB_X the number of threads to use per block for kernels
 * @param row_ind the input CSR row_ind array
 * @param total_rows number of total vertices in graph
 * @param nnz number of non-zeros
 * @param adj an adjacency array
 * @param row_ind_ptr output CSR row_ind_ptr for adjacency graph
 * @param stream cuda stream to use
 * @param fused_op the fused operation
 */
template <typename Index_, typename Lambda = auto(Index_, Index_, Index_)->void>
void csr_adj_graph(const Index_ *row_ind, Index_ total_rows, Index_ nnz,
                   const bool *adj, Index_ *row_ind_ptr, cudaStream_t stream,
                   Lambda fused_op) {
  detail::csr_adj_graph<Index_, 32, Lambda>(row_ind, total_rows, nnz, adj,
                                            row_ind_ptr, stream, fused_op);
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
void sorted_coo_to_csr(const T *rows, int nnz, T *row_ind, int m,
                       cudaStream_t stream) {
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
void sorted_coo_to_csr(COO<T> *coo, int *row_ind, cudaStream_t stream) {
  detail::sorted_coo_to_csr(coo->rows(), coo->nnz, row_ind, coo->n_rows,
                            stream);
}

};  // end NAMESPACE convert
};  // end NAMESPACE sparse
};  // end NAMESPACE raft