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

#include <cusparse_v2.h>

#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <raft/sparse/cusparse_wrappers.h>
#include <rmm/device_uvector.hpp>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cuda_runtime.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

#include <raft/sparse/coo.hpp>
#include <raft/sparse/detail/utils.h>
#include <raft/sparse/linalg/degree.hpp>
#include <raft/sparse/op/row_op.hpp>

namespace raft {
namespace sparse {
namespace convert {
namespace detail {

template <typename value_t>
void coo_to_csr(const raft::handle_t& handle,
                const int* srcRows,
                const int* srcCols,
                const value_t* srcVals,
                int nnz,
                int m,
                int* dst_offsets,
                int* dstCols,
                value_t* dstVals)
{
  auto stream         = handle.get_stream();
  auto cusparseHandle = handle.get_cusparse_handle();
  rmm::device_uvector<int> dstRows(nnz, stream);
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(dstRows.data(), srcRows, sizeof(int) * nnz, cudaMemcpyDeviceToDevice, stream));
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(dstCols, srcCols, sizeof(int) * nnz, cudaMemcpyDeviceToDevice, stream));
  auto buffSize = raft::sparse::cusparsecoosort_bufferSizeExt(
    cusparseHandle, m, m, nnz, srcRows, srcCols, stream);
  rmm::device_uvector<char> pBuffer(buffSize, stream);
  rmm::device_uvector<int> P(nnz, stream);
  RAFT_CUSPARSE_TRY(cusparseCreateIdentityPermutation(cusparseHandle, nnz, P.data()));
  raft::sparse::cusparsecoosortByRow(
    cusparseHandle, m, m, nnz, dstRows.data(), dstCols, P.data(), pBuffer.data(), stream);
  raft::sparse::cusparsegthr(cusparseHandle, nnz, srcVals, dstVals, P.data(), stream);
  raft::sparse::cusparsecoo2csr(cusparseHandle, dstRows.data(), nnz, m, dst_offsets, stream);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
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
template <typename Index_, int TPB_X = 32, typename Lambda = auto(Index_, Index_, Index_)->void>
void csr_adj_graph_batched(const Index_* row_ind,
                           Index_ total_rows,
                           Index_ nnz,
                           Index_ batchSize,
                           const bool* adj,
                           Index_* row_ind_ptr,
                           cudaStream_t stream,
                           Lambda fused_op)
{
  op::csr_row_op<Index_>(
    row_ind,
    batchSize,
    nnz,
    [fused_op, adj, total_rows, row_ind_ptr, batchSize, nnz] __device__(
      Index_ row, Index_ start_idx, Index_ stop_idx) {
      fused_op(row, start_idx, stop_idx);
      Index_ k = 0;
      for (Index_ i = 0; i < total_rows; i++) {
        // @todo: uncoalesced mem accesses!
        if (adj[batchSize * i + row]) {
          row_ind_ptr[start_idx + k] = i;
          k += 1;
        }
      }
    },
    stream);
}

template <typename Index_, int TPB_X = 32, typename Lambda = auto(Index_, Index_, Index_)->void>
void csr_adj_graph_batched(const Index_* row_ind,
                           Index_ total_rows,
                           Index_ nnz,
                           Index_ batchSize,
                           const bool* adj,
                           Index_* row_ind_ptr,
                           cudaStream_t stream)
{
  csr_adj_graph_batched(row_ind,
                        total_rows,
                        nnz,
                        batchSize,
                        adj,
                        row_ind_ptr,
                        stream,
                        [] __device__(Index_ row, Index_ start_idx, Index_ stop_idx) {});
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
template <typename Index_, int TPB_X = 32, typename Lambda = auto(Index_, Index_, Index_)->void>
void csr_adj_graph(const Index_* row_ind,
                   Index_ total_rows,
                   Index_ nnz,
                   const bool* adj,
                   Index_* row_ind_ptr,
                   cudaStream_t stream,
                   Lambda fused_op)
{
  csr_adj_graph_batched<Index_, TPB_X>(
    row_ind, total_rows, nnz, total_rows, adj, row_ind_ptr, stream, fused_op);
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
  rmm::device_uvector<T> row_counts(m, stream);

  RAFT_CUDA_TRY(cudaMemsetAsync(row_counts.data(), 0, m * sizeof(T), stream));

  linalg::coo_degree(rows, nnz, row_counts.data(), stream);

  // create csr compressed row index from row counts
  thrust::device_ptr<T> row_counts_d = thrust::device_pointer_cast(row_counts.data());
  thrust::device_ptr<T> c_ind_d      = thrust::device_pointer_cast(row_ind);
  exclusive_scan(rmm::exec_policy(stream), row_counts_d, row_counts_d + m, c_ind_d);
}

};  // end NAMESPACE detail
};  // end NAMESPACE convert
};  // end NAMESPACE sparse
};  // end NAMESPACE raft
