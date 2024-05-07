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

#pragma once

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cusparse_handle.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/sparse/detail/utils.h>
#include <raft/sparse/linalg/degree.cuh>
#include <raft/sparse/op/row_op.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cusparse_v2.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

namespace raft {
namespace sparse {
namespace convert {
namespace detail {

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
  auto stream         = resource::get_cuda_stream(handle);
  auto cusparseHandle = resource::get_cusparse_handle(handle);
  rmm::device_uvector<int> dstRows(nnz, stream);
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(dstRows.data(), srcRows, sizeof(int) * nnz, cudaMemcpyDeviceToDevice, stream));
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(dstCols, srcCols, sizeof(int) * nnz, cudaMemcpyDeviceToDevice, stream));
  auto buffSize = raft::sparse::detail::cusparsecoosort_bufferSizeExt(
    cusparseHandle, m, m, nnz, srcRows, srcCols, stream);
  rmm::device_uvector<char> pBuffer(buffSize, stream);
  rmm::device_uvector<int> P(nnz, stream);
  RAFT_CUSPARSE_TRY(cusparseCreateIdentityPermutation(cusparseHandle, nnz, P.data()));
  raft::sparse::detail::cusparsecoosortByRow(
    cusparseHandle, m, m, nnz, dstRows.data(), dstCols, P.data(), pBuffer.data(), stream);
  raft::sparse::detail::cusparsegthr(cusparseHandle, nnz, srcVals, dstVals, P.data(), stream);
  raft::sparse::detail::cusparsecoo2csr(
    cusparseHandle, dstRows.data(), nnz, m, dst_offsets, stream);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
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
