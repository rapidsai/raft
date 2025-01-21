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

#include <raft/sparse/coo.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/sparse/detail/utils.h>
#include <raft/sparse/linalg/degree.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include <cusparse_v2.h>

#include <algorithm>
#include <cstdio>
#include <iostream>

namespace raft {
namespace sparse {
namespace op {
namespace detail {

template <uint64_t TPB_X, typename T>
RAFT_KERNEL coo_remove_scalar_kernel(const int* in_rows,
                                     const int* in_cols,
                                     const T* in_vals,
                                     uint64_t nnz,
                                     int* out_rows,
                                     int* out_cols,
                                     T* out_vals,
                                     uint64_t* row_indices,
				                             int* rows_lenght_acc,
                                     T scalar,
				                             int n_rows)
{
  uint64_t in_idx = (blockIdx.x * TPB_X) + threadIdx.x;

  if (in_idx >= nnz)
    return;

  int val = in_vals[in_idx];

  if (val == scalar)
    return;

  int row = in_rows[in_idx];

  uint64_t row_start_index = row_indices[row];
  uint64_t out_idx = row_start_index + atomicAdd(rows_lenght_acc + row, 1);

  out_rows[out_idx] = row;
  out_cols[out_idx] = in_cols[in_idx];
  out_vals[out_idx] = val;
}

/**
 * @brief Removes the values matching a particular scalar from a COO formatted sparse matrix.
 *
 * @param rows: input array of rows (size n)
 * @param cols: input array of cols (size n)
 * @param vals: input array of vals (size n)
 * @param nnz: size of current rows/cols/vals arrays
 * @param crows: compressed array of rows
 * @param ccols: compressed array of cols
 * @param cvals: compressed array of vals
 * @param cnnz: array of non-zero counts per row
 * @param cur_cnnz array of counts per row
 * @param scalar: scalar to remove from arrays
 * @param n: number of rows in dense matrix
 * @param d_alloc device allocator for temporary buffers
 * @param stream: cuda stream to use
 */
template <uint64_t TPB_X, typename T>
void coo_remove_scalar(const int* rows,
                       const int* cols,
                       const T* vals,
                       uint64_t nnz,
                       int* crows,
                       int* ccols,
                       T* cvals,
                       uint64_t* cnnz,
                       T scalar,
                       int n,
                       cudaStream_t stream)
{
  rmm::device_uvector<uint64_t> ex_scan(n, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(ex_scan.data(), 0, (uint64_t)n * sizeof(uint64_t), stream));

  thrust::device_ptr<uint64_t> dev_cnnz    = thrust::device_pointer_cast(cnnz);
  thrust::device_ptr<uint64_t> dev_ex_scan = thrust::device_pointer_cast(ex_scan.data());
  thrust::exclusive_scan(rmm::exec_policy(stream), dev_cnnz, dev_cnnz + n, dev_ex_scan);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  rmm::device_uvector<int> rows_length_acc(n, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(rows_length_acc.data(), 0, (uint64_t)n * sizeof(int), stream));

  dim3 grid(raft::ceildiv(nnz, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  coo_remove_scalar_kernel<TPB_X><<<grid, blk, 0, stream>>>(rows,
                                                            cols,
                                                            vals,
                                                            nnz,
                                                            crows,
                                                            ccols,
                                                            cvals,
                                                            dev_ex_scan.get(),
							                                              rows_length_acc.data(),
                                                            scalar,
							                                              n);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * @brief Removes the values matching a particular scalar from a COO formatted sparse matrix.
 *
 * @param in: input COO matrix
 * @param out: output COO matrix
 * @param scalar: scalar to remove from arrays
 * @param stream: cuda stream to use
 */
template <int TPB_X, typename T>
void coo_remove_scalar(COO<T>* in, COO<T>* out, T scalar, cudaStream_t stream)
{
  rmm::device_uvector<uint64_t> row_count_nz(in->n_rows, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(row_count_nz.data(), 0, (uint64_t)in->n_rows * sizeof(uint64_t), stream));

  linalg::coo_degree_scalar(in->rows(), in->vals(), in->nnz, scalar, (unsigned long long int*)row_count_nz.data(), stream);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  thrust::device_ptr<uint64_t> d_row_count_nz = thrust::device_pointer_cast(row_count_nz.data());
  uint64_t out_nnz = thrust::reduce(rmm::exec_policy(stream), d_row_count_nz, d_row_count_nz + in->n_rows, (uint64_t)0);

  out->allocate(out_nnz, in->n_rows, in->n_cols, false, stream);

  coo_remove_scalar<TPB_X, T>(in->rows(),
                              in->cols(),
                              in->vals(),
                              in->nnz,
                              out->rows(),
                              out->cols(),
                              out->vals(),
                              row_count_nz.data(),
                              scalar,
                              in->n_rows,
                              stream);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * @brief Removes zeros from a COO formatted sparse matrix.
 *
 * @param in: input COO matrix
 * @param out: output COO matrix
 * @param stream: cuda stream to use
 */
template <int TPB_X, typename T>
void coo_remove_zeros(COO<T>* in, COO<T>* out, cudaStream_t stream)
{
  coo_remove_scalar<TPB_X, T>(in, out, T(0.0), stream);
}

};  // namespace detail
};  // namespace op
};  // end NAMESPACE sparse
};  // end NAMESPACE raft
