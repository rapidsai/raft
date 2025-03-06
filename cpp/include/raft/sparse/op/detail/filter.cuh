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

template <int TPB_X, typename T, typename nnz_t>
RAFT_KERNEL coo_remove_scalar_kernel(const int* rows,
                                     const int* cols,
                                     const T* vals,
                                     nnz_t nnz,
                                     int* out_rows,
                                     int* out_cols,
                                     T* out_vals,
                                     nnz_t* ex_scan,
                                     nnz_t* cur_ex_scan,
                                     int m,
                                     T scalar)
{
  int row = (blockIdx.x * TPB_X) + threadIdx.x;

  if (row < m) {
    nnz_t start       = cur_ex_scan[row];
    nnz_t stop        = get_stop_idx(row, m, nnz, cur_ex_scan);
    nnz_t cur_out_idx = ex_scan[row];

    for (nnz_t idx = start; idx < stop; idx++) {
      if (vals[idx] != scalar) {
        out_rows[cur_out_idx] = rows[idx];
        out_cols[cur_out_idx] = cols[idx];
        out_vals[cur_out_idx] = vals[idx];
        ++cur_out_idx;
      }
    }
  }
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
template <int TPB_X, typename T, typename idx_t, typename nnz_t>
void coo_remove_scalar(const idx_t* rows,
                       const idx_t* cols,
                       const T* vals,
                       nnz_t nnz,
                       idx_t* crows,
                       idx_t* ccols,
                       T* cvals,
                       nnz_t* cnnz,
                       nnz_t* cur_cnnz,
                       T scalar,
                       idx_t n,
                       cudaStream_t stream)
{
  rmm::device_uvector<nnz_t> ex_scan(n, stream);
  rmm::device_uvector<nnz_t> cur_ex_scan(n, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(ex_scan.data(), 0, static_cast<nnz_t>(n) * sizeof(nnz_t), stream));
  RAFT_CUDA_TRY(
    cudaMemsetAsync(cur_ex_scan.data(), 0, static_cast<nnz_t>(n) * sizeof(nnz_t), stream));

  thrust::device_ptr<nnz_t> dev_cnnz    = thrust::device_pointer_cast(cnnz);
  thrust::device_ptr<nnz_t> dev_ex_scan = thrust::device_pointer_cast(ex_scan.data());
  thrust::exclusive_scan(rmm::exec_policy(stream), dev_cnnz, dev_cnnz + n, dev_ex_scan);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  thrust::device_ptr<nnz_t> dev_cur_cnnz    = thrust::device_pointer_cast(cur_cnnz);
  thrust::device_ptr<nnz_t> dev_cur_ex_scan = thrust::device_pointer_cast(cur_ex_scan.data());
  thrust::exclusive_scan(rmm::exec_policy(stream), dev_cur_cnnz, dev_cur_cnnz + n, dev_cur_ex_scan);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  dim3 grid(raft::ceildiv(n, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  coo_remove_scalar_kernel<TPB_X><<<grid, blk, 0, stream>>>(rows,
                                                            cols,
                                                            vals,
                                                            nnz,
                                                            crows,
                                                            ccols,
                                                            cvals,
                                                            dev_ex_scan.get(),
                                                            dev_cur_ex_scan.get(),
                                                            n,
                                                            scalar);
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
template <int TPB_X, typename T, typename idx_t, typename nnz_t>
void coo_remove_scalar(COO<T, idx_t, nnz_t>* in,
                       COO<T, idx_t, nnz_t>* out,
                       T scalar,
                       cudaStream_t stream)
{
  rmm::device_uvector<nnz_t> row_count_nz(in->n_rows, stream);
  rmm::device_uvector<nnz_t> row_count(in->n_rows, stream);

  RAFT_CUDA_TRY(cudaMemsetAsync(
    row_count_nz.data(), 0, static_cast<nnz_t>(in->n_rows) * sizeof(nnz_t), stream));
  RAFT_CUDA_TRY(
    cudaMemsetAsync(row_count.data(), 0, static_cast<nnz_t>(in->n_rows) * sizeof(nnz_t), stream));

  linalg::coo_degree(in->rows(), in->nnz, row_count.data(), stream);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  linalg::coo_degree_scalar(
    in->rows(), in->vals(), in->nnz, scalar, (unsigned long long int*)row_count_nz.data(), stream);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  thrust::device_ptr<nnz_t> d_row_count_nz = thrust::device_pointer_cast(row_count_nz.data());
  nnz_t out_nnz =
    thrust::reduce(rmm::exec_policy(stream), d_row_count_nz, d_row_count_nz + in->n_rows);

  out->allocate(out_nnz, in->n_rows, in->n_cols, false, stream);

  coo_remove_scalar<TPB_X, T, idx_t, nnz_t>(in->rows(),
                                            in->cols(),
                                            in->vals(),
                                            in->nnz,
                                            out->rows(),
                                            out->cols(),
                                            out->vals(),
                                            row_count_nz.data(),
                                            row_count.data(),
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
template <int TPB_X, typename T, typename idx_t, typename nnz_t>
void coo_remove_zeros(COO<T, idx_t, nnz_t>* in, COO<T, idx_t, nnz_t>* out, cudaStream_t stream)
{
  coo_remove_scalar<TPB_X, T, idx_t, nnz_t>(in, out, T(0.0), stream);
}

};  // namespace detail
};  // namespace op
};  // end NAMESPACE sparse
};  // end NAMESPACE raft
