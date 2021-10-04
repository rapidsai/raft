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

#include <raft/cudart_utils.h>
#include <raft/sparse/cusparse_wrappers.h>
#include <raft/cuda_utils.cuh>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <raft/sparse/op/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <raft/device_atomics.cuh>

#include <cuda_runtime.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

#include <raft/sparse/utils.h>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/coo.cuh>
#include <raft/sparse/op/reduce.cuh>

namespace raft {
namespace sparse {
namespace linalg {

// TODO: value_idx param needs to be used for this once FAISS is updated to use float32
// for indices so that the index types can be uniform
template <int TPB_X = 128, typename T, typename Lambda>
__global__ void coo_symmetrize_kernel(int *row_ind, int *rows, int *cols,
                                      T *vals, int *orows, int *ocols, T *ovals,
                                      int n, int cnnz, Lambda reduction_op) {
  int row = (blockIdx.x * TPB_X) + threadIdx.x;

  if (row < n) {
    int start_idx = row_ind[row];  // each thread processes one row
    int stop_idx = get_stop_idx(row, n, cnnz, row_ind);

    int row_nnz = 0;
    int out_start_idx = start_idx * 2;

    for (int idx = 0; idx < stop_idx - start_idx; idx++) {
      int cur_row = rows[idx + start_idx];
      int cur_col = cols[idx + start_idx];
      T cur_val = vals[idx + start_idx];

      int lookup_row = cur_col;
      int t_start = row_ind[lookup_row];  // Start at
      int t_stop = get_stop_idx(lookup_row, n, cnnz, row_ind);

      T transpose = 0.0;

      bool found_match = false;
      for (int t_idx = t_start; t_idx < t_stop; t_idx++) {
        // If we find a match, let's get out of the loop. We won't
        // need to modify the transposed value, since that will be
        // done in a different thread.
        if (cols[t_idx] == cur_row && rows[t_idx] == cur_col) {
          // If it exists already, set transposed value to existing value
          transpose = vals[t_idx];
          found_match = true;
          break;
        }
      }

      // Custom reduction op on value and its transpose, which enables
      // specialized weighting.
      // If only simple X+X.T is desired, this op can just sum
      // the two values.
      T res = reduction_op(cur_row, cur_col, cur_val, transpose);

      // if we didn't find an exact match, we need to add
      // the computed res into our current matrix to guarantee
      // symmetry.
      // Note that if we did find a match, we don't need to
      // compute `res` on it here because it will be computed
      // in a different thread.
      if (!found_match && vals[idx] != 0.0) {
        orows[out_start_idx + row_nnz] = cur_col;
        ocols[out_start_idx + row_nnz] = cur_row;
        ovals[out_start_idx + row_nnz] = res;
        ++row_nnz;
      }

      if (res != 0.0) {
        orows[out_start_idx + row_nnz] = cur_row;
        ocols[out_start_idx + row_nnz] = cur_col;
        ovals[out_start_idx + row_nnz] = res;
        ++row_nnz;
      }
    }
  }
}

/**
 * @brief takes a COO matrix which may not be symmetric and symmetrizes
 * it, running a custom reduction function against the each value
 * and its transposed value.
 *
 * @param in: Input COO matrix
 * @param out: Output symmetrized COO matrix
 * @param reduction_op: a custom reduction function
 * @param stream: cuda stream to use
 */
template <int TPB_X = 128, typename T, typename Lambda>
void coo_symmetrize(COO<T> *in, COO<T> *out,
                    Lambda reduction_op,  // two-argument reducer
                    cudaStream_t stream) {
  dim3 grid(raft::ceildiv(in->n_rows, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  ASSERT(!out->validate_mem(), "Expecting unallocated COO for output");

  rmm::device_uvector<int> in_row_ind(in->n_rows, stream);

  convert::sorted_coo_to_csr(in, in_row_ind.data(), stream);

  out->allocate(in->nnz * 2, in->n_rows, in->n_cols, true, stream);

  coo_symmetrize_kernel<TPB_X, T><<<grid, blk, 0, stream>>>(
    in_row_ind.data(), in->rows(), in->cols(), in->vals(), out->rows(),
    out->cols(), out->vals(), in->n_rows, in->nnz, reduction_op);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename value_idx, typename value_t>
__global__ static void build_coo_k(value_idx *restrict ROW,
                                   value_idx *restrict COL,
                                   value_t *restrict VAL,
                                   const value_idx *restrict knn_indices,
                                   const value_t *restrict knn_dists,
                                   const value_idx total_nn, const int k) {
  const auto i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= total_nn) return;

  const value_idx sample_idx = i / k;

  if (i % k != 0) {
    const value_idx out_idx = i - sample_idx - 1;
    ROW[out_idx] = sample_idx;
    COL[out_idx] = knn_indices[i];
    VAL[out_idx] = knn_dists[i];
  }
}

/**
 * @brief Perform data + data.T on raw KNN data.
 * @param knn_indices: Input knn distances(n, k)
 * @param knn_dists: Input knn indices(n, k)
 * @param n: Number of rows
 * @param k: Number of n_neighbors
 * @param out: Output COO Matrix class
 * @param stream: Input cuda stream
 */
template <typename value_idx = int64_t, typename value_t = float,
          int TPB_X = 32, int TPB_Y = 32>
void from_knn_symmetrize_matrix(const value_idx *restrict knn_indices,
                                const value_t *restrict knn_dists,
                                const value_idx n, const int k,
                                COO<value_t, value_idx> *out,
                                cudaStream_t stream) {
  const value_idx total_nn_before = n * k;
  const value_idx total_nn_after = n * (k - 1);
  const value_idx NNZ = 2 * total_nn_after;
  out->allocate(NNZ, n, n, true, stream);

  build_coo_k<<<raft::ceildiv(total_nn_before, (value_idx)1024), 1024, 0,
                stream>>>(out->rows(), out->cols(), out->vals(), knn_indices,
                          knn_dists, total_nn_before, k);

  raft::copy(out->rows() + total_nn_after, out->cols(), total_nn_after, stream);
  raft::copy(out->cols() + total_nn_after, out->rows(), total_nn_after, stream);
  raft::copy(out->vals() + total_nn_after, out->vals(), total_nn_after, stream);
}

/**
 * Symmetrizes a COO matrix
 */
template <typename value_idx, typename value_t>
void symmetrize(const raft::handle_t &handle, const value_idx *rows,
                const value_idx *cols, const value_t *vals, size_t m, size_t n,
                size_t nnz, raft::sparse::COO<value_t, value_idx> &out) {
  auto stream = handle.get_stream();

  // copy rows to cols and cols to rows
  rmm::device_uvector<value_idx> symm_rows(nnz * 2, stream);
  rmm::device_uvector<value_idx> symm_cols(nnz * 2, stream);
  rmm::device_uvector<value_t> symm_vals(nnz * 2, stream);

  raft::copy_async(symm_rows.data(), rows, nnz, stream);
  raft::copy_async(symm_rows.data() + nnz, cols, nnz, stream);
  raft::copy_async(symm_cols.data(), cols, nnz, stream);
  raft::copy_async(symm_cols.data() + nnz, rows, nnz, stream);

  raft::copy_async(symm_vals.data(), vals, nnz, stream);
  raft::copy_async(symm_vals.data() + nnz, vals, nnz, stream);

  // sort COO
  raft::sparse::op::coo_sort((value_idx)m, (value_idx)n, (value_idx)nnz * 2,
                             symm_rows.data(), symm_cols.data(),
                             symm_vals.data(), stream);

  raft::sparse::op::max_duplicates(handle, out, symm_rows.data(),
                                   symm_cols.data(), symm_vals.data(), nnz * 2,
                                   m, n);
}

};  // end NAMESPACE linalg
};  // end NAMESPACE sparse
};  // end NAMESPACE raft
