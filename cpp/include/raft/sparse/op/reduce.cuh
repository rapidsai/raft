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
#include <raft/mr/device/allocator.hpp>
#include <raft/mr/device/buffer.hpp>

#include <raft/sparse/op/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <raft/device_atomics.cuh>

#include <cuda_runtime.h>
#include <stdio.h>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <algorithm>
#include <iostream>

#include <raft/sparse/utils.h>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/coo.cuh>

namespace raft {
namespace sparse {
namespace op {

template <typename value_idx>
__global__ void compute_duplicates_diffs_kernel(const value_idx *rows,
                                                const value_idx *cols,
                                                value_idx *diff, size_t nnz) {
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= nnz) return;

  value_idx d = 1;
  if (tid == 0 || (rows[tid - 1] == rows[tid] && cols[tid - 1] == cols[tid]))
    d = 0;
  diff[tid] = d;
}

template <typename value_idx, typename value_t>
__global__ void max_duplicates_kernel(const value_idx *src_rows,
                                      const value_idx *src_cols,
                                      const value_t *src_vals,
                                      const value_idx *index,
                                      value_idx *out_rows, value_idx *out_cols,
                                      value_t *out_vals, size_t nnz) {
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < nnz) {
    value_idx idx = index[tid];
    atomicMax(&out_vals[idx], src_vals[tid]);
    out_rows[idx] = src_rows[tid];
    out_cols[idx] = src_cols[tid];
  }
}

/**
 * Computes a mask from a sorted COO matrix where 0's denote
 * duplicate values and 1's denote new values. This mask can
 * be useful for computing an exclusive scan to pre-build offsets
 * for reducing duplicates, such as when symmetrizing
 * or taking the min of each duplicated value.
 *
 * Note that this function always marks the first value as 0 so that
 * a cumulative sum can be performed as a follow-on. However, even
 * if the mask is used direclty, any duplicates should always have a
 * 1 when first encountered so it can be assumed that the first element
 * is always a 1 otherwise.
 *
 * @tparam value_idx
 * @param[out] mask output mask, size nnz
 * @param[in] rows COO rows array, size nnz
 * @param[in] cols COO cols array, size nnz
 * @param[in] nnz number of nonzeros in input arrays
 * @param[in] stream cuda ops will be ordered wrt this stream
 */
template <typename value_idx>
void compute_duplicates_mask(value_idx *mask, const value_idx *rows,
                             const value_idx *cols, size_t nnz,
                             cudaStream_t stream) {
  CUDA_CHECK(cudaMemsetAsync(mask, 0, nnz * sizeof(value_idx), stream));

  compute_duplicates_diffs_kernel<<<raft::ceildiv(nnz, (size_t)256), 256, 0,
                                    stream>>>(rows, cols, mask, nnz);
}

/**
 * Performs a COO reduce of duplicate columns per row, taking the max weight
 * for duplicate columns in each row. This function assumes the input COO
 * has been sorted by both row and column but makes no assumption on
 * the sorting of values.
 * @tparam value_idx
 * @tparam value_t
 * @param[out] out output COO, the nnz will be computed allocate() will be called in this function.
 * @param[in] rows COO rows array, size nnz
 * @param[in] cols COO cols array, size nnz
 * @param[in] vals COO vals array, size nnz
 * @param[in] nnz number of nonzeros in COO input arrays
 * @param[in] m number of rows in COO input matrix
 * @param[in] n number of columns in COO input matrix
 * @param[in] stream cuda ops will be ordered wrt this stream
 */
template <typename value_idx, typename value_t>
void max_duplicates(const raft::handle_t &handle,
                    raft::sparse::COO<value_t, value_idx> &out,
                    const value_idx *rows, const value_idx *cols,
                    const value_t *vals, size_t nnz, size_t m, size_t n) {
  auto d_alloc = handle.get_device_allocator();
  auto stream = handle.get_stream();

  auto exec_policy = rmm::exec_policy(rmm::cuda_stream_view{stream});

  // compute diffs & take exclusive scan
  rmm::device_uvector<value_idx> diff(nnz + 1, stream);

  compute_duplicates_mask(diff.data(), rows, cols, nnz, stream);

  thrust::exclusive_scan(exec_policy, diff.data(), diff.data() + diff.size(),
                         diff.data());

  // compute final size
  value_idx size = 0;
  raft::update_host(&size, diff.data() + (diff.size() - 1), 1, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  size++;

  out.allocate(size, m, n, true, stream);

  // perform reduce
  max_duplicates_kernel<<<raft::ceildiv(nnz, (size_t)256), 256, 0, stream>>>(
    rows, cols, vals, diff.data() + 1, out.rows(), out.cols(), out.vals(), nnz);
}
};  // END namespace op
};  // END namespace sparse
};  // END namespace raft
