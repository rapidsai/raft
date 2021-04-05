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

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cuda_runtime.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

#include <raft/sparse/utils.h>
#include <raft/sparse/coo.cuh>

namespace raft {
namespace sparse {
namespace op {

struct TupleComp {
  template <typename one, typename two>
  __host__ __device__ bool operator()(const one &t1, const two &t2) {
    // sort first by each sample's color,
    if (thrust::get<0>(t1) < thrust::get<0>(t2)) return true;
    if (thrust::get<0>(t1) > thrust::get<0>(t2)) return false;

    // then sort by value in descending order
    return thrust::get<1>(t1) < thrust::get<1>(t2);
  }
};

/**
 * @brief Sorts the arrays that comprise the coo matrix
 * by row and then by column.
 *
 * @param m number of rows in coo matrix
 * @param n number of cols in coo matrix
 * @param nnz number of non-zeros
 * @param rows rows array from coo matrix
 * @param cols cols array from coo matrix
 * @param vals vals array from coo matrix
 * @param d_alloc device allocator for temporary buffers
 * @param stream: cuda stream to use
 */
template <typename T>
void coo_sort(int m, int n, int nnz, int *rows, int *cols, T *vals,
              // TODO: Remove this
              std::shared_ptr<raft::mr::device::allocator> d_alloc,
              cudaStream_t stream) {
  auto coo_indices = thrust::make_zip_iterator(thrust::make_tuple(rows, cols));

  // get all the colors in contiguous locations so we can map them to warps.
  thrust::sort_by_key(thrust::cuda::par.on(stream), coo_indices,
                      coo_indices + nnz, vals, TupleComp());
}

/**
 * @brief Sort the underlying COO arrays by row
 * @tparam T: the type name of the underlying value array
 * @param in: COO to sort by row
 * @param d_alloc device allocator for temporary buffers
 * @param stream: the cuda stream to use
 */
template <typename T>
void coo_sort(COO<T> *const in,
              // TODO: Remove this
              std::shared_ptr<raft::mr::device::allocator> d_alloc,
              cudaStream_t stream) {
  coo_sort<T>(in->n_rows, in->n_cols, in->nnz, in->rows(), in->cols(),
              in->vals(), d_alloc, stream);
}
};  // namespace op
};  // end NAMESPACE sparse
};  // end NAMESPACE raft