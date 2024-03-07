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
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <cusparse_v2.h>

#include <algorithm>

namespace raft {
namespace sparse {
namespace op {
namespace detail {

struct TupleComp {
  template <typename one, typename two>
  __host__ __device__

    bool
    operator()(const one& t1, const two& t2)
  {
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
 * @param stream: cuda stream to use
 */
template <typename T>
void coo_sort(int m, int n, int nnz, int* rows, int* cols, T* vals, cudaStream_t stream)
{
  auto coo_indices = thrust::make_zip_iterator(thrust::make_tuple(rows, cols));

  // get all the colors in contiguous locations so we can map them to warps.
  thrust::sort_by_key(rmm::exec_policy(stream), coo_indices, coo_indices + nnz, vals, TupleComp());
}

/**
 * @brief Sort the underlying COO arrays by row
 * @tparam T: the type name of the underlying value array
 * @param in: COO to sort by row
 * @param stream: the cuda stream to use
 */
template <typename T>
void coo_sort(COO<T>* const in, cudaStream_t stream)
{
  coo_sort<T>(in->n_rows, in->n_cols, in->nnz, in->rows(), in->cols(), in->vals(), stream);
}

/**
 * Sorts a COO by its weight
 * @tparam value_idx
 * @tparam value_t
 * @param[inout] rows source edges
 * @param[inout] cols dest edges
 * @param[inout] data edge weights
 * @param[in] nnz number of edges in edge list
 * @param[in] stream cuda stream for which to order cuda operations
 */
template <typename value_idx, typename value_t>
void coo_sort_by_weight(
  value_idx* rows, value_idx* cols, value_t* data, value_idx nnz, cudaStream_t stream)
{
  thrust::device_ptr<value_t> t_data = thrust::device_pointer_cast(data);

  auto first = thrust::make_zip_iterator(thrust::make_tuple(rows, cols));

  thrust::sort_by_key(rmm::exec_policy(stream), t_data, t_data + nnz, first);
}
};  // namespace detail
};  // namespace op
};  // end NAMESPACE sparse
};  // end NAMESPACE raft
