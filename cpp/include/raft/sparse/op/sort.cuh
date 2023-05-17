/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#ifndef __SPARSE_SORT_H
#define __SPARSE_SORT_H

#pragma once

#include <raft/core/resources.hpp>
#include <raft/sparse/op/detail/sort.h>

namespace raft {
namespace sparse {
namespace op {

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
  detail::coo_sort(m, n, nnz, rows, cols, vals, stream);
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
  detail::coo_sort_by_weight(rows, cols, data, nnz, stream);
}
};  // namespace op
};  // end NAMESPACE sparse
};  // end NAMESPACE raft

#endif