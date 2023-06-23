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
#ifndef __SPARSE_REDUCE_H
#define __SPARSE_REDUCE_H

#pragma once

#include <raft/core/resources.hpp>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/op/detail/reduce.cuh>

namespace raft {
namespace sparse {
namespace op {
/**
 * Computes a mask from a sorted COO matrix where 0's denote
 * duplicate values and 1's denote new values. This mask can
 * be useful for computing an exclusive scan to pre-build offsets
 * for reducing duplicates, such as when symmetrizing
 * or taking the min of each duplicated value.
 *
 * Note that this function always marks the first value as 0 so that
 * a cumulative sum can be performed as a follow-on. However, even
 * if the mask is used directly, any duplicates should always have a
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
void compute_duplicates_mask(
  value_idx* mask, const value_idx* rows, const value_idx* cols, size_t nnz, cudaStream_t stream)
{
  detail::compute_duplicates_mask(mask, rows, cols, nnz, stream);
}

/**
 * Performs a COO reduce of duplicate columns per row, taking the max weight
 * for duplicate columns in each row. This function assumes the input COO
 * has been sorted by both row and column but makes no assumption on
 * the sorting of values.
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle
 * @param[out] out output COO, the nnz will be computed allocate() will be called in this function.
 * @param[in] rows COO rows array, size nnz
 * @param[in] cols COO cols array, size nnz
 * @param[in] vals COO vals array, size nnz
 * @param[in] nnz number of nonzeros in COO input arrays
 * @param[in] m number of rows in COO input matrix
 * @param[in] n number of columns in COO input matrix
 */
template <typename value_idx, typename value_t>
void max_duplicates(raft::resources const& handle,
                    raft::sparse::COO<value_t, value_idx>& out,
                    const value_idx* rows,
                    const value_idx* cols,
                    const value_t* vals,
                    size_t nnz,
                    size_t m,
                    size_t n)
{
  detail::max_duplicates(handle, out, rows, cols, vals, nnz, m, n);
}
};  // END namespace op
};  // END namespace sparse
};  // END namespace raft

#endif