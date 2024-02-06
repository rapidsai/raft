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
#ifndef __SYMMETRIZE_H
#define __SYMMETRIZE_H

#pragma once

#include <raft/sparse/coo.hpp>
#include <raft/sparse/linalg/detail/symmetrize.cuh>

namespace raft {
namespace sparse {
namespace linalg {

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
template <typename T, typename Lambda>
void coo_symmetrize(COO<T>* in,
                    COO<T>* out,
                    Lambda reduction_op,  // two-argument reducer
                    cudaStream_t stream)
{
  detail::coo_symmetrize(in, out, reduction_op, stream);
}

/**
 * @brief Find how much space needed in each row.
 * We look through all datapoints and increment the count for each row.
 *
 * TODO: This isn't generalized. Remove in place of `symmetrize()`
 * @param data: Input knn distances(n, k)
 * @param indices: Input knn indices(n, k)
 * @param n: Number of rows
 * @param k: Number of n_neighbors
 * @param row_sizes: Input empty row sum 1 array(n)
 * @param row_sizes2: Input empty row sum 2 array(n) for faster reduction
 */
template <typename value_idx = int64_t, typename value_t = float>
RAFT_KERNEL symmetric_find_size(const value_t __restrict__* data,
                                const value_idx __restrict__* indices,
                                const value_idx n,
                                const int k,
                                value_idx __restrict__* row_sizes,
                                value_idx __restrict__* row_sizes2)
{
  detail::symmetric_find_size(data, indices, n, k, row_sizes, row_sizes2);
}

/**
 * @brief Reduce sum(row_sizes) + k
 * Reduction for symmetric_find_size kernel. Allows algo to be faster.
 *
 * TODO: This isn't generalized. Remove in place of `symmetrize()`
 * @param n: Number of rows
 * @param k: Number of n_neighbors
 * @param row_sizes: Input row sum 1 array(n)
 * @param row_sizes2: Input row sum 2 array(n) for faster reduction
 */
template <typename value_idx>
RAFT_KERNEL reduce_find_size(const value_idx n,
                             const int k,
                             value_idx __restrict__* row_sizes,
                             const value_idx __restrict__* row_sizes2)
{
  detail::reduce_find_size(n, k, row_sizes, row_sizes2);
}

/**
 * @brief Perform data + data.T operation.
 * Can only run once row_sizes from the CSR matrix of data + data.T has been
 * determined.
 *
 * TODO: This isn't generalized. Remove in place of `symmetrize()`
 *
 * @param edges: Input row sum array(n) after reduction
 * @param data: Input knn distances(n, k)
 * @param indices: Input knn indices(n, k)
 * @param VAL: Output values for data + data.T
 * @param COL: Output column indices for data + data.T
 * @param ROW: Output row indices for data + data.T
 * @param n: Number of rows
 * @param k: Number of n_neighbors
 */
template <typename value_idx = int64_t, typename value_t = float>
RAFT_KERNEL symmetric_sum(value_idx* __restrict__ edges,
                          const value_t* __restrict__ data,
                          const value_idx* __restrict__ indices,
                          value_t* __restrict__ VAL,
                          value_idx* __restrict__ COL,
                          value_idx* __restrict__ ROW,
                          const value_idx n,
                          const int k)
{
  detail::symmetric_sum(edges, data, indices, VAL, COL, ROW, n, k);
}

/**
 * @brief Perform data + data.T on raw KNN data.
 * The following steps are invoked:
 * (1) Find how much space needed in each row
 * (2) Compute final space needed (n*k + sum(row_sizes)) == 2*n*k
 * (3) Allocate new space
 * (4) Prepare edges for each new row
 * (5) Perform final data + data.T operation
 * (6) Return summed up VAL, COL, ROW
 *
 * TODO: This isn't generalized. Remove in place of `symmetrize()`
 *
 * @param knn_indices: Input knn distances(n, k)
 * @param knn_dists: Input knn indices(n, k)
 * @param n: Number of rows
 * @param k: Number of n_neighbors
 * @param out: Output COO Matrix class
 * @param stream: Input cuda stream
 */
template <typename value_idx = int64_t, typename value_t = float, int TPB_X = 32, int TPB_Y = 32>
void from_knn_symmetrize_matrix(const value_idx* __restrict__ knn_indices,
                                const value_t* __restrict__ knn_dists,
                                const value_idx n,
                                const int k,
                                COO<value_t, value_idx>* out,
                                cudaStream_t stream)
{
  detail::from_knn_symmetrize_matrix(knn_indices, knn_dists, n, k, out, stream);
}

/**
 * Symmetrizes a COO matrix
 */
template <typename value_idx, typename value_t>
void symmetrize(raft::resources const& handle,
                const value_idx* rows,
                const value_idx* cols,
                const value_t* vals,
                size_t m,
                size_t n,
                size_t nnz,
                raft::sparse::COO<value_t, value_idx>& out)
{
  detail::symmetrize(handle, rows, cols, vals, m, n, nnz, out);
}

};  // end NAMESPACE linalg
};  // end NAMESPACE sparse
};  // end NAMESPACE raft

#endif