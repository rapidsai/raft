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
#ifndef __FILTER_H
#define __FILTER_H

#pragma once

#include <raft/core/resources.hpp>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/op/detail/filter.cuh>

namespace raft {
namespace sparse {
namespace op {

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
 * @param stream: cuda stream to use
 */
template <typename T>
void coo_remove_scalar(const int* rows,
                       const int* cols,
                       const T* vals,
                       int nnz,
                       int* crows,
                       int* ccols,
                       T* cvals,
                       int* cnnz,
                       int* cur_cnnz,
                       T scalar,
                       int n,
                       cudaStream_t stream)
{
  detail::coo_remove_scalar<128, T>(
    rows, cols, vals, nnz, crows, ccols, cvals, cnnz, cur_cnnz, scalar, n, stream);
}

/**
 * @brief Removes the values matching a particular scalar from a COO formatted sparse matrix.
 *
 * @param in: input COO matrix
 * @param out: output COO matrix
 * @param scalar: scalar to remove from arrays
 * @param stream: cuda stream to use
 */
template <typename T>
void coo_remove_scalar(COO<T>* in, COO<T>* out, T scalar, cudaStream_t stream)
{
  detail::coo_remove_scalar<128, T>(in, out, scalar, stream);
}

/**
 * @brief Removes zeros from a COO formatted sparse matrix.
 *
 * @param in: input COO matrix
 * @param out: output COO matrix
 * @param stream: cuda stream to use
 */
template <typename T>
void coo_remove_zeros(COO<T>* in, COO<T>* out, cudaStream_t stream)
{
  coo_remove_scalar<T>(in, out, T(0.0), stream);
}

};  // namespace op
};  // end NAMESPACE sparse
};  // end NAMESPACE raft

#endif