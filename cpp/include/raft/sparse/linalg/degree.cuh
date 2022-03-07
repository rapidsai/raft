/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#ifndef __SPARSE_DEGREE_H
#define __SPARSE_DEGREE_H

#pragma once

#include <raft/sparse/coo.hpp>
#include <raft/sparse/linalg/detail/degree.cuh>

namespace raft {
namespace sparse {
namespace linalg {

/**
 * @brief Count the number of values for each row
 * @tparam TPB_X: number of threads to use per block
 * @param rows: rows array of the COO matrix
 * @param nnz: size of the rows array
 * @param results: output result array
 * @param stream: cuda stream to use
 */
template <typename T = int>
void coo_degree(const T* rows, int nnz, T* results, cudaStream_t stream)
{
  detail::coo_degree<64, T>(rows, nnz, results, stream);
}

/**
 * @brief Count the number of values for each row
 * @tparam TPB_X: number of threads to use per block
 * @tparam T: type name of underlying values array
 * @param in: input COO object for counting rows
 * @param results: output array with row counts (size=in->n_rows)
 * @param stream: cuda stream to use
 */
template <typename T>
void coo_degree(COO<T>* in, int* results, cudaStream_t stream)
{
  coo_degree(in->rows(), in->nnz, results, stream);
}

/**
 * @brief Count the number of values for each row that doesn't match a particular scalar
 * @tparam TPB_X: number of threads to use per block
 * @tparam T: the type name of the underlying value arrays
 * @param rows: Input COO row array
 * @param vals: Input COO val arrays
 * @param nnz: size of input COO arrays
 * @param scalar: scalar to match for counting rows
 * @param results: output row counts
 * @param stream: cuda stream to use
 */
template <typename T>
void coo_degree_scalar(
  const int* rows, const T* vals, int nnz, T scalar, int* results, cudaStream_t stream = 0)
{
  detail::coo_degree_scalar<64>(rows, vals, nnz, scalar, results, stream);
}

/**
 * @brief Count the number of values for each row that doesn't match a particular scalar
 * @tparam TPB_X: number of threads to use per block
 * @tparam T: the type name of the underlying value arrays
 * @param in: Input COO array
 * @param scalar: scalar to match for counting rows
 * @param results: output row counts
 * @param stream: cuda stream to use
 */
template <typename T>
void coo_degree_scalar(COO<T>* in, T scalar, int* results, cudaStream_t stream)
{
  coo_degree_scalar(in->rows(), in->vals(), in->nnz, scalar, results, stream);
}

/**
 * @brief Count the number of nonzeros for each row
 * @tparam TPB_X: number of threads to use per block
 * @tparam T: the type name of the underlying value arrays
 * @param rows: Input COO row array
 * @param vals: Input COO val arrays
 * @param nnz: size of input COO arrays
 * @param results: output row counts
 * @param stream: cuda stream to use
 */
template <typename T>
void coo_degree_nz(const int* rows, const T* vals, int nnz, int* results, cudaStream_t stream)
{
  detail::coo_degree_nz<64>(rows, vals, nnz, results, stream);
}

/**
 * @brief Count the number of nonzero values for each row
 * @tparam TPB_X: number of threads to use per block
 * @tparam T: the type name of the underlying value arrays
 * @param in: Input COO array
 * @param results: output row counts
 * @param stream: cuda stream to use
 */
template <typename T>
void coo_degree_nz(COO<T>* in, int* results, cudaStream_t stream)
{
  coo_degree_nz(in->rows(), in->vals(), in->nnz, results, stream);
}

};  // end NAMESPACE linalg
};  // end NAMESPACE sparse
};  // end NAMESPACE raft

#endif