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
#ifndef __SPARSE_NORM_H
#define __SPARSE_NORM_H

#pragma once

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/norm_types.hpp>
#include <raft/sparse/linalg/detail/norm.cuh>

namespace raft {
namespace sparse {
namespace linalg {

/**
 * @brief Perform L1 normalization on the rows of a given CSR-formatted sparse matrix
 *
 * @param ia: row_ind array
 * @param vals: data array
 * @param nnz: size of data array
 * @param m: size of row_ind array
 * @param result: l1 normalized data array
 * @param stream: cuda stream to use
 */
template <typename T>
void csr_row_normalize_l1(const int* ia,  // csr row ex_scan (sorted by row)
                          const T* vals,
                          int nnz,  // array of values and number of non-zeros
                          int m,    // num rows in csr
                          T* result,
                          cudaStream_t stream)
{  // output array
  detail::csr_row_normalize_l1(ia, vals, nnz, m, result, stream);
}

/**
 * @brief Perform L_inf normalization on a given CSR-formatted sparse matrix
 *
 * @param ia: row_ind array
 * @param vals: data array
 * @param nnz: size of data array
 * @param m: size of row_ind array
 * @param result: l1 normalized data array
 * @param stream: cuda stream to use
 */
template <typename T>
void csr_row_normalize_max(const int* ia,  // csr row ind array (sorted by row)
                           const T* vals,
                           int nnz,  // array of values and number of non-zeros
                           int m,    // num total rows in csr
                           T* result,
                           cudaStream_t stream)
{
  detail::csr_row_normalize_max(ia, vals, nnz, m, result, stream);
}

/**
 * @brief Compute row-wise norm of the input matrix and perform fin_op lambda
 *
 * Row-wise norm is useful while computing pairwise distance matrix, for
 * example.
 * This is used in many clustering algos like knn, kmeans, dbscan, etc...
 *
 * @tparam Type the data type
 * @tparam Lambda device final lambda
 * @tparam IdxType Integer type used to for addressing
 * @param handle raft handle
 * @param ia the input matrix row index array
 * @param data the input matrix nnz data
 * @param nnz number of elements in data
 * @param N number of rows
 * @param norm the output vector of row-wise norm, size [N]
 * @param type the type of norm to be applied
 * @param fin_op the final lambda op
 */
template <typename Type, typename IdxType = int, typename Lambda = raft::identity_op>
void rowNormCsr(raft::resources const& handle,
                const IdxType* ia,
                const Type* data,
                const IdxType nnz,
                const IdxType N,
                Type* norm,
                raft::linalg::NormType type,
                Lambda fin_op = raft::identity_op())
{
  detail::rowNormCsrCaller(ia, data, nnz, N, norm, type, fin_op, resource::get_cuda_stream(handle));
}

};  // end NAMESPACE linalg
};  // end NAMESPACE sparse
};  // end NAMESPACE raft

#endif