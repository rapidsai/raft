/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/matrix/detail/preprocessing.cuh>

#include <optional>

namespace raft::sparse::matrix {

/**
 * @brief Use BM25 algorithm to encode features in COO sparse matrix
 * @tparam IndexType is the type of the edges index in the coo matrix
 * @tparam ValueType is the type of the values array in the coo matrix
 * @tparam IdxT is the type of the indices of arrays in matrix
 * @param handle: raft resource handle
 * @param coo_in: Input COO matrix
 * @param values_out: Output values array
 * @param k_param: K value to use for BM25 algorithm
 * @param b_param: B value to use for BM25 algorithm
 */
template <typename IndexType, typename ValueType, typename IdxT>
void encode_bm25(raft::resources& handle,
                 raft::device_coo_matrix_view<ValueType, IndexType, IndexType, IndexType> coo_in,
                 raft::device_vector_view<ValueType, IdxT> values_out,
                 float k_param = 1.6f,
                 float b_param = 0.75)
{
  return matrix::detail::encode_bm25<IndexType, ValueType, IdxT>(
    handle, coo_in, values_out, k_param, b_param);
}

/**
 * @brief Use BM25 algorithm to encode features in CSR sparse matrix
 * @param handle: raft resource handle
 * @tparam IndexType is the type of the edges index in the csr matrix
 * @tparam ValueType is the type of the values array in the csr matrix
 * @tparam IdxT is the type of the indices of arrays in matrix
 * @param csr_in: Input CSR matrix
 * @param values_out: Output values array
 * @param k_param: K value to use for BM25 algorithm
 * @param b_param: B value to use for BM25 algorithm
 */
template <typename IndexType, typename ValueType, typename IdxT>
void encode_bm25(raft::resources& handle,
                 raft::device_csr_matrix_view<ValueType, IndexType, IndexType, IndexType> csr_in,
                 raft::device_vector_view<ValueType, IdxT> values_out,
                 float k_param = 1.6f,
                 float b_param = 0.75)
{
  return matrix::detail::encode_bm25<IndexType, ValueType, IdxT>(
    handle, csr_in, values_out, k_param, b_param);
}

/**
 * @brief Use TFIDF algorithm to encode features in COO sparse matrix
 * @tparam IndexType is the type of the edges index in the coo matrix
 * @tparam ValueType is the type of the values array in the coo matrix
 * @tparam IdxT is the type of the indices of arrays in matrix
 * @param handle: raft resource handle
 * @param coo_in: Input COO matrix
 * @param values_out: Output COO values array
 */
template <typename IndexType, typename ValueType, typename IdxT>
void encode_tfidf(raft::resources& handle,
                  raft::device_coo_matrix_view<ValueType, IndexType, IndexType, IndexType> coo_in,
                  raft::device_vector_view<ValueType, IdxT> values_out)
{
  return matrix::detail::encode_tfidf<IndexType, ValueType, IdxT>(handle, coo_in, values_out);
}

/**
 * @brief Use TFIDF algorithm to encode features in CSR sparse matrix
 * @tparam IndexType is the type of the edges index in the csr matrix
 * @tparam ValueType is the type of the values array in the csr matrix
 * @tparam IdxT is the type of the indices of arrays in matrix
 * @param handle: raft resource handle
 * @param csr_in: Input CSR matrix
 * @param values_out: Output values array
 */
template <typename IndexType, typename ValueType, typename IdxT>
void encode_tfidf(raft::resources& handle,
                  raft::device_csr_matrix_view<ValueType, IndexType, IndexType, IndexType> csr_in,
                  raft::device_vector_view<ValueType, IdxT> values_out)
{
  return matrix::detail::encode_tfidf<IndexType, ValueType, IdxT>(handle, csr_in, values_out);
}

}  // namespace raft::sparse::matrix
