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

#include <raft/linalg/detail/reduce_rows_by_key.cuh>

namespace raft {
namespace linalg {

/**
 Small helper function to convert from int->char and char->int
 Transform ncols*nrows read of int in 2*nrows reads of int + ncols*rows reads of chars
**/
template <typename IteratorT1, typename IteratorT2>
void convert_array(IteratorT1 dst, IteratorT2 src, int n, cudaStream_t st)
{
  detail::convert_array(dst, src, st);
}

/**
 * @brief Computes the weighted reduction of matrix rows for each given key
 *
 * @tparam DataIteratorT Random-access iterator type, for reading input matrix
 *                       (may be a simple pointer type)
 * @tparam KeysIteratorT Random-access iterator type, for reading input keys
 *                       (may be a simple pointer type)
 *
 * @param[in]  d_A         Input data array (lda x nrows)
 * @param[in]  lda         Real row size for input data, d_A
 * @param[in]  d_keys      Keys for each row (1 x nrows)
 * @param[in]  d_weights   Weights for each observation in d_A (1 x nrows)
 * @param[out] d_keys_char Scratch memory for conversion of keys to char
 * @param[in]  nrows       Number of rows in d_A and d_keys
 * @param[in]  ncols       Number of data columns in d_A
 * @param[in]  nkeys       Number of unique keys in d_keys
 * @param[out] d_sums      Row sums by key (ncols x d_keys)
 * @param[in]  stream      CUDA stream
 */
template <typename DataIteratorT, typename KeysIteratorT, typename WeightT>
void reduce_rows_by_key(const DataIteratorT d_A,
                        int lda,
                        const KeysIteratorT d_keys,
                        const WeightT* d_weights,
                        char* d_keys_char,
                        int nrows,
                        int ncols,
                        int nkeys,
                        DataIteratorT d_sums,
                        cudaStream_t stream)
{
  detail::reduce_rows_by_key(
    d_A, lda, d_keys, d_weights, d_keys_char, nrows, ncols, nkeys, d_sums, stream);
}

/**
 * @brief Computes the reduction of matrix rows for each given key
 * @tparam DataIteratorT Random-access iterator type, for reading input matrix (may be a simple
 * pointer type)
 * @tparam KeysIteratorT Random-access iterator type, for reading input keys (may be a simple
 * pointer type)
 * @param[in]  d_A         Input data array (lda x nrows)
 * @param[in]  lda         Real row size for input data, d_A
 * @param[in]  d_keys      Keys for each row (1 x nrows)
 * @param      d_keys_char Scratch memory for conversion of keys to char
 * @param[in]  nrows       Number of rows in d_A and d_keys
 * @param[in]  ncols       Number of data columns in d_A
 * @param[in]  nkeys       Number of unique keys in d_keys
 * @param[out] d_sums      Row sums by key (ncols x d_keys)
 * @param[in]  stream      CUDA stream
 */
template <typename DataIteratorT, typename KeysIteratorT>
void reduce_rows_by_key(const DataIteratorT d_A,
                        int lda,
                        const KeysIteratorT d_keys,
                        char* d_keys_char,
                        int nrows,
                        int ncols,
                        int nkeys,
                        DataIteratorT d_sums,
                        cudaStream_t stream)
{
  typedef typename std::iterator_traits<DataIteratorT>::value_type DataType;
  reduce_rows_by_key(d_A,
                     lda,
                     d_keys,
                     static_cast<DataType*>(nullptr),
                     d_keys_char,
                     nrows,
                     ncols,
                     nkeys,
                     d_sums,
                     stream);
}

};  // end namespace linalg
};  // end namespace raft
