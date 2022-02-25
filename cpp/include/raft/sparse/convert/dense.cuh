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
#ifndef __DENSE_H
#define __DENSE_H

#pragma once

#include <raft/sparse/convert/detail/dense.cuh>

namespace raft {
namespace sparse {
namespace convert {

/**
 * Convert CSR arrays to a dense matrix in either row-
 * or column-major format. A custom kernel is used when
 * row-major output is desired since cusparse does not
 * output row-major.
 * @tparam value_idx : data type of the CSR index arrays
 * @tparam value_t : data type of the CSR value array
 * @param[in] handle : cusparse handle for conversion
 * @param[in] nrows : number of rows in CSR
 * @param[in] ncols : number of columns in CSR
 * @param[in] nnz : number of nonzeros in CSR
 * @param[in] csr_indptr : CSR row index pointer array
 * @param[in] csr_indices : CSR column indices array
 * @param[in] csr_data : CSR data array
 * @param[in] lda : Leading dimension (used for col-major only)
 * @param[out] out : Dense output array of size nrows * ncols
 * @param[in] stream : Cuda stream for ordering events
 * @param[in] row_major : Is row-major output desired?
 */
template <typename value_idx, typename value_t>
void csr_to_dense(cusparseHandle_t handle,
                  value_idx nrows,
                  value_idx ncols,
                  value_idx nnz,
                  const value_idx* csr_indptr,
                  const value_idx* csr_indices,
                  const value_t* csr_data,
                  value_idx lda,
                  value_t* out,
                  cudaStream_t stream,
                  bool row_major = true)
{
  detail::csr_to_dense<value_idx, value_t>(
    handle, nrows, ncols, nnz, csr_indptr, csr_indices, csr_data, lda, out, stream, row_major);
}

};  // end NAMESPACE convert
};  // end NAMESPACE sparse
};  // end NAMESPACE raft

#endif