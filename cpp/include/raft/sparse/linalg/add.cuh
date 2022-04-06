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
#ifndef __SPARSE_ADD_H
#define __SPARSE_ADD_H

#pragma once

#include <raft/sparse/linalg/detail/add.cuh>

namespace raft {
namespace sparse {
namespace linalg {

/**
 * @brief Calculate the CSR row_ind array that would result
 * from summing together two CSR matrices
 * @param a_ind: left hand row_ind array
 * @param a_indptr: left hand index_ptr array
 * @param a_val: left hand data array
 * @param nnz1: size of left hand index_ptr and val arrays
 * @param b_ind: right hand row_ind array
 * @param b_indptr: right hand index_ptr array
 * @param b_val: right hand data array
 * @param nnz2: size of right hand index_ptr and val arrays
 * @param m: size of output array (number of rows in final matrix)
 * @param out_ind: output row_ind array
 * @param stream: cuda stream to use
 */
template <typename T>
size_t csr_add_calc_inds(const int* a_ind,
                         const int* a_indptr,
                         const T* a_val,
                         int nnz1,
                         const int* b_ind,
                         const int* b_indptr,
                         const T* b_val,
                         int nnz2,
                         int m,
                         int* out_ind,
                         cudaStream_t stream)
{
  return detail::csr_add_calc_inds(
    a_ind, a_indptr, a_val, nnz1, b_ind, b_indptr, b_val, nnz2, m, out_ind, stream);
}

/**
 * @brief Calculate the CSR row_ind array that would result
 * from summing together two CSR matrices
 * @param a_ind: left hand row_ind array
 * @param a_indptr: left hand index_ptr array
 * @param a_val: left hand data array
 * @param nnz1: size of left hand index_ptr and val arrays
 * @param b_ind: right hand row_ind array
 * @param b_indptr: right hand index_ptr array
 * @param b_val: right hand data array
 * @param nnz2: size of right hand index_ptr and val arrays
 * @param m: size of output array (number of rows in final matrix)
 * @param c_ind: output row_ind array
 * @param c_indptr: output ind_ptr array
 * @param c_val: output data array
 * @param stream: cuda stream to use
 */
template <typename T>
void csr_add_finalize(const int* a_ind,
                      const int* a_indptr,
                      const T* a_val,
                      int nnz1,
                      const int* b_ind,
                      const int* b_indptr,
                      const T* b_val,
                      int nnz2,
                      int m,
                      int* c_ind,
                      int* c_indptr,
                      T* c_val,
                      cudaStream_t stream)
{
  detail::csr_add_finalize(
    a_ind, a_indptr, a_val, nnz1, b_ind, b_indptr, b_val, nnz2, m, c_ind, c_indptr, c_val, stream);
}

};  // end NAMESPACE linalg
};  // end NAMESPACE sparse
};  // end NAMESPACE raft

#endif