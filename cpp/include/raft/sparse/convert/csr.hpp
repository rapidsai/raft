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
/**
 * This file is deprecated and will be removed in release 22.06.
 * Please use the cuh version instead.
 */

#ifndef __CSR_H
#define __CSR_H

#pragma once

#include <raft/sparse/convert/detail/csr.cuh>
#include <raft/sparse/csr.hpp>

namespace raft {
namespace sparse {
namespace convert {

template <typename value_t>
void coo_to_csr(const raft::handle_t& handle,
                const int* srcRows,
                const int* srcCols,
                const value_t* srcVals,
                int nnz,
                int m,
                int* dst_offsets,
                int* dstCols,
                value_t* dstVals)
{
  detail::coo_to_csr(handle, srcRows, srcCols, srcVals, nnz, m, dst_offsets, dstCols, dstVals);
}

/**
 * @brief Generate the row indices array for a sorted COO matrix
 *
 * @param rows: COO rows array
 * @param nnz: size of COO rows array
 * @param row_ind: output row indices array
 * @param m: number of rows in dense matrix
 * @param stream: cuda stream to use
 */
template <typename T>
void sorted_coo_to_csr(const T* rows, int nnz, T* row_ind, int m, cudaStream_t stream)
{
  detail::sorted_coo_to_csr(rows, nnz, row_ind, m, stream);
}

/**
 * @brief Generate the row indices array for a sorted COO matrix
 *
 * @param coo: Input COO matrix
 * @param row_ind: output row indices array
 * @param stream: cuda stream to use
 */
template <typename T>
void sorted_coo_to_csr(COO<T>* coo, int* row_ind, cudaStream_t stream)
{
  detail::sorted_coo_to_csr(coo->rows(), coo->nnz, row_ind, coo->n_rows, stream);
}

};  // end NAMESPACE convert
};  // end NAMESPACE sparse
};  // end NAMESPACE raft

#endif
