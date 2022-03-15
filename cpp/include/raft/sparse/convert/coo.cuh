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
#ifndef __COO_H
#define __COO_H

#pragma once

#include <raft/sparse/convert/detail/coo.cuh>

namespace raft {
namespace sparse {
namespace convert {

/**
 * @brief Convert a CSR row_ind array to a COO rows array
 * @param row_ind: Input CSR row_ind array
 * @param m: size of row_ind array
 * @param coo_rows: Output COO row array
 * @param nnz: size of output COO row array
 * @param stream: cuda stream to use
 */
template <typename value_idx = int>
void csr_to_coo(
  const value_idx* row_ind, value_idx m, value_idx* coo_rows, value_idx nnz, cudaStream_t stream)
{
  detail::csr_to_coo<value_idx, 32>(row_ind, m, coo_rows, nnz, stream);
}

};  // end NAMESPACE convert
};  // end NAMESPACE sparse
};  // end NAMESPACE raft

#endif