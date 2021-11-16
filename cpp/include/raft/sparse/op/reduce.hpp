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

#include <raft/handle.hpp>
#include <raft/sparse/op/detail/reduce.cuh>
#include <raft/sparse/coo.hpp>

namespace raft {
namespace sparse {
namespace op {

/**
 * Performs a COO reduce of duplicate columns per row, taking the max weight
 * for duplicate columns in each row. This function assumes the input COO
 * has been sorted by both row and column but makes no assumption on
 * the sorting of values.
 * @tparam value_idx
 * @tparam value_t
 * @param[out] out output COO, the nnz will be computed allocate() will be called in this function.
 * @param[in] rows COO rows array, size nnz
 * @param[in] cols COO cols array, size nnz
 * @param[in] vals COO vals array, size nnz
 * @param[in] nnz number of nonzeros in COO input arrays
 * @param[in] m number of rows in COO input matrix
 * @param[in] n number of columns in COO input matrix
 * @param[in] stream cuda ops will be ordered wrt this stream
 */
template <typename value_idx, typename value_t>
void max_duplicates(const raft::handle_t &handle,
                    raft::sparse::COO<value_t, value_idx> &out,
                    const value_idx *rows, const value_idx *cols,
                    const value_t *vals, size_t nnz, size_t m, size_t n) {
    detail::max_duplicates(handle, out, rows, cols, vals, nnz, m, n,);
}
};  // END namespace op
};  // END namespace sparse
};  // END namespace raft
