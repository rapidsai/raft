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

#ifndef __COL_WISE_SORT_H
#define __COL_WISE_SORT_H

#pragma once

#include <raft/matrix/detail/columnWiseSort.cuh>

namespace raft {
    namespace matrix {

/**
 * @brief sort columns within each row of row-major input matrix and return sorted indexes
 * modelled as key-value sort with key being input matrix and value being index of values
 * @param in: input matrix
 * @param out: output value(index) matrix
 * @param n_rows: number rows of input matrix
 * @param n_columns: number columns of input matrix
 * @param bAllocWorkspace: check returned value, if true allocate workspace passed in workspaceSize
 * @param workspacePtr: pointer to workspace memory
 * @param workspaceSize: Size of workspace to be allocated
 * @param stream: cuda stream to execute prim on
 * @param sortedKeys: Optional, output matrix for sorted keys (input)
 */
        template <typename InType, typename OutType>
        void sort_cols_per_row(const InType* in,
                               OutType* out,
                               int n_rows,
                               int n_columns,
                               bool& bAllocWorkspace,
                               void* workspacePtr,
                               size_t& workspaceSize,
                               cudaStream_t stream,
                               InType* sortedKeys = nullptr)
        {
            detail::sortColumnsPerRow<InType, OutType>(
                    in, out, n_rows, n_columns, bAllocWorkspace, workspacePtr, workspaceSize, stream, sortedKeys);
        }
    };  // end namespace matrix
};  // end namespace raft

#endif