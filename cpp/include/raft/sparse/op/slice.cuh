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
#ifndef __SLICE_H
#define __SLICE_H

#pragma once

#include <raft/core/resources.hpp>
#include <raft/sparse/op/detail/slice.cuh>

namespace raft {
namespace sparse {
namespace op {

/**
 * Slice consecutive rows from a CSR array and populate newly sliced indptr array
 * @tparam value_idx
 * @param[in] start_row : beginning row to slice
 * @param[in] stop_row : ending row to slice
 * @param[in] indptr : indptr of input CSR to slice
 * @param[out] indptr_out : output sliced indptr to populate
 * @param[in] start_offset : beginning column offset of input indptr
 * @param[in] stop_offset : ending column offset of input indptr
 * @param[in] stream : cuda stream for ordering events
 */
template <typename value_idx>
void csr_row_slice_indptr(value_idx start_row,
                          value_idx stop_row,
                          const value_idx* indptr,
                          value_idx* indptr_out,
                          value_idx* start_offset,
                          value_idx* stop_offset,
                          cudaStream_t stream)
{
  detail::csr_row_slice_indptr(
    start_row, stop_row, indptr, indptr_out, start_offset, stop_offset, stream);
}

/**
 * Slice rows from a CSR, populate column and data arrays
 * @tparam value_idx : data type of CSR index arrays
 * @tparam value_t : data type of CSR data array
 * @param[in] start_offset : beginning column offset to slice
 * @param[in] stop_offset : ending column offset to slice
 * @param[in] indices : column indices array from input CSR
 * @param[in] data : data array from input CSR
 * @param[out] indices_out : output column indices array
 * @param[out] data_out : output data array
 * @param[in] stream : cuda stream for ordering events
 */
template <typename value_idx, typename value_t>
void csr_row_slice_populate(value_idx start_offset,
                            value_idx stop_offset,
                            const value_idx* indices,
                            const value_t* data,
                            value_idx* indices_out,
                            value_t* data_out,
                            cudaStream_t stream)
{
  detail::csr_row_slice_populate(
    start_offset, stop_offset, indices, data, indices_out, data_out, stream);
}

};  // namespace op
};  // end NAMESPACE sparse
};  // end NAMESPACE raft

#endif