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
#include <raft/core/nvtx.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/select_k_types.hpp>
#include <raft/sparse/matrix/detail/select_k.cuh>

#include <optional>

namespace raft::sparse::matrix {

using SelectAlgo = raft::matrix::SelectAlgo;

/**
 * @defgroup select_k Batched-select k smallest or largest key/values
 * @{
 */

/**
 * Selects the k smallest or largest keys/values from each row of the input matrix.
 *
 * This function operates on a row-major matrix `in_val` with dimensions `batch_size` x `len`,
 * selecting the k smallest or largest elements from each row. The selected elements are then stored
 * in a row-major output matrix `out_val` with dimensions `batch_size` x k.
 * If the total number of values in a row is less than K, then the extra position in the
 * corresponding row of out_val will maintain the original value. This applies to out_idx
 *
 * @tparam T
 *   Type of the elements being compared (keys).
 * @tparam IdxT
 *   Type of the indices associated with the keys.
 *
 * @param[in] handle
 *   Container for managing reusable resources.
 * @param[in] in_val
 *   Input matrix in CSR format with a logical dense shape of [batch_size, len],
 *   containing the elements to be compared and selected.
 * @param[in] in_idx
 *   Optional input indices [in_val.nnz] associated with `in_val.values`.
 *   If `in_idx` is `std::nullopt`, it defaults to a contiguous array from 0 to len-1.
 * @param[out] out_val
 *   Output matrix [in_val.get_n_row(), k] storing the selected k smallest/largest elements
 *   from each row of `in_val`.
 * @param[out] out_idx
 *   Output indices [in_val.get_n_row(), k] corresponding to the selected elements in `out_val`.
 * @param[in] select_min
 *   Flag indicating whether to select the k smallest (true) or largest (false) elements.
 * @param[in] sorted
 *   whether to make sure selected pairs are sorted by value
 * @param[in] algo
 *   the selection algorithm to use
 */
template <typename T, typename IdxT>
void select_k(raft::resources const& handle,
              raft::device_csr_matrix_view<const T, IdxT, IdxT, IdxT> in_val,
              std::optional<raft::device_vector_view<const IdxT, IdxT>> in_idx,
              raft::device_matrix_view<T, IdxT, raft::row_major> out_val,
              raft::device_matrix_view<IdxT, IdxT, raft::row_major> out_idx,
              bool select_min,
              bool sorted     = false,
              SelectAlgo algo = SelectAlgo::kAuto)
{
  return detail::select_k<T, IdxT>(
    handle, in_val, in_idx, out_val, out_idx, select_min, sorted, algo);
}
/** @} */  // end of group select_k

}  // namespace raft::sparse::matrix
