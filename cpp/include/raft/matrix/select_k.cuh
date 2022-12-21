/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "detail/select_k.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>

#include <optional>

namespace raft::matrix {

/**
 * Select k smallest or largest key/values from each row in the input data.
 *
 * If you think of the input data `in_val` as a row-major matrix with `len` columns and
 * `batch_size` rows, then this function selects `k` smallest/largest values in each row and fills
 * in the row-major matrix `out_val` of size (batch_size, k).
 *
 * Example usage
 * @code{.cpp}
 *   using namespace raft;
 *   // get a 2D row-major array of values to search through
 *   auto in_values = {... input device_matrix_view<const float, size_t, row_major> ...}
 *   // prepare output arrays
 *   auto out_extents = make_extents<size_t>(in_values.extent(0), k);
 *   auto out_values  = make_device_mdarray<float>(handle, out_extents);
 *   auto out_indices = make_device_mdarray<size_t>(handle, out_extents);
 *   // search `k` smallest values in each row
 *   matrix::select_k<float, size_t>(
 *     handle, in_values, std::nullopt, out_values.view(), out_indices.view(), true);
 * @endcode
 *
 * @tparam T
 *   the type of the keys (what is being compared).
 * @tparam IdxT
 *   the index type (what is being selected together with the keys).
 *
 * @param handle
 * @param[in] in_val
 *   inputs values [batch_size, len];
 *   these are compared and selected.
 * @param[in] in_idx
 *   optional input payload [batch_size, len];
 *   typically, these are indices of the corresponding `in_val`.
 *   If `in_idx` is `std::nullopt`, a contiguous array `0...len-1` is implied.
 * @param[out] out_val
 *   output values [batch_size, k];
 *   the k smallest/largest values from each row of the `in_val`.
 * @param[out] out_idx
 *   output payload (e.g. indices) [batch_size, k];
 *   the payload selected together with `out_val`.
 * @param select_min
 *   whether to select k smallest (true) or largest (false) keys.
 * @param mr an optional memory resource to use across the calls (you can provide a large enough
 *           memory pool here to avoid memory allocations within the call).
 */
template <typename T, typename IdxT>
void select_k(const handle_t& handle,
              raft::device_matrix_view<const T, size_t, row_major> in_val,
              std::optional<raft::device_matrix_view<const IdxT, size_t, row_major>> in_idx,
              raft::device_matrix_view<T, size_t, row_major> out_val,
              raft::device_matrix_view<IdxT, size_t, row_major> out_idx,
              bool select_min,
              rmm::mr::device_memory_resource* mr = nullptr)
{
  RAFT_EXPECTS(out_val.extent(1) <= size_t(std::numeric_limits<int>::max()),
               "output k must fit the int type.");
  auto batch_size = in_val.extent(0);
  auto len        = in_val.extent(1);
  auto k          = int(out_val.extent(1));
  RAFT_EXPECTS(batch_size == out_val.extent(0), "batch sizes must be equal");
  RAFT_EXPECTS(batch_size == out_idx.extent(0), "batch sizes must be equal");
  if (in_idx.has_value()) {
    RAFT_EXPECTS(batch_size == in_idx->extent(0), "batch sizes must be equal");
    RAFT_EXPECTS(len == in_idx->extent(1), "value and index input lengths must be equal");
  }
  RAFT_EXPECTS(size_t(k) == out_idx.extent(1), "value and index output lengths must be equal");
  return detail::select_k<T, IdxT>(in_val.data_handle(),
                                   in_idx.has_value() ? in_idx->data_handle() : nullptr,
                                   batch_size,
                                   len,
                                   k,
                                   out_val.data_handle(),
                                   out_idx.data_handle(),
                                   select_min,
                                   handle.get_stream(),
                                   mr);
}

}  // namespace raft::matrix
