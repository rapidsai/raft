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

#include "topk/radix_topk.cuh"
#include "topk/warpsort_topk.cuh"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace raft::spatial::knn::detail {

/**
 * Select k smallest or largest key/values from each row in the input data.
 *
 * If you think of the input data `in_keys` as a row-major matrix with len columns and
 * batch_size rows, then this function selects k smallest/largest values in each row and fills
 * in the row-major matrix `out` of size (batch_size, k).
 *
 * @tparam T
 *   the type of the keys (what is being compared).
 * @tparam IdxT
 *   the index type (what is being selected together with the keys).
 *
 * @param[in] in
 *   contiguous device array of inputs of size (len * batch_size);
 *   these are compared and selected.
 * @param[in] in_idx
 *   contiguous device array of inputs of size (len * batch_size);
 *   typically, these are indices of the corresponding in_keys.
 * @param batch_size
 *   number of input rows, i.e. the batch size.
 * @param len
 *   length of a single input array (row); also sometimes referred as n_cols.
 *   Invariant: len >= k.
 * @param k
 *   the number of outputs to select in each input row.
 * @param[out] out
 *   contiguous device array of outputs of size (k * batch_size);
 *   the k smallest/largest values from each row of the `in_keys`.
 * @param[out] out_idx
 *   contiguous device array of outputs of size (k * batch_size);
 *   the payload selected together with `out`.
 * @param select_min
 *   whether to select k smallest (true) or largest (false) keys.
 * @param stream
 * @param mr an optional memory resource to use across the calls (you can provide a large enough
 *           memory pool here to avoid memory allocations within the call).
 */
template <typename T, typename IdxT>
void select_topk(const T* in,
                 const IdxT* in_idx,
                 size_t batch_size,
                 size_t len,
                 int k,
                 T* out,
                 IdxT* out_idx,
                 bool select_min,
                 rmm::cuda_stream_view stream,
                 rmm::mr::device_memory_resource* mr = nullptr)
{
  if (k <= raft::spatial::knn::detail::topk::kMaxCapacity) {
    topk::warp_sort_topk<T, IdxT>(
      in, in_idx, batch_size, len, k, out, out_idx, select_min, stream, mr);
  } else {
    topk::radix_topk<T, IdxT, (sizeof(T) >= 4 ? 11 : 8), 512>(
      in, in_idx, batch_size, len, k, out, out_idx, select_min, stream, mr);
  }
}

}  // namespace raft::spatial::knn::detail
