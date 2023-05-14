/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cstdint>
#include <cstddef>

namespace raft::neighbors::ivf_pq::detail {

/* A filter that filters nothing. */
struct NoneSampleFilter {
  inline __device__ __host__ bool operator()(
    // query index
    const uint32_t query_ix,
    // the current inverted list index
    const uint32_t cluster_ix,
    // the index of the current sample inside the current inverted list
    const uint32_t sample_ix
  ) const {
    return true;
  }
};

/**
If the filtering depends on the index of a sample, then the following
filter template can be used:

template <typename IdxT>
struct IndexSampleFilter {
  using index_type = IdxT;

  const index_type* const* inds_ptr = nullptr;

  IndexSampleFilter() {}
  IndexSampleFilter(const index_type* const* _inds_ptr)
      : inds_ptr{_inds_ptr} {}
  IndexSampleFilter(const IndexSampleFilter&) = default;
  IndexSampleFilter(IndexSampleFilter&&) = default;
  IndexSampleFilter& operator=(const IndexSampleFilter&) = default;
  IndexSampleFilter& operator=(IndexSampleFilter&&) = default;

  inline __device__ __host__ bool operator()(
      const uint32_t query_ix,
      const uint32_t cluster_ix,
      const uint32_t sample_ix) const {
    index_type database_idx = inds_ptr[cluster_ix][sample_ix];

    // return true or false, depending on the database_idx
    return true;
  }
};

Initialize it as:
  using filter_type = IndexSampleFilter<idx_t>;
  filter_type filter(raft_ivfpq_index.inds_ptrs().data_handle());

Use it as:
  raft::neighbors::ivf_pq::search_with_filtering<data_t, idx_t, filter_type>(
    ...regular parameters here...,
    filter
  );
*/

/* A filter that selects samples according to a bit mask. */
template <typename IdxT>
struct BitMaskSampleFilter {
  using index_type = IdxT;

  const index_type* const* inds_ptr = nullptr;
  const uint64_t* const bit_mask_ptr = nullptr;
  const int64_t bit_mask_stride_64 = 0;

  BitMaskSampleFilter() {}
  BitMaskSampleFilter(
      const index_type* const* _inds_ptr,
      const uint64_t* const _bit_mask_ptr,
      const int64_t _bit_mask_stride_64)
      : inds_ptr{_inds_ptr},
        bit_mask_ptr{_bit_mask_ptr},
        bit_mask_stride_64{_bit_mask_stride_64} {}
  BitMaskSampleFilter(const BitMaskSampleFilter&) = default;
  BitMaskSampleFilter(BitMaskSampleFilter&&) = default;
  BitMaskSampleFilter& operator=(const BitMaskSampleFilter&) = default;
  BitMaskSampleFilter& operator=(BitMaskSampleFilter&&) = default;

  inline __device__ __host__ bool operator()(
      const uint32_t query_ix,
      const uint32_t cluster_ix,
      const uint32_t sample_ix) const {
    const index_type database_idx = inds_ptr[cluster_ix][sample_ix];
    const uint64_t bit_mask_element =
        bit_mask_ptr[query_ix * bit_mask_stride_64 + database_idx / 64];
    const uint64_t masked_bool =
        bit_mask_element & (1ULL << (uint64_t)(database_idx % 64));
    const bool is_bit_set = (masked_bool != 0);

    return is_bit_set;
  }
};

}
