/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cstddef>
#include <cstdint>

#include <raft/core/bitset.cuh>
#include <raft/core/detail/macros.hpp>

namespace raft::neighbors::filtering {

/* A filter that filters nothing. This is the default behavior. */
struct none_ivf_sample_filter {
  inline _RAFT_HOST_DEVICE bool operator()(
    // query index
    const uint32_t query_ix,
    // the current inverted list index
    const uint32_t cluster_ix,
    // the index of the current sample inside the current inverted list
    const uint32_t sample_ix) const
  {
    return true;
  }
};
/**
 * @brief Filter a CAGRA index with a bitset
 *
 * @tparam index_t Indexing type
 */
template <typename index_t>
struct removed_cagra_filter {
  // Pointers to the inverted lists (clusters) indices  [n_lists]
  const raft::core::bitset_view<const std::uint32_t, index_t> removed_bitset_;

  removed_cagra_filter(const raft::core::bitset_view<const std::uint32_t, index_t> removed_bitset)
    : removed_bitset_{removed_bitset}
  {
  }
  inline _RAFT_HOST_DEVICE bool operator()(
    // query index
    const uint32_t query_ix,
    // the index of the current sample
    const uint32_t sample_ix) const
  {
    return removed_bitset_.test(sample_ix);
  }
};

/* A filter that filters nothing. This is the default behavior. */
struct none_cagra_sample_filter {
  inline _RAFT_HOST_DEVICE bool operator()(
    // query index
    const uint32_t query_ix,
    // the index of the current sample
    const uint32_t sample_ix) const
  {
    return true;
  }
};

template <class CagraSampleFilterT>
struct CagraSampleFilterWithQueryIdOffset {
  const uint32_t offset;
  CagraSampleFilterT filter;

  CagraSampleFilterWithQueryIdOffset(const uint32_t offset, const CagraSampleFilterT filter)
    : offset(offset), filter(filter)
  {
  }

  _RAFT_DEVICE auto operator()(const uint32_t query_id, const uint32_t sample_id)
  {
    return filter(query_id + offset, sample_id);
  }
};

/** Utility to add an offset to the query ud */
template <class CagraSampleFilterT>
struct CagraSampleFilterT_Selector {
  using type = CagraSampleFilterWithQueryIdOffset<CagraSampleFilterT>;
};
template <>
struct CagraSampleFilterT_Selector<raft::neighbors::filtering::none_cagra_sample_filter> {
  using type = raft::neighbors::filtering::none_cagra_sample_filter;
};

// A helper function to set a query id offset
template <class CagraSampleFilterT>
inline typename CagraSampleFilterT_Selector<CagraSampleFilterT>::type set_offset(
  CagraSampleFilterT filter, const uint32_t offset)
{
  typename CagraSampleFilterT_Selector<CagraSampleFilterT>::type new_filter(offset, filter);
  return new_filter;
}
template <>
inline
  typename CagraSampleFilterT_Selector<raft::neighbors::filtering::none_cagra_sample_filter>::type
  set_offset<raft::neighbors::filtering::none_cagra_sample_filter>(
    raft::neighbors::filtering::none_cagra_sample_filter filter, const uint32_t)
{
  return filter;
}

using removed_cagra_filter_with_offset_u32 =
  CagraSampleFilterWithQueryIdOffset<removed_cagra_filter<uint32_t>>;
using removed_cagra_filter_with_offset_u64 =
  CagraSampleFilterWithQueryIdOffset<removed_cagra_filter<uint64_t>>;
/**
 * If the filtering depends on the index of a sample, then the following
 * filter template can be used:
 *
 * template <typename IdxT>
 * struct index_ivf_sample_filter {
 *   using index_type = IdxT;
 *
 *   const index_type* const* inds_ptr = nullptr;
 *
 *   index_ivf_sample_filter() {}
 *   index_ivf_sample_filter(const index_type* const* _inds_ptr)
 *       : inds_ptr{_inds_ptr} {}
 *   index_ivf_sample_filter(const index_ivf_sample_filter&) = default;
 *   index_ivf_sample_filter(index_ivf_sample_filter&&) = default;
 *   index_ivf_sample_filter& operator=(const index_ivf_sample_filter&) = default;
 *   index_ivf_sample_filter& operator=(index_ivf_sample_filter&&) = default;
 *
 *   inline _RAFT_HOST_DEVICE bool operator()(
 *       const uint32_t query_ix,
 *       const uint32_t cluster_ix,
 *       const uint32_t sample_ix) const {
 *     index_type database_idx = inds_ptr[cluster_ix][sample_ix];
 *
 *     // return true or false, depending on the database_idx
 *     return true;
 *   }
 * };
 *
 * Initialize it as:
 *   using filter_type = index_ivf_sample_filter<idx_t>;
 *   filter_type filter(raft_ivfpq_index.inds_ptrs().data_handle());
 *
 * Use it as:
 *   raft::neighbors::ivf_pq::search_with_filtering<data_t, idx_t, filter_type>(
 *     ...regular parameters here...,
 *     filter
 *   );
 *
 * Another example would be the following filter that greenlights samples according
 * to a contiguous bit mask vector.
 *
 * template <typename IdxT>
 * struct bitmask_ivf_sample_filter {
 *   using index_type = IdxT;
 *
 *   const index_type* const* inds_ptr = nullptr;
 *   const uint64_t* const bit_mask_ptr = nullptr;
 *   const int64_t bit_mask_stride_64 = 0;
 *
 *   bitmask_ivf_sample_filter() {}
 *   bitmask_ivf_sample_filter(
 *       const index_type* const* _inds_ptr,
 *       const uint64_t* const _bit_mask_ptr,
 *       const int64_t _bit_mask_stride_64)
 *       : inds_ptr{_inds_ptr},
 *         bit_mask_ptr{_bit_mask_ptr},
 *         bit_mask_stride_64{_bit_mask_stride_64} {}
 *   bitmask_ivf_sample_filter(const bitmask_ivf_sample_filter&) = default;
 *   bitmask_ivf_sample_filter(bitmask_ivf_sample_filter&&) = default;
 *   bitmask_ivf_sample_filter& operator=(const bitmask_ivf_sample_filter&) = default;
 *   bitmask_ivf_sample_filter& operator=(bitmask_ivf_sample_filter&&) = default;
 *
 *   inline _RAFT_HOST_DEVICE bool operator()(
 *       const uint32_t query_ix,
 *       const uint32_t cluster_ix,
 *       const uint32_t sample_ix) const {
 *     const index_type database_idx = inds_ptr[cluster_ix][sample_ix];
 *     const uint64_t bit_mask_element =
 *         bit_mask_ptr[query_ix * bit_mask_stride_64 + database_idx / 64];
 *     const uint64_t masked_bool =
 *         bit_mask_element & (1ULL << (uint64_t)(database_idx % 64));
 *     const bool is_bit_set = (masked_bool != 0);
 *
 *     return is_bit_set;
 *   }
 * };
 */
}  // namespace raft::neighbors::filtering
