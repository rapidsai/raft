/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <utility>

namespace raft::matrix::detail::select {

/**
 * Row-layout policy for dense matrices where every row has the same length.
 * Returns the linear offset as size_t to avoid overflow when the product exceeds IdxT range.
 */
template <typename IdxT>
struct dense_layout {
  static constexpr bool is_uniform = true;

  static __device__ __forceinline__ auto compute(IdxT len,
                                                 size_t offset,
                                                 size_t batch_id,
                                                 const IdxT*) -> std::pair<IdxT, size_t>
  {
    return {len, (offset + batch_id) * len};
  }
};

/**
 * Row-layout policy for CSR-like matrices where row lengths are given by an
 * indptr array: row `i` spans `[indptr[i], indptr[i+1])`.
 */
template <typename IdxT>
struct csr_layout {
  static constexpr bool is_uniform = false;

  static __device__ __forceinline__ auto compute(IdxT, size_t, size_t batch_id, const IdxT* indptr)
    -> std::pair<IdxT, IdxT>
  {
    return {indptr[batch_id + 1] - indptr[batch_id], indptr[batch_id]};
  }
};

}  // namespace raft::matrix::detail::select
