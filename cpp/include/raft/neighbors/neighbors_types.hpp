/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>

namespace raft::neighbors {

/** A single batch of nearest neighbors in device memory */
template <typename T, typename IdxT>
class batch {
 public:
  /** Create a new empty batch of data */
  batch(raft::resources const& res, int64_t rows, int64_t cols)
    : indices_(make_device_matrix<IdxT, int64_t>(res, rows, cols)),
      distances_(make_device_matrix<T, int64_t>(res, rows, cols))
  {
  }

  void resize(raft::resources const& res, int64_t rows, int64_t cols)
  {
    indices_   = make_device_matrix<IdxT, int64_t>(res, rows, cols);
    distances_ = make_device_matrix<T, int64_t>(res, rows, cols);
  }

  /** Returns the indices for the batch */
  device_matrix_view<const IdxT, int64_t> indices() const
  {
    return raft::make_const_mdspan(indices_.view());
  }
  device_matrix_view<IdxT, int64_t> indices() { return indices_.view(); }

  /** Returns the distances for the batch */
  device_matrix_view<const T, int64_t> distances() const
  {
    return raft::make_const_mdspan(distances_.view());
  }
  device_matrix_view<T, int64_t> distances() { return distances_.view(); }

  /** Returns the size of the batch */
  int64_t batch_size() const { return indices().extent(1); }

 protected:
  raft::device_matrix<IdxT, int64_t> indices_;
  raft::device_matrix<T, int64_t> distances_;
};
}  // namespace raft::neighbors
