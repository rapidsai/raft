/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file thirdparty/LICENSES/LICENSE.faiss
 */

#pragma once

namespace raft::neighbors::detail::faiss_select {

template <typename T>
inline __device__ void swap(bool swap, T& x, T& y)
{
  T tmp = x;
  x     = swap ? y : x;
  y     = swap ? tmp : y;
}

template <typename T>
inline __device__ void assign(bool assign, T& x, T y)
{
  x = assign ? y : x;
}
}  // namespace raft::neighbors::detail::faiss_select
