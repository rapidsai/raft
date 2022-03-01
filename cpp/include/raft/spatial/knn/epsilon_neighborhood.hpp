/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
/**
 * @warning This file is deprecated and will be removed in release 22.06.
 * Please use the cuh version instead.
 */

#ifndef __EPSILON_NEIGH_H
#define __EPSILON_NEIGH_H

#pragma once

#include <raft/spatial/knn/detail/epsilon_neighborhood.cuh>

namespace raft {
namespace spatial {
namespace knn {

/**
 * @brief Computes epsilon neighborhood for the L2-Squared distance metric
 *
 * @tparam DataT   IO and math type
 * @tparam IdxT    Index type
 *
 * @param[out] adj    adjacency matrix [row-major] [on device] [dim = m x n]
 * @param[out] vd     vertex degree array [on device] [len = m + 1]
 *                    `vd + m` stores the total number of edges in the adjacency
 *                    matrix. Pass a nullptr if you don't need this info.
 * @param[in]  x      first matrix [row-major] [on device] [dim = m x k]
 * @param[in]  y      second matrix [row-major] [on device] [dim = n x k]
 * @param[in]  m      number of rows in x
 * @param[in]  n      number of rows in y
 * @param[in]  k      number of columns in x and k
 * @param[in]  eps    defines epsilon neighborhood radius (should be passed as
 *                    squared as we compute L2-squared distance in this method)
 * @param[in]  stream cuda stream
 */
template <typename DataT, typename IdxT>
void epsUnexpL2SqNeighborhood(bool* adj,
                              IdxT* vd,
                              const DataT* x,
                              const DataT* y,
                              IdxT m,
                              IdxT n,
                              IdxT k,
                              DataT eps,
                              cudaStream_t stream)
{
  detail::epsUnexpL2SqNeighborhood<DataT, IdxT>(adj, vd, x, y, m, n, k, eps, stream);
}
}  // namespace knn
}  // namespace spatial
}  // namespace raft

#endif