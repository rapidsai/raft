/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#ifndef __EPSILON_NEIGH_H
#define __EPSILON_NEIGH_H

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/spatial/knn/detail/epsilon_neighborhood.cuh>

namespace raft::neighbors::epsilon_neighborhood {

/**
 * @brief Computes epsilon neighborhood for the L2-Squared distance metric
 *
 * @tparam value_t   IO and math type
 * @tparam idx_t    Index type
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
template <typename value_t, typename idx_t>
void epsUnexpL2SqNeighborhood(bool* adj,
                              idx_t* vd,
                              const value_t* x,
                              const value_t* y,
                              idx_t m,
                              idx_t n,
                              idx_t k,
                              value_t eps,
                              cudaStream_t stream)
{
  spatial::knn::detail::epsUnexpL2SqNeighborhood<value_t, idx_t>(
    adj, vd, x, y, m, n, k, eps, stream);
}

/**
 * @defgroup epsilon_neighbors Epislon Neighborhood Operations
 * @{
 */

/**
 * @brief Computes epsilon neighborhood for the L2-Squared distance metric and given ball size.
 * The epsilon neighbors is represented by a dense boolean adjacency matrix of size m * n and
 * an array of degrees for each vertex, which can be used as a compressed sparse row (CSR)
 * indptr array.
 *
 * @code{.cpp}
 *  #include <raft/neighbors/epsilon_neighborhood.cuh>
 *  #include <raft/core/resources.hpp>
 *  #include <raft/core/device_mdarray.hpp>
 *  using namespace raft::neighbors;
 *  raft::raft::resources handle;
 *  ...
 *  auto adj = raft::make_device_matrix<bool>(handle, m * n);
 *  auto vd = raft::make_device_vector<int>(handle, m+1);
 *  epsilon_neighborhood::eps_neighbors_l2sq(handle, x, y, adj.view(), vd.view(), eps);
 * @endcode
 *
 * @tparam value_t   IO and math type
 * @tparam idx_t    Index type
 * @tparam matrix_idx_t matrix indexing type
 *
 * @param[in]  handle raft handle to manage library resources
 * @param[in]  x      first matrix [row-major] [on device] [dim = m x k]
 * @param[in]  y      second matrix [row-major] [on device] [dim = n x k]
 * @param[out] adj    adjacency matrix [row-major] [on device] [dim = m x n]
 * @param[out] vd     vertex degree array [on device] [len = m + 1]
 *                    `vd + m` stores the total number of edges in the adjacency
 *                    matrix. Pass a nullptr if you don't need this info.
 * @param[in]  eps    defines epsilon neighborhood radius (should be passed as
 *                    squared as we compute L2-squared distance in this method)
 */
template <typename value_t, typename idx_t, typename matrix_idx_t>
void eps_neighbors_l2sq(raft::resources const& handle,
                        raft::device_matrix_view<const value_t, matrix_idx_t, row_major> x,
                        raft::device_matrix_view<const value_t, matrix_idx_t, row_major> y,
                        raft::device_matrix_view<bool, matrix_idx_t, row_major> adj,
                        raft::device_vector_view<idx_t, matrix_idx_t> vd,
                        value_t eps)
{
  epsUnexpL2SqNeighborhood<value_t, idx_t>(adj.data_handle(),
                                           vd.data_handle(),
                                           x.data_handle(),
                                           y.data_handle(),
                                           x.extent(0),
                                           y.extent(0),
                                           x.extent(1),
                                           eps,
                                           resource::get_cuda_stream(handle));
}

/** @} */  // end group epsilon_neighbors

}  // namespace raft::neighbors::epsilon_neighborhood

#endif