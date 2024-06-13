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

#include "detail/nn_descent.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>

namespace raft::neighbors::experimental::nn_descent {

/**
 * @defgroup nn-descent CUDA gradient descent nearest neighbor
 * @{
 */

/**
 * @brief Build nn-descent Index with dataset in device memory
 *
 * The following distance metrics are supported:
 * - L2
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace raft::neighbors::experimental;
 *   // use default index parameters
 *   nn_descent::index_params index_params;
 *   // create and fill the index from a [N, D] raft::device_matrix_view dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *   // index.graph() provides a raft::host_matrix_view of an
 *   // all-neighbors knn graph of dimensions [N, k] of the input
 *   // dataset
 * @endcode
 *
 * @tparam T data-type of the input dataset
 * @tparam IdxT data-type for the output index
 * @param[in] res raft::resources is an object mangaging resources
 * @param[in] params an instance of nn_descent::index_params that are parameters
 *               to run the nn-descent algorithm
 * @param[in] dataset raft::device_matrix_view input dataset expected to be located
 *                in device memory
 * @return index<IdxT> index containing all-neighbors knn graph in host memory
 */
template <typename T, typename IdxT = uint32_t>
index<IdxT> build(raft::resources const& res,
                  index_params const& params,
                  raft::device_matrix_view<const T, int64_t, row_major> dataset)
{
  return detail::build<T, IdxT>(res, params, dataset);
}

/**
 * @brief Build nn-descent Index with dataset in device memory
 *
 * The following distance metrics are supported:
 * - L2
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace raft::neighbors::experimental;
 *   // use default index parameters
 *   nn_descent::index_params index_params;
 *   // create and fill the index from a [N, D] raft::device_matrix_view dataset
 *   auto knn_graph = raft::make_host_matrix<uint32_t, int64_t>(N, D);
 *   auto index = nn_descent::index{res, knn_graph.view()};
 *   cagra::build(res, index_params, dataset, index);
 *   // index.graph() provides a raft::host_matrix_view of an
 *   // all-neighbors knn graph of dimensions [N, k] of the input
 *   // dataset
 * @endcode
 *
 * @tparam T data-type of the input dataset
 * @tparam IdxT data-type for the output index
 * @param res raft::resources is an object mangaging resources
 * @param[in] params an instance of nn_descent::index_params that are parameters
 *               to run the nn-descent algorithm
 * @param[in] dataset raft::device_matrix_view input dataset expected to be located
 *                in device memory
 * @param[out] idx raft::neighbors::experimental::nn_descentindex containing all-neighbors knn graph
 * in host memory
 */
template <typename T, typename IdxT = uint32_t>
void build(raft::resources const& res,
           index_params const& params,
           raft::device_matrix_view<const T, int64_t, row_major> dataset,
           index<IdxT>& idx)
{
  detail::build<T, IdxT>(res, params, dataset, idx);
}

/**
 * @brief Build nn-descent Index with dataset in host memory
 *
 * The following distance metrics are supported:
 * - L2
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace raft::neighbors::experimental;
 *   // use default index parameters
 *   nn_descent::index_params index_params;
 *   // create and fill the index from a [N, D] raft::host_matrix_view dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *   // index.graph() provides a raft::host_matrix_view of an
 *   // all-neighbors knn graph of dimensions [N, k] of the input
 *   // dataset
 * @endcode
 *
 * @tparam T data-type of the input dataset
 * @tparam IdxT data-type for the output index
 * @param res raft::resources is an object mangaging resources
 * @param[in] params an instance of nn_descent::index_params that are parameters
 *               to run the nn-descent algorithm
 * @param[in] dataset raft::host_matrix_view input dataset expected to be located
 *                in host memory
 * @return index<IdxT> index containing all-neighbors knn graph in host memory
 */
template <typename T, typename IdxT = uint32_t>
index<IdxT> build(raft::resources const& res,
                  index_params const& params,
                  raft::host_matrix_view<const T, int64_t, row_major> dataset)
{
  return detail::build<T, IdxT>(res, params, dataset);
}

/**
 * @brief Build nn-descent Index with dataset in host memory
 *
 * The following distance metrics are supported:
 * - L2
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace raft::neighbors::experimental;
 *   // use default index parameters
 *   nn_descent::index_params index_params;
 *   // create and fill the index from a [N, D] raft::host_matrix_view dataset
 *   auto knn_graph = raft::make_host_matrix<uint32_t, int64_t>(N, D);
 *   auto index = nn_descent::index{res, knn_graph.view()};
 *   cagra::build(res, index_params, dataset, index);
 *   // index.graph() provides a raft::host_matrix_view of an
 *   // all-neighbors knn graph of dimensions [N, k] of the input
 *   // dataset
 * @endcode
 *
 * @tparam T data-type of the input dataset
 * @tparam IdxT data-type for the output index
 * @param[in] res raft::resources is an object mangaging resources
 * @param[in] params an instance of nn_descent::index_params that are parameters
 *               to run the nn-descent algorithm
 * @param[in] dataset raft::host_matrix_view input dataset expected to be located
 *                in host memory
 * @param[out] idx raft::neighbors::experimental::nn_descentindex containing all-neighbors knn graph
 * in host memory
 */
template <typename T, typename IdxT = uint32_t>
void build(raft::resources const& res,
           index_params const& params,
           raft::host_matrix_view<const T, int64_t, row_major> dataset,
           index<IdxT>& idx)
{
  detail::build<T, IdxT>(res, params, dataset, idx);
}

/** @} */  // end group nn-descent

}  // namespace raft::neighbors::experimental::nn_descent
