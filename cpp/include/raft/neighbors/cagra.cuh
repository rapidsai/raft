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

#include "detail/cagra/cagra_build.cuh"
#include "detail/cagra/cagra_search.cuh"
#include "detail/cagra/graph_core.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/neighbors/cagra_types.hpp>
#include <rmm/cuda_stream_view.hpp>

namespace raft::neighbors::experimental::cagra {

/**
 * @defgroup cagra CUDA ANN Graph-based nearest neighbor search
 * @{
 */

/**
 * @brief Build a kNN graph.
 *
 * The kNN graph is the first building block for CAGRA index.
 * This function uses the IVF-PQ method to build a kNN graph.
 *
 * The output is a dense matrix that stores the neighbor indices for each pont in the dataset.
 * Each point has the same number of neighbors.
 *
 * See [cagra::build](#cagra::build) for an alternative method.
 *
 * The following distance metrics are supported:
 * - L2Expanded
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace raft::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params build_params;
 *   ivf_pq::search_params search_params
 *   auto knn_graph      = raft::make_host_matrix<IdxT, IdxT>(dataset.extent(0), 128);
 *   // create knn graph
 *   cagra::build_knn_graph(res, dataset, knn_graph.view(), 2, build_params, search_params);
 *   auto pruned_gaph      = raft::make_host_matrix<IdxT, IdxT>(dataset.extent(0), 64);
 *   cagra::prune(res, dataset, knn_graph.view(), pruned_graph.view());
 *   // Construct an index from dataset and pruned knn_graph
 *   auto index = cagra::index<T, IdxT>(res, build_params.metric(), dataset, pruned_graph.view());
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] res raft resources
 * @param[in] dataset a matrix view (host or device) to a row-major matrix [n_rows, dim]
 * @param[out] knn_graph a host matrix view to store the output knn graph [n_rows, graph_degree]
 * @param[in] refine_rate refinement rate for ivf-pq search
 * @param[in] build_params (optional) ivf_pq index building parameters for knn graph
 * @param[in] search_params (optional) ivf_pq search parameters
 */
template <typename DataT, typename IdxT, typename accessor>
void build_knn_graph(raft::device_resources const& res,
                     mdspan<const DataT, matrix_extent<IdxT>, row_major, accessor> dataset,
                     raft::host_matrix_view<IdxT, IdxT, row_major> knn_graph,
                     std::optional<float> refine_rate                   = std::nullopt,
                     std::optional<ivf_pq::index_params> build_params   = std::nullopt,
                     std::optional<ivf_pq::search_params> search_params = std::nullopt)
{
  detail::build_knn_graph(res, dataset, knn_graph, refine_rate, build_params, search_params);
}

/**
 * @brief Prune a KNN graph.
 *
 * Decrease the number of neighbors for each node.
 *
 * See [cagra::build_knn_graph](#cagra::build_knn_graph) for usage example
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] res raft resources
 * @param[in] dataset a matrix view (host or device) to a row-major matrix [n_rows, dim]
 * @param[in] knn_graph a matrix view (host or device) of the input knn graph [n_rows,
 * knn_graph_degree]
 * @param[out] new_graph a host matrix view of the pruned knn graph [n_rows, graph_degree]
 */
template <class DATA_T,
          typename IdxT = uint32_t,
          typename d_accessor =
            host_device_accessor<std::experimental::default_accessor<DATA_T>, memory_type::device>,
          typename g_accessor =
            host_device_accessor<std::experimental::default_accessor<DATA_T>, memory_type::host>>
void prune(raft::device_resources const& res,
           mdspan<const DATA_T, matrix_extent<IdxT>, row_major, d_accessor> dataset,
           mdspan<IdxT, matrix_extent<IdxT>, row_major, g_accessor> knn_graph,
           raft::host_matrix_view<IdxT, IdxT, row_major> new_graph)
{
  detail::graph::prune(res, dataset, knn_graph, new_graph);
}

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * The build consist of two steps: build an intermediate knn-graph, and prune it to
 * create the final graph. The index_params struct controls the node degree of these
 * graphs.
 *
 * It is required that dataset and the pruned graph fit the GPU memory.
 *
 * To customize the parameters for knn-graph building and pruning, and to reuse the
 * intermediate results, you could build the index in two steps using
 * [cagra::build_knn_graph](#cagra::build_knn_graph) and [cagra::prune](#cagra::prune).
 *
 * The following distance metrics are supported:
 * - L2
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace raft::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *   // use default search parameters
 *   ivf_pq::search_params search_params;
 *   // search K nearest neighbours
 *   auto neighbors = raft::make_device_matrix<uint32_t>(res, n_queries, k);
 *   auto distances = raft::make_device_matrix<float>(res, n_queries, k);
 *   ivf_pq::search(res, search_params, index, queries, neighbors, distances);
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] res
 * @param[in] params parameters for building the index
 * @param[in] dataset a matrix view (host or device) to a row-major matrix [n_rows, dim]
 *
 * @return the constructed cagra index
 */
template <typename T,
          typename IdxT = uint32_t,
          typename Accessor =
            host_device_accessor<std::experimental::default_accessor<T>, memory_type::host>>
index<T, IdxT> build(raft::device_resources const& res,
                     const index_params& params,
                     mdspan<const T, matrix_extent<IdxT>, row_major, Accessor> dataset)
{
  size_t degree = params.intermediate_graph_degree;
  if (degree >= dataset.extent(0)) {
    RAFT_LOG_WARN(
      "Intermediate graph degree cannot be larger than dataset size, reducing it to %lu",
      dataset.extent(0));
    degree = dataset.extent(0) - 1;
  }
  RAFT_EXPECTS(degree >= params.graph_degree,
               "Intermediate graph degree cannot be smaller than final graph degree");

  auto knn_graph = raft::make_host_matrix<IdxT, IdxT>(dataset.extent(0), degree);

  build_knn_graph(res, dataset, knn_graph.view());

  auto cagra_graph = raft::make_host_matrix<IdxT, IdxT>(dataset.extent(0), params.graph_degree);

  prune<T, IdxT>(res, dataset, knn_graph.view(), cagra_graph.view());

  // Construct an index from dataset and pruned knn graph.
  return index<T, IdxT>(res, params.metric, dataset, cagra_graph.view());
}

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] idx cagra index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 */
template <typename T, typename IdxT>
void search(raft::device_resources const& res,
            const search_params& params,
            const index<T, IdxT>& idx,
            raft::device_matrix_view<const T, IdxT, row_major> queries,
            raft::device_matrix_view<IdxT, IdxT, row_major> neighbors,
            raft::device_matrix_view<float, IdxT, row_major> distances)
{
  RAFT_EXPECTS(
    queries.extent(0) == neighbors.extent(0) && queries.extent(0) == distances.extent(0),
    "Number of rows in output neighbors and distances matrices must equal the number of queries.");

  RAFT_EXPECTS(neighbors.extent(1) == distances.extent(1),
               "Number of columns in output neighbors and distances matrices must equal k");

  RAFT_EXPECTS(queries.extent(1) == idx.dim(),
               "Number of query dimensions should equal number of dimensions in the index.");

  detail::search_main(res, params, idx, queries, neighbors, distances);
}
/** @} */  // end group cagra

}  // namespace raft::neighbors::experimental::cagra
