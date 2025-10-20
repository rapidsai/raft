/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/cagra_types.hpp>
#include <raft/neighbors/dataset.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace raft::neighbors::cagra {

/**
 * @defgroup cagra CUDA ANN Graph-based nearest neighbor search
 * @{
 */

/**
 * @brief Build a kNN graph using IVF-PQ.
 *
 * The kNN graph is the first building block for CAGRA index.
 *
 * The output is a dense matrix that stores the neighbor indices for each point in the dataset.
 * Each point has the same number of neighbors.
 *
 * See [cagra::build](#cagra::build) for an alternative method.
 *
 * The following distance metrics are supported:
 * - L2Expanded
 * - InnerProduct
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace raft::neighbors;
 *   // use default index parameters based on shape of the dataset
 *   ivf_pq::index_params build_params = ivf_pq::index_params::from_dataset(dataset);
 *   ivf_pq::search_params search_params;
 *   auto knn_graph      = raft::make_host_matrix<IdxT, IdxT>(dataset.extent(0), 128);
 *   // create knn graph
 *   cagra::build_knn_graph(res, dataset, knn_graph.view(), 2, build_params, search_params);
 *   auto optimized_gaph = raft::make_host_matrix<IdxT, IdxT>(dataset.extent(0), 64);
 *   cagra::optimize(res, dataset, knn_graph.view(), optimized_graph.view());
 *   // Construct an index from dataset and optimized knn_graph
 *   auto index = cagra::index<T, IdxT>(res, build_params.metric(), dataset,
 *                                      optimized_graph.view());
 * @endcode
 *
 * @tparam DataT data element type
 * @tparam IdxT type of the dataset vector indices
 *
 * @param[in] res raft resources
 * @param[in] dataset a matrix view (host or device) to a row-major matrix [n_rows, dim]
 * @param[out] knn_graph a host matrix view to store the output knn graph [n_rows, graph_degree]
 * @param[in] refine_rate (optional) refinement rate for ivf-pq search
 * @param[in] build_params (optional) ivf_pq index building parameters for knn graph
 * @param[in] search_params (optional) ivf_pq search parameters
 */
template <typename DataT, typename IdxT, typename accessor>
void build_knn_graph(raft::resources const& res,
                     mdspan<const DataT, matrix_extent<int64_t>, row_major, accessor> dataset,
                     raft::host_matrix_view<IdxT, int64_t, row_major> knn_graph,
                     std::optional<float> refine_rate                   = std::nullopt,
                     std::optional<ivf_pq::index_params> build_params   = std::nullopt,
                     std::optional<ivf_pq::search_params> search_params = std::nullopt)
{
  using internal_IdxT = typename std::make_unsigned<IdxT>::type;

  auto knn_graph_internal = make_host_matrix_view<internal_IdxT, int64_t>(
    reinterpret_cast<internal_IdxT*>(knn_graph.data_handle()),
    knn_graph.extent(0),
    knn_graph.extent(1));
  auto dataset_internal = mdspan<const DataT, matrix_extent<int64_t>, row_major, accessor>(
    dataset.data_handle(), dataset.extent(0), dataset.extent(1));

  cagra::detail::build_knn_graph(
    res, dataset_internal, knn_graph_internal, refine_rate, build_params, search_params);
}

/**
 * @brief Build a kNN graph using NN-descent.
 *
 * The kNN graph is the first building block for CAGRA index.
 *
 * The output is a dense matrix that stores the neighbor indices for each point in the dataset.
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
 *   using namespace raft::neighbors::experimental;
 *   // use default index parameters
 *   nn_descent::index_params build_params;
 *   build_params.graph_degree = 128;
 *   auto knn_graph      = raft::make_host_matrix<IdxT, IdxT>(dataset.extent(0), 128);
 *   // create knn graph
 *   cagra::build_knn_graph(res, dataset, knn_graph.view(), build_params);
 *   auto optimized_gaph      = raft::make_host_matrix<IdxT, int64_t>(dataset.extent(0), 64);
 *   cagra::optimize(res, dataset, nn_descent_index.graph.view(), optimized_graph.view());
 *   // Construct an index from dataset and optimized knn_graph
 *   auto index = cagra::index<T, IdxT>(res, build_params.metric(), dataset,
 * optimized_graph.view());
 * @endcode
 *
 * @tparam DataT data element type
 * @tparam IdxT type of the dataset vector indices
 * @tparam accessor host or device accessor_type for the dataset
 * @param[in] res raft::resources is an object mangaging resources
 * @param[in] dataset input raft::host/device_matrix_view that can be located in
 *                in host or device memory
 * @param[out] knn_graph a host matrix view to store the output knn graph [n_rows, graph_degree]
 * @param[in] build_params an instance of experimental::nn_descent::index_params that are parameters
 *                     to run the nn-descent algorithm
 */
template <typename DataT,
          typename IdxT = uint32_t,
          typename accessor =
            host_device_accessor<std::experimental::default_accessor<DataT>, memory_type::device>>
void build_knn_graph(raft::resources const& res,
                     mdspan<const DataT, matrix_extent<int64_t>, row_major, accessor> dataset,
                     raft::host_matrix_view<IdxT, int64_t, row_major> knn_graph,
                     experimental::nn_descent::index_params build_params)
{
  detail::build_knn_graph<DataT, IdxT>(res, dataset, knn_graph, build_params);
}

/**
 * @brief Sort a KNN graph index.
 * Preprocessing step for `cagra::optimize`: If a KNN graph is not built using
 * `cagra::build_knn_graph`, then it is necessary to call this function before calling
 * `cagra::optimize`. If the graph is built by `cagra::build_knn_graph`, it is already sorted and
 * you do not need to call this function.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace raft::neighbors;
 *   cagra::index_params build_params;
 *   auto knn_graph = raft::make_host_matrix<IdxT, IdxT>(dataset.extent(0), 128);
 *   // build KNN graph not using `cagra::build_knn_graph`
 *   // build(knn_graph, dataset, ...);
 *   // sort graph index
 *   sort_knn_graph(res, dataset.view(), knn_graph.view());
 *   // optimize graph
 *   cagra::optimize(res, dataset, knn_graph.view(), optimized_graph.view());
 *   // Construct an index from dataset and optimized knn_graph
 *   auto index = cagra::index<T, IdxT>(res, build_params.metric(), dataset,
 *                                      optimized_graph.view());
 * @endcode
 *
 * @tparam DataT type of the data in the source dataset
 * @tparam IdxT type of the dataset vector indices
 *
 * @param[in] res raft resources
 * @param[in] dataset a matrix view (host or device) to a row-major matrix [n_rows, dim]
 * @param[in,out] knn_graph a matrix view (host or device) of the input knn graph [n_rows,
 * knn_graph_degree]
 */
template <typename DataT,
          typename IdxT = uint32_t,
          typename d_accessor =
            host_device_accessor<std::experimental::default_accessor<DataT>, memory_type::device>,
          typename g_accessor =
            host_device_accessor<std::experimental::default_accessor<IdxT>, memory_type::host>>
void sort_knn_graph(raft::resources const& res,
                    mdspan<const DataT, matrix_extent<int64_t>, row_major, d_accessor> dataset,
                    mdspan<IdxT, matrix_extent<int64_t>, row_major, g_accessor> knn_graph)
{
  using internal_IdxT = typename std::make_unsigned<IdxT>::type;

  using g_accessor_internal =
    host_device_accessor<std::experimental::default_accessor<internal_IdxT>, g_accessor::mem_type>;
  auto knn_graph_internal =
    mdspan<internal_IdxT, matrix_extent<int64_t>, row_major, g_accessor_internal>(
      reinterpret_cast<internal_IdxT*>(knn_graph.data_handle()),
      knn_graph.extent(0),
      knn_graph.extent(1));

  auto dataset_internal = mdspan<const DataT, matrix_extent<int64_t>, row_major, d_accessor>(
    dataset.data_handle(), dataset.extent(0), dataset.extent(1));

  cagra::detail::graph::sort_knn_graph(res, dataset_internal, knn_graph_internal);
}

/**
 * @brief Prune a KNN graph.
 *
 * Decrease the number of neighbors for each node.
 *
 * See [cagra::build_knn_graph](#cagra::build_knn_graph) for usage example
 *
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] res raft resources
 * @param[in] knn_graph a matrix view (host or device) of the input knn graph [n_rows,
 * knn_graph_degree]
 * @param[out] new_graph a host matrix view of the optimized knn graph [n_rows, graph_degree]
 */
template <typename IdxT = uint32_t,
          typename g_accessor =
            host_device_accessor<std::experimental::default_accessor<IdxT>, memory_type::host>>
void optimize(raft::resources const& res,
              mdspan<IdxT, matrix_extent<int64_t>, row_major, g_accessor> knn_graph,
              raft::host_matrix_view<IdxT, int64_t, row_major> new_graph)
{
  detail::optimize(res, knn_graph, new_graph);
}

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * The build consist of two steps: build an intermediate knn-graph, and optimize it to
 * create the final graph. The index_params struct controls the node degree of these
 * graphs.
 *
 * It is required that dataset and the optimized graph fit the GPU memory.
 *
 * To customize the parameters for knn-graph building and pruning, and to reuse the
 * intermediate results, you could build the index in two steps using
 * [cagra::build_knn_graph](#cagra::build_knn_graph) and [cagra::optimize](#cagra::optimize).
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
 *   cagra::search_params search_params;
 *   // search K nearest neighbours
 *   auto neighbors = raft::make_device_matrix<uint32_t>(res, n_queries, k);
 *   auto distances = raft::make_device_matrix<float>(res, n_queries, k);
 *   cagra::search(res, search_params, index, queries, neighbors, distances);
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
index<T, IdxT> build(raft::resources const& res,
                     const index_params& params,
                     mdspan<const T, matrix_extent<int64_t>, row_major, Accessor> dataset)
{
  return detail::build<T, IdxT, Accessor>(res, params, dataset);
}

/**
 * @brief Search ANN using the constructed index with the given sample filter.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace raft::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *   // use default search parameters
 *   cagra::search_params search_params;
 *   // create a bitset to filter the search
 *   auto removed_indices = raft::make_device_vector<IdxT>(res, n_removed_indices);
 *   raft::core::bitset<std::uint32_t, IdxT> removed_indices_bitset(
 *     res, removed_indices.view(), dataset.extent(0));
 *   // search K nearest neighbours according to a bitset
 *   auto neighbors = raft::make_device_matrix<uint32_t>(res, n_queries, k);
 *   auto distances = raft::make_device_matrix<float>(res, n_queries, k);
 *   cagra::search_with_filtering(res, search_params, index, queries, neighbors, distances,
 *     filtering::bitset_filter(removed_indices_bitset.view()));
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices
 * @tparam CagraSampleFilterT Device filter function, with the signature
 *         `(uint32_t query ix, uint32_t sample_ix) -> bool`
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] idx cagra index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter a device filter function that greenlights samples for a given query
 */
template <typename T, typename IdxT, typename CagraSampleFilterT>
void search_with_filtering(raft::resources const& res,
                           const search_params& params,
                           const index<T, IdxT>& idx,
                           raft::device_matrix_view<const T, int64_t, row_major> queries,
                           raft::device_matrix_view<IdxT, int64_t, row_major> neighbors,
                           raft::device_matrix_view<float, int64_t, row_major> distances,
                           CagraSampleFilterT sample_filter = CagraSampleFilterT())
{
  RAFT_EXPECTS(
    queries.extent(0) == neighbors.extent(0) && queries.extent(0) == distances.extent(0),
    "Number of rows in output neighbors and distances matrices must equal the number of queries.");

  RAFT_EXPECTS(neighbors.extent(1) == distances.extent(1),
               "Number of columns in output neighbors and distances matrices must equal k");
  RAFT_EXPECTS(queries.extent(1) == idx.dim(),
               "Number of query dimensions should equal number of dimensions in the index.");

  using internal_IdxT   = typename std::make_unsigned<IdxT>::type;
  auto queries_internal = raft::make_device_matrix_view<const T, int64_t, row_major>(
    queries.data_handle(), queries.extent(0), queries.extent(1));
  auto neighbors_internal = raft::make_device_matrix_view<internal_IdxT, int64_t, row_major>(
    reinterpret_cast<internal_IdxT*>(neighbors.data_handle()),
    neighbors.extent(0),
    neighbors.extent(1));
  auto distances_internal = raft::make_device_matrix_view<float, int64_t, row_major>(
    distances.data_handle(), distances.extent(0), distances.extent(1));

  return cagra::detail::search_main<T, internal_IdxT, CagraSampleFilterT, IdxT>(
    res, params, idx, queries_internal, neighbors_internal, distances_internal, sample_filter);
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
void search(raft::resources const& res,
            const search_params& params,
            const index<T, IdxT>& idx,
            raft::device_matrix_view<const T, int64_t, row_major> queries,
            raft::device_matrix_view<IdxT, int64_t, row_major> neighbors,
            raft::device_matrix_view<float, int64_t, row_major> distances)
{
  using none_filter_type = raft::neighbors::filtering::none_cagra_sample_filter;
  return cagra::search_with_filtering<T, IdxT, none_filter_type>(
    res, params, idx, queries, neighbors, distances, none_filter_type{});
}

/** @} */  // end group cagra

}  // namespace raft::neighbors::cagra

// TODO: Remove deprecated experimental namespace in 23.12 release
namespace raft::neighbors::experimental::cagra {
using raft::neighbors::cagra::build;
using raft::neighbors::cagra::build_knn_graph;
using raft::neighbors::cagra::optimize;
using raft::neighbors::cagra::search;
using raft::neighbors::cagra::sort_knn_graph;
}  // namespace raft::neighbors::experimental::cagra
