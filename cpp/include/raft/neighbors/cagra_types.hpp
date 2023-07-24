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

#include "ann_types.hpp"
#include <raft/core/resource/cuda_stream.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/util/integer_utils.hpp>
#include <raft/util/pow2_utils.cuh>

#include <memory>
#include <optional>
#include <string>
#include <thrust/fill.h>
#include <type_traits>

#include <raft/core/logger.hpp>
namespace raft::neighbors::cagra {
/**
 * @ingroup cagra
 * @{
 */

struct index_params : ann::index_params {
  size_t intermediate_graph_degree = 128;  // Degree of input graph for pruning.
  size_t graph_degree              = 64;   // Degree of output graph.
};

enum class search_algo {
  SINGLE_CTA,  // for large batch
  MULTI_CTA,   // for small batch
  MULTI_KERNEL,
  AUTO
};

enum class hash_mode { HASH, SMALL, AUTO };

struct search_params : ann::search_params {
  /** Maximum number of queries to search at the same time (batch size). Auto select when 0.*/
  size_t max_queries = 0;

  /** Number of intermediate search results retained during the search.
   *
   *  This is the main knob to adjust trade off between accuracy and search speed.
   *  Higher values improve the search accuracy.
   */
  size_t itopk_size = 64;

  /** Upper limit of search iterations. Auto select when 0.*/
  size_t max_iterations = 0;

  // In the following we list additional search parameters for fine tuning.
  // Reasonable default values are automatically chosen.

  /** Which search implementation to use. */
  search_algo algo = search_algo::AUTO;

  /** Number of threads used to calculate a single distance. 4, 8, 16, or 32. */
  size_t team_size = 0;

  /*/ Number of graph nodes to select as the starting point for the search in each iteration. aka
   * search width?*/
  size_t num_parents = 1;
  /** Lower limit of search iterations. */
  size_t min_iterations = 0;

  /** Thread block size. 0, 64, 128, 256, 512, 1024. Auto selection when 0. */
  size_t thread_block_size = 0;
  /** Hashmap type. Auto selection when AUTO. */
  hash_mode hashmap_mode = hash_mode::AUTO;
  /** Lower limit of hashmap bit length. More than 8. */
  size_t hashmap_min_bitlen = 0;
  /** Upper limit of hashmap fill rate. More than 0.1, less than 0.9.*/
  float hashmap_max_fill_rate = 0.5;

  /* Number of iterations of initial random seed node selection. 1 or more. */
  uint32_t num_random_samplings = 1;
  // Bit mask used for initial random seed node selection. */
  uint64_t rand_xor_mask = 0x128394;
};

static_assert(std::is_aggregate_v<index_params>);
static_assert(std::is_aggregate_v<search_params>);

/**
 * @brief CAGRA index.
 *
 * The index stores the dataset and a kNN graph in device memory.
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 */
template <typename T, typename IdxT>
struct index : ann::index {
  using AlignDim = raft::Pow2<16 / sizeof(T)>;
  static_assert(!raft::is_narrowing_v<uint32_t, IdxT>,
                "IdxT must be able to represent all values of uint32_t");

 public:
  /** Distance metric used for clustering. */
  [[nodiscard]] constexpr inline auto metric() const noexcept -> raft::distance::DistanceType
  {
    return metric_;
  }

  // /** Total length of the index. */
  [[nodiscard]] constexpr inline auto size() const noexcept -> IdxT
  {
    return dataset_view_.extent(0);
  }

  /** Dimensionality of the data. */
  [[nodiscard]] constexpr inline auto dim() const noexcept -> uint32_t
  {
    return dataset_view_.extent(1);
  }
  /** Graph degree */
  [[nodiscard]] constexpr inline auto graph_degree() const noexcept -> uint32_t
  {
    return graph_view_.extent(1);
  }

  /** Dataset [size, dim] */
  [[nodiscard]] inline auto dataset() const noexcept
    -> device_matrix_view<const T, IdxT, layout_stride>
  {
    return dataset_view_;
  }

  /** neighborhood graph [size, graph-degree] */
  inline auto graph() noexcept -> device_matrix_view<IdxT, IdxT, row_major> { return graph_view_; }

  [[nodiscard]] inline auto graph() const noexcept
    -> device_matrix_view<const IdxT, IdxT, row_major>
  {
    return graph_view_;
  }

  // Don't allow copying the index for performance reasons (try avoiding copying data)
  index(const index&)                    = delete;
  index(index&&)                         = default;
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index&      = default;
  ~index()                               = default;

  /** Construct an empty index. */
  index(raft::resources const& res)
    : ann::index(),
      metric_(raft::distance::DistanceType::L2Expanded),
      dataset_(make_device_matrix<T, IdxT>(res, 0, 0)),
      graph_(make_device_matrix<IdxT, IdxT>(res, 0, 0))
  {
  }

  /** Construct an index from dataset and knn_graph arrays
   *
   * If the dataset and graph is already in GPU memory, then the index is just a thin wrapper around
   * these that stores a non-owning a reference to the arrays.
   *
   * The constructor also accepts host arrays. In that case they are copied to the device, and the
   * device arrays will be owned by the index.
   *
   * In case the dasates rows are not 16 bytes aligned, then we create a padded copy in device
   * memory to ensure alignment for vectorized load.
   *
   * Usage examples:
   *
   * - Cagra index is normally created by the cagra::build
   * @code{.cpp}
   *   using namespace raft::neighbors::experimental;
   *   auto dataset = raft::make_host_matrix<float>(n_rows, n_cols);
   *   load_dataset(dataset.view());
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
   *   In the above example, we have passed a host dataset to build. The returned index will own a
   * device copy of the dataset and the knn_graph. In contrast, if we pass the dataset as a
   * device_mdspan to build, then it will only store a reference to it.
   *
   * - Constructing index using existing knn-graph
   * @code{.cpp}
   *   using namespace raft::neighbors::experimental;
   *
   *   auto dataset = raft::make_device_matrix<float>(res, n_rows, n_cols);
   *   auto knn_graph = raft::make_device_matrix<uint32_n>(res, n_rows, graph_degree);
   *
   *   // custom loading and graph creation
   *   // load_dataset(dataset.view());
   *   // create_knn_graph(knn_graph.view());
   *
   *   // Wrap the existing device arrays into an index structure
   *   cagra::index<T, IdxT> index(res, metric, raft::make_const_mdspan(dataset.view()),
   *                               raft::make_const_mdspan(knn_graph.view()));
   *
   *   // Both knn_graph and dataset objects have to be in scope while the index is used because
   *   // the index only stores a reference to these.
   *   cagra::search(res, search_params, index, queries, neighbors, distances);
   * @endcode
   *
   */
  template <typename data_accessor, typename graph_accessor>
  index(raft::resources const& res,
        raft::distance::DistanceType metric,
        mdspan<const T, matrix_extent<IdxT>, row_major, data_accessor> dataset,
        mdspan<const IdxT, matrix_extent<IdxT>, row_major, graph_accessor> knn_graph)
    : ann::index(),
      metric_(metric),
      dataset_(make_device_matrix<T, IdxT>(res, 0, 0)),
      graph_(make_device_matrix<IdxT, IdxT>(res, 0, 0))
  {
    RAFT_EXPECTS(dataset.extent(0) == knn_graph.extent(0),
                 "Dataset and knn_graph must have equal number of rows");
    update_dataset(res, dataset);
    update_graph(res, knn_graph);
    resource::sync_stream(res);
  }

  /**
   * Replace the dataset with a new dataset.
   *
   * If the new dataset rows are aligned on 16 bytes, then only a reference is stored to the
   * dataset. It is the caller's responsibility to ensure that dataset stays alive as long as the
   * index.
   */
  void update_dataset(raft::resources const& res,
                      raft::device_matrix_view<const T, IdxT, row_major> dataset)
  {
    if (dataset.extent(1) % AlignDim::Value != 0) {
      RAFT_LOG_DEBUG("Creating a padded copy of CAGRA dataset in device memory");
      copy_padded(res, dataset);
    } else {
      dataset_view_ = make_device_strided_matrix_view<const T, IdxT>(
        dataset.data_handle(), dataset.extent(0), dataset.extent(1), dataset.extent(1));
    }
  }

  /**
   * Replace the dataset with a new dataset.
   *
   * We create a copy of the dataset on the device. The index manages the lifetime of this copy.
   */
  void update_dataset(raft::resources const& res,
                      raft::host_matrix_view<const T, IdxT, row_major> dataset)
  {
    RAFT_LOG_DEBUG("Copying CAGRA dataset from host to device");
    copy_padded(res, dataset);
  }

  /**
   * Replace the graph with a new graph.
   *
   * Since the new graph is a device array, we store a reference to that, and it is
   * the caller's responsibility to ensure that knn_graph stays alive as long as the index.
   */
  void update_graph(raft::resources const& res,
                    raft::device_matrix_view<const IdxT, IdxT, row_major> knn_graph)
  {
    graph_view_ = knn_graph;
  }

  /**
   * Replace the graph with a new graph.
   *
   * We create a copy of the graph on the device. The index manages the lifetime of this copy.
   */
  void update_graph(raft::resources const& res,
                    raft::host_matrix_view<const IdxT, IdxT, row_major> knn_graph)
  {
    RAFT_LOG_DEBUG("Copying CAGRA knn graph from host to device");
    graph_ = make_device_matrix<IdxT, IdxT>(res, knn_graph.extent(0), knn_graph.extent(1));
    raft::copy(graph_.data_handle(),
               knn_graph.data_handle(),
               knn_graph.size(),
               resource::get_cuda_stream(res));
    graph_view_ = graph_.view();
  }

 private:
  /** Create a device copy of the dataset, and pad it if necessary. */
  template <typename data_accessor>
  void copy_padded(raft::resources const& res,
                   mdspan<const T, matrix_extent<IdxT>, row_major, data_accessor> dataset)
  {
    dataset_ =
      make_device_matrix<T, IdxT>(res, dataset.extent(0), AlignDim::roundUp(dataset.extent(1)));
    if (dataset_.extent(1) == dataset.extent(1)) {
      raft::copy(dataset_.data_handle(),
                 dataset.data_handle(),
                 dataset.size(),
                 resource::get_cuda_stream(res));
    } else {
      // copy with padding
      RAFT_CUDA_TRY(cudaMemsetAsync(
        dataset_.data_handle(), 0, dataset_.size() * sizeof(T), resource::get_cuda_stream(res)));
      RAFT_CUDA_TRY(cudaMemcpy2DAsync(dataset_.data_handle(),
                                      sizeof(T) * dataset_.extent(1),
                                      dataset.data_handle(),
                                      sizeof(T) * dataset.extent(1),
                                      sizeof(T) * dataset.extent(1),
                                      dataset.extent(0),
                                      cudaMemcpyDefault,
                                      resource::get_cuda_stream(res)));
    }
    dataset_view_ = make_device_strided_matrix_view<const T, IdxT>(
      dataset_.data_handle(), dataset_.extent(0), dataset.extent(1), dataset_.extent(1));
    RAFT_LOG_DEBUG("CAGRA dataset strided matrix view %zux%zu, stride %zu",
                   static_cast<size_t>(dataset_view_.extent(0)),
                   static_cast<size_t>(dataset_view_.extent(1)),
                   static_cast<size_t>(dataset_view_.stride(0)));
  }

  raft::distance::DistanceType metric_;
  raft::device_matrix<T, IdxT, row_major> dataset_;
  raft::device_matrix<IdxT, IdxT, row_major> graph_;
  raft::device_matrix_view<const T, IdxT, layout_stride> dataset_view_;
  raft::device_matrix_view<const IdxT, IdxT, row_major> graph_view_;
};

/** @} */

}  // namespace raft::neighbors::cagra

// TODO: Remove deprecated experimental namespace in 23.12 release
namespace raft::neighbors::experimental::cagra {
using raft::neighbors::cagra::hash_mode;
using raft::neighbors::cagra::index;
using raft::neighbors::cagra::index_params;
using raft::neighbors::cagra::search_algo;
using raft::neighbors::cagra::search_params;
}  // namespace raft::neighbors::experimental::cagra
