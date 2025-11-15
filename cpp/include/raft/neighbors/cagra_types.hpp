/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ann_types.hpp"
#include "dataset.hpp"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/detail/cagra/utils.hpp>
#include <raft/util/integer_utils.hpp>

#include <thrust/fill.h>

#include <memory>
#include <optional>
#include <string>
#include <type_traits>

namespace raft::neighbors::cagra {
/**
 * @addtogroup cagra
 * @{
 */

/**
 * @brief ANN algorithm used by CAGRA to build knn graph
 *
 */
enum class graph_build_algo {
  /* Use IVF-PQ to build all-neighbors knn graph */
  IVF_PQ,
  /* Experimental, use NN-Descent to build all-neighbors knn graph */
  NN_DESCENT
};

struct index_params : ann::index_params {
  /** Degree of input graph for pruning. */
  size_t intermediate_graph_degree = 128;
  /** Degree of output graph. */
  size_t graph_degree = 64;
  /** ANN algorithm to build knn graph. */
  graph_build_algo build_algo = graph_build_algo::IVF_PQ;
  /** Number of Iterations to run if building with NN_DESCENT */
  size_t nn_descent_niter = 20;
  /**
   * Specify compression params if compression is desired.
   *
   * NOTE: this is experimental new API, consider it unsafe.
   */
  std::optional<vpq_params> compression = std::nullopt;
};

enum class search_algo {
  /** For large batch sizes. */
  SINGLE_CTA,
  /** For small batch sizes. */
  MULTI_CTA,
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

  /** Number of graph nodes to select as the starting point for the search in each iteration. aka
   * search width?*/
  size_t search_width = 1;
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

  /** Number of iterations of initial random seed node selection. 1 or more. */
  uint32_t num_random_samplings = 1;
  /** Bit mask used for initial random seed node selection. */
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
 * @tparam IdxT type of the vector indices (represent dataset.extent(0))
 *
 */
template <typename T, typename IdxT>
struct index : ann::index {
  static_assert(!raft::is_narrowing_v<uint32_t, IdxT>,
                "IdxT must be able to represent all values of uint32_t");

 public:
  /** Distance metric used for clustering. */
  [[nodiscard]] constexpr inline auto metric() const noexcept -> raft::distance::DistanceType
  {
    return metric_;
  }

  /** Total length of the index (number of vectors). */
  [[nodiscard]] constexpr inline auto size() const noexcept -> IdxT
  {
    auto data_rows = dataset_->n_rows();
    return data_rows > 0 ? data_rows : graph_view_.extent(0);
  }

  /** Dimensionality of the data. */
  [[nodiscard]] constexpr inline auto dim() const noexcept -> uint32_t { return dataset_->dim(); }
  /** Graph degree */
  [[nodiscard]] constexpr inline auto graph_degree() const noexcept -> uint32_t
  {
    return graph_view_.extent(1);
  }

  /** Dataset [size, dim] */
  [[nodiscard]] inline auto data() const noexcept -> const neighbors::dataset<int64_t>&
  {
    return *dataset_;
  }

  /** neighborhood graph [size, graph-degree] */
  [[nodiscard]] inline auto graph() const noexcept
    -> device_matrix_view<const IdxT, int64_t, row_major>
  {
    return graph_view_;
  }

  // Don't allow copying the index for performance reasons (try avoiding copying data)
  index(const index&)                    = delete;
  index(index&&)                         = default;
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index&      = default;
  ~index()                               = default;

  /**
   * Replace the dataset with a new dataset.
   *
   * If the new dataset rows are aligned on 16 bytes, then only a reference is stored to the
   * dataset. It is the caller's responsibility to ensure that dataset stays alive as long as the
   * index.
   */
  void update_dataset(raft::resources const& res,
                      raft::device_matrix_view<const T, int64_t, row_major> dataset)
  {
    dataset_ = make_aligned_dataset(res, dataset, 16);
  }

  /** Set the dataset reference explicitly to a device matrix view with padding. */
  void update_dataset(raft::resources const& res,
                      raft::device_matrix_view<const T, int64_t, layout_stride> dataset)
  {
    dataset_ = make_aligned_dataset(res, dataset, 16);
  }

  /**
   * Replace the dataset with a new dataset.
   *
   * We create a copy of the dataset on the device. The index manages the lifetime of this copy.
   */
  void update_dataset(raft::resources const& res,
                      raft::host_matrix_view<const T, int64_t, row_major> dataset)
  {
    dataset_ = make_aligned_dataset(res, dataset, 16);
  }

  /** Replace the dataset with a new dataset. */
  template <typename DatasetT>
  auto update_dataset(raft::resources const& res, DatasetT&& dataset)
    -> std::enable_if_t<std::is_base_of_v<neighbors::dataset<int64_t>, DatasetT>>
  {
    dataset_ = std::make_unique<DatasetT>(std::move(dataset));
  }

  template <typename DatasetT>
  auto update_dataset(raft::resources const& res, std::unique_ptr<DatasetT>&& dataset)
    -> std::enable_if_t<std::is_base_of_v<neighbors::dataset<int64_t>, DatasetT>>
  {
    dataset_ = std::move(dataset);
  }

  /**
   * Replace the graph with a new graph.
   *
   * Since the new graph is a device array, we store a reference to that, and it is
   * the caller's responsibility to ensure that knn_graph stays alive as long as the index.
   */
  void update_graph(raft::resources const& res,
                    raft::device_matrix_view<const IdxT, int64_t, row_major> knn_graph)
  {
    graph_view_ = knn_graph;
  }

  /**
   * Replace the graph with a new graph.
   *
   * We create a copy of the graph on the device. The index manages the lifetime of this copy.
   */
  void update_graph(raft::resources const& res,
                    raft::host_matrix_view<const IdxT, int64_t, row_major> knn_graph)
  {
    RAFT_LOG_DEBUG("Copying CAGRA knn graph from host to device");
    if ((graph_.extent(0) != knn_graph.extent(0)) || (graph_.extent(1) != knn_graph.extent(1))) {
      // clear existing memory before allocating to prevent OOM errors on large graphs
      if (graph_.size()) { graph_ = make_device_matrix<IdxT, int64_t>(res, 0, 0); }
      graph_ = make_device_matrix<IdxT, int64_t>(res, knn_graph.extent(0), knn_graph.extent(1));
    }
    raft::copy(graph_.data_handle(),
               knn_graph.data_handle(),
               knn_graph.size(),
               resource::get_cuda_stream(res));
    graph_view_ = graph_.view();
  }

 private:
  raft::distance::DistanceType metric_;
  raft::device_matrix<IdxT, int64_t, row_major> graph_;
  raft::device_matrix_view<const IdxT, int64_t, row_major> graph_view_;
  std::unique_ptr<neighbors::dataset<int64_t>> dataset_;
};

/** @} */

}  // namespace raft::neighbors::cagra
