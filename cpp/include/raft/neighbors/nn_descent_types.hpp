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

#include "ann_types.hpp"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance_types.hpp>

#include <optional>

namespace raft::neighbors::experimental::nn_descent {
/**
 * @ingroup nn-descent
 * @{
 */

/**
 * @brief Parameters used to build an nn-descent index
 *
 * `graph_degree`: For an input dataset of dimensions (N, D),
 * determines the final dimensions of the all-neighbors knn graph
 * which turns out to be of dimensions (N, graph_degree)
 * `intermediate_graph_degree`: Internally, nn-descent builds an
 * all-neighbors knn graph of dimensions (N, intermediate_graph_degree)
 * before selecting the final `graph_degree` neighbors. It's recommended
 * that `intermediate_graph_degree` >= 1.5 * graph_degree
 * `max_iterations`: The number of iterations that nn-descent will refine
 * the graph for. More iterations produce a better quality graph at cost of performance
 * `termination_threshold`: The delta at which nn-descent will terminate its iterations
 *
 */
struct index_params : ann::index_params {
  size_t graph_degree              = 64;      // Degree of output graph.
  size_t intermediate_graph_degree = 128;     // Degree of input graph for pruning.
  size_t max_iterations            = 20;      // Number of nn-descent iterations.
  float termination_threshold      = 0.0001;  // Termination threshold of nn-descent.
  bool return_distances            = false;   // return distances if true
};

/**
 * @brief nn-descent Build an nn-descent index
 * The index contains an all-neighbors graph of the input dataset
 * stored in host memory of dimensions (n_rows, n_cols)
 *
 * Reference:
 * Hui Wang, Wan-Lei Zhao, Xiangxiang Zeng, and Jianye Yang. 2021.
 * Fast k-NN Graph Construction by GPU based NN-Descent. In Proceedings of the 30th ACM
 * International Conference on Information & Knowledge Management (CIKM '21). Association for
 * Computing Machinery, New York, NY, USA, 1929–1938. https://doi.org/10.1145/3459637.3482344
 *
 * @tparam IdxT dtype to be used for constructing knn-graph
 */
template <typename IdxT>
struct index : ann::index {
 public:
  /**
   * @brief Construct a new index object
   *
   * This constructor creates an nn-descent index which is a knn-graph in host memory.
   * The type of the knn-graph is a dense raft::host_matrix and dimensions are
   * (n_rows, n_cols).
   *
   * @param res raft::resources is an object mangaging resources
   * @param n_rows number of rows in knn-graph
   * @param n_cols number of cols in knn-graph
   * @param return_distances whether to allocate and get distances information
   */
  index(raft::resources const& res, int64_t n_rows, int64_t n_cols, bool return_distances = false)
    : ann::index(),
      res_{res},
      metric_{raft::distance::DistanceType::L2Expanded},
      graph_{raft::make_host_matrix<IdxT, int64_t, row_major>(n_rows, n_cols)},
      graph_view_{graph_.view()},
      return_distances_(return_distances)
  {
    if (return_distances) {
      distances_      = raft::make_device_matrix<float, int64_t>(res_, n_rows, n_cols);
      distances_view_ = distances_.value().view();
    }
  }

  /**
   * @brief Construct a new index object
   *
   * This constructor creates an nn-descent index using a user allocated host memory knn-graph.
   * The type of the knn-graph is a dense raft::host_matrix and dimensions are
   * (n_rows, n_cols).
   *
   * @param res raft::resources is an object mangaging resources
   * @param graph_view raft::host_matrix_view<IdxT, int64_t, raft::row_major> for storing knn-graph
   * @param distances_view std::optional<raft::device_matrix_view<T, int64_t, row_major>> for
   * storing knn-graph distances
   * @param return_distances whether to allocate and get distances information
   */
  index(raft::resources const& res,
        raft::host_matrix_view<IdxT, int64_t, raft::row_major> graph_view,
        std::optional<raft::device_matrix_view<float, int64_t, row_major>> distances_view =
          std::nullopt,
        bool return_distances = false)
    : ann::index(),
      res_{res},
      metric_{raft::distance::DistanceType::L2Expanded},
      graph_{raft::make_host_matrix<IdxT, int64_t, row_major>(0, 0)},
      distances_{raft::make_device_matrix<float, int64_t>(res_, 0, 0)},
      graph_view_{graph_view},
      distances_view_(distances_view),
      return_distances_(return_distances)
  {
  }

  /** Distance metric used for clustering. */
  [[nodiscard]] constexpr inline auto metric() const noexcept -> raft::distance::DistanceType
  {
    return metric_;
  }

  // /** Total length of the index (number of vectors). */
  [[nodiscard]] constexpr inline auto size() const noexcept -> IdxT
  {
    return graph_view_.extent(0);
  }

  /** Graph degree */
  [[nodiscard]] constexpr inline auto graph_degree() const noexcept -> uint32_t
  {
    return graph_view_.extent(1);
  }

  /** neighborhood graph [size, graph-degree] */
  [[nodiscard]] inline auto graph() noexcept -> host_matrix_view<IdxT, int64_t, row_major>
  {
    return graph_view_;
  }

  /** neighborhood graph distances [size, graph-degree] */
  [[nodiscard]] inline auto distances() noexcept
    -> std::optional<device_matrix_view<float, int64_t, row_major>>
  {
    return distances_view_;
  }

  // Don't allow copying the index for performance reasons (try avoiding copying data)
  index(const index&)                    = delete;
  index(index&&)                         = default;
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index&      = default;
  ~index()                               = default;

 private:
  raft::resources const& res_;
  raft::distance::DistanceType metric_;
  raft::host_matrix<IdxT, int64_t, row_major> graph_;  // graph to return for non-int IdxT
  std::optional<raft::device_matrix<float, int64_t, row_major>> distances_;
  raft::host_matrix_view<IdxT, int64_t, row_major>
    graph_view_;  // view of graph for user provided matrix
  std::optional<raft::device_matrix_view<float, int64_t, row_major>> distances_view_;
  bool return_distances_;
};

/** @} */

}  // namespace raft::neighbors::experimental::nn_descent
