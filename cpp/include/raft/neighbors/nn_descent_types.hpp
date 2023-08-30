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

#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance_types.hpp>

namespace raft::neighbors::nn_descent {
/**
 * @ingroup nn_descent
 * @{
 */

struct index_params : ann::index_params {
  size_t intermediate_graph_degree = 128;     // Degree of input graph for pruning.
  size_t graph_degree              = 64;      // Degree of output graph.
  size_t max_iterations            = 15;      // Number of nn-descent iterations.
  float termination_threshold      = 0.0001;  // Termination threshold of nn-descent.
};

/**
 * @brief nn-descent Index
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
   * @param res raft::resources
   * @param n_rows number of rows in knn-graph
   * @param n_cols number of cols in knn-graph
   */
  index(raft::resources const& res, int64_t n_rows, int64_t n_cols)
    : ann::index(),
      res_{res},
      metric_{raft::distance::DistanceType::L2Expanded},
      graph_{raft::make_host_matrix<IdxT, int64_t, row_major>(n_rows, n_cols)}
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
    return graph_.view().extent(0);
  }

  /** Graph degree */
  [[nodiscard]] constexpr inline auto graph_degree() const noexcept -> uint32_t
  {
    return graph_.view().extent(1);
  }

  /** neighborhood graph [size, graph-degree] */
  [[nodiscard]] inline auto graph() noexcept -> host_matrix_view<IdxT, int64_t, row_major>
  {
    return graph_.view();
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
};

/** @} */

}  // namespace raft::neighbors::nn_descent
