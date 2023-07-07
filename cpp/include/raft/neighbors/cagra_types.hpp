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

namespace raft::neighbors::experimental::cagra {
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
    return graph_.extent(1);
  }

  /** Dataset [size, dim] */
  [[nodiscard]] inline auto dataset() const noexcept
    -> device_matrix_view<const T, IdxT, layout_stride>
  {
    return dataset_view_;
  }

  /** neighborhood graph [size, graph-degree] */
  inline auto graph() noexcept -> device_matrix_view<IdxT, IdxT, row_major>
  {
    return graph_.view();
  }

  [[nodiscard]] inline auto graph() const noexcept
    -> device_matrix_view<const IdxT, IdxT, row_major>
  {
    return graph_.view();
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

  /** Construct an index from dataset and knn_graph arrays */
  template <typename data_accessor, typename graph_accessor>
  index(raft::resources const& res,
        raft::distance::DistanceType metric,
        mdspan<const T, matrix_extent<IdxT>, row_major, data_accessor> dataset,
        mdspan<IdxT, matrix_extent<IdxT>, row_major, graph_accessor> knn_graph)
    : ann::index(),
      metric_(metric),
      dataset_(
        make_device_matrix<T, IdxT>(res, dataset.extent(0), AlignDim::roundUp(dataset.extent(1)))),
      graph_(make_device_matrix<IdxT, IdxT>(res, knn_graph.extent(0), knn_graph.extent(1)))
  {
    RAFT_EXPECTS(dataset.extent(0) == knn_graph.extent(0),
                 "Dataset and knn_graph must have equal number of rows");
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
    dataset_view_ = make_device_strided_matrix_view<T, IdxT>(
      dataset_.data_handle(), dataset_.extent(0), dataset.extent(1), dataset_.extent(1));
    RAFT_LOG_DEBUG("CAGRA dataset strided matrix view %zux%zu, stride %zu",
                   static_cast<size_t>(dataset_view_.extent(0)),
                   static_cast<size_t>(dataset_view_.extent(1)),
                   static_cast<size_t>(dataset_view_.stride(0)));
    raft::copy(graph_.data_handle(),
               knn_graph.data_handle(),
               knn_graph.size(),
               resource::get_cuda_stream(res));
    resource::sync_stream(res);
  }

 private:
  raft::distance::DistanceType metric_;
  raft::device_matrix<T, IdxT, row_major> dataset_;
  raft::device_matrix<IdxT, IdxT, row_major> graph_;
  raft::device_matrix_view<T, IdxT, layout_stride> dataset_view_;
};

/** @} */

}  // namespace raft::neighbors::experimental::cagra
