/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "../ann_common.h"
#include "ann_ivf_flat_kernel.cuh"
#include "ann_kmeans_balanced.cuh"
#include "ann_utils.cuh"
#include "topk/radix_topk.cuh"
#include "topk/warpsort_topk.cuh"

#include <raft/common/nvtx.hpp>
#include <raft/core/logger.hpp>
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/distance/distance.cuh>
#include <raft/distance/distance_type.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/spatial/knn/ann_common.h>
#include <raft/stats/histogram.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

namespace raft::spatial::knn::detail {

template <typename T>
class ivf_flat_handle {
 public:
  ivf_flat_handle(const handle_t& handle, ivf_flat_params params)
    : handle_(handle), params_(std::move(params))
  {
  }

  /**
   * @brief Build the index from the dataset for efficient search.
   *
   * @param[in] dataset a device pointer to a row-major matrix [n_rows, dim]
   * @param n_rows number of samples
   * @param dim the dimensionality of the data
   * @param metric distance type
   * @param stream
   */
  void build(const T* dataset,
             uint32_t n_rows,
             uint32_t dim,
             raft::distance::DistanceType metric,
             rmm::cuda_stream_view stream);

  /**
   * @brief Search ANN using the constructed index.
   *
   * @param[in] queries a device pointer to a row-major matrix [n_queries, dim]
   * @param n_queries is the batch size
   * @param k is the number of neighbors to find for each query.
   * @param n_probes number of clusters to look at for each query (affects speed vs recall).
   * @param[out] neighbors a device pointer to the indices of the neighbors in the source dataset
   * [n_queries, k]
   * @param[out] distances a device pointer to the distances to the selected neighbors [n_queries,
   * k]
   * @param stream
   */
  void search(const T* queries,
              uint32_t n_queries,
              uint32_t k,
              uint32_t n_probes,
              size_t* neighbors,
              float* distances,
              rmm::cuda_stream_view stream);

  /** Whether `build` method has already been succesfully invoked. */
  [[nodiscard]] auto is_trained() const -> bool { return index_.has_value(); }

  /** Dimensionality of the data, on which the index has been built. */
  [[nodiscard]] auto data_dim() const -> uint32_t { return is_trained() ? index_->dim() : 0; }

 private:
  const handle_t& handle_;
  const ivf_flat_params params_;

  // The built index
  std::optional<const ivf_flat_index<T>> index_ = std::nullopt;

  // Memory pool for use during search; after the first search is done the pool is not likely to
  // resize, saving the costs of allocations.
  std::optional<rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>> search_mem_res_;

  template <typename AccT>
  void search_impl(const T* queries,
                   uint32_t n_queries,
                   uint32_t k,
                   uint32_t n_probes,
                   bool select_min,
                   size_t* neighbors,
                   AccT* distances,
                   rmm::cuda_stream_view stream);
};

/**
 * @brief Record the dataset into the index, one source row at a time.
 *
 * The index consists of the dataset rows, grouped by their labels (into clusters/lists).
 * Within each cluster (list), the data is grouped into blocks of `WarpSize` interleaved
 * vectors. Note, the total index length is slightly larger than the dataset length, because
 * each cluster is padded by `WarpSize` elements
 *
 * CUDA launch grid:
 *   X dimension must cover the dataset (n_rows), YZ are not used;
 *   there are no dependencies between threads, hence no constraints on the block size.
 *
 * @tparam T the element type.
 *
 * @param[in] labels device pointer to the cluster ids for each row [n_rows]
 * @param[in] list_offsets device pointer to the cluster offsets in the output (index) [n_lists]
 * @param[in] dataset device poitner to the input data [n_rows, dim]
 * @param[out] list_data device pointer to the output [index_size, dim]
 * @param[out] list_index device pointer to the source ids corr. to the output [index_size]
 * @param[out] list_sizes_ptr device pointer to the cluster sizes [n_lists];
 *                          it's used as an atomic counter, and must be initialized with zeros.
 * @param n_rows source length
 * @param dim the dimensionality of the data
 * @param veclen size of vectorized loads/stores; must satisfy `dim % veclen == 0`.
 *
 */
template <typename T>
__global__ void build_index_kernel(const uint32_t* labels,
                                   const uint32_t* list_offsets,
                                   const T* dataset,
                                   T* list_data,
                                   uint32_t* list_index,
                                   uint32_t* list_sizes_ptr,
                                   uint32_t n_rows,
                                   uint32_t dim,
                                   uint32_t veclen)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_rows) { return; }

  auto list_id     = labels[i];
  auto inlist_id   = atomicAdd(list_sizes_ptr + list_id, 1);
  auto list_offset = list_offsets[list_id];

  // Record the source vector id in the index
  list_index[list_offset + inlist_id] = i;

  // The data is written in interleaved groups of `WarpSize` vectors
  using interleaved_group = Pow2<WarpSize>;
  auto group_offset       = interleaved_group::roundDown(inlist_id);
  auto ingroup_id         = interleaved_group::mod(inlist_id) * veclen;

  // Point to the location of the interleaved group of vectors
  list_data += (list_offset + group_offset) * dim;

  // Point to the source vector
  dataset += i * dim;

  // Interleave dimensions of the source vector while recording it.
  // NB: such `veclen` is selected, that `dim % veclen == 0`
  for (uint32_t l = 0; l < dim; l += veclen) {
    for (uint32_t j = 0; j < veclen; j++) {
      list_data[l * WarpSize + ingroup_id + j] = dataset[l + j];
    }
  }
}

template <typename T>
void ivf_flat_handle<T>::build(const T* dataset,
                               uint32_t n_rows,
                               uint32_t dim,
                               raft::distance::DistanceType metric,
                               rmm::cuda_stream_view stream)
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_flat_handle::build(%u, %u)", n_rows, dim);
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
                "unsupported data type");
  RAFT_EXPECTS(n_rows > 0 && dim > 0, "empty dataset");

  // TODO: consider padding the dimensions and fixing veclen to its maximum possible value as a
  // template parameter (https://github.com/rapidsai/raft/issues/711)
  uint32_t veclen = 16 / sizeof(T);
  while (dim % veclen != 0) {
    veclen = veclen >> 1;
  }
  auto n_lists = static_cast<uint32_t>(params_.nlist);

  // kmeans cluster ids for the dataset
  rmm::device_uvector<uint32_t> labels(n_rows, stream);
  auto&& centers      = make_array_for_index<float>(stream, n_lists, dim);
  auto&& list_sizes   = make_array_for_index<uint32_t>(stream, n_lists);
  auto list_sizes_ptr = list_sizes.data();

  // Predict labels of the whole dataset
  kmeans::build_optimized_kmeans(handle_,
                                 params_.kmeans_n_iters,
                                 dim,
                                 dataset,
                                 n_rows,
                                 labels.data(),
                                 list_sizes_ptr,
                                 centers.data(),
                                 n_lists,
                                 params_.kmeans_trainset_fraction,
                                 metric,
                                 stream);

  // Calculate offsets into cluster data using exclusive scan
  auto&& list_offsets   = make_array_for_index<uint32_t>(stream, n_lists + 1);
  auto list_offsets_ptr = list_offsets.data();

  thrust::exclusive_scan(
    rmm::exec_policy(stream),
    list_sizes_ptr,
    list_sizes_ptr + n_lists + 1,
    list_offsets_ptr,
    uint32_t(0),
    [] __device__(uint32_t s, uint32_t l) { return s + Pow2<WarpSize>::roundUp(l); });

  uint32_t index_size;
  update_host(&index_size, list_offsets_ptr + n_lists, 1, stream);
  handle_.sync_stream(stream);

  auto&& data    = make_array_for_index<T>(stream, index_size, dim);
  auto&& indices = make_array_for_index<uint32_t>(stream, index_size);

  // we'll rebuild the `list_sizes_ptr` in the following kernel, using it as an atomic counter.
  utils::memset(list_sizes_ptr, 0, sizeof(uint32_t) * n_lists, stream);

  const dim3 block_dim(256);
  const dim3 grid_dim(raft::ceildiv<uint32_t>(n_rows, block_dim.x));
  build_index_kernel<<<grid_dim, block_dim, 0, stream>>>(labels.data(),
                                                         list_offsets_ptr,
                                                         dataset,
                                                         data.data(),
                                                         indices.data(),
                                                         list_sizes_ptr,
                                                         n_rows,
                                                         dim,
                                                         veclen);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Precompute the centers vector norms for L2Expanded distance
  auto compute_norms = [&]() {
    auto&& r = make_array_for_index<float>(stream, n_lists);
    utils::dots_along_rows(n_lists, dim, centers.data(), r.data(), stream);
    RAFT_LOG_TRACE_VEC(r.data(), 20);
    return r;
  };
  auto&& center_norms = metric == raft::distance::DistanceType::L2Expanded
                          ? std::optional(compute_norms())
                          : std::nullopt;

  // assemble the index
  index_.emplace(ivf_flat_index<T>{
    veclen, metric, data, indices, list_sizes, list_offsets, centers, center_norms});

  // check index invariants
  index_->check_consistency();
}

template <typename T>
void ivf_flat_handle<T>::search(const T* queries,
                                uint32_t n_queries,
                                uint32_t k,
                                uint32_t n_probes,
                                size_t* neighbors,
                                float* distances,
                                rmm::cuda_stream_view stream)
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_flat_handle::search(%u, %u, %zu)", n_queries, k, neighbors);

  RAFT_EXPECTS(is_trained(),
               "The index must be trained before the search (ivf_flat_handle::build)");
  RAFT_EXPECTS(n_probes > 0,
               "n_probes (number of clusters to probe in the search) must be positive.");

  bool select_min;
  switch (index_->metric) {
    case raft::distance::DistanceType::InnerProduct:
    case raft::distance::DistanceType::CosineExpanded:
    case raft::distance::DistanceType::CorrelationExpanded:
      // Similarity metrics have the opposite meaning, i.e. nearest neigbours are those with larger
      // similarity (See the same logic at cpp/include/raft/sparse/selection/detail/knn.cuh:362
      // {perform_k_selection})
      select_min = false;
      break;
    default: select_min = true;
  }

  // Set memory buffer to be reused across searches
  auto cur_memory_resource = rmm::mr::get_current_device_resource();
  if (!search_mem_res_.has_value() || search_mem_res_->get_upstream() != cur_memory_resource) {
    search_mem_res_.emplace(cur_memory_resource, Pow2<256>::roundUp(n_queries * n_probes * k * 16));
  }

  search_impl<float>(queries, n_queries, k, n_probes, select_min, neighbors, distances, stream);
}

template <typename T>
template <typename AccT>
void ivf_flat_handle<T>::search_impl(const T* queries,
                                     uint32_t n_queries,
                                     uint32_t k,
                                     uint32_t n_probes,
                                     bool select_min,
                                     size_t* neighbors,
                                     AccT* distances,
                                     rmm::cuda_stream_view stream)
{
  auto n_lists   = index_->n_lists();
  n_probes       = std::min<uint32_t>(n_probes, n_lists);
  auto search_mr = &(search_mem_res_.value());
  // The norm of query
  rmm::device_uvector<float> query_norm_dev(n_queries, stream, search_mr);
  // The distance value of cluster(list) and queries
  rmm::device_uvector<float> distance_buffer_dev(n_queries * n_lists, stream, search_mr);
  // The topk distance value of cluster(list) and queries
  rmm::device_uvector<float> coarse_distances_dev(n_queries * n_probes, stream, search_mr);
  // The topk  index of cluster(list) and queries
  rmm::device_uvector<uint32_t> coarse_indices_dev(n_queries * n_probes, stream, search_mr);
  // The topk distance value of candicate vectors from each cluster(list)
  rmm::device_uvector<AccT> refined_distances_dev(n_queries * n_probes * k, stream, search_mr);
  // The topk index of candicate vectors from each cluster(list)
  rmm::device_uvector<size_t> refined_indices_dev(n_queries * n_probes * k, stream, search_mr);

  size_t float_query_size;
  if constexpr (std::is_integral_v<T>) {
    float_query_size = n_queries * index_->dim();
  } else {
    float_query_size = 0;
  }
  rmm::device_uvector<float> converted_queries_dev(float_query_size, stream, search_mr);
  float* converted_queries_ptr = converted_queries_dev.data();

  if constexpr (std::is_same_v<T, float>) {
    converted_queries_ptr = const_cast<float*>(queries);
  } else {
    linalg::unaryOp(
      converted_queries_ptr, queries, n_queries * index_->dim(), utils::mapping<float>{}, stream);
  }

  float alpha = 1.0f;
  float beta  = 0.0f;

  if (index_->metric == raft::distance::DistanceType::L2Expanded) {
    alpha = -2.0f;
    beta  = 1.0f;
    utils::dots_along_rows(
      n_queries, index_->dim(), converted_queries_ptr, query_norm_dev.data(), stream);
    utils::outer_add(query_norm_dev.data(),
                     n_queries,
                     index_->center_norms->data(),
                     n_lists,
                     distance_buffer_dev.data(),
                     stream);
    RAFT_LOG_TRACE_VEC(index_->center_norms->data(), 20);
    RAFT_LOG_TRACE_VEC(distance_buffer_dev.data(), 20);
  } else {
    alpha = 1.0f;
    beta  = 0.0f;
  }

  linalg::gemm(handle_,
               true,
               false,
               n_lists,
               n_queries,
               index_->dim(),
               &alpha,
               index_->centers.data(),
               index_->dim(),
               converted_queries_ptr,
               index_->dim(),
               &beta,
               distance_buffer_dev.data(),
               n_lists,
               stream);

  RAFT_LOG_TRACE_VEC(distance_buffer_dev.data(), 20);
  if (n_probes <= raft::spatial::knn::detail::topk::kMaxCapacity) {
    topk::warp_sort_topk<AccT, uint32_t>(distance_buffer_dev.data(),
                                         nullptr,
                                         n_queries,
                                         n_lists,
                                         n_probes,
                                         coarse_distances_dev.data(),
                                         coarse_indices_dev.data(),
                                         select_min,
                                         stream);
  } else {
    topk::radix_topk<AccT, uint32_t, 11, 512>(distance_buffer_dev.data(),
                                              nullptr,
                                              n_queries,
                                              n_lists,
                                              n_probes,
                                              coarse_distances_dev.data(),
                                              coarse_indices_dev.data(),
                                              select_min,
                                              stream,
                                              &(search_mem_res_.value()));
  }
  RAFT_LOG_TRACE_VEC(coarse_indices_dev.data(), 1 * n_probes);
  RAFT_LOG_TRACE_VEC(coarse_distances_dev.data(), 1 * n_probes);

  AccT* distances_dev_ptr = refined_distances_dev.data();
  size_t* indices_dev_ptr = refined_indices_dev.data();

  uint32_t grid_dim_x = 0;
  if (n_probes > 1) {
    // query the gridDimX size to store probes topK output
    ivfflat_interleaved_scan<T, typename utils::config<T>::value_t>(index_.value(),
                                                                    nullptr,
                                                                    nullptr,
                                                                    n_queries,
                                                                    index_->metric,
                                                                    n_probes,
                                                                    k,
                                                                    select_min,
                                                                    nullptr,
                                                                    nullptr,
                                                                    grid_dim_x,
                                                                    stream);
  } else {
    grid_dim_x = 1;
  }

  if (grid_dim_x == 1) {
    distances_dev_ptr = distances;
    indices_dev_ptr   = neighbors;
  }

  ivfflat_interleaved_scan<T, typename utils::config<T>::value_t>(index_.value(),
                                                                  queries,
                                                                  coarse_indices_dev.data(),
                                                                  n_queries,
                                                                  index_->metric,
                                                                  n_probes,
                                                                  k,
                                                                  select_min,
                                                                  indices_dev_ptr,
                                                                  distances_dev_ptr,
                                                                  grid_dim_x,
                                                                  stream);

  RAFT_LOG_TRACE_VEC(distances_dev_ptr, 2 * k);
  RAFT_LOG_TRACE_VEC(indices_dev_ptr, 2 * k);

  // Merge topk values from different blocks
  if (grid_dim_x > 1) {
    if (k <= raft::spatial::knn::detail::topk::kMaxCapacity) {
      topk::warp_sort_topk<AccT, size_t>(refined_distances_dev.data(),
                                         refined_indices_dev.data(),
                                         n_queries,
                                         k * grid_dim_x,
                                         k,
                                         distances,
                                         neighbors,
                                         select_min,
                                         stream);
    } else {
      topk::radix_topk<AccT, size_t, 11, 512>(refined_distances_dev.data(),
                                              refined_indices_dev.data(),
                                              n_queries,
                                              k * grid_dim_x,
                                              k,
                                              distances,
                                              neighbors,
                                              select_min,
                                              stream,
                                              &(search_mem_res_.value()));
    }
  }
}

}  // namespace raft::spatial::knn::detail
