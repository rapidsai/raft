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
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

namespace raft::spatial::knn::detail {

template <typename T>
class cuivflHandle {
 public:
  cuivflHandle(const handle_t& handle,
               raft::distance::DistanceType metric_type,
               const ivf_flat_params& params);

  void cuivflBuildIndex(const T* dataset, uint32_t n_rows, uint32_t dim);

  void cuivflSetSearchParameters(const uint32_t n_probes,
                                 const uint32_t max_batch,
                                 const uint32_t max_k);

  void cuivflSearch(
    const T* queries, uint32_t n_queries, uint32_t k, size_t* neighbors, float* distances);

  void queryIVFFlatGridSize(const uint32_t n_probes, const uint32_t n_queries, const uint32_t k);
  uint32_t getDim() { return index_.has_value() ? index_->dim() : 0; }

 private:
  const handle_t& handle_;
  const rmm::cuda_stream_view stream_;
  ivf_flat_params params_;

  const raft::distance::DistanceType metric_type_;
  bool greater_;
  uint32_t grid_dim_x_;  // The number of blocks launched across n_probes.
  // The built index
  std::optional<const ivf_flat_index<T>> index_ = std::nullopt;

  // Memory pool for use during search; after the first search is done the pool is not likely to
  // resize, saving the costs of allocations.
  std::optional<rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>> search_mem_res;

  template <typename AccT>
  void cuivflSearchImpl(
    const T* queries, uint32_t n_queries, uint32_t k, size_t* neighbors, AccT* distances);
};

template <typename T>
cuivflHandle<T>::cuivflHandle(const handle_t& handle,
                              raft::distance::DistanceType metric_type,
                              const ivf_flat_params& params)
  : handle_(handle),
    stream_(handle_.get_stream()),
    params_(params),
    grid_dim_x_(0),
    metric_type_(metric_type)
{
}

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
void cuivflHandle<T>::cuivflBuildIndex(const T* dataset, uint32_t n_rows, uint32_t dim)
{
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
  rmm::device_uvector<uint32_t> labels(n_rows, stream_);
  auto&& centers = make_array_for_index<float>(stream_, n_lists, dim);

  // Predict labels of the whole dataset
  kmeans::build_optimized_kmeans(handle_,
                                 params_.kmeans_n_iters,
                                 dim,
                                 dataset,
                                 n_rows,
                                 labels.data(),
                                 centers.data(),
                                 n_lists,
                                 params_.kmeans_trainset_fraction,
                                 metric_type_,
                                 stream_);

  auto&& list_sizes   = make_array_for_index<uint32_t>(stream_, n_lists);
  auto list_sizes_ptr = list_sizes.data();
  stats::histogram(stats::HistType::HistTypeAuto,
                   reinterpret_cast<int*>(list_sizes_ptr),
                   n_lists,
                   labels.data(),
                   n_rows,
                   uint32_t(1),
                   stream_);

  // NB: stream_ must be equal to handle_.get_stream() to have the thrust functions executed in
  // order with the rest
  auto thrust_policy = handle_.get_thrust_policy();

  auto&& list_offsets   = make_array_for_index<uint32_t>(stream_, n_lists + 1);
  auto list_offsets_ptr = list_offsets.data();

  thrust::exclusive_scan(
    thrust_policy,
    list_sizes_ptr,
    list_sizes_ptr + n_lists + 1,
    list_offsets_ptr,
    uint32_t(0),
    [] __device__(uint32_t s, uint32_t l) { return s + Pow2<WarpSize>::roundUp(l); });

  uint32_t index_size;
  update_host(&index_size, list_offsets_ptr + n_lists, 1, stream_);
  handle_.sync_stream(stream_);

  auto&& data    = make_array_for_index<T>(stream_, index_size, dim);
  auto&& indices = make_array_for_index<uint32_t>(stream_, index_size);

  // we'll rebuild the `list_sizes_ptr` in the following kernel, using it as an atomic counter.
  utils::memset(list_sizes_ptr, 0, sizeof(uint32_t) * n_lists, stream_);

  const dim3 block_dim(256);
  const dim3 grid_dim(raft::ceildiv<uint32_t>(n_rows, block_dim.x));
  build_index_kernel<<<grid_dim, block_dim, 0, stream_>>>(labels.data(),
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
    auto&& r = make_array_for_index<float>(stream_, params_.nlist);
    utils::dots_along_rows(params_.nlist, dim, centers.data(), r.data(), stream_);
    RAFT_LOG_TRACE_VEC(r.data(), 20);
    return r;
  };
  auto&& center_norms = metric_type_ == raft::distance::DistanceType::L2Expanded
                          ? std::optional(compute_norms())
                          : std::nullopt;

  // assemble the index
  index_.emplace(
    ivf_flat_index<T>{veclen, data, indices, list_sizes, list_offsets, centers, center_norms});

  // check index invariants
  index_->check_consistency();
}

template <typename T>
void cuivflHandle<T>::queryIVFFlatGridSize(const uint32_t n_probes,
                                           const uint32_t n_queries,
                                           const uint32_t k)
{
  // query the gridDimX size to store probes topK output
  ivfflat_interleaved_scan<T, typename utils::config<T>::value_t>(index_.value(),
                                                                  nullptr,
                                                                  nullptr,
                                                                  n_queries,
                                                                  metric_type_,
                                                                  n_probes,
                                                                  k,
                                                                  greater_,
                                                                  nullptr,
                                                                  nullptr,
                                                                  grid_dim_x_,
                                                                  stream_);
}

template <typename T>
void cuivflHandle<T>::cuivflSetSearchParameters(const uint32_t n_probes,
                                                const uint32_t max_batch,
                                                const uint32_t max_k)
{
  RAFT_EXPECTS(n_probes > 0,
               "n_probes (number of clusters to probe in the search) must be positive.");
  params_.nprobe = n_probes;
  // Set the greater_
  if (metric_type_ == raft::distance::DistanceType::L2Expanded ||
      metric_type_ == raft::distance::DistanceType::L2Unexpanded) {
    greater_ = false;
  } else {
    // Need to set this to true for inner product if need FAISS like behavior for inner product
    greater_ = false;
  }

  // Set memory buffer to be reused across searches
  auto cur_memory_resource = rmm::mr::get_current_device_resource();
  if (!search_mem_res.has_value() || search_mem_res->get_upstream() != cur_memory_resource) {
    search_mem_res.emplace(cur_memory_resource,
                           Pow2<256>::roundUp(max_batch * n_probes * max_k * 16));
  }
}

template <typename T>
void cuivflHandle<T>::cuivflSearch(const T* queries,  // [numQueries, dim]
                                   uint32_t n_queries,
                                   uint32_t k,
                                   size_t* neighbors,  // [numQueries, topK]
                                   float* distances)
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "cuivflSearch(%u, %u, %zu)", n_queries, k, neighbors);
  cuivflSearchImpl<float>(queries, n_queries, k, neighbors, distances);
}

template <typename T>
template <typename AccT>
void cuivflHandle<T>::cuivflSearchImpl(const T* queries,  // [numQueries, dim]
                                       uint32_t n_queries,
                                       uint32_t k,
                                       size_t* neighbors,  // [numQueries, topK]
                                       AccT* distances)
{
  uint32_t n_probes = std::min(params_.nprobe, params_.nlist);
  grid_dim_x_       = 0;
  queryIVFFlatGridSize(n_probes, n_queries, k);
  auto search_mr = &(search_mem_res.value());
  // The norm of query
  rmm::device_uvector<float> query_norm_dev(n_queries, stream_, search_mr);
  // The distance value of cluster(list) and queries
  rmm::device_uvector<float> distance_buffer_dev(n_queries * params_.nlist, stream_, search_mr);
  // The topk distance value of cluster(list) and queries
  rmm::device_uvector<float> coarse_distances_dev(n_queries * n_probes, stream_, search_mr);
  // The topk  index of cluster(list) and queries
  rmm::device_uvector<uint32_t> coarse_indices_dev(n_queries * n_probes, stream_, search_mr);
  // The topk distance value of candicate vectors from each cluster(list)
  rmm::device_uvector<AccT> refined_distances_dev(n_queries * n_probes * k, stream_, search_mr);
  // The topk index of candicate vectors from each cluster(list)
  rmm::device_uvector<size_t> refined_indices_dev(n_queries * n_probes * k, stream_, search_mr);

  size_t float_query_size;
  if constexpr (std::is_integral_v<T>) {
    float_query_size = n_queries * index_->dim();
  } else {
    float_query_size = 0;
  }
  rmm::device_uvector<float> converted_queries_dev(float_query_size, stream_, search_mr);
  float* converted_queries_ptr = converted_queries_dev.data();

  if constexpr (std::is_same_v<T, float>) {
    converted_queries_ptr = const_cast<float*>(queries);
  } else {
    linalg::unaryOp(
      converted_queries_ptr, queries, n_queries * index_->dim(), utils::mapping<float>{}, stream_);
  }

  float alpha = 1.0f;
  float beta  = 0.0f;

  if (metric_type_ == raft::distance::DistanceType::L2Expanded) {
    alpha = -2.0f;
    beta  = 1.0f;
    utils::dots_along_rows(
      n_queries, index_->dim(), converted_queries_ptr, query_norm_dev.data(), stream_);
    utils::outer_add(query_norm_dev.data(),
                     n_queries,
                     index_->center_norms->data(),
                     params_.nlist,
                     distance_buffer_dev.data(),
                     stream_);
    RAFT_LOG_TRACE_VEC(index_->center_norms->data(), 20);
    RAFT_LOG_TRACE_VEC(distance_buffer_dev.data(), 20);
  } else {
    alpha = 1.0f;
    beta  = 0.0f;
  }

  linalg::gemm(handle_,
               true,
               false,
               params_.nlist,
               n_queries,
               index_->dim(),
               &alpha,
               index_->centers.data(),
               index_->dim(),
               converted_queries_ptr,
               index_->dim(),
               &beta,
               distance_buffer_dev.data(),
               params_.nlist,
               stream_);

  RAFT_LOG_TRACE_VEC(distance_buffer_dev.data(), 20);
  if (n_probes <= raft::spatial::knn::detail::topk::kMaxCapacity) {
    topk::warp_sort_topk<AccT, uint32_t>(distance_buffer_dev.data(),
                                         nullptr,
                                         n_queries,
                                         params_.nlist,
                                         n_probes,
                                         coarse_distances_dev.data(),
                                         coarse_indices_dev.data(),
                                         !greater_,
                                         stream_);
  } else {
    topk::radix_topk<AccT, uint32_t, 11, 512>(distance_buffer_dev.data(),
                                              nullptr,
                                              n_queries,
                                              params_.nlist,
                                              n_probes,
                                              coarse_distances_dev.data(),
                                              coarse_indices_dev.data(),
                                              !greater_,
                                              stream_,
                                              &(search_mem_res.value()));
  }
  RAFT_LOG_TRACE_VEC(coarse_indices_dev.data(), 1 * n_probes);
  RAFT_LOG_TRACE_VEC(coarse_distances_dev.data(), 1 * n_probes);

  AccT* distances_dev_ptr = refined_distances_dev.data();
  size_t* indices_dev_ptr = refined_indices_dev.data();
  if (n_probes == 1 || grid_dim_x_ == 1) {
    distances_dev_ptr = distances;
    indices_dev_ptr   = neighbors;
  }

  ivfflat_interleaved_scan<T, typename utils::config<T>::value_t>(index_.value(),
                                                                  queries,
                                                                  coarse_indices_dev.data(),
                                                                  n_queries,
                                                                  metric_type_,
                                                                  n_probes,
                                                                  k,
                                                                  greater_,
                                                                  indices_dev_ptr,
                                                                  distances_dev_ptr,
                                                                  grid_dim_x_,
                                                                  stream_);

  RAFT_LOG_TRACE_VEC(distances_dev_ptr, 2 * k);
  RAFT_LOG_TRACE_VEC(indices_dev_ptr, 2 * k);

  // Merge topk values from different blocks
  if (grid_dim_x_ > 1) {
    if (k <= raft::spatial::knn::detail::topk::kMaxCapacity) {
      topk::warp_sort_topk<AccT, size_t>(refined_distances_dev.data(),
                                         refined_indices_dev.data(),
                                         n_queries,
                                         k * grid_dim_x_,
                                         k,
                                         distances,
                                         neighbors,
                                         !greater_,
                                         stream_);
    } else {
      topk::radix_topk<AccT, size_t, 11, 512>(refined_distances_dev.data(),
                                              refined_indices_dev.data(),
                                              n_queries,
                                              k * grid_dim_x_,
                                              k,
                                              distances,
                                              neighbors,
                                              !greater_,
                                              stream_,
                                              &(search_mem_res.value()));
    }
  }
}

}  // namespace raft::spatial::knn::detail
