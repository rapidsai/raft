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
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

namespace raft::spatial::knn::detail {

template <typename T>
class cuivflHandle {
 public:
  cuivflHandle(const handle_t& handle,
               raft::distance::DistanceType metric_type,
               uint32_t dim,
               uint32_t n_lists,
               uint32_t n_iters);

  void cuivflBuildIndex(const T* dataset, T* trainset, uint32_t n_rows, uint32_t nTrainset);

  void cuivflSetSearchParameters(const uint32_t n_probes,
                                 const uint32_t max_batch,
                                 const uint32_t max_k);

  void cuivflSearch(
    const T* queries, uint32_t n_queries, uint32_t k, size_t* neighbors, float* distances);

  void queryIVFFlatGridSize(const uint32_t n_probes, const uint32_t n_queries, const uint32_t k);
  uint32_t getDim() { return dim_; }

 private:
  const handle_t& handle_;
  const rmm::cuda_stream_view stream_;

  raft::distance::DistanceType metric_type_;
  bool greater_;
  uint32_t n_lists_;     // The number of inverted lists= the number of centriods
  uint32_t n_iters_;     // The number of uint32_terations for kmeans to build the indexs
  uint32_t dim_;         // The dimension of vectors for input dataset
  uint32_t n_probes_;    // The number of clusters for searching
  uint32_t n_rows_;      // The number of elements for input dataset
  uint32_t index_size_;  // The number of elements in 32 interleaved group for input dataset
  uint32_t veclen_;      // The vectorization length of dataset in index.
  uint32_t grid_dim_x_;  // The number of blocks launched across n_probes.

  // device pointer
  //  The device memory pointer; inverted list for data; size [index_size_, dim_]
  rmm::device_uvector<T> list_data_dev_;
  // The device memory pointer; inverted list for index; size [index_size_]
  rmm::device_uvector<uint32_t> list_index_dev_;
  // The device memory pointer; Used for list_data_manage_ptr_; size [n_lists_]
  rmm::device_uvector<uint32_t> list_offsets_dev_;
  // The device memory pointer; the number of each cluster(list); size [n_lists_]
  rmm::device_uvector<uint32_t> list_lengths_dev_;
  // The device memory pointer; centriod; size [n_lists_, dim_]
  rmm::device_uvector<float> centriod_dev_;
  // The device memory pointer; centriod norm ; size [n_lists_, dim_]
  rmm::device_uvector<float> centriod_norm_dev_;
  // Memory pool for use during search; after the first search is done the pool is not likely to
  // resize, saving the costs of allocations.
  std::optional<rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>> search_mem_res;

  void cuivflBuildOptimizedKmeans(float* centriod_managed_ptr,
                                  const T* dataset,
                                  T* trainset,
                                  uint32_t* clusterSize,
                                  uint32_t n_rows,
                                  uint32_t n_rows_train);

  template <typename AccT>
  void cuivflSearchImpl(
    const T* queries, uint32_t n_queries, uint32_t k, size_t* neighbors, AccT* distances);
};

template <typename T>
cuivflHandle<T>::cuivflHandle(const handle_t& handle,
                              raft::distance::DistanceType metric_type,
                              uint32_t dim,
                              uint32_t n_lists,
                              uint32_t n_iters)
  : handle_(handle),
    stream_(handle_.get_stream()),
    dim_(dim),
    n_lists_(n_lists),
    n_iters_(n_iters),
    metric_type_(metric_type),
    grid_dim_x_(0),
    list_data_dev_(0, stream_),
    list_index_dev_(0, stream_),
    list_offsets_dev_(0, stream_),
    list_lengths_dev_(0, stream_),
    centriod_dev_(0, stream_),
    centriod_norm_dev_(0, stream_)
{
  // TODO: consider padding the dimensions and fixing veclen to its maximum possible value as a
  // template parameter (https://github.com/rapidsai/raft/issues/711)
  veclen_ = 16 / sizeof(T);
  while (dim % veclen_ != 0) {
    veclen_ = veclen_ >> 1;
  }
}

/**
 * NB: `dataset` is accessed only by GPU code, `trainset` accessed by CPU and GPU.
 *
 */
template <typename T>
void cuivflHandle<T>::cuivflBuildOptimizedKmeans(float* centriod_managed_ptr,
                                                 const T* dataset,
                                                 T* trainset,
                                                 uint32_t* labels,
                                                 uint32_t n_rows,
                                                 uint32_t n_rows_train)
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "cuivflBuildOptimizedKmeans(%u, %u)", n_rows, n_rows_train);

  rmm::device_uvector<uint32_t> trainset_labels(n_rows_train, stream_);

  float* cluster_centers = centriod_managed_ptr;

  uint32_t n_mesoclusters = std::pow<double>(n_lists_, 0.5) + 0.5;
  RAFT_LOG_DEBUG("(%s) # n_mesoclusters: %u", __func__, n_mesoclusters);

  rmm::mr::managed_memory_resource managed_memory;
  rmm::device_uvector<float> mesocluster_centers(n_mesoclusters * dim_, stream_, &managed_memory);
  rmm::device_uvector<uint32_t> mesocluster_labels(n_rows_train, stream_, &managed_memory);
  rmm::device_uvector<uint32_t> mesocluster_sizes_buf(n_mesoclusters, stream_, &managed_memory);
  rmm::device_uvector<float> mesocluster_centers_tmp(
    n_mesoclusters * dim_, stream_, &managed_memory);

  auto mesocluster_sizes = mesocluster_sizes_buf.data();

  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> kmeans_mem_res(
    rmm::mr::get_current_device_resource(),
    // an arbitrary guess on the upper bound of the workspace size
    Pow2<256>::roundUp(kmeans::calc_minibatch_size(n_mesoclusters, n_rows) * dim_ * 4));

  // Training meso-clusters
  for (uint32_t iter = 0; iter < 2 * n_iters_; iter += 2) {
    RAFT_LOG_TRACE("Training kmeans of meso-clusters: %.1f / %u", (float)iter / 2, n_iters_);
    kmeans::predict(handle_,
                    mesocluster_centers.data(),
                    n_mesoclusters,
                    dim_,
                    trainset,
                    n_rows_train,
                    mesocluster_labels.data(),
                    metric_type_,
                    (iter != 0),
                    mesocluster_centers_tmp.data(),
                    mesocluster_sizes,
                    true,
                    stream_,
                    &kmeans_mem_res);

    if (iter + 1 < 2 * n_iters_) {
      if (kmeans::adjust_centers(mesocluster_centers.data(),
                                 n_mesoclusters,
                                 dim_,
                                 trainset,
                                 n_rows_train,
                                 mesocluster_labels.data(),
                                 metric_type_,
                                 mesocluster_sizes,
                                 (float)1.0 / 4,
                                 stream_)) {
        iter -= 1;
      }
    }
  }

  handle_.sync_stream(stream_);

  std::vector<uint32_t> fine_clusters_nums(n_mesoclusters);
  std::vector<uint32_t> fine_clusters_csum(n_mesoclusters + 1);
  fine_clusters_csum[0] = 0;

  uint32_t n_lists_rem            = n_lists_;
  uint32_t n_rows_train_rem       = n_rows_train;
  uint32_t mesocluster_size_max   = 0;
  uint32_t mesocluster_size_sum   = 0;
  uint32_t fine_clusters_nums_sum = 0;  // checking
  uint32_t fine_clusters_nums_max = 0;
  for (uint32_t i = 0; i < n_mesoclusters; i++) {
    if (i < n_mesoclusters - 1) {
      fine_clusters_nums[i] = (double)n_lists_rem * mesocluster_sizes[i] / n_rows_train_rem + .5;
    } else {
      fine_clusters_nums[i] = n_lists_rem;
    }
    n_lists_rem -= fine_clusters_nums[i];
    n_rows_train_rem -= mesocluster_sizes[i];
    mesocluster_size_max = max(mesocluster_size_max, mesocluster_sizes[i]);
    mesocluster_size_sum += mesocluster_sizes[i];
    fine_clusters_nums_sum += fine_clusters_nums[i];
    fine_clusters_nums_max    = max(fine_clusters_nums_max, fine_clusters_nums[i]);
    fine_clusters_csum[i + 1] = fine_clusters_csum[i] + fine_clusters_nums[i];
  }

  RAFT_LOG_DEBUG("(%s) # mesocluster_size_sum: %u", __func__, mesocluster_size_sum);
  RAFT_LOG_DEBUG("(%s) # fine_clusters_nums_sum: %u", __func__, fine_clusters_nums_sum);
  assert(mesocluster_size_sum == n_rows_train);
  assert(fine_clusters_nums_sum == n_lists_);
  assert(fine_clusters_csum[n_mesoclusters] == n_lists_);

  rmm::device_uvector<uint32_t> mc_trainset_ids_buf(mesocluster_size_max, stream_, &managed_memory);
  rmm::device_uvector<float> mc_trainset_buf(mesocluster_size_max * dim_, stream_, &managed_memory);
  auto mc_trainset_ids = mc_trainset_ids_buf.data();
  auto mc_trainset     = mc_trainset_buf.data();

  // label (cluster ID) of each vector
  rmm::device_uvector<uint32_t> mc_trainset_labels(mesocluster_size_max, stream_, &managed_memory);

  rmm::device_uvector<float> mc_trainset_ccenters(
    fine_clusters_nums_max * dim_, stream_, &managed_memory);
  rmm::device_uvector<float> mc_trainset_ccenters_tmp(
    fine_clusters_nums_max * dim_, stream_, &managed_memory);
  // number of vectors in each cluster
  rmm::device_uvector<uint32_t> mc_trainset_csizes_tmp(
    fine_clusters_nums_max, stream_, &managed_memory);

  // Training clusters in each meso-clusters
  uint32_t n_clusters_done = 0;
  for (uint32_t i = 0; i < n_mesoclusters; i++) {
    uint32_t k = 0;
    for (uint32_t j = 0; j < n_rows_train; j++) {
      if (mesocluster_labels.data()[j] != i) continue;
      mc_trainset_ids[k++] = j;
    }
    assert(k == mesocluster_sizes[i]);

    utils::copy_selected<T>(
      mesocluster_sizes[i], dim_, trainset, mc_trainset_ids, dim_, mc_trainset, dim_, stream_);

    for (uint32_t iter = 0; iter < 2 * n_iters_; iter += 2) {
      RAFT_LOG_TRACE("Training kmeans of clusters in meso-cluster %u (n_lists: %u): %.1f / %u",
                     i,
                     fine_clusters_nums[i],
                     (float)iter / 2,
                     n_iters_);

      kmeans::predict(handle_,
                      mc_trainset_ccenters.data(),
                      fine_clusters_nums[i],
                      dim_,
                      mc_trainset,
                      mesocluster_sizes[i],
                      mc_trainset_labels.data(),
                      metric_type_,
                      (iter != 0),
                      mc_trainset_ccenters_tmp.data(),
                      mc_trainset_csizes_tmp.data(),
                      true,
                      stream_,
                      &kmeans_mem_res);

      if (iter + 1 < 2 * n_iters_) {
        if (kmeans::adjust_centers(mc_trainset_ccenters.data(),
                                   fine_clusters_nums[i],
                                   dim_,
                                   mc_trainset,
                                   mesocluster_sizes[i],
                                   mc_trainset_labels.data(),
                                   metric_type_,
                                   mc_trainset_csizes_tmp.data(),
                                   (float)1.0 / 4,
                                   stream_)) {
          iter -= 1;
        }
      }
    }
    copy(cluster_centers + (dim_ * fine_clusters_csum[i]),
         mc_trainset_ccenters.data(),
         fine_clusters_nums[i] * dim_,
         stream_);
    handle_.sync_stream(stream_);
    n_clusters_done += fine_clusters_nums[i];
  }  // end for (uint32_t i = 0; i < n_mesoclusters; i++)
  assert(n_clusters_done == n_lists_);

  mc_trainset_ccenters_tmp.resize(n_lists_ * dim_, stream_);
  mc_trainset_csizes_tmp.resize(n_lists_, stream_);

  // Fitting whole clusters using whole trainset.
  for (int iter = 0; iter < 2; iter++) {
    kmeans::predict(handle_,
                    cluster_centers,
                    n_lists_,
                    dim_,
                    trainset,
                    n_rows_train,
                    trainset_labels.data(),
                    metric_type_,
                    true,
                    mc_trainset_ccenters_tmp.data(),
                    mc_trainset_csizes_tmp.data(),
                    true,
                    stream_,
                    &kmeans_mem_res);
  }  // end for (int iter = 0; iter < 2; iter++)

  RAFT_LOG_DEBUG("(%s) Final fitting.", __func__);

  kmeans::predict(handle_,
                  cluster_centers,
                  n_lists_,
                  dim_,
                  dataset,
                  n_rows_,
                  labels,
                  metric_type_,
                  true,
                  mc_trainset_ccenters_tmp.data(),
                  mc_trainset_csizes_tmp.data(),
                  true,
                  stream_,
                  &kmeans_mem_res);

  kmeans::predict(handle_,
                  cluster_centers,
                  n_lists_,
                  dim_,
                  dataset,
                  n_rows_,
                  labels,
                  metric_type_,
                  true,
                  mc_trainset_ccenters_tmp.data(),
                  mc_trainset_csizes_tmp.data(),
                  false,
                  stream_,
                  &kmeans_mem_res);
}

/**
 * @brief Record the dataset into the index, one source row at a time.
 *
 * The index concists of the dataset rows, grouped by their labels (into clusters/lists).
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
 * @param[out] list_data device pointer to the output [index_size_, dim]
 * @param[out] list_index device pointer to the source ids corr. to the output [index_size_]
 * @param[out] list_lengths device pointer to the cluster sizes [n_lists];
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
                                   uint32_t* list_lengths,
                                   uint32_t n_rows,
                                   uint32_t dim,
                                   uint32_t veclen)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n_rows) { return; }

  auto list_id     = labels[i];
  auto inlist_id   = atomicAdd(list_lengths + list_id, 1);
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
void cuivflHandle<T>::cuivflBuildIndex(const T* dataset,
                                       T* trainset,
                                       uint32_t n_rows,
                                       uint32_t n_rows_train)
{
  n_rows_ = n_rows;
  RAFT_EXPECTS(n_rows_ > 0, "empty dataset");

  rmm::mr::managed_memory_resource managed_memory;
  rmm::device_uvector<float> centriod_managed_buf(n_lists_ * dim_, stream_, &managed_memory);
  auto centriod_managed_ptr = centriod_managed_buf.data();

  static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
                "unsupported data type");

  // Alloc manage memory for centriods, trainset and workspace
  rmm::device_uvector<uint32_t> labels_buf(n_rows_, stream_, &managed_memory);  // [numDataset]
  auto labels = labels_buf.data();

  // Predict labels of the whole dataset
  cuivflBuildOptimizedKmeans(centriod_managed_ptr, dataset, trainset, labels, n_rows, n_rows_train);

  // Calculate the L2 related result
  centriod_norm_dev_.resize(n_lists_, stream_);

  if (metric_type_ == raft::distance::DistanceType::L2Expanded) {
    utils::dots_along_rows(
      n_lists_, dim_, centriod_managed_ptr, centriod_norm_dev_.data(), stream_);
    RAFT_LOG_TRACE_VEC(centriod_norm_dev_.data(), 20);
  }
  centriod_dev_.resize(n_lists_ * dim_, stream_);
  copy(centriod_dev_.data(), centriod_managed_ptr, n_lists_ * dim_, stream_);

  list_lengths_dev_.resize(n_lists_, stream_);
  auto list_lengths = list_lengths_dev_.data();
  stats::histogram(stats::HistType::HistTypeAuto,
                   reinterpret_cast<int*>(list_lengths),
                   n_lists_,
                   labels,
                   n_rows_,
                   uint32_t(1),
                   stream_);

  // NB: stream_ must be equal to handle_.get_stream() to have the thrust functions executed in
  // order with the rest
  auto thrust_policy = handle_.get_thrust_policy();

  list_offsets_dev_.resize(n_lists_ + 1, stream_);
  auto list_offsets = list_offsets_dev_.data();

  thrust::exclusive_scan(
    thrust_policy,
    list_lengths,
    list_lengths + n_lists_ + 1,
    list_offsets,
    uint32_t(0),
    [] __device__(uint32_t s, uint32_t l) { return s + Pow2<WarpSize>::roundUp(l); });

  update_host(&index_size_, list_offsets + n_lists_, 1, stream_);
  handle_.sync_stream(stream_);

  list_data_dev_.resize(index_size_ * dim_, stream_);
  list_index_dev_.resize(index_size_, stream_);

  // we'll rebuild the `list_lengths` in the following kernel, using it as an atomic counter.
  utils::memset(list_lengths, 0, sizeof(uint32_t) * n_lists_, stream_);

  const dim3 block_dim(256);
  const dim3 grid_dim(raft::ceildiv<uint32_t>(n_rows_, block_dim.x));
  build_index_kernel<<<grid_dim, block_dim, 0, stream_>>>(labels,
                                                          list_offsets,
                                                          dataset,
                                                          list_data_dev_.data(),
                                                          list_index_dev_.data(),
                                                          list_lengths,
                                                          n_rows_,
                                                          dim_,
                                                          veclen_);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename T>
void cuivflHandle<T>::queryIVFFlatGridSize(const uint32_t n_probes,
                                           const uint32_t n_queries,
                                           const uint32_t k)
{
  // query the gridDimX size to store probes topK output
  ivfflat_interleaved_scan<T, typename utils::config<T>::value_t>(nullptr,
                                                                  nullptr,
                                                                  nullptr,
                                                                  nullptr,
                                                                  nullptr,
                                                                  nullptr,
                                                                  metric_type_,
                                                                  n_probes,
                                                                  k,
                                                                  n_queries,
                                                                  dim_,
                                                                  nullptr,
                                                                  nullptr,
                                                                  stream_,
                                                                  greater_,
                                                                  veclen_,
                                                                  grid_dim_x_);
}

template <typename T>
void cuivflHandle<T>::cuivflSetSearchParameters(const uint32_t n_probes,
                                                const uint32_t max_batch,
                                                const uint32_t max_k)
{
  RAFT_EXPECTS(n_probes > 0,
               "n_probes (number of clusters to probe in the search) must be positive.");
  n_probes_ = n_probes;
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
  uint32_t n_probes = std::min(n_probes_, n_lists_);
  grid_dim_x_       = 0;
  queryIVFFlatGridSize(n_probes, n_queries, k);
  auto search_mr = &(search_mem_res.value());
  // The norm of query
  rmm::device_uvector<float> query_norm_dev(n_queries, stream_, search_mr);
  // The distance value of cluster(list) and queries
  rmm::device_uvector<float> distance_buffer_dev(n_queries * n_lists_, stream_, search_mr);
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
    float_query_size = n_queries * dim_;
  } else {
    float_query_size = 0;
  }
  rmm::device_uvector<float> converted_queries_dev(float_query_size, stream_, search_mr);
  float* converted_queries_ptr = converted_queries_dev.data();

  if constexpr (std::is_same_v<T, float>) {
    converted_queries_ptr = const_cast<float*>(queries);
  } else {
    linalg::unaryOp(
      converted_queries_ptr, queries, n_queries * dim_, utils::mapping<float>{}, stream_);
  }

  float alpha = 1.0f;
  float beta  = 0.0f;

  if (metric_type_ == raft::distance::DistanceType::L2Expanded) {
    alpha = -2.0f;
    beta  = 1.0f;
    utils::dots_along_rows(n_queries, dim_, converted_queries_ptr, query_norm_dev.data(), stream_);
    utils::outer_add(query_norm_dev.data(),
                     n_queries,
                     centriod_norm_dev_.data(),
                     n_lists_,
                     distance_buffer_dev.data(),
                     stream_);
    RAFT_LOG_TRACE_VEC(centriod_norm_dev_.data(), 20);
    RAFT_LOG_TRACE_VEC(distance_buffer_dev.data(), 20);
  } else {
    alpha = 1.0f;
    beta  = 0.0f;
  }

  linalg::gemm(handle_,
               true,
               false,
               n_lists_,
               n_queries,
               dim_,
               &alpha,
               centriod_dev_.data(),
               dim_,
               converted_queries_ptr,
               dim_,
               &beta,
               distance_buffer_dev.data(),
               n_lists_,
               stream_);

  RAFT_LOG_TRACE_VEC(distance_buffer_dev.data(), 20);
  if (n_probes <= raft::spatial::knn::detail::topk::kMaxCapacity) {
    topk::warp_sort_topk<AccT, uint32_t>(distance_buffer_dev.data(),
                                         nullptr,
                                         n_queries,
                                         n_lists_,
                                         n_probes,
                                         coarse_distances_dev.data(),
                                         coarse_indices_dev.data(),
                                         !greater_,
                                         stream_);
  } else {
    topk::radix_topk<AccT, uint32_t, 11, 512>(distance_buffer_dev.data(),
                                              nullptr,
                                              n_queries,
                                              n_lists_,
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

  ivfflat_interleaved_scan<T, typename utils::config<T>::value_t>(queries,
                                                                  coarse_indices_dev.data(),
                                                                  list_index_dev_.data(),
                                                                  list_data_dev_.data(),
                                                                  list_lengths_dev_.data(),
                                                                  list_offsets_dev_.data(),
                                                                  metric_type_,
                                                                  n_probes,
                                                                  k,
                                                                  n_queries,
                                                                  dim_,
                                                                  indices_dev_ptr,
                                                                  distances_dev_ptr,
                                                                  stream_,
                                                                  greater_,
                                                                  veclen_,
                                                                  grid_dim_x_);

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
