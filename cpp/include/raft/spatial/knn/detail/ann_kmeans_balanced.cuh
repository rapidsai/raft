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

#include "ann_utils.cuh"

#include <raft/common/nvtx.hpp>
#include <raft/core/cudart_utils.hpp>
#include <raft/core/logger.hpp>
#include <raft/cuda_utils.cuh>
#include <raft/distance/distance.hpp>
#include <raft/distance/distance_type.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/pow2_utils.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

namespace raft::spatial::knn::detail::kmeans {

/**
 * @brief Predict labels for the dataset; floats only.
 *
 * NB: no minibatch splitting is done here, it may require large amount of temporary memory (n_rows
 * * n_cluster * sizeof(float)).
 *
 * @param handle
 * @param[in] centers a pointer to the row-major matrix of cluster centers [n_clusters, dim]
 * @param n_clusters number of clusters/centers
 * @param dim dimensionality of the data
 * @param[in] dataset a pointer to the data [n_rows, dim]
 * @param n_rows number samples in the `dataset`
 * @param[out] labels output predictions [n_rows]
 * @param metric
 * @param stream
 * @param mr (optional) memory resource to use for temporary allocations
 */
void predict_float_core(const handle_t& handle,
                        const float* centers,
                        uint32_t n_clusters,
                        uint32_t dim,
                        const float* dataset,
                        size_t n_rows,
                        uint32_t* labels,
                        raft::distance::DistanceType metric,
                        rmm::cuda_stream_view stream,
                        rmm::mr::device_memory_resource* mr)
{
  rmm::device_uvector<float> distances(n_rows * n_clusters, stream, mr);

  float alpha;
  float beta;
  switch (metric) {
    case raft::distance::DistanceType::InnerProduct: {
      alpha = -1.0;
      beta  = 0.0;
    } break;
    case raft::distance::DistanceType::L2Expanded:
    case raft::distance::DistanceType::L2Unexpanded: {
      rmm::device_uvector<float> sqsum_centers(n_clusters, stream, mr);
      rmm::device_uvector<float> sqsum_data(n_rows, stream, mr);
      utils::dots_along_rows(n_clusters, dim, centers, sqsum_centers.data(), stream);
      utils::dots_along_rows(n_rows, dim, dataset, sqsum_data.data(), stream);
      utils::outer_add(
        sqsum_data.data(), n_rows, sqsum_centers.data(), n_clusters, distances.data(), stream);
      alpha = -2.0;
      beta  = 1.0;
    } break;
    // NB: update the description of `knn::ivf_flat::build` when adding here a new metric.
    default: RAFT_FAIL("The chosen distance metric is not supported (%d)", int(metric));
  }
  linalg::gemm(handle,
               true,
               false,
               n_clusters,
               n_rows,
               dim,
               &alpha,
               centers,
               dim,
               dataset,
               dim,
               &beta,
               distances.data(),
               n_clusters,
               stream);
  utils::argmin_along_rows(n_rows, n_clusters, distances.data(), labels, stream);
}

/**
 * @brief Suggest a minibatch size for kmeans prediction.
 *
 * This function is used as a heuristic to split the work over a large dataset
 * to reduce the size of temporary memory allocations.
 *
 * @param n_clusters number of clusters in kmeans clustering
 * @param n_rows dataset size
 * @return a suggested minibatch size
 */
constexpr auto calc_minibatch_size(uint32_t n_clusters, size_t n_rows) -> uint32_t
{
  n_clusters              = std::max<uint32_t>(1, n_clusters);
  uint32_t minibatch_size = (1 << 20);
  if (minibatch_size > (1 << 28) / n_clusters) {
    minibatch_size = (1 << 28) / n_clusters;
    minibatch_size += 32;
    minibatch_size -= minibatch_size % 64;
  }
  minibatch_size = uint32_t(std::min<size_t>(minibatch_size, n_rows));
  return minibatch_size;
}

/**
 * @brief Given the data and labels, calculate cluster centers and sizes in one sweep.
 *
 * Let S_i = {x_k | x_k \in dataset & labels[k] == i} be the vectors in the dataset with label i.
 *   On exit centers_i = normalize(\sum_{x \in S_i} x), where `normalize` depends on the distance
 * type.
 *
 * NB: `centers` and `cluster_sizes` must be accessible on GPU due to
 * divide_along_rows/normalize_rows. The rest can be both, under assumption that all pointers are
 * accessible from the same place.
 *
 * i.e. two variants are possible:
 *
 *   1. All pointers are on the device.
 *   2. All pointers are on the host, but `centers` and `cluster_sizes` are accessible from GPU.
 *
 * @tparam T element type
 *
 * @param[out] centers pointer to the output [n_clusters, dim]
 * @param[out] cluster_sizes number of rows in each cluster [n_clusters]
 * @param n_clusters number of clusters/centers
 * @param dim dimensionality of the data
 * @param[in] dataset a pointer to the data [n_rows, dim]
 * @param n_rows number samples in the `dataset`
 * @param[in] labels output predictions [n_rows]
 * @param metric
 * @param stream
 */
template <typename T>
void calc_centers_and_sizes(float* centers,
                            uint32_t* cluster_sizes,
                            uint32_t n_clusters,
                            uint32_t dim,
                            const T* dataset,
                            size_t n_rows,
                            const uint32_t* labels,
                            raft::distance::DistanceType metric,
                            rmm::cuda_stream_view stream)
{
  utils::memset(centers, 0, sizeof(float) * n_clusters * dim, stream);
  utils::memset(cluster_sizes, 0, sizeof(uint32_t) * n_clusters, stream);
  utils::accumulate_into_selected(n_rows, dim, centers, cluster_sizes, dataset, labels, stream);

  if (metric == raft::distance::DistanceType::InnerProduct) {
    // normalize
    utils::normalize_rows(n_clusters, dim, centers, stream);
  } else {
    // average
    utils::divide_along_rows(n_clusters, dim, centers, cluster_sizes, stream);
  }
}

/**
 * @brief Predict labels for the dataset.
 *
 * NB: no minibatch splitting is done here, it may require large amount of temporary memory (n_rows
 * * n_cluster * sizeof(float)).
 *
 * @tparam T element type
 *
 * @param handle
 * @param[in] centers a pointer to the row-major matrix of cluster centers [n_clusters, dim]
 * @param n_clusters number of clusters/centers
 * @param dim dimensionality of the data
 * @param[in] dataset a pointer to the data [n_rows, dim]
 * @param n_rows number samples in the `dataset`
 * @param[out] labels output predictions [n_rows]
 * @param metric
 * @param stream
 * @param mr (optional) memory resource to use for temporary allocations
 */

template <typename T>
void predict(const handle_t& handle,
             const float* centers,
             uint32_t n_clusters,
             uint32_t dim,
             const T* dataset,
             size_t n_rows,
             uint32_t* labels,
             raft::distance::DistanceType metric,
             rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource* mr = nullptr)
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "kmeans::predict(%u, %u)", n_rows, n_clusters);
  if (mr == nullptr) { mr = rmm::mr::get_current_device_resource(); }
  const uint32_t max_minibatch_size = calc_minibatch_size(n_clusters, n_rows);
  rmm::device_uvector<float> cur_dataset(
    std::is_same_v<T, float> ? 0 : max_minibatch_size * dim, stream, mr);
  auto cur_dataset_ptr = cur_dataset.data();
  for (size_t offset = 0; offset < n_rows; offset += max_minibatch_size) {
    auto minibatch_size = std::min<uint32_t>(max_minibatch_size, n_rows - offset);

    if constexpr (std::is_same_v<T, float>) {
      cur_dataset_ptr = const_cast<float*>(dataset + offset * dim);
    } else {
      linalg::unaryOp(cur_dataset_ptr,
                      dataset + offset * dim,
                      minibatch_size * dim,
                      utils::mapping<float>{},
                      stream);
    }

    predict_float_core(handle,
                       centers,
                       n_clusters,
                       dim,
                       cur_dataset_ptr,
                       minibatch_size,
                       labels + offset,
                       metric,
                       stream,
                       mr);
  }
}

/**
 * @brief Adjust centers for clusters that have small number of entries.
 *
 * For each cluster, where the cluster size is not bigger than a threshold, the center is moved
 * towards a data point that belongs to a large cluster.
 *
 * NB: if this function returns `true`, you should update the labels.
 *
 * NB: all pointers are used on the host side.
 *
 * @tparam T element type
 *
 * @param[inout] centers cluster centers [n_clusters, dim]
 * @param n_clusters number of rows in `centers`
 * @param dim number of columns in `centers` and `dataset`
 * @param[in] dataset a host pointer to the row-major data matrix [n_rows, dim]
 * @param n_rows number of rows in `dataset`
 * @param[in] labels a host pointer to the cluster indices [n_rows]
 * @param metric
 * @param[in] cluster_sizes number of rows in each cluster [n_clusters]
 * @param threshold defines a criterion for adjusting a cluster
 *                   (cluster_sizes <= average_size * threshold)
 *                   0 <= threshold < 1
 * @param stream
 *
 * @return whether any of the centers has been updated (and thus, `labels` need to be recalculated).
 */
template <typename T>
auto adjust_centers(float* centers,
                    size_t n_clusters,
                    size_t dim,
                    const T* dataset,
                    size_t n_rows,
                    const uint32_t* labels,
                    raft::distance::DistanceType metric,
                    const uint32_t* cluster_sizes,
                    float threshold,
                    rmm::cuda_stream_view stream) -> bool
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "kmeans::adjust_centers(%u, %u)", n_rows, n_clusters);
  stream.synchronize();
  if (n_clusters == 0) { return false; }
  constexpr static std::array kPrimes{29,   71,   113,  173,  229,  281,  349,  409,  463,  541,
                                      601,  659,  733,  809,  863,  941,  1013, 1069, 1151, 1223,
                                      1291, 1373, 1451, 1511, 1583, 1657, 1733, 1811, 1889, 1987,
                                      2053, 2129, 2213, 2287, 2357, 2423, 2531, 2617, 2687, 2741};
  static size_t i        = 0;
  static size_t i_primes = 0;

  bool adjusted    = false;
  uint32_t average = static_cast<uint32_t>(n_rows / size_t(n_clusters));
  uint32_t ofst;

  do {
    i_primes = (i_primes + 1) % kPrimes.size();
    ofst     = kPrimes[i_primes];
  } while (n_rows % ofst == 0);

  for (size_t l = 0; l < n_clusters; l++) {
    // skip big clusters
    if (cluster_sizes[l] > static_cast<uint32_t>(average * threshold)) continue;
    // choose a "random" i that belongs to a rather large cluster
    do {
      i = (i + ofst) % n_rows;
    } while (cluster_sizes[labels[i]] < average);
    // Adjust the center of the selected smaller cluster to gravitate towards
    // a sample from the selected larger cluster.
    size_t li   = labels[i];
    float sqsum = 0.0;
    for (size_t j = 0; j < dim; j++) {
      constexpr float kWc = 7.0;
      constexpr float kWd = 1.0;
      float val           = 0;
      val += kWc * centers[j + dim * li];
      val += kWd * utils::mapping<float>{}(dataset[j + dim * i]);
      val /= kWc + kWd;
      sqsum += val * val;
      centers[j + dim * l] = val;
    }
    if (metric == raft::distance::DistanceType::InnerProduct) {
      sqsum = sqrt(sqsum);
      for (size_t j = 0; j < dim; j++) {
        centers[j + dim * l] /= sqsum;
      }
    }
    adjusted = true;
  }
  stream.synchronize();
  return adjusted;
}

/** predict & adjust_centers combined in an iterative process. */
template <typename T>
void build_clusters(const handle_t& handle,
                    uint32_t n_iters,
                    size_t dim,
                    const T* dataset,  // managedl [n_rows, dim]
                    size_t n_rows,
                    size_t n_clusters,
                    float* cluster_centers,    // managed; [n_clusters, dim]
                    uint32_t* cluster_labels,  // managed; [n_rows]
                    uint32_t* cluster_sizes,   // managed; [n_clusters]
                    raft::distance::DistanceType metric,
                    rmm::mr::device_memory_resource* device_memory,
                    rmm::cuda_stream_view stream)
{
  // "randomly initialize labels"
  linalg::writeOnlyUnaryOp(
    cluster_labels,
    n_rows,
    [n_clusters] __device__(uint32_t * out, uint32_t i) { *out = i % n_clusters; },
    stream);

  // update centers to match the initialized labels.
  calc_centers_and_sizes(cluster_centers,
                         cluster_sizes,
                         n_clusters,
                         dim,
                         dataset,
                         n_rows,
                         cluster_labels,
                         metric,
                         stream);

  for (uint32_t iter = 0; iter < 2 * n_iters; iter += 2) {
    predict(handle,
            cluster_centers,
            n_clusters,
            dim,
            dataset,
            n_rows,
            cluster_labels,
            metric,
            stream,
            device_memory);
    calc_centers_and_sizes(cluster_centers,
                           cluster_sizes,
                           n_clusters,
                           dim,
                           dataset,
                           n_rows,
                           cluster_labels,
                           metric,
                           stream);

    if (iter + 1 < 2 * n_iters) {
      if (kmeans::adjust_centers(cluster_centers,
                                 n_clusters,
                                 dim,
                                 dataset,
                                 n_rows,
                                 cluster_labels,
                                 metric,
                                 cluster_sizes,
                                 (float)1.0 / 4,
                                 stream)) {
        iter -= 1;
      }
    }
  }
}

/** Calculate how many fine clusters should belong to each mesocluster. */
auto arrange_fine_clusters(size_t n_clusters,
                           size_t n_mesoclusters,
                           size_t n_rows,
                           const uint32_t* mesocluster_sizes)
{
  std::vector<uint32_t> fine_clusters_nums(n_mesoclusters);
  std::vector<uint32_t> fine_clusters_csum(n_mesoclusters + 1);
  fine_clusters_csum[0] = 0;

  uint32_t n_lists_rem            = n_clusters;
  size_t n_rows_rem               = n_rows;
  uint32_t mesocluster_size_sum   = 0;
  uint32_t mesocluster_size_max   = 0;
  uint32_t fine_clusters_nums_max = 0;
  for (uint32_t i = 0; i < n_mesoclusters; i++) {
    if (i < n_mesoclusters - 1) {
      fine_clusters_nums[i] = (double)n_lists_rem * mesocluster_sizes[i] / n_rows_rem + .5;
    } else {
      fine_clusters_nums[i] = n_lists_rem;
    }
    n_lists_rem -= fine_clusters_nums[i];
    n_rows_rem -= mesocluster_sizes[i];
    mesocluster_size_max = max(mesocluster_size_max, mesocluster_sizes[i]);
    mesocluster_size_sum += mesocluster_sizes[i];
    fine_clusters_nums_max    = max(fine_clusters_nums_max, fine_clusters_nums[i]);
    fine_clusters_csum[i + 1] = fine_clusters_csum[i] + fine_clusters_nums[i];
  }

  RAFT_EXPECTS(mesocluster_size_sum == n_rows,
               "mesocluster sizes do not add up (%u) to the total trainset size (%zu)",
               mesocluster_size_sum,
               n_rows);
  RAFT_EXPECTS(fine_clusters_csum[n_mesoclusters] == n_clusters,
               "fine cluster numbers do not add up (%u) to the total number of clusters (%zu)",
               fine_clusters_csum[n_mesoclusters],
               n_rows

  );

  return std::make_tuple(mesocluster_size_max,
                         fine_clusters_nums_max,
                         std::move(fine_clusters_nums),
                         std::move(fine_clusters_csum));
}

/**
 *  Given the (coarse) mesoclusters and the distribution of fine clusters within them,
 *  build the fine clusters.
 *
 *  Processing one mesocluster at a time:
 *   1. Copy mesocluster data into a separate buffer
 *   2. Predict fine cluster
 *   3. Refince the fine cluster centers
 *
 *  As a result, the fine clusters are what is returned by `build_optimized_kmeans`;
 *  this function returns the total number of fine clusters, which can be checked to be
 *  the same as the requested number of clusters.
 */
template <typename T>
auto build_fine_clusters(const handle_t& handle,
                         uint32_t n_iters,
                         size_t dim,
                         const T* dataset_mptr,
                         const uint32_t* labels_mptr,
                         size_t n_rows,
                         const uint32_t* fine_clusters_nums,
                         const uint32_t* fine_clusters_csum,
                         const uint32_t* mesocluster_sizes,
                         size_t n_mesoclusters,
                         size_t mesocluster_size_max,
                         size_t fine_clusters_nums_max,
                         float* cluster_centers,
                         raft::distance::DistanceType metric,
                         rmm::mr::managed_memory_resource* managed_memory,
                         rmm::mr::device_memory_resource* device_memory,
                         rmm::cuda_stream_view stream) -> uint32_t
{
  rmm::device_uvector<uint32_t> mc_trainset_ids_buf(mesocluster_size_max, stream, managed_memory);
  rmm::device_uvector<float> mc_trainset_buf(mesocluster_size_max * dim, stream, managed_memory);
  auto mc_trainset_ids = mc_trainset_ids_buf.data();
  auto mc_trainset     = mc_trainset_buf.data();

  // label (cluster ID) of each vector
  rmm::device_uvector<uint32_t> mc_trainset_labels(mesocluster_size_max, stream, managed_memory);

  rmm::device_uvector<float> mc_trainset_ccenters(
    fine_clusters_nums_max * dim, stream, managed_memory);
  // number of vectors in each cluster
  rmm::device_uvector<uint32_t> mc_trainset_csizes_tmp(
    fine_clusters_nums_max, stream, managed_memory);

  // Training clusters in each meso-cluster
  uint32_t n_clusters_done = 0;
  for (uint32_t i = 0; i < n_mesoclusters; i++) {
    uint32_t k = 0;
    for (size_t j = 0; j < n_rows; j++) {
      if (labels_mptr[j] == i) { mc_trainset_ids[k++] = j; }
    }
    RAFT_EXPECTS(k == mesocluster_sizes[i], "Incorrect mesocluster size at %d.", i);
    if (k == 0) {
      RAFT_LOG_DEBUG("Empty cluster %d", i);
      RAFT_EXPECTS(fine_clusters_nums[i] == 0,
                   "Number of fine clusters must be zero for the empty mesocluster (got %d)",
                   fine_clusters_nums[i]);
      continue;
    }

    utils::copy_selected(
      mesocluster_sizes[i], dim, dataset_mptr, mc_trainset_ids, dim, mc_trainset, dim, stream);

    build_clusters(handle,
                   n_iters,
                   dim,
                   mc_trainset,
                   mesocluster_sizes[i],
                   fine_clusters_nums[i],
                   mc_trainset_ccenters.data(),
                   mc_trainset_labels.data(),
                   mc_trainset_csizes_tmp.data(),
                   metric,
                   device_memory,
                   stream);

    raft::copy(cluster_centers + (dim * fine_clusters_csum[i]),
               mc_trainset_ccenters.data(),
               fine_clusters_nums[i] * dim,
               stream);
    handle.sync_stream(stream);
    n_clusters_done += fine_clusters_nums[i];
  }
  return n_clusters_done;
}

/**
 * kmeans
 *
 * @tparam T element type
 *
 * @param handle
 * @param n_iters number of training iterations
 * @param dim number of columns in `centers` and `dataset`
 * @param[in] dataset a device pointer to the source dataset [n_rows, dim]
 * @param n_rows number of rows in the input
 * @param[out] labels a device pointer to the output labels [n_rows]
 * @param[out] cluster_centers a device pointer to the found cluster centers [n_cluster, dim]
 * @param n_cluster
 * @param trainset_fraction a fraction of rows in the `dataset` to sample for kmeans training;
 *                            0 < trainset_fraction <= 1.
 * @param metric the distance metric
 * @param stream
 */
template <typename T>
void build_optimized_kmeans(const handle_t& handle,
                            uint32_t n_iters,
                            size_t dim,
                            const T* dataset,
                            size_t n_rows,
                            uint32_t* labels,
                            float* cluster_centers,
                            size_t n_clusters,
                            double trainset_fraction,
                            raft::distance::DistanceType metric,
                            rmm::cuda_stream_view stream)
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "kmeans::build_optimized_kmeans(%u, %u)", n_rows, n_clusters);

  auto trainset_ratio =
    std::max<size_t>(1, n_rows / std::max<size_t>(trainset_fraction * n_rows, n_clusters));
  auto n_rows_train = n_rows / trainset_ratio;

  uint32_t n_mesoclusters = std::min<uint32_t>(n_clusters, std::sqrt(n_clusters) + 0.5);
  RAFT_LOG_DEBUG("(%s) # n_mesoclusters: %u", __func__, n_mesoclusters);

  rmm::mr::managed_memory_resource managed_memory;
  rmm::mr::device_memory_resource* device_memory = nullptr;
  auto pool_guard                                = raft::get_pool_memory_resource(
    device_memory, kmeans::calc_minibatch_size(n_mesoclusters, n_rows) * dim * 4);
  if (pool_guard) {
    RAFT_LOG_DEBUG(
      "kmeans::build_optimized_kmeans: using pool memory resource with initial size %zu bytes",
      pool_guard->pool_size());
  }

  rmm::device_uvector<T> trainset(n_rows_train * dim, stream, &managed_memory);
  // TODO: a proper sampling
  RAFT_CUDA_TRY(cudaMemcpy2DAsync(trainset.data(),
                                  sizeof(T) * dim,
                                  dataset,
                                  sizeof(T) * dim * trainset_ratio,
                                  sizeof(T) * dim,
                                  n_rows_train,
                                  cudaMemcpyDefault,
                                  stream));

  // build coarse clusters (mesoclusters)
  rmm::device_uvector<uint32_t> mesocluster_labels_buf(n_rows_train, stream, &managed_memory);
  rmm::device_uvector<uint32_t> mesocluster_sizes_buf(n_mesoclusters, stream, &managed_memory);
  {
    rmm::device_uvector<float> mesocluster_centers_buf(
      n_mesoclusters * dim, stream, &managed_memory);
    build_clusters(handle,
                   n_iters,
                   dim,
                   trainset.data(),
                   n_rows_train,
                   n_mesoclusters,
                   mesocluster_centers_buf.data(),
                   mesocluster_labels_buf.data(),
                   mesocluster_sizes_buf.data(),
                   metric,
                   device_memory,
                   stream);
  }

  auto mesocluster_sizes  = mesocluster_sizes_buf.data();
  auto mesocluster_labels = mesocluster_labels_buf.data();

  handle.sync_stream(stream);

  // build fine clusters
  auto [mesocluster_size_max, fine_clusters_nums_max, fine_clusters_nums, fine_clusters_csum] =
    arrange_fine_clusters(n_clusters, n_mesoclusters, n_rows_train, mesocluster_sizes);

  auto n_clusters_done = build_fine_clusters(handle,
                                             n_iters,
                                             dim,
                                             trainset.data(),
                                             mesocluster_labels,
                                             n_rows_train,
                                             fine_clusters_nums.data(),
                                             fine_clusters_csum.data(),
                                             mesocluster_sizes,
                                             n_mesoclusters,
                                             mesocluster_size_max,
                                             fine_clusters_nums_max,
                                             cluster_centers,
                                             metric,
                                             &managed_memory,
                                             device_memory,
                                             stream);
  RAFT_EXPECTS(n_clusters_done == n_clusters, "Didn't process all clusters.");

  rmm::device_uvector<uint32_t> cluster_sizes(n_clusters, stream, device_memory);

  // fit clusters using the trainset
  for (int iter = 0; iter < 2; iter++) {
    // NB: labels.size == n_rows >= n_rows_train; the output is not used.
    predict(handle,
            cluster_centers,
            n_clusters,
            dim,
            trainset.data(),
            n_rows_train,
            labels,
            metric,
            stream,
            device_memory);
    calc_centers_and_sizes(cluster_centers,
                           cluster_sizes.data(),
                           n_clusters,
                           dim,
                           trainset.data(),
                           n_rows_train,
                           labels,
                           metric,
                           stream);
  }

  RAFT_LOG_DEBUG("(%s) Final fitting.", __func__);

  // fit clusters using the whole dataset
  predict(handle,
          cluster_centers,
          n_clusters,
          dim,
          dataset,
          n_rows,
          labels,
          metric,
          stream,
          device_memory);
  // update the cluster centers one last time to better represent the whole dataset
  calc_centers_and_sizes(cluster_centers,
                         cluster_sizes.data(),
                         n_clusters,
                         dim,
                         dataset,
                         n_rows,
                         labels,
                         metric,
                         stream);

  predict(handle,
          cluster_centers,
          n_clusters,
          dim,
          dataset,
          n_rows,
          labels,
          metric,
          stream,
          device_memory);
}

}  // namespace raft::spatial::knn::detail::kmeans
