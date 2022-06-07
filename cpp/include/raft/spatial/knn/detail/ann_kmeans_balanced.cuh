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
#include "ann_utils.cuh"

#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/distance/distance.hpp>
#include <raft/distance/distance_type.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/pow2_utils.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

namespace raft::spatial::knn::detail::kmeans {

void predict_core_(const handle_t& handle,
                   const float* centers,  // [n_clusters, dim]
                   uint32_t n_clusters,
                   uint32_t dim,
                   const float* dataset,  // [n_rows, dim]
                   uint32_t n_rows,
                   uint32_t* labels,  // [n_rows]
                   raft::distance::DistanceType metric,
                   float* workspace,
                   rmm::cuda_stream_view stream)
{
  float* sqsum_centers = workspace;                   // [n_clusters]
  float* sqsum_data    = sqsum_centers + n_clusters;  // [n_rows]
  float* distances     = sqsum_data + n_rows;         // [n_rows, n_clusters]

  float alpha;
  float beta;
  if (metric == raft::distance::DistanceType::InnerProduct) {
    alpha = -1.0;
    beta  = 0.0;
  } else {
    utils::dots_along_rows(n_clusters, dim, centers, sqsum_centers, stream);
    utils::dots_along_rows(n_rows, dim, dataset, sqsum_data, stream);
    utils::outer_add(sqsum_data, n_rows, sqsum_centers, n_clusters, distances, stream);
    alpha = -2.0;
    beta  = 1.0;
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
               distances,
               n_clusters,
               stream);
  utils::argmin_along_rows(n_rows, n_clusters, distances, labels, stream);
}

auto predict_chunk_size_(uint32_t n_clusters, uint32_t n_rows) -> uint32_t
{
  n_clusters     = max(1, n_clusters);
  uint32_t chunk = (1 << 20);
  if (chunk > (1 << 28) / n_clusters) {
    chunk = (1 << 28) / n_clusters;
    chunk += 32;
    chunk -= chunk % 64;
  }
  chunk = min(chunk, n_rows);
  return chunk;
}

/**
 * @brief Calculate the required workspace size for the `predict`.
 */
auto predict_buffer_size(uint32_t n_clusters, uint32_t dim, uint32_t n_rows) -> size_t
{
  uint32_t chunk = predict_chunk_size_(n_clusters, n_rows);
  size_t size    = 0;
  using align_t  = Pow2<128>;
  // float *cur_dataset;  // [chunk, dim]
  size += align_t::roundUp(sizeof(float) * chunk * dim);
  // void *buf_dataset;  // [chunk, dim]
  size += align_t::roundUp(sizeof(float) * chunk * dim);
  // float *workspace;
  size += align_t::roundUp(sizeof(float) * (n_clusters + chunk + (n_clusters * chunk)));
  return size;
}

/**
 * @brief update kmeans centers
 *
 * Let S_i = {x_k | x_k \in dataset & labels[k] == i} be the vectors in the dataset with label i.
 *   On exit centers_i = normalize(\sum_{x \in S_i} x), where `normalize` depends on the distance
 * type.
 *
 * If accumulated_centers is not null, then it is expected that the summation is already done and
 * the results are stored in accumulated_centers. In that case only the normalization will be
 * applied.
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
 * @param n_clusters number of clusters/centers
 * @param dim dimensionality of the data
 * @param[in] dataset a pointer to the data [n_rows, dim]
 * @param n_rows number samples in the `dataset`
 * @param[in] labels output predictions [n_rows]
 * @param metric
 * @param[inout] cluster_sizes (optional) number of rows in each cluster [n_clusters]
 * @param[in] accumulated_centers (optional) pre-computed accumulated sums
 *                                (non-normalized centers) [n_clusters, dim]
 * @param stream
 */
template <typename T>
void update_centers(float* centers,
                    uint32_t n_clusters,
                    uint32_t dim,
                    const T* dataset,
                    uint32_t n_rows,
                    const uint32_t* labels,
                    raft::distance::DistanceType metric,
                    uint32_t* cluster_sizes,
                    const float* accumulated_centers,
                    rmm::cuda_stream_view stream)
{
  if (accumulated_centers == nullptr) {
    // accumulate
    utils::memset(centers, 0, sizeof(float) * n_clusters * dim, stream);
    utils::memset(cluster_sizes, 0, sizeof(uint32_t) * n_clusters, stream);
    utils::accumulate_into_selected<T>(
      n_rows, dim, centers, cluster_sizes, dataset, labels, stream);
  } else {
    copy(centers, accumulated_centers, n_clusters * dim, stream);
  }

  if (metric == raft::distance::DistanceType::InnerProduct) {
    // normalize
    utils::normalize_rows(n_clusters, dim, centers, stream);
  } else {
    // average
    utils::divide_along_rows(n_clusters, dim, centers, cluster_sizes, stream);
  }
}

/**
 * @brief Predict labels for the dataset. For each point we assign the label of the nearest center.
 *
 * NB: seems that all pointers here are accessed by devicie code only
 *
 * @tparam T element type
 *
 * @param handle
 * @param[inout] centers a pointer to the row-major matrix of cluster centers [n_clusters, dim]
 * @param n_clusters number of clusters/centers
 * @param dim dimensionality of the data
 * @param[in] dataset a pointer to the data [n_rows, dim]
 * @param n_rows number samples in the `dataset`
 * @param[out] labels output predictions [n_rows]
 * @param metric
 * @param is_center_set
 * @param[in] _workspace optional
 * @param[in] centers_temp optional [n_clusters, dim]
 * @param[inout] cluster_sizes (optional) number of rows in each cluster [n_clusters]
 * @param shall_update_centers
 * @param stream
 */
template <typename T>
void predict(const handle_t& handle,
             float* centers,
             uint32_t n_clusters,
             uint32_t dim,
             const T* dataset,
             uint32_t n_rows,
             uint32_t* labels,
             raft::distance::DistanceType metric,
             bool is_center_set,
             void* _workspace,
             float* centers_temp,
             uint32_t* cluster_sizes,
             bool shall_update_centers,
             rmm::cuda_stream_view stream)
{
  if (n_rows == 0) {
    RAFT_LOG_WARN(
      "cuann_kmeans_predict: empty dataset (n_rows = %d, n_clusters = %d)", n_rows, n_clusters);
    return;
  }
  if (!is_center_set) {
    // If centers are not set, the labels will be determined randomly.
    linalg::writeOnlyUnaryOp(
      labels,
      n_rows,
      [n_clusters] __device__(uint32_t * out, uint32_t i) { *out = i % n_clusters; },
      stream);
    if (centers_temp != nullptr && cluster_sizes != nullptr) {
      // update centers
      update_centers(
        centers, n_clusters, dim, dataset, n_rows, labels, metric, cluster_sizes, nullptr, stream);
    }
    return;
  }

  uint32_t chunk_max = predict_chunk_size_(n_clusters, n_rows);
  void* workspace    = _workspace;
  rmm::device_buffer sub_workspace(0, stream);

  if (_workspace == nullptr) {
    sub_workspace.resize(predict_buffer_size(n_clusters, dim, n_rows), stream);
    workspace = sub_workspace.data();
  }

  // [chunk_max, dim]
  auto cur_dataset = reinterpret_cast<float*>(workspace);
  // [chunk_max, dim]
  auto buf_dataset    = reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(cur_dataset) +
                                          Pow2<128>::roundUp(sizeof(float) * chunk_max * dim));
  auto workspace_core = reinterpret_cast<float*>(
    reinterpret_cast<uint8_t*>(buf_dataset) + Pow2<128>::roundUp(sizeof(float) * chunk_max * dim));

  if (centers_temp != nullptr && cluster_sizes != nullptr) {
    utils::memset(centers_temp, 0, sizeof(float) * n_clusters * dim, stream);
    utils::memset(cluster_sizes, 0, sizeof(uint32_t) * n_clusters, stream);
  }

  for (uint32_t offset = 0; offset < n_rows; offset += chunk_max) {
    auto chunk = std::min<uint32_t>(chunk_max, n_rows - offset);

    copy(buf_dataset, dataset + offset * dim, chunk * dim, stream);
    handle.sync_stream(stream);

    if constexpr (std::is_same_v<T, float>) {
      // No need to copy floats
      cur_dataset = buf_dataset;
    } else {
      linalg::unaryOp(cur_dataset, buf_dataset, chunk * dim, utils::mapping<float>{}, stream);
    }

    // predict
    predict_core_(handle,
                  centers,
                  n_clusters,
                  dim,
                  cur_dataset,
                  chunk,
                  labels + offset,
                  metric,
                  workspace_core,
                  stream);

    if ((centers_temp != nullptr) && (cluster_sizes != nullptr)) {
      // accumulate
      utils::accumulate_into_selected<float>(
        chunk, dim, centers_temp, cluster_sizes, cur_dataset, labels + offset, stream);
    }
  }

  if ((centers_temp != nullptr) && (cluster_sizes != nullptr) && shall_update_centers) {
    update_centers(centers,
                   n_clusters,
                   dim,
                   dataset,
                   n_rows,
                   labels,
                   metric,
                   cluster_sizes,
                   centers_temp,
                   stream);
  }
}

/**
 * @brief Adjust centers which have small number of entries.
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
  stream.synchronize();
  if (n_clusters == 0) { return false; }
  constexpr static std::array kPrimes{29,   71,   113,  173,  229,  281,  349,  409,  463,  541,
                                      601,  659,  733,  809,  863,  941,  1013, 1069, 1151, 1223,
                                      1291, 1373, 1451, 1511, 1583, 1657, 1733, 1811, 1889, 1987,
                                      2053, 2129, 2213, 2287, 2357, 2423, 2531, 2617, 2687, 2741};
  static size_t i        = 0;
  static size_t i_primes = 0;

  bool adjusted    = false;
  uint32_t average = n_rows / n_clusters;
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

}  // namespace raft::spatial::knn::detail::kmeans
