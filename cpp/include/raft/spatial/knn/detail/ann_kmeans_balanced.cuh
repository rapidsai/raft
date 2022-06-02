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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

namespace raft::spatial::knn::detail {

// predict label of dataset
void _cuann_kmeans_predict_core(const handle_t& handle,
                                const float* centers,  // [numCenters, dimCenters]
                                uint32_t numCenters,
                                uint32_t dimCenters,
                                const float* dataset,  // [numDataset, dimCenters]
                                uint32_t numDataset,
                                uint32_t* labels,  // [numDataset]
                                raft::distance::DistanceType metric,
                                float* workspace,
                                rmm::cuda_stream_view stream)
{
  const uint32_t dimDataset = dimCenters;
  float* sqsumCenters;  // [numCenters]
  float* sqsumDataset;  // [numDataset]
  float* distances;     // [numDataset, numCenters]

  sqsumCenters = workspace;
  sqsumDataset = sqsumCenters + numCenters;
  distances    = sqsumDataset + numDataset;

  float alpha;
  float beta;
  if (metric == raft::distance::DistanceType::InnerProduct) {
    alpha = -1.0;
    beta  = 0.0;
  } else {
    utils::_cuann_sqsum(numCenters, dimCenters, centers, sqsumCenters, stream);
    utils::_cuann_sqsum(numDataset, dimDataset, dataset, sqsumDataset, stream);
    utils::_cuann_outer_add(sqsumDataset, numDataset, sqsumCenters, numCenters, distances, stream);
    alpha = -2.0;
    beta  = 1.0;
  }
  linalg::gemm(handle,
               true,
               false,
               numCenters,
               numDataset,
               dimCenters,
               &alpha,
               centers,
               dimCenters,
               dataset,
               dimDataset,
               &beta,
               distances,
               numCenters,
               stream);
  utils::_cuann_argmin(numDataset, numCenters, distances, labels, stream);
}

//
uint32_t _cuann_kmeans_predict_chunkSize(uint32_t numCenters, uint32_t numDataset)
{
  numCenters     = max(1, numCenters);
  uint32_t chunk = (1 << 20);
  if (chunk > (1 << 28) / numCenters) {
    chunk = (1 << 28) / numCenters;
    chunk += 32;
    chunk -= chunk % 64;
  }
  chunk = min(chunk, numDataset);
  return chunk;
}

//
size_t _cuann_kmeans_predict_bufferSize(uint32_t numCenters,
                                        uint32_t dimCenters,
                                        uint32_t numDataset)
{
  uint32_t chunk = _cuann_kmeans_predict_chunkSize(numCenters, numDataset);
  size_t size    = 0;
  // float *curDataset;  // [chunk, dimCenters]
  size += utils::_cuann_aligned(sizeof(float) * chunk * dimCenters);
  // void *bufDataset;  // [chunk, dimCenters]
  size += utils::_cuann_aligned(sizeof(float) * chunk * dimCenters);
  // float *workspace;
  size += utils::_cuann_aligned(sizeof(float) * (numCenters + chunk + (numCenters * chunk)));
  return size;
}

/**
 * @brief update kmeans centers
 *
 * NB: `centers` and `clusterSize` must be accessible on GPU due to _cuann_divide/_cuann_normalize.
 *      The rest can be both, under assumption that all pointer are accessible from the same place.
 *
 * i.e. two variants are possible:
 *
 *   1. All pointers are on the device.
 *   2. All pointers are on the host, but `centers` and `clusterSize` are accessible from GPU.
 *
 */
template <typename T>
void _cuann_kmeans_update_centers(float* centers,  // [numCenters, dimCenters]
                                  uint32_t numCenters,
                                  uint32_t dimCenters,
                                  const T* dataset,  // [numDataset, dimCenters]
                                  uint32_t numDataset,
                                  uint32_t* labels,  // [numDataset]
                                  raft::distance::DistanceType metric,
                                  uint32_t* clusterSize,  // [numCenters]
                                  float* accumulatedCenters    = nullptr,
                                  rmm::cuda_stream_view stream = rmm::cuda_stream_default)
{
  if (accumulatedCenters == nullptr) {
    // accumulate
    utils::_cuann_memset(centers, 0, sizeof(float) * numCenters * dimCenters, stream);
    utils::_cuann_memset(clusterSize, 0, sizeof(uint32_t) * numCenters, stream);
    utils::_cuann_accumulate_with_label<T>(
      numCenters, dimCenters, centers, clusterSize, numDataset, dataset, labels, stream);
  } else {
    copy(centers, accumulatedCenters, numCenters * dimCenters, stream);
  }

  if (metric == raft::distance::DistanceType::InnerProduct) {
    // normalize
    utils::_cuann_normalize(numCenters, dimCenters, centers, clusterSize, stream);
  } else {
    // average
    utils::_cuann_divide(numCenters, dimCenters, centers, clusterSize, stream);
  }
}

/**
 * @brief predict label of dataset
 *
 * NB: seems that all pointers here are accessed by devicie code only
 *
 */
template <typename T>
void _cuann_kmeans_predict(const handle_t& handle,
                           float* centers,  // [numCenters, dimCenters]
                           uint32_t numCenters,
                           uint32_t dimCenters,
                           const T* dataset,  // [numDataset, dimCenters]
                           uint32_t numDataset,
                           uint32_t* labels,  // [numDataset]
                           raft::distance::DistanceType metric,
                           bool isCenterSet             = true,
                           void* _workspace             = nullptr,
                           float* tempCenters           = nullptr,  // [numCenters, dimCenters]
                           uint32_t* clusterSize        = nullptr,  // [numCenters,]
                           bool updateCenter            = true,
                           rmm::cuda_stream_view stream = rmm::cuda_stream_default)
{
  if (numDataset == 0) {
    RAFT_LOG_WARN("cuann_kmeans_predict: empty dataset (numDataset = %d, numCenters = %d)",
                  numDataset,
                  numCenters);
    return;
  }
  if (!isCenterSet) {
    // If centers are not set, the labels will be determined randomly.
    linalg::writeOnlyUnaryOp(
      labels,
      numDataset,
      [numCenters] __device__(uint32_t * out, uint32_t i) { *out = i % numCenters; },
      stream);
    if (tempCenters != nullptr && clusterSize != nullptr) {
      // update centers
      _cuann_kmeans_update_centers(centers,
                                   numCenters,
                                   dimCenters,
                                   dataset,
                                   numDataset,
                                   labels,
                                   metric,
                                   clusterSize,
                                   nullptr,
                                   stream);
    }
    return;
  }

  uint32_t chunk  = _cuann_kmeans_predict_chunkSize(numCenters, numDataset);
  void* workspace = _workspace;
  rmm::device_buffer sub_workspace(0, stream);

  if (_workspace == nullptr) {
    sub_workspace.resize(_cuann_kmeans_predict_bufferSize(numCenters, dimCenters, numDataset),
                         stream);
    workspace = sub_workspace.data();
  }
  float* curDataset;  // [chunk, dimCenters]
  T* bufDataset;      // [chunk, dimCenters]
  float* workspace_core;
  curDataset = (float*)workspace;
  bufDataset =
    (T*)((uint8_t*)curDataset + utils::_cuann_aligned(sizeof(float) * chunk * dimCenters));
  workspace_core =
    (float*)((uint8_t*)bufDataset + utils::_cuann_aligned(sizeof(float) * chunk * dimCenters));

  if (tempCenters != nullptr && clusterSize != nullptr) {
    utils::_cuann_memset(tempCenters, 0, sizeof(float) * numCenters * dimCenters, stream);
    utils::_cuann_memset(clusterSize, 0, sizeof(uint32_t) * numCenters, stream);
  }

  for (uint64_t is = 0; is < numDataset; is += chunk) {
    uint64_t ie       = min(is + chunk, (uint64_t)numDataset);
    uint32_t nDataset = ie - is;

    copy(bufDataset, dataset + is * dimCenters, nDataset * dimCenters, stream);
    handle.sync_stream(stream);

    if constexpr (std::is_same_v<T, float>) {
      // No need to copy floats
      curDataset = bufDataset;
    } else {
      linalg::unaryOp(
        curDataset, bufDataset, nDataset * dimCenters, utils::mapping<T, float>{}, stream);
    }

    // predict
    _cuann_kmeans_predict_core(handle,
                               centers,
                               numCenters,
                               dimCenters,
                               curDataset,
                               nDataset,
                               labels + is,
                               metric,
                               workspace_core,
                               stream);

    if ((tempCenters != nullptr) && (clusterSize != nullptr)) {
      // accumulate
      utils::_cuann_accumulate_with_label<float>(numCenters,
                                                 dimCenters,
                                                 tempCenters,
                                                 clusterSize,
                                                 nDataset,
                                                 curDataset,
                                                 labels + is,
                                                 stream);
    }
  }

  if ((tempCenters != nullptr) && (clusterSize != nullptr) && updateCenter) {
    _cuann_kmeans_update_centers(centers,
                                 numCenters,
                                 dimCenters,
                                 dataset,
                                 numDataset,
                                 labels,
                                 metric,
                                 clusterSize,
                                 tempCenters,
                                 stream);
  }
}

/**
 * @brief adjust centers which have small number of entries
 *
 * NB: all pointers are used on the host side.
 */
template <typename T>
bool _cuann_kmeans_adjust_centers(float* centers,  // [numCenters, dimCenters]
                                  uint32_t numCenters,
                                  uint32_t dimCenters,
                                  const T* dataset,  // [numDataset, dimCenters]
                                  uint32_t numDataset,
                                  const uint32_t* labels,  // [numDataset]
                                  raft::distance::DistanceType metric,
                                  const uint32_t* clusterSize,  // [numCenters]
                                  float threshold,
                                  rmm::cuda_stream_view stream)
{
  stream.synchronize();
  if (numCenters == 0) { return false; }
  bool adjusted                = false;
  static uint32_t i            = 0;
  static uint32_t iPrimes      = 0;
  constexpr uint32_t numPrimes = 40;
  uint32_t primes[numPrimes]   = {29,   71,   113,  173,  229,  281,  349,  409,  463,  541,
                                601,  659,  733,  809,  863,  941,  1013, 1069, 1151, 1223,
                                1291, 1373, 1451, 1511, 1583, 1657, 1733, 1811, 1889, 1987,
                                2053, 2129, 2213, 2287, 2357, 2423, 2531, 2617, 2687, 2741};
  uint32_t average             = numDataset / numCenters;
  uint32_t ofst;

  do {
    iPrimes = (iPrimes + 1) % numPrimes;
    ofst    = primes[iPrimes];
  } while (numDataset % ofst == 0);
  uint32_t count = 0;

  for (uint32_t l = 0; l < numCenters; l++) {
    if (clusterSize[l] > (uint32_t)(average * threshold)) continue;
    do {
      i = (i + ofst) % numDataset;
    } while (clusterSize[labels[i]] < average);
    uint32_t li = labels[i];
    float sqsum = 0.0;
    for (uint32_t j = 0; j < dimCenters; j++) {
      constexpr float kWc = 7.0;
      constexpr float kWd = 1.0;
      float val           = 0;
      val += kWc * centers[j + ((uint64_t)dimCenters * li)];
      val += kWd * dataset[j + ((uint64_t)dimCenters * i)] / utils::config<T>::kDivisor;
      val /= kWc + kWd;
      sqsum += val * val;
      centers[j + ((uint64_t)dimCenters * l)] = val;
    }
    if (metric == raft::distance::DistanceType::InnerProduct) {
      sqsum = sqrt(sqsum);
      for (uint32_t j = 0; j < dimCenters; j++) {
        centers[j + ((uint64_t)dimCenters * l)] /= sqsum;
      }
    }
    adjusted = true;
    count += 1;
  }
  stream.synchronize();
  return adjusted;
}

}  // namespace raft::spatial::knn::detail
