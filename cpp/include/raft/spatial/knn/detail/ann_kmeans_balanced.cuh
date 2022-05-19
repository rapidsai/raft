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
#include "knn_brute_force_faiss.cuh"

#include "common_faiss.h"
#include "processing.hpp"

#include "processing.hpp"
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/interruptible.hpp>

//#include <label/classlabels.cuh>
#include <raft/distance/distance.hpp>
#include <raft/spatial/knn/faiss_mr.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/Select.cuh>
#include <faiss/gpu/utils/Tensor.cuh>
#include <faiss/utils/Heap.h>

#include <thrust/iterator/transform_iterator.h>

#include <raft/distance/distance_type.hpp>

#include <iostream>
#include <set>

namespace raft {
namespace spatial {
namespace knn {
namespace detail {
// namespace kmeans {

// predict label of dataset
void _cuann_kmeans_predict_core(cublasHandle_t cublasHandle,
                                const float* centers,  // [numCenters, dimCenters]
                                uint32_t numCenters,
                                uint32_t dimCenters,
                                const float* dataset,  // [numDataset, dimCenters]
                                uint32_t numDataset,
                                uint32_t* labels,  // [numDataset]
                                raft::distance::DistanceType metric,
                                float* workspace)
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
    utils::_cuann_sqsum(numCenters, dimCenters, centers, sqsumCenters);
    utils::_cuann_sqsum(numDataset, dimDataset, dataset, sqsumDataset);
    utils::_cuann_outer_add(sqsumDataset, numDataset, sqsumCenters, numCenters, distances);
    alpha = -2.0;
    beta  = 1.0;
  }
  cublasGemmEx(cublasHandle,
               CUBLAS_OP_T,
               CUBLAS_OP_N,
               numCenters,
               numDataset,
               dimCenters,
               &alpha,
               centers,
               CUDA_R_32F,
               dimCenters,
               dataset,
               CUDA_R_32F,
               dimDataset,
               &beta,
               distances,
               CUDA_R_32F,
               numCenters,
               CUBLAS_COMPUTE_32F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  utils::_cuann_argmin(numDataset, numCenters, distances, labels);
}

//
uint32_t _cuann_kmeans_predict_chunkSize(uint32_t numCenters, uint32_t numDataset)
{
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
void _cuann_kmeans_update_centers(float* centers,  // [numCenters, dimCenters]
                                  uint32_t numCenters,
                                  uint32_t dimCenters,
                                  const void* dataset,  // [numDataset, dimCenters]
                                  cudaDataType_t dtype,
                                  uint32_t numDataset,
                                  uint32_t* labels,  // [numDataset]
                                  raft::distance::DistanceType metric,
                                  uint32_t* clusterSize,  // [numCenters]
                                  float* accumulatedCenters = NULL)
{
  if (accumulatedCenters == NULL) {
    // accumulate
    utils::_cuann_memset(centers, 0, sizeof(float) * numCenters * dimCenters);
    utils::_cuann_memset(clusterSize, 0, sizeof(uint32_t) * numCenters);
    if (dtype == CUDA_R_32F) {
      utils::_cuann_accumulate_with_label<float>(
        numCenters, dimCenters, centers, clusterSize, numDataset, (const float*)dataset, labels);
    } else if (dtype == CUDA_R_8U) {
      constexpr float divisor = 256.0;
      utils::_cuann_accumulate_with_label<uint8_t>(numCenters,
                                                   dimCenters,
                                                   centers,
                                                   clusterSize,
                                                   numDataset,
                                                   (const uint8_t*)dataset,
                                                   labels,
                                                   divisor);
    } else if (dtype == CUDA_R_8I) {
      constexpr float divisor = 128.0;
      utils::_cuann_accumulate_with_label<int8_t>(numCenters,
                                                  dimCenters,
                                                  centers,
                                                  clusterSize,
                                                  numDataset,
                                                  (const int8_t*)dataset,
                                                  labels,
                                                  divisor);
    }
  } else {
    copy(centers, accumulatedCenters, numCenters * dimCenters, rmm::cuda_stream_default);
    interruptible::synchronize(rmm::cuda_stream_default);
  }

  if (metric == raft::distance::DistanceType::InnerProduct) {
    // normalize
    utils::_cuann_normalize(numCenters, dimCenters, centers, clusterSize);
  } else {
    // average
    utils::_cuann_divide(numCenters, dimCenters, centers, clusterSize);
  }
}


/**
 * @brief predict label of dataset
 *
 * NB: seems that all pointers here are accessed by devicie code only
 *
 */
void _cuann_kmeans_predict(cublasHandle_t cublasHandle,
                           float* centers,  // [numCenters, dimCenters]
                           uint32_t numCenters,
                           uint32_t dimCenters,
                           const void* dataset,  // [numDataset, dimCenters]
                           cudaDataType_t dtype,
                           uint32_t numDataset,
                           uint32_t* labels,  // [numDataset]
                           raft::distance::DistanceType metric,
                           bool isCenterSet      = true,
                           void* _workspace      = NULL,
                           float* tempCenters    = NULL,  // [numCenters, dimCenters]
                           uint32_t* clusterSize = NULL,  // [numCenters,]
                           bool updateCenter     = true)
{
  rmm::cuda_stream_view stream = rmm::cuda_stream_default;
  if (!isCenterSet) {
    // If centers are not set, the labels will be determined randomly.
    linalg::writeOnlyUnaryOp(
      labels,
      numDataset,
      [numCenters] __device__(uint32_t * out, uint32_t i) { *out = i % numCenters; },
      stream);
    if (tempCenters != NULL && clusterSize != NULL) {
      // update centers
      _cuann_kmeans_update_centers(
        centers, numCenters, dimCenters, dataset, dtype, numDataset, labels, metric, clusterSize);
    }
    return;
  }

  uint32_t chunk  = _cuann_kmeans_predict_chunkSize(numCenters, numDataset);
  void* workspace = _workspace;
  rmm::device_buffer sub_workspace(0, stream);

  if (_workspace == NULL) {
    sub_workspace.resize(_cuann_kmeans_predict_bufferSize(numCenters, dimCenters, numDataset),
                         stream);
    workspace = sub_workspace.data();
  }
  float* curDataset;  // [chunk, dimCenters]
  void* bufDataset;   // [chunk, dimCenters]
  float* workspace_core;
  curDataset = (float*)workspace;
  bufDataset =
    (void*)((uint8_t*)curDataset + utils::_cuann_aligned(sizeof(float) * chunk * dimCenters));
  workspace_core =
    (float*)((uint8_t*)bufDataset + utils::_cuann_aligned(sizeof(float) * chunk * dimCenters));

  if (tempCenters != NULL && clusterSize != NULL) {
    utils::_cuann_memset(tempCenters, 0, sizeof(float) * numCenters * dimCenters);
    utils::_cuann_memset(clusterSize, 0, sizeof(uint32_t) * numCenters);
  }

  auto elem_size = utils::cuda_datatype_size(dtype);
  for (uint64_t is = 0; is < numDataset; is += chunk) {
    uint64_t ie       = min(is + chunk, (uint64_t)numDataset);
    uint32_t nDataset = ie - is;

    RAFT_CUDA_TRY(
      cudaMemcpy(bufDataset,
                 reinterpret_cast<const uint8_t*>(dataset) + is * dimCenters * elem_size,
                 elem_size * nDataset * dimCenters,
                 cudaMemcpyDefault));

    if (dtype == CUDA_R_32F) {
      // No need to copy when dtype is CUDA_R_32F
      curDataset = (float*)bufDataset;
    } else if (dtype == CUDA_R_8U) {
      float divisor = 256.0;
      utils::_cuann_copy<uint8_t, float>(nDataset,
                                         dimCenters,
                                         (const uint8_t*)bufDataset,
                                         dimCenters,
                                         curDataset,
                                         dimCenters,
                                         divisor);
    } else if (dtype == CUDA_R_8I) {
      float divisor = 128.0;
      utils::_cuann_copy<int8_t, float>(nDataset,
                                        dimCenters,
                                        (const int8_t*)bufDataset,
                                        dimCenters,
                                        curDataset,
                                        dimCenters,
                                        divisor);
    }

    // predict
    _cuann_kmeans_predict_core(cublasHandle,
                               centers,
                               numCenters,
                               dimCenters,
                               curDataset,
                               nDataset,
                               labels + is,
                               metric,
                               workspace_core);

    if ((tempCenters != NULL) && (clusterSize != NULL)) {
      // accumulate
      utils::_cuann_accumulate_with_label<float>(
        numCenters, dimCenters, tempCenters, clusterSize, nDataset, curDataset, labels + is);
    }
  }

  if ((tempCenters != NULL) && (clusterSize != NULL) && updateCenter) {
    _cuann_kmeans_update_centers(centers,
                                 numCenters,
                                 dimCenters,
                                 dataset,
                                 dtype,
                                 numDataset,
                                 labels,
                                 metric,
                                 clusterSize,
                                 tempCenters);
  }
}

/**
 * @brief adjust centers which have small number of entries
 *
 * NB: all pointers are used on the CPU side.
 */
bool _cuann_kmeans_adjust_centers(float* centers,  // [numCenters, dimCenters]
                                  uint32_t numCenters,
                                  uint32_t dimCenters,
                                  const void* dataset,  // [numDataset, dimCenters]
                                  cudaDataType_t dtype,
                                  uint32_t numDataset,
                                  const uint32_t* labels,  // [numDataset]
                                  raft::distance::DistanceType metric,
                                  const uint32_t* clusterSize,  // [numCenters]
                                  float threshold)
{
  // cudaDeviceSynchronize();
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
  if (dtype != CUDA_R_32F && dtype != CUDA_R_8U && dtype != CUDA_R_8I) {
    fprintf(stderr, "(%s, %d) Unsupported dtype (%d)\n", __func__, __LINE__, dtype);
  }
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
      float val = centers[j + ((uint64_t)dimCenters * li)] * 7.0;
      if (dtype == CUDA_R_32F) {
        val += ((float*)dataset)[j + ((uint64_t)dimCenters * i)];
      } else if (dtype == CUDA_R_8U) {
        float divisor = 256.0;
        val += ((uint8_t*)dataset)[j + ((uint64_t)dimCenters * i)] / divisor;
      } else if (dtype == CUDA_R_8I) {
        float divisor = 128.0;
        val += ((int8_t*)dataset)[j + ((uint64_t)dimCenters * i)] / divisor;
      }
      val /= 8.0;
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

#ifdef CUANN_DEBUG
  if (count > 0) {
    fprintf(stderr,
            "(%s) num adjusted: %u / %u, threshold: %d\n",
            __func__,
            count,
            numCenters,
            (int)(average * threshold));
  }
#endif
  return adjusted;
}

//}  // namespace kmeans
}  // namespace detail
}  // namespace knn
}  // namespace spatial
}  // namespace raft
