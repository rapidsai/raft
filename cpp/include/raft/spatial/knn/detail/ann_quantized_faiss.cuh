/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include "knn_brute_force_faiss.cuh"

#include "common_faiss.h"
#include "processing.hpp"

#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>
#include "processing.hpp"

#include <label/classlabels.cuh>
#include <raft/distance/distance.hpp>

#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/utils/Heap.h>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/Select.cuh>
#include <faiss/gpu/utils/Tensor.cuh>

#include <thrust/iterator/transform_iterator.h>

#include <raft/distance/distance_type.hpp>

#include <cuml/neighbors/knn.hpp>

#include <iostream>
#include <set>

namespace raft {
namespace spatial {
namespace knn {
namespace detail {

inline faiss::ScalarQuantizer::QuantizerType build_faiss_qtype(QuantizerType qtype)
{
  switch (qtype) {
    case QuantizerType::QT_8bit: return faiss::ScalarQuantizer::QuantizerType::QT_8bit;
    case QuantizerType::QT_8bit_uniform:
      return faiss::ScalarQuantizer::QuantizerType::QT_8bit_uniform;
    case QuantizerType::QT_4bit_uniform:
      return faiss::ScalarQuantizer::QuantizerType::QT_4bit_uniform;
    case QuantizerType::QT_fp16: return faiss::ScalarQuantizer::QuantizerType::QT_fp16;
    case QuantizerType::QT_8bit_direct:
      return faiss::ScalarQuantizer::QuantizerType::QT_8bit_direct;
    case QuantizerType::QT_6bit: return faiss::ScalarQuantizer::QuantizerType::QT_6bit;
    default: return (faiss::ScalarQuantizer::QuantizerType)qtype;
  }
}

template <typename IntType = int>
void approx_knn_ivfflat_build_index(
  knnIndex* index, IVFParam* params, raft::distance::DistanceType metric, IntType n, IntType D)
{
  faiss::gpu::GpuIndexIVFFlatConfig config;
  config.device                  = index->device;
  faiss::MetricType faiss_metric = build_faiss_metric(metric);
  faiss::gpu::GpuIndexIVFFlat* faiss_index =
    new faiss::gpu::GpuIndexIVFFlat(index->gpu_res, D, params->nlist, faiss_metric, config);
  faiss_index->setNumProbes(params->nprobe);
  index->index = faiss_index;
}

template <typename IntType = int>
void approx_knn_ivfpq_build_index(
  knnIndex* index, IVFPQParam* params, raft::distance::DistanceType metric, IntType n, IntType D)
{
  faiss::gpu::GpuIndexIVFPQConfig config;
  config.device                          = index->device;
  config.usePrecomputedTables            = params->usePrecomputedTables;
  config.interleavedLayout               = params->n_bits != 8;
  faiss::MetricType faiss_metric         = build_faiss_metric(metric);
  faiss::gpu::GpuIndexIVFPQ* faiss_index = new faiss::gpu::GpuIndexIVFPQ(
    index->gpu_res, D, params->nlist, params->M, params->n_bits, faiss_metric, config);
  faiss_index->setNumProbes(params->nprobe);
  index->index = faiss_index;
}

template <typename IntType = int>
void approx_knn_ivfsq_build_index(
  knnIndex* index, IVFSQParam* params, raft::distance::DistanceType metric, IntType n, IntType D)
{
  faiss::gpu::GpuIndexIVFScalarQuantizerConfig config;
  config.device                                       = index->device;
  faiss::MetricType faiss_metric                      = build_faiss_metric(metric);
  faiss::ScalarQuantizer::QuantizerType faiss_qtype   = build_faiss_qtype(params->qtype);
  faiss::gpu::GpuIndexIVFScalarQuantizer* faiss_index = new faiss::gpu::GpuIndexIVFScalarQuantizer(
    index->gpu_res, D, params->nlist, faiss_qtype, faiss_metric, params->encodeResidual);
  faiss_index->setNumProbes(params->nprobe);
  index->index = faiss_index;
}

template <typename IntType = int>
void approx_knn_build_index(raft::handle_t& handle,
                            raft::spatial::knn::knnIndex* index,
                            raft::spatial::knn::knnIndexParam* params,
                            raft::distance::DistanceType metric,
                            float metricArg,
                            float* index_array,
                            IntType n,
                            IntType D)
{
  int device;
  RAFT_CUDA_TRY(cudaGetDevice(&device));

  faiss::gpu::StandardGpuResources* gpu_res = new faiss::gpu::StandardGpuResources();
  gpu_res->noTempMemory();
  gpu_res->setDefaultStream(device, handle.get_stream());
  index->gpu_res   = gpu_res;
  index->device    = device;
  index->index     = nullptr;
  index->metric    = metric;
  index->metricArg = metricArg;

  // perform preprocessing
  // k set to 0 (unused during preprocessing / revertion)
  std::unique_ptr<MetricProcessor<float>> query_metric_processor =
    create_processor<float>(metric, n, D, 0, false, handle.get_stream());

  query_metric_processor->preprocess(index_array);

  if (dynamic_cast<IVFFlatParam*>(params)) {
    IVFFlatParam* IVFFlat_param = dynamic_cast<IVFFlatParam*>(params);
    approx_knn_ivfflat_build_index(index, IVFFlat_param, metric, n, D);
    std::vector<float> h_index_array(n * D);
    raft::update_host(h_index_array.data(), index_array, h_index_array.size(), handle.get_stream());
    query_metric_processor->revert(index_array);
    index->index->train(n, h_index_array.data());
    index->index->add(n, h_index_array.data());
  } else {
    if (dynamic_cast<IVFPQParam*>(params)) {
      IVFPQParam* IVFPQ_param = dynamic_cast<IVFPQParam*>(params);
      approx_knn_ivfpq_build_index(index, IVFPQ_param, metric, n, D);
    } else if (dynamic_cast<IVFSQParam*>(params)) {
      IVFSQParam* IVFSQ_param = dynamic_cast<IVFSQParam*>(params);
      approx_knn_ivfsq_build_index(index, IVFSQ_param, metric, n, D);
    } else {
      ASSERT(index->index, "KNN index could not be initialized");
    }

    index->index->train(n, index_array);
    index->index->add(n, index_array);
    query_metric_processor->revert(index_array);
  }
}

template <typename IntType = int>
void approx_knn_search(raft::handle_t& handle,
                       float* distances,
                       int64_t* indices,
                       raft::spatial::knn::knnIndex* index,
                       IntType k,
                       float* query_array,
                       IntType n)
{
  // perform preprocessing
  std::unique_ptr<MetricProcessor<float>> query_metric_processor =
    create_processor<float>(index->metric, n, index->index->d, k, false, handle.get_stream());

  query_metric_processor->preprocess(query_array);
  index->index->search(n, query_array, k, distances, indices);
  query_metric_processor->revert(query_array);

  // Perform necessary post-processing
  if (index->metric == raft::distance::DistanceType::L2SqrtExpanded ||
      index->metric == raft::distance::DistanceType::L2SqrtUnexpanded ||
      index->metric == raft::distance::DistanceType::LpUnexpanded) {
    /**
     * post-processing
     */
    float p = 0.5;  // standard l2
    if (index->metric == raft::distance::DistanceType::LpUnexpanded) p = 1.0 / index->metricArg;
    raft::linalg::unaryOp<float>(
      distances,
      distances,
      n * k,
      [p] __device__(float input) { return powf(input, p); },
      handle.get_stream());
  }
  query_metric_processor->postprocess(distances);
}

}  // namespace detail
}  // namespace knn
}  // namespace spatial
}  // namespace raft