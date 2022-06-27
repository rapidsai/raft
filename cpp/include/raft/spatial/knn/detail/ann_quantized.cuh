/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include "../ann_common.hpp"
#include "../ivf_flat.cuh"
#include "knn_brute_force_faiss.cuh"

#include "common_faiss.h"
#include "processing.hpp"
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>

#include <raft/distance/distance.cuh>
#include <raft/distance/distance_type.hpp>
#include <raft/label/classlabels.cuh>
#include <raft/spatial/knn/faiss_mr.hpp>

#include <rmm/cuda_stream_view.hpp>

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

namespace raft ::spatial ::knn::detail {

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
void approx_knn_ivfflat_build_index(knnIndex* index,
                                    const ivf_index_params& params,
                                    IntType n,
                                    IntType D)
{
  faiss::gpu::GpuIndexIVFFlatConfig config;
  config.device                  = index->device;
  faiss::MetricType faiss_metric = build_faiss_metric(params.metric);
  index->index.reset(
    new faiss::gpu::GpuIndexIVFFlat(index->gpu_res.get(), D, params.n_lists, faiss_metric, config));
}

template <typename IntType = int>
void approx_knn_ivfpq_build_index(knnIndex* index,
                                  const ivf_pq_index_params& params,
                                  IntType n,
                                  IntType D)
{
  faiss::gpu::GpuIndexIVFPQConfig config;
  config.device                  = index->device;
  config.usePrecomputedTables    = params.use_precomputed_tables;
  config.interleavedLayout       = params.n_bits != 8;
  faiss::MetricType faiss_metric = build_faiss_metric(params.metric);
  index->index.reset(new faiss::gpu::GpuIndexIVFPQ(index->gpu_res.get(),
                                                   D,
                                                   params.n_lists,
                                                   params.n_subquantizers,
                                                   params.n_bits,
                                                   faiss_metric,
                                                   config));
}

template <typename IntType = int>
void approx_knn_ivfsq_build_index(knnIndex* index,
                                  const ivf_sq_index_params& params,
                                  IntType n,
                                  IntType D)
{
  faiss::gpu::GpuIndexIVFScalarQuantizerConfig config;
  config.device                                     = index->device;
  faiss::MetricType faiss_metric                    = build_faiss_metric(params.metric);
  faiss::ScalarQuantizer::QuantizerType faiss_qtype = build_faiss_qtype(params.qtype);
  index->index.reset(new faiss::gpu::GpuIndexIVFScalarQuantizer(
    index->gpu_res.get(), D, params.n_lists, faiss_qtype, faiss_metric, params.encode_residual));
}

template <typename T = float, typename IntType = int>
void approx_knn_build_index(const handle_t& handle,
                            knnIndex* index,
                            const knn_index_params& params,
                            T* index_array,
                            IntType n,
                            IntType D)
{
  auto stream      = handle.get_stream();
  auto metric      = params.metric;
  index->index     = nullptr;
  index->metric    = metric;
  index->metricArg = params.metric_arg;
  auto ivf_ft_pams = dynamic_cast<const ivf_flat::index_params*>(&params);
  auto ivf_pq_pams = dynamic_cast<const ivf_pq_index_params*>(&params);
  auto ivf_sq_pams = dynamic_cast<const ivf_sq_index_params*>(&params);

  if constexpr (std::is_same<T, float>{}) {
    index->metric_processor = create_processor<float>(metric, n, D, 0, false, stream);
  }
  if constexpr (std::is_same<T, float>{}) { index->metric_processor->preprocess(index_array); }

  if (ivf_ft_pams && (metric == raft::distance::DistanceType::L2SqrtExpanded ||
                      metric == raft::distance::DistanceType::L2SqrtUnexpanded ||
                      metric == raft::distance::DistanceType::L2Unexpanded ||
                      metric == raft::distance::DistanceType::L2Expanded ||
                      metric == raft::distance::DistanceType::InnerProduct)) {
    index->ivf_flat<T>() = std::make_unique<ivf_flat::index<T>>(
      ivf_flat::build(handle, *ivf_ft_pams, index_array, n, D, stream));
  } else {
    RAFT_CUDA_TRY(cudaGetDevice(&(index->device)));
    index->gpu_res.reset(new raft::spatial::knn::RmmGpuResources());
    index->gpu_res->noTempMemory();
    index->gpu_res->setDefaultStream(index->device, stream);
    if (ivf_ft_pams) {
      approx_knn_ivfflat_build_index(index, *ivf_ft_pams, n, D);
    } else if (ivf_pq_pams) {
      approx_knn_ivfpq_build_index(index, *ivf_pq_pams, n, D);
    } else if (ivf_sq_pams) {
      approx_knn_ivfsq_build_index(index, *ivf_sq_pams, n, D);
    } else {
      RAFT_FAIL("Unrecognized index type.");
    }
    if constexpr (std::is_same<T, float>{}) {
      index->index->train(n, index_array);
      index->index->add(n, index_array);
    } else {
      RAFT_FAIL("FAISS-based index supports only float data.");
    }
  }

  if constexpr (std::is_same<T, float>{}) { index->metric_processor->revert(index_array); }
}

template <typename T = float, typename IntType = int>
void approx_knn_search(const handle_t& handle,
                       float* distances,
                       int64_t* indices,
                       knnIndex* index,
                       const knn_search_params& params,
                       IntType k,
                       T* query_array,
                       IntType n)
{
  auto ivf_ft_pams = dynamic_cast<const ivf_flat::search_params*>(&params);
  auto ivf_pams    = dynamic_cast<const ivf_search_params*>(&params);
  auto faiss_ivf   = dynamic_cast<GpuIndexIVF*>(index->index.get());
  if (ivf_pams && faiss_ivf) { faiss_ivf->setNumProbes(ivf_pams->n_probes); }

  if constexpr (std::is_same<T, float>{}) { index->metric_processor->preprocess(query_array); }

  // search
  if (faiss_ivf) {
    if constexpr (std::is_same<T, float>{}) {
      faiss_ivf->search(n, query_array, k, distances, indices);
    } else {
      RAFT_FAIL("FAISS-based index supports only float data.");
    }
  } else if (ivf_ft_pams) {
    ivf_flat::search(handle,
                     *ivf_ft_pams,
                     *(index->ivf_flat<T>()),
                     query_array,
                     n,
                     k,
                     (size_t*)indices,
                     distances,
                     handle.get_stream());
  } else {
    RAFT_FAIL("The model is not trained");
  }

  // revert changes to the query
  if constexpr (std::is_same<T, float>{}) { index->metric_processor->revert(query_array); }

  // perform post-processing to show the real distances
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
  if constexpr (std::is_same<T, float>{}) { index->metric_processor->postprocess(distances); }
}

}  // namespace raft::spatial::knn::detail
