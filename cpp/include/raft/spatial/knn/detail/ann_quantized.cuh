/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
#include "../ivf_flat.cuh"
#include "knn_brute_force_faiss.cuh"

#include "common_faiss.h"
#include "processing.cuh"
#include <raft/core/operators.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <raft/distance/distance.cuh>
#include <raft/distance/distance_types.hpp>
#include <raft/label/classlabels.cuh>
#include <raft/spatial/knn/faiss_mr.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuIndexIVFScalarQuantizer.h>

#include <thrust/iterator/transform_iterator.h>

namespace raft::spatial::knn::detail {

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
                                    const IVFFlatParam& params,
                                    IntType n,
                                    IntType D)
{
  faiss::gpu::GpuIndexIVFFlatConfig config;
  config.device                  = index->device;
  faiss::MetricType faiss_metric = build_faiss_metric(index->metric);
  index->index.reset(
    new faiss::gpu::GpuIndexIVFFlat(index->gpu_res.get(), D, params.nlist, faiss_metric, config));
}

template <typename IntType = int>
void approx_knn_ivfpq_build_index(knnIndex* index, const IVFPQParam& params, IntType n, IntType D)
{
  faiss::gpu::GpuIndexIVFPQConfig config;
  config.device                  = index->device;
  config.usePrecomputedTables    = params.usePrecomputedTables;
  config.interleavedLayout       = params.n_bits != 8;
  faiss::MetricType faiss_metric = build_faiss_metric(index->metric);
  index->index.reset(new faiss::gpu::GpuIndexIVFPQ(
    index->gpu_res.get(), D, params.nlist, params.M, params.n_bits, faiss_metric, config));
}

template <typename IntType = int>
void approx_knn_ivfsq_build_index(knnIndex* index, const IVFSQParam& params, IntType n, IntType D)
{
  faiss::gpu::GpuIndexIVFScalarQuantizerConfig config;
  config.device                                     = index->device;
  faiss::MetricType faiss_metric                    = build_faiss_metric(index->metric);
  faiss::ScalarQuantizer::QuantizerType faiss_qtype = build_faiss_qtype(params.qtype);
  index->index.reset(new faiss::gpu::GpuIndexIVFScalarQuantizer(
    index->gpu_res.get(), D, params.nlist, faiss_qtype, faiss_metric, params.encodeResidual));
}

template <typename T = float, typename IntType = int>
void approx_knn_build_index(const handle_t& handle,
                            knnIndex* index,
                            knnIndexParam* params,
                            raft::distance::DistanceType metric,
                            float metricArg,
                            T* index_array,
                            IntType n,
                            IntType D)
{
  auto stream      = handle.get_stream();
  index->index     = nullptr;
  index->metric    = metric;
  index->metricArg = metricArg;
  if (dynamic_cast<const IVFParam*>(params)) {
    index->nprobe = dynamic_cast<const IVFParam*>(params)->nprobe;
  }
  auto ivf_ft_pams = dynamic_cast<IVFFlatParam*>(params);
  auto ivf_pq_pams = dynamic_cast<IVFPQParam*>(params);
  auto ivf_sq_pams = dynamic_cast<IVFSQParam*>(params);

  if constexpr (std::is_same_v<T, float>) {
    index->metric_processor = create_processor<float>(metric, n, D, 0, false, stream);
  }
  if constexpr (std::is_same_v<T, float>) { index->metric_processor->preprocess(index_array); }

  if (ivf_ft_pams && (metric == raft::distance::DistanceType::L2Unexpanded ||
                      metric == raft::distance::DistanceType::L2Expanded ||
                      metric == raft::distance::DistanceType::L2SqrtExpanded ||
                      metric == raft::distance::DistanceType::L2SqrtUnexpanded ||
                      metric == raft::distance::DistanceType::InnerProduct)) {
    auto new_params               = from_legacy_index_params(*ivf_ft_pams, metric, metricArg);
    index->ivf_flat<T, int64_t>() = std::make_unique<const ivf_flat::index<T, int64_t>>(
      ivf_flat::build(handle, new_params, index_array, int64_t(n), D));
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
    if constexpr (std::is_same_v<T, float>) {
      index->index->train(n, index_array);
      index->index->add(n, index_array);
    } else {
      RAFT_FAIL("FAISS-based index supports only float data.");
    }
  }

  if constexpr (std::is_same_v<T, float>) { index->metric_processor->revert(index_array); }
}

template <typename T = float, typename IntType = int>
void approx_knn_search(const handle_t& handle,
                       float* distances,
                       int64_t* indices,
                       knnIndex* index,
                       IntType k,
                       T* query_array,
                       IntType n)
{
  auto faiss_ivf = dynamic_cast<GpuIndexIVF*>(index->index.get());
  if (faiss_ivf) { faiss_ivf->setNumProbes(index->nprobe); }

  if constexpr (std::is_same_v<T, float>) {
    index->metric_processor->preprocess(query_array);
    index->metric_processor->set_num_queries(k);
  }

  // search
  if (faiss_ivf) {
    if constexpr (std::is_same_v<T, float>) {
      faiss_ivf->search(n, query_array, k, distances, indices);
    } else {
      RAFT_FAIL("FAISS-based index supports only float data.");
    }
  } else if (index->ivf_flat<T, int64_t>()) {
    ivf_flat::search_params params;
    params.n_probes = index->nprobe;
    ivf_flat::search(
      handle, params, *(index->ivf_flat<T, int64_t>()), query_array, n, k, indices, distances);
  } else {
    RAFT_FAIL("The model is not trained");
  }

  // revert changes to the query
  if constexpr (std::is_same_v<T, float>) { index->metric_processor->revert(query_array); }

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
      distances, distances, n * k, raft::pow_const_op<float>(p), handle.get_stream());
  }
  if constexpr (std::is_same_v<T, float>) { index->metric_processor->postprocess(distances); }
}

}  // namespace raft::spatial::knn::detail
