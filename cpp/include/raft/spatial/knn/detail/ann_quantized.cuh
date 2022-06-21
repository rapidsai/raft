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
#include "knn_brute_force_faiss.cuh"

#include "common_faiss.h"
#include "processing.hpp"

#include "processing.hpp"
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>

#include <raft/distance/distance.cuh>
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

#include <raft/distance/distance_type.hpp>

#include "ann_ivf_flat.cuh"

#include <iostream>
#include <set>

#define IVF_FAISS 0

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
void approx_knn_ivfflat_build_index(knnIndex* index,
                                    const ivf_index_params& params,
                                    IntType n,
                                    IntType D)
{
  faiss::gpu::GpuIndexIVFFlatConfig config;
  config.device                  = index->device;
  faiss::MetricType faiss_metric = build_faiss_metric(params.metric);
  faiss::gpu::GpuIndexIVFFlat* faiss_index =
    new faiss::gpu::GpuIndexIVFFlat(index->gpu_res, D, params.n_lists, faiss_metric, config);
  index->index = faiss_index;
}

template <typename T = float, typename IntType = int>
void approx_knn_cuivfl_ivfflat_build_index(const raft::handle_t& handle,
                                           knnIndex* index,
                                           const ivf_flat_index_params& params,
                                           T* dataset,
                                           IntType n,
                                           IntType D)
{
  index->ivf_flat<T>() = std::make_unique<detail::ivf_flat_handle<T>>(handle);
  index->ivf_flat<T>()->build(dataset, n, D, params, handle.get_stream());
}

template <typename IntType = int>
void approx_knn_ivfpq_build_index(knnIndex* index,
                                  const ivf_pq_index_params& params,
                                  IntType n,
                                  IntType D)
{
  faiss::gpu::GpuIndexIVFPQConfig config;
  config.device                          = index->device;
  config.usePrecomputedTables            = params.use_precomputed_tables;
  config.interleavedLayout               = params.n_bits != 8;
  faiss::MetricType faiss_metric         = build_faiss_metric(params.metric);
  faiss::gpu::GpuIndexIVFPQ* faiss_index = new faiss::gpu::GpuIndexIVFPQ(
    index->gpu_res, D, params.n_lists, params.n_subquantizers, params.n_bits, faiss_metric, config);
  index->index = faiss_index;
}

template <typename IntType = int>
void approx_knn_ivfsq_build_index(knnIndex* index,
                                  const ivf_sq_index_params& params,
                                  IntType n,
                                  IntType D)
{
  faiss::gpu::GpuIndexIVFScalarQuantizerConfig config;
  config.device                                       = index->device;
  faiss::MetricType faiss_metric                      = build_faiss_metric(params.metric);
  faiss::ScalarQuantizer::QuantizerType faiss_qtype   = build_faiss_qtype(params.qtype);
  faiss::gpu::GpuIndexIVFScalarQuantizer* faiss_index = new faiss::gpu::GpuIndexIVFScalarQuantizer(
    index->gpu_res, D, params.n_lists, faiss_qtype, faiss_metric, params.encode_residual);
  index->index = faiss_index;
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
  int device;
  RAFT_CUDA_TRY(cudaGetDevice(&device));
  index->device    = device;
  auto ivf_ft_pams = dynamic_cast<const ivf_flat_index_params*>(&params);
  auto ivf_pq_pams = dynamic_cast<const ivf_pq_index_params*>(&params);
  auto ivf_sq_pams = dynamic_cast<const ivf_sq_index_params*>(&params);

  // perform preprocessing
  // k set to 0 (unused during preprocessing / revertion)
  if constexpr (std::is_same<T, uint8_t>{} || std::is_same<T, int8_t>{}) {
    if (ivf_ft_pams) {
      approx_knn_cuivfl_ivfflat_build_index(handle, index, *ivf_ft_pams, index_array, n, D);
    } else {
      RAFT_FAIL("IVF Flat algorithm required to fit int8 data");
    }
  } else if constexpr (std::is_same<T, float>{}) {
    std::unique_ptr<MetricProcessor<float>> query_metric_processor =
      create_processor<float>(metric, n, D, 0, false, stream);

    if (ivf_ft_pams) {
      // cuivfl only supports L2/Inner product for now.
      if (metric == raft::distance::DistanceType::L2SqrtExpanded ||
          metric == raft::distance::DistanceType::L2SqrtUnexpanded ||
          metric == raft::distance::DistanceType::L2Unexpanded ||
          metric == raft::distance::DistanceType::L2Expanded ||
          metric == raft::distance::DistanceType::InnerProduct) {
        approx_knn_cuivfl_ivfflat_build_index(handle, index, *ivf_ft_pams, index_array, n, D);
      } else {
        raft::spatial::knn::RmmGpuResources* gpu_res = new raft::spatial::knn::RmmGpuResources();
        gpu_res->noTempMemory();
        gpu_res->setDefaultStream(device, stream);
        index->gpu_res = gpu_res;
        approx_knn_ivfflat_build_index(index, *ivf_ft_pams, n, D);
        std::vector<float> h_index_array(n * D);
        raft::update_host(h_index_array.data(), index_array, h_index_array.size(), stream);
        query_metric_processor->revert(index_array);
        index->index->train(n, h_index_array.data());
        index->index->add(n, h_index_array.data());
      }
    } else {
      int device;
      RAFT_CUDA_TRY(cudaGetDevice(&device));
      raft::spatial::knn::RmmGpuResources* gpu_res = new raft::spatial::knn::RmmGpuResources();
      gpu_res->noTempMemory();
      gpu_res->setDefaultStream(device, stream);
      index->gpu_res = gpu_res;
      query_metric_processor->preprocess(index_array);
      if (ivf_pq_pams) {
        approx_knn_ivfpq_build_index(index, *ivf_pq_pams, n, D);
      } else if (ivf_sq_pams) {
        approx_knn_ivfsq_build_index(index, *ivf_sq_pams, n, D);
      } else {
        ASSERT(index->index, "KNN index could not be initialized");
      }

      index->index->train(n, index_array);
      index->index->add(n, index_array);
      query_metric_processor->revert(index_array);
    }
  }
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
  if (dynamic_cast<GpuIndexIVF*>(index->index) && dynamic_cast<const ivf_search_params*>(&params)) {
    dynamic_cast<GpuIndexIVF*>(index->index)
      ->setNumProbes(dynamic_cast<const ivf_search_params&>(params).n_probes);
  }
  // perform preprocessing
#if 0
  std::unique_ptr<MetricProcessor<float>> query_metric_processor =
  create_processor<float>(index->metric, n, index->index->d, k, false, handle.get_stream());
  query_metric_processor->preprocess(query_array);
    index->index->search(n, query_array, k, distances, indices);
#else
  auto ivf_ft_pams = dynamic_cast<const ivf_flat_search_params*>(&params);
  if constexpr (std::is_same<T, uint8_t>{} || std::is_same<T, int8_t>{}) {
    if (ivf_ft_pams) {
      index->ivf_flat<T>()->search(
        query_array, n, k, *ivf_ft_pams, (size_t*)indices, distances, handle.get_stream());
    }
  } else if constexpr (std::is_same<T, float>{}) {
    std::unique_ptr<MetricProcessor<float>> query_metric_processor = create_processor<float>(
      index->metric, n, index->ivf_flat<T>()->data_dim(), k, false, handle.get_stream());
    query_metric_processor->preprocess(query_array);

    if (ivf_ft_pams) {
      index->ivf_flat<T>()->search(
        query_array, n, k, *ivf_ft_pams, (size_t*)indices, distances, handle.get_stream());
    }
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
#endif
}

}  // namespace detail
}  // namespace knn
}  // namespace spatial
}  // namespace raft
