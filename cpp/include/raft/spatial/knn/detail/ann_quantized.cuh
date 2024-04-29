/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
#include "processing.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/distance/distance.cuh>
#include <raft/distance/distance_types.hpp>
#include <raft/label/classlabels.cuh>
#include <raft/neighbors/ivf_pq.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/transform_iterator.h>

namespace raft::spatial::knn::detail {

template <typename T = float, typename IntType = int>
void approx_knn_build_index(raft::resources const& handle,
                            knnIndex* index,
                            knnIndexParam* params,
                            raft::distance::DistanceType metric,
                            float metricArg,
                            T* index_array,
                            IntType n,
                            IntType D)
{
  auto stream      = resource::get_cuda_stream(handle);
  index->metric    = metric;
  index->metricArg = metricArg;
  if (dynamic_cast<const IVFParam*>(params)) {
    index->nprobe = dynamic_cast<const IVFParam*>(params)->nprobe;
  }
  auto ivf_ft_pams = dynamic_cast<IVFFlatParam*>(params);
  auto ivf_pq_pams = dynamic_cast<IVFPQParam*>(params);

  if constexpr (std::is_same_v<T, float>) {
    index->metric_processor = create_processor<float>(metric, n, D, 0, false, stream);
    // For cosine/correlation distance, the metric processor translates distance
    // to inner product via pre/post processing - pass the translated metric to
    // ANN index
    if (metric == raft::distance::DistanceType::CosineExpanded ||
        metric == raft::distance::DistanceType::CorrelationExpanded) {
      metric = index->metric = raft::distance::DistanceType::InnerProduct;
    }
  }
  if constexpr (std::is_same_v<T, float>) { index->metric_processor->preprocess(index_array); }

  if (ivf_ft_pams) {
    auto new_params               = from_legacy_index_params(*ivf_ft_pams, metric, metricArg);
    index->ivf_flat<T, int64_t>() = std::make_unique<const ivf_flat::index<T, int64_t>>(
      ivf_flat::build(handle, new_params, index_array, int64_t(n), D));
  } else if (ivf_pq_pams) {
    neighbors::ivf_pq::index_params params;
    params.metric     = metric;
    params.metric_arg = metricArg;
    params.n_lists    = ivf_pq_pams->nlist;
    params.pq_bits    = ivf_pq_pams->n_bits;
    params.pq_dim     = ivf_pq_pams->M;
    // TODO: handle ivf_pq_pams.usePrecomputedTables ?

    auto index_view = raft::make_device_matrix_view<const T, int64_t>(index_array, n, D);
    index->ivf_pq   = std::make_unique<const neighbors::ivf_pq::index<int64_t>>(
      neighbors::ivf_pq::build(handle, params, index_view));
  } else {
    RAFT_FAIL("Unrecognized index type.");
  }

  if constexpr (std::is_same_v<T, float>) { index->metric_processor->revert(index_array); }
}

template <typename T = float, typename IntType = int>
void approx_knn_search(raft::resources const& handle,
                       float* distances,
                       int64_t* indices,
                       knnIndex* index,
                       IntType k,
                       T* query_array,
                       IntType n)
{
  if constexpr (std::is_same_v<T, float>) {
    index->metric_processor->preprocess(query_array);
    index->metric_processor->set_num_queries(k);
  }

  // search
  if (index->ivf_flat<T, int64_t>()) {
    ivf_flat::search_params params;
    params.n_probes = index->nprobe;
    ivf_flat::search(handle,
                     params,
                     *(index->ivf_flat<T, int64_t>()),
                     query_array,
                     n,
                     k,
                     indices,
                     distances,
                     resource::get_workspace_resource(handle));
  } else if (index->ivf_pq) {
    neighbors::ivf_pq::search_params params;
    params.n_probes = index->nprobe;

    auto query_view =
      raft::make_device_matrix_view<const T, uint32_t>(query_array, n, index->ivf_pq->dim());
    auto indices_view   = raft::make_device_matrix_view<int64_t, uint32_t>(indices, n, k);
    auto distances_view = raft::make_device_matrix_view<float, uint32_t>(distances, n, k);
    neighbors::ivf_pq::search(
      handle, params, *index->ivf_pq, query_view, indices_view, distances_view);
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
      distances, distances, n * k, raft::pow_const_op<float>(p), resource::get_cuda_stream(handle));
  }
  if constexpr (std::is_same_v<T, float>) { index->metric_processor->postprocess(distances); }
}

}  // namespace raft::spatial::knn::detail
