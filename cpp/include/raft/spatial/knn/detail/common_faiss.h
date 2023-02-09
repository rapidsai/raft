/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <faiss/MetricType.h>
#include <raft/distance/distance_types.hpp>

namespace raft {
namespace spatial {
namespace knn {
namespace detail {

inline faiss::MetricType build_faiss_metric(raft::distance::DistanceType metric)
{
  switch (metric) {
    case raft::distance::DistanceType::CosineExpanded:
      return faiss::MetricType::METRIC_INNER_PRODUCT;
    case raft::distance::DistanceType::CorrelationExpanded:
      return faiss::MetricType::METRIC_INNER_PRODUCT;
    case raft::distance::DistanceType::L2Expanded: return faiss::MetricType::METRIC_L2;
    case raft::distance::DistanceType::L2Unexpanded: return faiss::MetricType::METRIC_L2;
    case raft::distance::DistanceType::L2SqrtExpanded: return faiss::MetricType::METRIC_L2;
    case raft::distance::DistanceType::L2SqrtUnexpanded: return faiss::MetricType::METRIC_L2;
    case raft::distance::DistanceType::L1: return faiss::MetricType::METRIC_L1;
    case raft::distance::DistanceType::InnerProduct: return faiss::MetricType::METRIC_INNER_PRODUCT;
    case raft::distance::DistanceType::LpUnexpanded: return faiss::MetricType::METRIC_Lp;
    case raft::distance::DistanceType::Linf: return faiss::MetricType::METRIC_Linf;
    case raft::distance::DistanceType::Canberra: return faiss::MetricType::METRIC_Canberra;
    case raft::distance::DistanceType::BrayCurtis: return faiss::MetricType::METRIC_BrayCurtis;
    case raft::distance::DistanceType::JensenShannon:
      return faiss::MetricType::METRIC_JensenShannon;
    default: THROW("MetricType not supported: %d", metric);
  }
}

}  // namespace detail
}  // namespace knn
}  // namespace spatial
}  // namespace raft
