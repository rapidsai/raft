/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RAFT_HIDE_DEPRECATION_WARNINGS
#pragma message(__FILE__                                                  \
                " is deprecated and will be removed in a future release." \
                " Please use the other approximate KNN implementations defined in spatial/knn/*.")
#endif

#pragma once

#include "detail/processing.hpp"
#include "ivf_flat_types.hpp"

#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/ivf_pq_types.hpp>

namespace raft {
namespace spatial {
namespace knn {

struct knnIndex {
  raft::distance::DistanceType metric;
  float metricArg;
  int nprobe;
  std::unique_ptr<MetricProcessor<float>> metric_processor;

  std::unique_ptr<const ivf_flat::index<float, int64_t>> ivf_flat_float_;
  std::unique_ptr<const ivf_flat::index<uint8_t, int64_t>> ivf_flat_uint8_t_;
  std::unique_ptr<const ivf_flat::index<int8_t, int64_t>> ivf_flat_int8_t_;

  std::unique_ptr<const raft::neighbors::ivf_pq::index<int64_t>> ivf_pq;

  int device;

  template <typename T, typename IdxT>
  auto ivf_flat() -> std::unique_ptr<const ivf_flat::index<T, IdxT>>&;
};

template <>
inline auto knnIndex::ivf_flat<float, int64_t>()
  -> std::unique_ptr<const ivf_flat::index<float, int64_t>>&
{
  return ivf_flat_float_;
}

template <>
inline auto knnIndex::ivf_flat<uint8_t, int64_t>()
  -> std::unique_ptr<const ivf_flat::index<uint8_t, int64_t>>&
{
  return ivf_flat_uint8_t_;
}

template <>
inline auto knnIndex::ivf_flat<int8_t, int64_t>()
  -> std::unique_ptr<const ivf_flat::index<int8_t, int64_t>>&
{
  return ivf_flat_int8_t_;
}

struct knnIndexParam {
  virtual ~knnIndexParam() {}
};

struct IVFParam : knnIndexParam {
  int nlist;
  int nprobe;
};

struct IVFFlatParam : IVFParam {};

struct IVFPQParam : IVFParam {
  int M;
  int n_bits;
  bool usePrecomputedTables;
};

inline auto from_legacy_index_params(const IVFFlatParam& legacy,
                                     raft::distance::DistanceType metric,
                                     float metric_arg)
{
  ivf_flat::index_params params;
  params.metric     = metric;
  params.metric_arg = metric_arg;
  params.n_lists    = legacy.nlist;
  return params;
}

};  // namespace knn
};  // namespace spatial
};  // namespace raft
