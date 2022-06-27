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

#include "detail/processing.hpp"
#include <faiss/gpu/GpuIndex.h>
#include <raft/distance/distance_type.hpp>
#include <raft/spatial/knn/faiss_mr.hpp>

namespace raft::spatial::knn {

namespace ivf_flat {
template <typename T>
class index;
};

struct knnIndex {
  raft::distance::DistanceType metric;
  float metricArg;
  std::unique_ptr<faiss::gpu::GpuIndex> index;
  std::unique_ptr<MetricProcessor<float>> metric_processor;
  std::unique_ptr<ivf_flat::index<float>> ivf_flat_float_;
  std::unique_ptr<ivf_flat::index<uint8_t>> ivf_flat_uint8_t_;
  std::unique_ptr<ivf_flat::index<int8_t>> ivf_flat_int8_t_;

  std::unique_ptr<raft::spatial::knn::RmmGpuResources> gpu_res;
  int device;

  template <typename T>
  auto ivf_flat() -> std::unique_ptr<ivf_flat::index<T>>&;
};

template <>
auto knnIndex::ivf_flat<float>() -> std::unique_ptr<ivf_flat::index<float>>&
{
  return ivf_flat_float_;
}

template <>
auto knnIndex::ivf_flat<uint8_t>() -> std::unique_ptr<ivf_flat::index<uint8_t>>&
{
  return ivf_flat_uint8_t_;
}

template <>
auto knnIndex::ivf_flat<int8_t>() -> std::unique_ptr<ivf_flat::index<int8_t>>&
{
  return ivf_flat_int8_t_;
}

enum QuantizerType : unsigned int {
  QT_8bit,
  QT_4bit,
  QT_8bit_uniform,
  QT_4bit_uniform,
  QT_fp16,
  QT_8bit_direct,
  QT_6bit
};

struct knn_index_params {
  /** Distance type. */
  raft::distance::DistanceType metric = distance::DistanceType::L2Expanded;
  /** The argument used by some distance metrics. */
  float metric_arg = 2.0f;

  virtual ~knn_index_params() = default;
};

struct knn_search_params {
  virtual ~knn_search_params() = default;
};

struct ivf_index_params : knn_index_params {
  /** The number of inverted lists (clusters) */
  uint32_t n_lists = 1024;
};

struct ivf_search_params : knn_search_params {
  /** The number of clusters to search. */
  uint32_t n_probes = 20;
};

// TODO: move to ivf_pq
struct ivf_pq_index_params : ivf_index_params {
  int n_subquantizers;
  int n_bits;
  bool use_precomputed_tables;
};

// TODO: move to ivf_sq
struct ivf_sq_index_params : ivf_index_params {
  QuantizerType qtype;
  bool encode_residual;
};

};  // namespace raft::spatial::knn
