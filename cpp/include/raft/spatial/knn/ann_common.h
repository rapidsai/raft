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

#include <faiss/gpu/GpuIndex.h>
#include <raft/distance/distance_type.hpp>
#include <raft/spatial/knn/faiss_mr.hpp>

namespace raft::spatial::knn {

namespace detail {
template <typename T>
class ivf_flat_handle;
};

struct knnIndex {
  faiss::gpu::GpuIndex* index;
  raft::distance::DistanceType metric;
  float metricArg;
  std::unique_ptr<detail::ivf_flat_handle<float>> ivf_flat_float_;
  std::unique_ptr<detail::ivf_flat_handle<uint8_t>> ivf_flat_uint8_t_;
  std::unique_ptr<detail::ivf_flat_handle<int8_t>> ivf_flat_int8_t_;

  raft::spatial::knn::RmmGpuResources* gpu_res;
  int device;
  ~knnIndex()
  {
    delete index;
    delete gpu_res;
  }

  template <typename T>
  auto ivf_flat() -> std::unique_ptr<detail::ivf_flat_handle<T>>&;
};

template <>
auto knnIndex::ivf_flat<float>() -> std::unique_ptr<detail::ivf_flat_handle<float>>&
{
  return ivf_flat_float_;
}

template <>
auto knnIndex::ivf_flat<uint8_t>() -> std::unique_ptr<detail::ivf_flat_handle<uint8_t>>&
{
  return ivf_flat_uint8_t_;
}

template <>
auto knnIndex::ivf_flat<int8_t>() -> std::unique_ptr<detail::ivf_flat_handle<int8_t>>&
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

struct knnIndexParam {
  virtual ~knnIndexParam() {}
};

struct IVFParam : knnIndexParam {
  /** The number of inverted lists (clusters) */
  int nlist;
  /** The number of clusters to search. */
  int nprobe;
};

struct ivf_flat_params : IVFParam {
  /** The number of iterations searching for kmeans centers (index building). */
  uint32_t kmeans_n_iters = 20;
  /** The fraction of data to use during iterative kmeans building. */
  double kmeans_trainset_fraction = 0.5;
};

struct IVFPQParam : IVFParam {
  int M;
  int n_bits;
  bool usePrecomputedTables;
};

struct IVFSQParam : IVFParam {
  QuantizerType qtype;
  bool encodeResidual;
};

};  // namespace raft::spatial::knn
