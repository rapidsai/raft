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
class cuivflHandle;
};

struct cuivfl_handle_t {
  template <typename T>
  auto get() -> std::unique_ptr<detail::cuivflHandle<T>>&;

  cuivfl_handle_t() {}

  ~cuivfl_handle_t()
  {
    if (dtype_.has_value()) {
      switch (*dtype_) {
        case CUDA_R_32F: impl.float_.~unique_ptr(); break;
        case CUDA_R_8U: impl.uint8_t_.~unique_ptr(); break;
        case CUDA_R_8I: impl.int8_t_.~unique_ptr(); break;
        default: break;
      }
    }
  }

 private:
  union handle {
    void* dummy;
    std::unique_ptr<detail::cuivflHandle<float>> float_;
    std::unique_ptr<detail::cuivflHandle<uint8_t>> uint8_t_;
    std::unique_ptr<detail::cuivflHandle<int8_t>> int8_t_;
    handle() : dummy(nullptr){};
    ~handle(){};
  } impl;
  std::optional<cudaDataType_t> dtype_;
};

template <>
auto cuivfl_handle_t::get<float>() -> std::unique_ptr<detail::cuivflHandle<float>>&
{
  if (dtype_.has_value()) {
    RAFT_EXPECTS(*dtype_ == CUDA_R_32F, "wrong element type");
  } else {
    *dtype_ = CUDA_R_32F;
  }
  return impl.float_;
}

template <>
auto cuivfl_handle_t::get<uint8_t>() -> std::unique_ptr<detail::cuivflHandle<uint8_t>>&
{
  if (dtype_.has_value()) {
    RAFT_EXPECTS(*dtype_ == CUDA_R_8U, "wrong element type");
  } else {
    *dtype_ = CUDA_R_8U;
  }
  return impl.uint8_t_;
}

template <>
auto cuivfl_handle_t::get<int8_t>() -> std::unique_ptr<detail::cuivflHandle<int8_t>>&
{
  if (dtype_.has_value()) {
    RAFT_EXPECTS(*dtype_ == CUDA_R_8I, "wrong element type");
  } else {
    *dtype_ = CUDA_R_8I;
  }
  return impl.int8_t_;
}

struct knnIndex {
  faiss::gpu::GpuIndex* index;
  raft::distance::DistanceType metric;
  float metricArg;
  cuivfl_handle_t handle_;

  raft::spatial::knn::RmmGpuResources* gpu_res;
  int device;
  ~knnIndex()
  {
    delete index;
    delete gpu_res;
  }
};

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
