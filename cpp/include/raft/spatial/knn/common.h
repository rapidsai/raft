/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndex.h>

namespace raft {
namespace spatial {
namespace knn {

struct knnIndex {
  faiss::gpu::GpuIndex *index;
  raft::distance::DistanceType metric;
  float metricArg;

  faiss::gpu::StandardGpuResources *gpu_res;
  int device;
  ~knnIndex() {
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
  int nlist;
  int nprobe;
};

struct IVFFlatParam : IVFParam {};

struct IVFPQParam : IVFParam {
  int M;
  int n_bits;
  bool usePrecomputedTables;
};

struct IVFSQParam : IVFParam {
  QuantizerType qtype;
  bool encodeResidual;
};

};  // namespace knn
};  // namespace spatial
};  // namespace raft
