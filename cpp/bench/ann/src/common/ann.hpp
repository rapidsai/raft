

/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

namespace raft::bench::ann {

enum class Metric {
  kInnerProduct,
  kEuclidean,
};

enum class MemoryType {
  Host,
  HostMmap,
  Device,
};

struct AlgoProperty {
  MemoryType dataset_memory_type;
  // neighbors/distances should have same memory type as queries
  MemoryType query_memory_type;
  bool need_dataset_when_search;
};

template <typename T>
class ANN {
 public:
  struct AnnSearchParam {
    virtual ~AnnSearchParam() = default;
  };

  ANN(Metric metric, int dim) : metric_(metric), dim_(dim) {}
  virtual ~ANN() = default;

  virtual void build(const T* dataset, size_t nrow, cudaStream_t stream = 0) = 0;

  virtual void set_search_param(const AnnSearchParam& param) = 0;
  // TODO: this assumes that an algorithm can always return k results.
  // This is not always possible.
  virtual void search(const T* queries,
                      int batch_size,
                      int k,
                      size_t* neighbors,
                      float* distances,
                      cudaStream_t stream = 0) const = 0;

  virtual void save(const std::string& file) const = 0;
  virtual void load(const std::string& file)       = 0;

  virtual AlgoProperty get_property() const = 0;

  // Some algorithms don't save the building dataset in their indices.
  // So they should be given the access to that dataset during searching.
  // The advantage of this way is that index has smaller size
  // and many indices can share one dataset.
  //
  // AlgoProperty::need_dataset_when_search of such algorithm should be true,
  // and set_search_dataset() should save the passed-in pointer somewhere.
  // The client code should call set_search_dataset() before searching,
  // and should not release dataset before searching is finished.
  virtual void set_search_dataset(const T* /*dataset*/, size_t /*nrow*/){};

 protected:
  Metric metric_;
  int dim_;
};

}  // namespace raft::bench::ann
