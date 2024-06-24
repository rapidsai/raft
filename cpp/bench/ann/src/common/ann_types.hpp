/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "cuda_stub.hpp"  // cudaStream_t

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace raft::bench::ann {

enum Objective {
  THROUGHPUT,  // See how many vectors we can push through
  LATENCY      // See how fast we can push a vector through
};

enum class MemoryType {
  Host,
  HostMmap,
  HostPinned,
  Device,
};

enum class Metric {
  kInnerProduct,
  kEuclidean,
};

inline auto parse_metric(const std::string& metric_str) -> Metric
{
  if (metric_str == "inner_product") {
    return raft::bench::ann::Metric::kInnerProduct;
  } else if (metric_str == "euclidean") {
    return raft::bench::ann::Metric::kEuclidean;
  } else {
    throw std::runtime_error("invalid metric: '" + metric_str + "'");
  }
}

inline auto parse_memory_type(const std::string& memory_type) -> MemoryType
{
  if (memory_type == "host") {
    return MemoryType::Host;
  } else if (memory_type == "mmap") {
    return MemoryType::HostMmap;
  } else if (memory_type == "pinned") {
    return MemoryType::HostPinned;
  } else if (memory_type == "device") {
    return MemoryType::Device;
  } else {
    throw std::runtime_error("invalid memory type: '" + memory_type + "'");
  }
}

struct AlgoProperty {
  MemoryType dataset_memory_type;
  // neighbors/distances should have same memory type as queries
  MemoryType query_memory_type;
};

class AnnBase {
 public:
  using index_type = uint32_t;

  inline AnnBase(Metric metric, int dim) : metric_(metric), dim_(dim) {}
  virtual ~AnnBase() noexcept = default;

 protected:
  Metric metric_;
  int dim_;
};

/**
 * The GPU-based algorithms, which do not perform CPU synchronization at the end of their build or
 * search methods, must implement this interface.
 *
 * The `cuda_timer` / `cuda_lap`  from `util.hpp` uses this stream to record GPU times with events
 * and, if necessary, also synchronize (via events) between iterations.
 *
 * If the algo does not implement this interface, GPU timings are disabled.
 */
class AnnGPU {
 public:
  /**
   * Return the main cuda stream for this algorithm.
   * If any work is done in multiple streams, they should synchornize with the main stream at the
   * end.
   */
  [[nodiscard]] virtual auto get_sync_stream() const noexcept -> cudaStream_t = 0;
  /**
   * By default a GPU algorithm uses a fixed stream to order GPU operations.
   * However, an algorithm may need to synchronize with the host at the end of its execution.
   * In that case, also synchronizing with a benchmark event would put it at disadvantage.
   *
   * We can disable event sync by passing `false` here
   *   - ONLY IF THE ALGORITHM HAS PRODUCED ITS OUTPUT BY THE TIME IT SYNCHRONIZES WITH CPU.
   */
  [[nodiscard]] virtual auto uses_stream() const noexcept -> bool { return true; }
  virtual ~AnnGPU() noexcept = default;
};

template <typename T>
class ANN : public AnnBase {
 public:
  struct AnnSearchParam {
    Objective metric_objective = Objective::LATENCY;
    virtual ~AnnSearchParam()  = default;
    [[nodiscard]] virtual auto needs_dataset() const -> bool { return false; };
  };

  inline ANN(Metric metric, int dim) : AnnBase(metric, dim) {}
  virtual ~ANN() noexcept override = default;

  virtual void build(const T* dataset, size_t nrow) = 0;

  virtual void set_search_param(const AnnSearchParam& param) = 0;
  // TODO: this assumes that an algorithm can always return k results.
  // This is not always possible.
  virtual void search(const T* queries,
                      int batch_size,
                      int k,
                      AnnBase::index_type* neighbors,
                      float* distances) const = 0;

  virtual void save(const std::string& file) const = 0;
  virtual void load(const std::string& file)       = 0;

  virtual AlgoProperty get_preference() const = 0;

  // Some algorithms don't save the building dataset in their indices.
  // So they should be given the access to that dataset during searching.
  // The advantage of this way is that index has smaller size
  // and many indices can share one dataset.
  //
  // SearchParam::needs_dataset() of such algorithm should be true,
  // and set_search_dataset() should save the passed-in pointer somewhere.
  // The client code should call set_search_dataset() before searching,
  // and should not release dataset before searching is finished.
  virtual void set_search_dataset(const T* /*dataset*/, size_t /*nrow*/){};

  /**
   * Make a shallow copy of the ANN wrapper that shares the resources and ensures thread-safe access
   * to them. */
  virtual auto copy() -> std::unique_ptr<ANN<T>> = 0;
};

}  // namespace raft::bench::ann

#define REGISTER_ALGO_INSTANCE(DataT)                                                            \
  template auto raft::bench::ann::create_algo<DataT>(                                            \
    const std::string&, const std::string&, int, const nlohmann::json&, const std::vector<int>&) \
    ->std::unique_ptr<raft::bench::ann::ANN<DataT>>;                                             \
  template auto raft::bench::ann::create_search_param<DataT>(const std::string&,                 \
                                                             const nlohmann::json&)              \
    ->std::unique_ptr<typename raft::bench::ann::ANN<DataT>::AnnSearchParam>;
