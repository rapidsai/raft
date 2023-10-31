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

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/distance/distance_types.hpp>
#include <stdexcept>
#include <string>

#include <raft/util/cudart_utils.hpp>

#include <chrono>
#include <thread>

namespace raft::bench::ann {

// This kernel sleeps for 10ms
__global__ inline void kernel_sleep(int sleep_ms)
{
  for (int i = 0; i < sleep_ms; i++)
    __nanosleep(1000000U);  // lms
}

inline void workload(bool use_gpu, bool sync_stream, int sleep_ms, cudaStream_t stream)
{
  if (use_gpu) {
    kernel_sleep<<<1, 1, 0, stream>>>(sleep_ms);
    if (sync_stream) { cudaStreamSynchronize(stream); }
  } else {
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
  }
}

class FixLatencyWorkload : public ANN<float> {
 public:
  using typename ANN<float>::AnnSearchParam;

  struct SearchParam : public AnnSearchParam {
    bool use_gpu     = true;
    bool sync_stream = true;
    int sleep_ms     = 10;
  };

  using BuildParam = SearchParam;

  FixLatencyWorkload(Metric metric, int dim, const BuildParam& param)
    : ANN<float>(metric, dim), build_param_{param}
  {
  }

  ~FixLatencyWorkload() noexcept {}

  void build(const float* dataset, size_t nrow, cudaStream_t stream) final
  {
    workload(build_param_.use_gpu,
             build_param_.sync_stream,
             build_param_.sleep_ms,
             raft::resource::get_cuda_stream(handle_));
  }

  void set_search_param(const AnnSearchParam& param) override
  {
    search_param_ = dynamic_cast<const SearchParam&>(param);
  }

  // TODO: if the number of results is less than k, the remaining elements of 'neighbors'
  // will be filled with (size_t)-1
  void search(const float* queries,
              int batch_size,
              int k,
              size_t* neighbors,
              float* distances,
              cudaStream_t stream = 0) const override
  {
    workload(search_param_.use_gpu,
             search_param_.sync_stream,
             search_param_.sleep_ms,
             raft::resource::get_cuda_stream(handle_));
  }

  // to enable dataset access from GPU memory
  AlgoProperty get_preference() const override
  {
    AlgoProperty property;
    property.dataset_memory_type = MemoryType::HostMmap;
    property.query_memory_type   = MemoryType::Device;
    return property;
  }

  void save(const std::string& file) const override
  {
    std::ofstream of(file, std::ios::out | std::ios::binary);
    of.close();
  }

  void load(const std::string&) override {}

 private:
  raft::device_resources handle_;
  BuildParam build_param_;
  SearchParam search_param_;
};
}  // namespace raft::bench::ann
