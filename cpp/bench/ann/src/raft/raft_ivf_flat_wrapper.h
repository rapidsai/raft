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

#include "../common/ann_types.hpp"
#include "raft_ann_bench_utils.h"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/neighbors/ivf_flat.cuh>
#include <raft/neighbors/ivf_flat_types.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace raft::bench::ann {

template <typename T, typename IdxT>
class RaftIvfFlatGpu : public ANN<T> {
 public:
  using typename ANN<T>::AnnSearchParam;

  struct SearchParam : public AnnSearchParam {
    raft::neighbors::ivf_flat::search_params ivf_flat_params;
  };

  using BuildParam = raft::neighbors::ivf_flat::index_params;

  RaftIvfFlatGpu(Metric metric, int dim, const BuildParam& param)
    : ANN<T>(metric, dim), index_params_(param), dimension_(dim)
  {
    index_params_.metric                         = parse_metric_type(metric);
    index_params_.conservative_memory_allocation = true;
    RAFT_CUDA_TRY(cudaGetDevice(&device_));
  }

  void build(const T* dataset, size_t nrow, cudaStream_t stream) final;

  void set_search_param(const AnnSearchParam& param) override;

  // TODO: if the number of results is less than k, the remaining elements of 'neighbors'
  // will be filled with (size_t)-1
  void search(const T* queries,
              int batch_size,
              int k,
              size_t* neighbors,
              float* distances,
              cudaStream_t stream = 0) const override;

  // to enable dataset access from GPU memory
  AlgoProperty get_preference() const override
  {
    AlgoProperty property;
    property.dataset_memory_type = MemoryType::HostMmap;
    property.query_memory_type   = MemoryType::Device;
    return property;
  }
  void save(const std::string& file) const override;
  void load(const std::string&) override;
  std::unique_ptr<ANN<T>> copy() override;

 private:
  // handle_ must go first to make sure it dies last and all memory allocated in pool
  configured_raft_resources handle_{};
  BuildParam index_params_;
  raft::neighbors::ivf_flat::search_params search_params_;
  std::shared_ptr<raft::neighbors::ivf_flat::index<T, IdxT>> index_;
  int device_;
  int dimension_;
};

template <typename T, typename IdxT>
void RaftIvfFlatGpu<T, IdxT>::build(const T* dataset, size_t nrow, cudaStream_t stream)
{
  index_ = std::make_shared<raft::neighbors::ivf_flat::index<T, IdxT>>(std::move(
    raft::neighbors::ivf_flat::build(handle_, index_params_, dataset, IdxT(nrow), dimension_)));
  handle_.stream_wait(stream);  // RAFT stream -> bench stream
}

template <typename T, typename IdxT>
void RaftIvfFlatGpu<T, IdxT>::set_search_param(const AnnSearchParam& param)
{
  auto search_param = dynamic_cast<const SearchParam&>(param);
  search_params_    = search_param.ivf_flat_params;
  assert(search_params_.n_probes <= index_params_.n_lists);
}

template <typename T, typename IdxT>
void RaftIvfFlatGpu<T, IdxT>::save(const std::string& file) const
{
  raft::neighbors::ivf_flat::serialize(handle_, file, *index_);
  return;
}

template <typename T, typename IdxT>
void RaftIvfFlatGpu<T, IdxT>::load(const std::string& file)
{
  index_ = std::make_shared<raft::neighbors::ivf_flat::index<T, IdxT>>(
    std::move(raft::neighbors::ivf_flat::deserialize<T, IdxT>(handle_, file)));
  return;
}

template <typename T, typename IdxT>
std::unique_ptr<ANN<T>> RaftIvfFlatGpu<T, IdxT>::copy()
{
  return std::make_unique<RaftIvfFlatGpu<T, IdxT>>(*this);  // use copy constructor
}

template <typename T, typename IdxT>
void RaftIvfFlatGpu<T, IdxT>::search(const T* queries,
                                     int batch_size,
                                     int k,
                                     size_t* neighbors,
                                     float* distances,
                                     cudaStream_t stream) const
{
  static_assert(sizeof(size_t) == sizeof(IdxT), "IdxT is incompatible with size_t");
  raft::neighbors::ivf_flat::search(
    handle_, search_params_, *index_, queries, batch_size, k, (IdxT*)neighbors, distances);
  handle_.stream_wait(stream);  // RAFT stream -> bench stream
}
}  // namespace raft::bench::ann
