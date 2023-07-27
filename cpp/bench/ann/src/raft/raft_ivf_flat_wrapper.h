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
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/distance/detail/distance.cuh>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/neighbors/ivf_flat.cuh>
#include <raft/neighbors/ivf_flat_types.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "../common/ann_types.hpp"
#include "raft_ann_bench_utils.h"
#include <raft/util/cudart_utils.hpp>

namespace raft::bench::ann {

template <typename T, typename IdxT>
class RaftIvfFlatGpu : public ANN<T> {
 public:
  using typename ANN<T>::AnnSearchParam;

  struct SearchParam : public AnnSearchParam {
    raft::neighbors::ivf_flat::search_params ivf_flat_params;
  };

  using BuildParam = raft::neighbors::ivf_flat::index_params;

  RaftIvfFlatGpu(Metric metric, int dim, const BuildParam& param);

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
  AlgoProperty get_property() const override
  {
    AlgoProperty property;
    property.dataset_memory_type      = MemoryType::Device;
    property.query_memory_type        = MemoryType::Device;
    property.need_dataset_when_search = false;
    return property;
  }
  void save(const std::string& file) const override;
  void load(const std::string&) override;

  ~RaftIvfFlatGpu() noexcept { rmm::mr::set_current_device_resource(mr_.get_upstream()); }

 private:
  raft::device_resources handle_;
  BuildParam index_params_;
  raft::neighbors::ivf_flat::search_params search_params_;
  std::optional<raft::neighbors::ivf_flat::index<T, IdxT>> index_;
  int device_;
  int dimension_;
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> mr_;
};

template <typename T, typename IdxT>
RaftIvfFlatGpu<T, IdxT>::RaftIvfFlatGpu(Metric metric, int dim, const BuildParam& param)
  : ANN<T>(metric, dim),
    index_params_(param),
    dimension_(dim),
    mr_(rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull)
{
  index_params_.metric                         = parse_metric_type(metric);
  index_params_.conservative_memory_allocation = true;
  rmm::mr::set_current_device_resource(&mr_);
  RAFT_CUDA_TRY(cudaGetDevice(&device_));
}

template <typename T, typename IdxT>
void RaftIvfFlatGpu<T, IdxT>::build(const T* dataset, size_t nrow, cudaStream_t)
{
  index_.emplace(
    raft::neighbors::ivf_flat::build(handle_, index_params_, dataset, IdxT(nrow), dimension_));
  return;
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
  index_ = raft::neighbors::ivf_flat::deserialize<T, IdxT>(handle_, file);
  return;
}

template <typename T, typename IdxT>
void RaftIvfFlatGpu<T, IdxT>::search(
  const T* queries, int batch_size, int k, size_t* neighbors, float* distances, cudaStream_t) const
{
  rmm::mr::device_memory_resource* mr_ptr = &const_cast<RaftIvfFlatGpu*>(this)->mr_;
  static_assert(sizeof(size_t) == sizeof(IdxT), "IdxT is incompatible with size_t");
  raft::neighbors::ivf_flat::search(
    handle_, search_params_, *index_, queries, batch_size, k, (IdxT*)neighbors, distances, mr_ptr);
  resource::sync_stream(handle_);
  return;
}
}  // namespace raft::bench::ann
