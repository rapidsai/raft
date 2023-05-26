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
#include <raft/distance/detail/distance.cuh>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/neighbors/cagra.cuh>
#include <raft/neighbors/cagra_serialize.cuh>
#include <raft/neighbors/cagra_types.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "../common/ann_types.hpp"
#include "raft_ann_bench_utils.h"
#include <raft/util/cudart_utils.hpp>

namespace raft::bench::ann {

template <typename T, typename IdxT>
class RaftCagra : public ANN<T> {
 public:
  using typename ANN<T>::AnnSearchParam;

  struct SearchParam : public AnnSearchParam {
    raft::neighbors::experimental::cagra::search_params search_params;
  };

  using BuildParam = raft::neighbors::experimental::cagra::index_params;

  RaftCagra(Metric metric, int dim, const BuildParam& param);

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
    property.need_dataset_when_search = true;
    return property;
  }
  void save(const std::string& file) const override;
  void load(const std::string&) override;

 private:
  raft::device_resources handle_;
  BuildParam index_params_;
  raft::neighbors::experimental::cagra::search_params search_params_;
  std::optional<raft::neighbors::experimental::cagra::index<T, IdxT>> index_;
  int device_;
  int dimension_;
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> mr_;
};

template <typename T, typename IdxT>
RaftCagra<T, IdxT>::RaftCagra(Metric metric, int dim, const BuildParam& param)
  : ANN<T>(metric, dim),
    index_params_(param),
    dimension_(dim),
    mr_(rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull)
{
  index_params_.metric = parse_metric_type(metric);
  RAFT_CUDA_TRY(cudaGetDevice(&device_));
}

template <typename T, typename IdxT>
void RaftCagra<T, IdxT>::build(const T* dataset, size_t nrow, cudaStream_t)
{
  auto dataset_view = raft::make_device_matrix_view<const T, IdxT>(dataset, IdxT(nrow), dimension_);
  index_.emplace(raft::neighbors::experimental::cagra::build(handle_, index_params_, dataset_view));
  return;
}

template <typename T, typename IdxT>
void RaftCagra<T, IdxT>::set_search_param(const AnnSearchParam& param)
{
  auto search_param = dynamic_cast<const SearchParam&>(param);
  search_params_    = search_param.search_params;

  if (search_params_.team_size != 0 && search_params_.team_size != 2 &&
      search_params_.team_size != 4 && search_params_.team_size != 8 &&
      search_params_.team_size != 16 && search_params_.team_size != 32) {
    throw std::runtime_error("team size must be 2, 4, 8, 16, or 32.");
  }
}

template <typename T, typename IdxT>
void RaftCagra<T, IdxT>::save(const std::string& file) const
{
  raft::neighbors::experimental::cagra::serialize(handle_, file, *index_);
  return;
}

template <typename T, typename IdxT>
void RaftCagra<T, IdxT>::load(const std::string& file)
{
  index_ = raft::neighbors::experimental::cagra::deserialize<T, IdxT>(handle_, file);
  return;
}

namespace {
template <typename IdxT>
__global__ void convert_neighbor_index_type_kernel(size_t* const dst_ptr,
                                                   const IdxT* const src_ptr,
                                                   const size_t len)
{
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= len) { return; }
  dst_ptr[tid] = src_ptr[tid];
}

template <typename IdxT>
void convert_neighbor_index_type(size_t* const dst_ptr,
                                 const IdxT* const src_ptr,
                                 const size_t len,
                                 cudaStream_t cuda_stream)
{
  const size_t block_size = 256;
  const size_t grid_size  = (len + block_size - 1) / block_size;
  convert_neighbor_index_type_kernel<IdxT>
    <<<grid_size, block_size, 0, cuda_stream>>>(dst_ptr, src_ptr, len);
}
}  // anonymous namespace

template <typename T, typename IdxT>
void RaftCagra<T, IdxT>::search(
  const T* queries, int batch_size, int k, size_t* neighbors, float* distances, cudaStream_t) const
{
  rmm::mr::device_memory_resource* mr_ptr = &const_cast<RaftCagra*>(this)->mr_;

  IdxT* neighbors_IdxT;
  if constexpr (std::is_same<IdxT, size_t>::value) {
    neighbors_IdxT = neighbors;
  } else {
    neighbors_IdxT = reinterpret_cast<IdxT*>(mr_ptr->allocate(
      batch_size * this->dimension_ * sizeof(IdxT), resource::get_cuda_stream(handle_)));
  }

  auto queries_view = raft::make_device_matrix_view<const T, IdxT>(queries, batch_size, dimension_);
  auto neighbors_view =
    raft::make_device_matrix_view<IdxT, IdxT>(neighbors_IdxT, batch_size, dimension_);
  auto distances_view =
    raft::make_device_matrix_view<float, IdxT>(distances, batch_size, dimension_);

  raft::neighbors::experimental::cagra::search(
    handle_, search_params_, *index_, queries_view, neighbors_view, distances_view);

  if (!std::is_same<IdxT, size_t>::value) {
    convert_neighbor_index_type(
      neighbors, neighbors_IdxT, batch_size * this->dimension_, resource::get_cuda_stream(handle_));
    mr_ptr->deallocate(neighbors_IdxT,
                       batch_size * this->dimension_ * sizeof(IdxT),
                       resource::get_cuda_stream(handle_));
  }

  handle_.sync_stream();
  return;
}
}  // namespace raft::bench::ann
