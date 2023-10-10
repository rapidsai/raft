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
#include <raft/core/operators.hpp>
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
    raft::neighbors::experimental::cagra::search_params p;
    auto needs_dataset() const -> bool override { return true; }
  };

  using BuildParam = raft::neighbors::cagra::index_params;

  RaftCagra(Metric metric, int dim, const BuildParam& param, int concurrent_searches = 1)
    : ANN<T>(metric, dim),
      index_params_(param),
      dimension_(dim),
      mr_(rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull)
  {
    rmm::mr::set_current_device_resource(&mr_);
    index_params_.metric = parse_metric_type(metric);
    RAFT_CUDA_TRY(cudaGetDevice(&device_));
  }

  ~RaftCagra() noexcept { rmm::mr::set_current_device_resource(mr_.get_upstream()); }

  void build(const T* dataset, size_t nrow, cudaStream_t stream) final;

  void set_search_param(const AnnSearchParam& param) override;

  void set_search_dataset(const T* dataset, size_t nrow) override;

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

 private:
  // `mr_` must go first to make sure it dies last
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> mr_;
  raft::device_resources handle_;
  BuildParam index_params_;
  raft::neighbors::cagra::search_params search_params_;
  std::optional<raft::neighbors::cagra::index<T, IdxT>> index_;
  int device_;
  int dimension_;
};

template <typename T, typename IdxT>
void RaftCagra<T, IdxT>::build(const T* dataset, size_t nrow, cudaStream_t)
{
  if (raft::get_device_for_address(dataset) == -1) {
    auto dataset_view =
      raft::make_host_matrix_view<const T, int64_t>(dataset, IdxT(nrow), dimension_);
    index_.emplace(raft::neighbors::cagra::build(handle_, index_params_, dataset_view));
    return;
  } else {
    auto dataset_view =
      raft::make_device_matrix_view<const T, int64_t>(dataset, IdxT(nrow), dimension_);
    index_.emplace(raft::neighbors::cagra::build(handle_, index_params_, dataset_view));
    return;
  }
}

template <typename T, typename IdxT>
void RaftCagra<T, IdxT>::set_search_param(const AnnSearchParam& param)
{
  auto search_param = dynamic_cast<const SearchParam&>(param);
  search_params_    = search_param.p;
}

template <typename T, typename IdxT>
void RaftCagra<T, IdxT>::set_search_dataset(const T* dataset, size_t nrow)
{
  index_->update_dataset(handle_,
                         raft::make_host_matrix_view<const T, int64_t>(dataset, nrow, this->dim_));
}

template <typename T, typename IdxT>
void RaftCagra<T, IdxT>::save(const std::string& file) const
{
  raft::neighbors::cagra::serialize(handle_, file, *index_, false);
}

template <typename T, typename IdxT>
void RaftCagra<T, IdxT>::load(const std::string& file)
{
  index_ = raft::neighbors::cagra::deserialize<T, IdxT>(handle_, file);
}

template <typename T, typename IdxT>
void RaftCagra<T, IdxT>::search(
  const T* queries, int batch_size, int k, size_t* neighbors, float* distances, cudaStream_t) const
{
  IdxT* neighbors_IdxT;
  rmm::device_uvector<IdxT> neighbors_storage(0, resource::get_cuda_stream(handle_));
  if constexpr (std::is_same<IdxT, size_t>::value) {
    neighbors_IdxT = neighbors;
  } else {
    neighbors_storage.resize(batch_size * k, resource::get_cuda_stream(handle_));
    neighbors_IdxT = neighbors_storage.data();
  }

  auto queries_view =
    raft::make_device_matrix_view<const T, int64_t>(queries, batch_size, dimension_);
  auto neighbors_view = raft::make_device_matrix_view<IdxT, int64_t>(neighbors_IdxT, batch_size, k);
  auto distances_view = raft::make_device_matrix_view<float, int64_t>(distances, batch_size, k);

  raft::neighbors::cagra::search(
    handle_, search_params_, *index_, queries_view, neighbors_view, distances_view);

  if (!std::is_same<IdxT, size_t>::value) {
    raft::linalg::unaryOp(neighbors,
                          neighbors_IdxT,
                          batch_size * k,
                          raft::cast_op<size_t>(),
                          resource::get_cuda_stream(handle_));
  }

  handle_.sync_stream();
}
}  // namespace raft::bench::ann
