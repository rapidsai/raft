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

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/neighbors/ivf_pq_types.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft_runtime/neighbors/ivf_pq.hpp>
#include <raft_runtime/neighbors/refine.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <type_traits>

namespace raft::bench::ann {

template <typename T, typename IdxT>
class RaftIvfPQ : public ANN<T> {
 public:
  using typename ANN<T>::AnnSearchParam;
  using ANN<T>::dim_;

  struct SearchParam : public AnnSearchParam {
    raft::neighbors::ivf_pq::search_params pq_param;
    float refine_ratio = 1.0f;
    auto needs_dataset() const -> bool override { return refine_ratio > 1.0f; }
  };

  using BuildParam = raft::neighbors::ivf_pq::index_params;

  RaftIvfPQ(Metric metric, int dim, const BuildParam& param)
    : ANN<T>(metric, dim), index_params_(param), dimension_(dim)
  {
    index_params_.metric = parse_metric_type(metric);
  }

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
    property.dataset_memory_type = MemoryType::Host;
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
  raft::neighbors::ivf_pq::search_params search_params_;
  std::shared_ptr<raft::neighbors::ivf_pq::index<IdxT>> index_;
  int dimension_;
  float refine_ratio_ = 1.0;
  raft::device_matrix_view<const T, IdxT> dataset_;
};

template <typename T, typename IdxT>
void RaftIvfPQ<T, IdxT>::save(const std::string& file) const
{
  raft::runtime::neighbors::ivf_pq::serialize(handle_, file, *index_);
}

template <typename T, typename IdxT>
void RaftIvfPQ<T, IdxT>::load(const std::string& file)
{
  std::make_shared<raft::neighbors::ivf_pq::index<IdxT>>(handle_, index_params_, dimension_)
    .swap(index_);
  raft::runtime::neighbors::ivf_pq::deserialize(handle_, file, index_.get());
  return;
}

template <typename T, typename IdxT>
void RaftIvfPQ<T, IdxT>::build(const T* dataset, size_t nrow, cudaStream_t stream)
{
  auto dataset_v = raft::make_device_matrix_view<const T, IdxT>(dataset, IdxT(nrow), dim_);
  std::make_shared<raft::neighbors::ivf_pq::index<IdxT>>(
    std::move(raft::runtime::neighbors::ivf_pq::build(handle_, index_params_, dataset_v)))
    .swap(index_);
  handle_.stream_wait(stream);  // RAFT stream -> bench stream
}

template <typename T, typename IdxT>
std::unique_ptr<ANN<T>> RaftIvfPQ<T, IdxT>::copy()
{
  return std::make_unique<RaftIvfPQ<T, IdxT>>(*this);  // use copy constructor
}

template <typename T, typename IdxT>
void RaftIvfPQ<T, IdxT>::set_search_param(const AnnSearchParam& param)
{
  auto search_param = dynamic_cast<const SearchParam&>(param);
  search_params_    = search_param.pq_param;
  refine_ratio_     = search_param.refine_ratio;
  assert(search_params_.n_probes <= index_params_.n_lists);
}

template <typename T, typename IdxT>
void RaftIvfPQ<T, IdxT>::set_search_dataset(const T* dataset, size_t nrow)
{
  dataset_ = raft::make_device_matrix_view<const T, IdxT>(dataset, nrow, index_->dim());
}

template <typename T, typename IdxT>
void RaftIvfPQ<T, IdxT>::search(const T* queries,
                                int batch_size,
                                int k,
                                size_t* neighbors,
                                float* distances,
                                cudaStream_t stream) const
{
  if (refine_ratio_ > 1.0f) {
    uint32_t k0 = static_cast<uint32_t>(refine_ratio_ * k);
    auto queries_v =
      raft::make_device_matrix_view<const T, IdxT>(queries, batch_size, index_->dim());
    auto distances_tmp = raft::make_device_matrix<float, IdxT>(handle_, batch_size, k0);
    auto candidates    = raft::make_device_matrix<IdxT, IdxT>(handle_, batch_size, k0);

    raft::runtime::neighbors::ivf_pq::search(
      handle_, search_params_, *index_, queries_v, candidates.view(), distances_tmp.view());

    if (raft::get_device_for_address(dataset_.data_handle()) >= 0) {
      auto queries_v =
        raft::make_device_matrix_view<const T, IdxT>(queries, batch_size, index_->dim());
      auto neighbors_v = raft::make_device_matrix_view<IdxT, IdxT>((IdxT*)neighbors, batch_size, k);
      auto distances_v = raft::make_device_matrix_view<float, IdxT>(distances, batch_size, k);

      raft::runtime::neighbors::refine(handle_,
                                       dataset_,
                                       queries_v,
                                       candidates.view(),
                                       neighbors_v,
                                       distances_v,
                                       index_->metric());
      handle_.stream_wait(stream);  // RAFT stream -> bench stream
    } else {
      auto queries_host    = raft::make_host_matrix<T, IdxT>(batch_size, index_->dim());
      auto candidates_host = raft::make_host_matrix<IdxT, IdxT>(batch_size, k0);
      auto neighbors_host  = raft::make_host_matrix<IdxT, IdxT>(batch_size, k);
      auto distances_host  = raft::make_host_matrix<float, IdxT>(batch_size, k);

      raft::copy(queries_host.data_handle(), queries, queries_host.size(), stream);
      raft::copy(candidates_host.data_handle(),
                 candidates.data_handle(),
                 candidates_host.size(),
                 resource::get_cuda_stream(handle_));

      auto dataset_v = raft::make_host_matrix_view<const T, IdxT>(
        dataset_.data_handle(), dataset_.extent(0), dataset_.extent(1));

      // wait for the queries to copy to host in 'stream` and for IVF-PQ::search to finish
      RAFT_CUDA_TRY(cudaEventRecord(handle_.get_sync_event(), resource::get_cuda_stream(handle_)));
      RAFT_CUDA_TRY(cudaEventRecord(handle_.get_sync_event(), stream));
      RAFT_CUDA_TRY(cudaEventSynchronize(handle_.get_sync_event()));
      raft::runtime::neighbors::refine(handle_,
                                       dataset_v,
                                       queries_host.view(),
                                       candidates_host.view(),
                                       neighbors_host.view(),
                                       distances_host.view(),
                                       index_->metric());

      raft::copy(neighbors, (size_t*)neighbors_host.data_handle(), neighbors_host.size(), stream);
      raft::copy(distances, distances_host.data_handle(), distances_host.size(), stream);
    }
  } else {
    auto queries_v =
      raft::make_device_matrix_view<const T, IdxT>(queries, batch_size, index_->dim());
    auto neighbors_v = raft::make_device_matrix_view<IdxT, IdxT>((IdxT*)neighbors, batch_size, k);
    auto distances_v = raft::make_device_matrix_view<float, IdxT>(distances, batch_size, k);

    raft::runtime::neighbors::ivf_pq::search(
      handle_, search_params_, *index_, queries_v, neighbors_v, distances_v);
    handle_.stream_wait(stream);  // RAFT stream -> bench stream
  }
}
}  // namespace raft::bench::ann
