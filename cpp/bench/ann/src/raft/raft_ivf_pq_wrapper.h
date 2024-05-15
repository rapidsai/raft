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
#include <raft/neighbors/ivf_pq.cuh>
#include <raft/neighbors/ivf_pq_types.hpp>
#include <raft/neighbors/refine.cuh>
#include <raft/util/cudart_utils.hpp>

#include <type_traits>

namespace raft::bench::ann {

template <typename T, typename IdxT>
class RaftIvfPQ : public ANN<T>, public AnnGPU {
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

  void build(const T* dataset, size_t nrow) final;

  void set_search_param(const AnnSearchParam& param) override;
  void set_search_dataset(const T* dataset, size_t nrow) override;

  void search(const T* queries,
              int batch_size,
              int k,
              AnnBase::index_type* neighbors,
              float* distances) const override;
  void search_base(const T* queries,
                   int batch_size,
                   int k,
                   AnnBase::index_type* neighbors,
                   float* distances) const;

  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    return handle_.get_sync_stream();
  }

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
  raft::neighbors::ivf_pq::serialize(handle_, file, *index_);
}

template <typename T, typename IdxT>
void RaftIvfPQ<T, IdxT>::load(const std::string& file)
{
  index_ = std::make_shared<raft::neighbors::ivf_pq::index<IdxT>>(
    std::move(raft::neighbors::ivf_pq::deserialize<IdxT>(handle_, file)));
}

template <typename T, typename IdxT>
void RaftIvfPQ<T, IdxT>::build(const T* dataset, size_t nrow)
{
  auto dataset_v = raft::make_device_matrix_view<const T, IdxT>(dataset, IdxT(nrow), dim_);
  std::make_shared<raft::neighbors::ivf_pq::index<IdxT>>(
    std::move(raft::neighbors::ivf_pq::build(handle_, index_params_, dataset_v)))
    .swap(index_);
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
void RaftIvfPQ<T, IdxT>::search_base(
  const T* queries, int batch_size, int k, AnnBase::index_type* neighbors, float* distances) const
{
  static_assert(std::is_integral_v<AnnBase::index_type>);
  static_assert(std::is_integral_v<IdxT>);

  IdxT* neighbors_IdxT;
  std::optional<rmm::device_uvector<IdxT>> neighbors_storage{std::nullopt};
  if constexpr (sizeof(IdxT) == sizeof(AnnBase::index_type)) {
    neighbors_IdxT = reinterpret_cast<IdxT*>(neighbors);
  } else {
    neighbors_storage.emplace(batch_size * k, resource::get_cuda_stream(handle_));
    neighbors_IdxT = neighbors_storage->data();
  }

  auto queries_view =
    raft::make_device_matrix_view<const T, uint32_t>(queries, batch_size, dimension_);
  auto neighbors_view =
    raft::make_device_matrix_view<IdxT, uint32_t>(neighbors_IdxT, batch_size, k);
  auto distances_view = raft::make_device_matrix_view<float, uint32_t>(distances, batch_size, k);

  raft::neighbors::ivf_pq::search(
    handle_, search_params_, *index_, queries_view, neighbors_view, distances_view);

  if constexpr (sizeof(IdxT) != sizeof(AnnBase::index_type)) {
    raft::linalg::unaryOp(neighbors,
                          neighbors_IdxT,
                          batch_size * k,
                          raft::cast_op<AnnBase::index_type>(),
                          raft::resource::get_cuda_stream(handle_));
  }
}

template <typename T, typename IdxT>
void RaftIvfPQ<T, IdxT>::search(
  const T* queries, int batch_size, int k, AnnBase::index_type* neighbors, float* distances) const
{
  auto k0                       = static_cast<size_t>(refine_ratio_ * k);
  const bool disable_refinement = k0 <= static_cast<size_t>(k);
  const raft::resources& res    = handle_;

  if (disable_refinement) {
    search_base(queries, batch_size, k, neighbors, distances);
  } else {
    auto queries_v =
      raft::make_device_matrix_view<const T, AnnBase::index_type>(queries, batch_size, dimension_);
    auto candidate_ixs =
      raft::make_device_matrix<AnnBase::index_type, AnnBase::index_type>(res, batch_size, k0);
    auto candidate_dists =
      raft::make_device_matrix<float, AnnBase::index_type>(res, batch_size, k0);
    search_base(
      queries, batch_size, k0, candidate_ixs.data_handle(), candidate_dists.data_handle());
    refine_helper(
      res, dataset_, queries_v, candidate_ixs, k, neighbors, distances, index_->metric());
  }
}
}  // namespace raft::bench::ann
