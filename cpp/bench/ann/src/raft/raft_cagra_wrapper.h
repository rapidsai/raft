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
#include "../common/cuda_huge_page_resource.hpp"
#include "../common/cuda_pinned_resource.hpp"
#include "raft_ann_bench_utils.h"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/operators.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/neighbors/cagra.cuh>
#include <raft/neighbors/cagra_serialize.cuh>
#include <raft/neighbors/cagra_types.hpp>
#include <raft/neighbors/dataset.hpp>
#include <raft/neighbors/detail/cagra/cagra_build.cuh>
#include <raft/neighbors/ivf_pq_types.hpp>
#include <raft/neighbors/nn_descent_types.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace raft::bench::ann {

enum class AllocatorType { HostPinned, HostHugePage, Device };
template <typename T, typename IdxT>
class RaftCagra : public ANN<T>, public AnnGPU {
 public:
  using typename ANN<T>::AnnSearchParam;

  struct SearchParam : public AnnSearchParam {
    raft::neighbors::experimental::cagra::search_params p;
    float refine_ratio;
    AllocatorType graph_mem   = AllocatorType::Device;
    AllocatorType dataset_mem = AllocatorType::Device;
    auto needs_dataset() const -> bool override { return true; }
  };

  struct BuildParam {
    raft::neighbors::cagra::index_params cagra_params;
    std::optional<raft::neighbors::experimental::nn_descent::index_params> nn_descent_params =
      std::nullopt;
    std::optional<float> ivf_pq_refine_rate                                    = std::nullopt;
    std::optional<raft::neighbors::ivf_pq::index_params> ivf_pq_build_params   = std::nullopt;
    std::optional<raft::neighbors::ivf_pq::search_params> ivf_pq_search_params = std::nullopt;
  };

  RaftCagra(Metric metric, int dim, const BuildParam& param, int concurrent_searches = 1)
    : ANN<T>(metric, dim),
      index_params_(param),
      dimension_(dim),
      need_dataset_update_(true),
      dataset_(std::make_shared<raft::device_matrix<T, int64_t, row_major>>(
        std::move(make_device_matrix<T, int64_t>(handle_, 0, 0)))),
      graph_(std::make_shared<raft::device_matrix<IdxT, int64_t, row_major>>(
        std::move(make_device_matrix<IdxT, int64_t>(handle_, 0, 0)))),
      input_dataset_v_(
        std::make_shared<raft::device_matrix_view<const T, int64_t, row_major>>(nullptr, 0, 0)),
      graph_mem_(AllocatorType::Device),
      dataset_mem_(AllocatorType::Device)
  {
    index_params_.cagra_params.metric         = parse_metric_type(metric);
    index_params_.ivf_pq_build_params->metric = parse_metric_type(metric);
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
    property.dataset_memory_type = MemoryType::HostMmap;
    property.query_memory_type   = MemoryType::Device;
    return property;
  }
  void save(const std::string& file) const override;
  void load(const std::string&) override;
  void save_to_hnswlib(const std::string& file) const;
  std::unique_ptr<ANN<T>> copy() override;

 private:
  // handle_ must go first to make sure it dies last and all memory allocated in pool
  configured_raft_resources handle_{};
  raft::mr::cuda_pinned_resource mr_pinned_;
  raft::mr::cuda_huge_page_resource mr_huge_page_;
  AllocatorType graph_mem_;
  AllocatorType dataset_mem_;
  float refine_ratio_;
  BuildParam index_params_;
  bool need_dataset_update_;
  raft::neighbors::cagra::search_params search_params_;
  std::shared_ptr<raft::neighbors::cagra::index<T, IdxT>> index_;
  int dimension_;
  std::shared_ptr<raft::device_matrix<IdxT, int64_t, row_major>> graph_;
  std::shared_ptr<raft::device_matrix<T, int64_t, row_major>> dataset_;
  std::shared_ptr<raft::device_matrix_view<const T, int64_t, row_major>> input_dataset_v_;

  inline rmm::device_async_resource_ref get_mr(AllocatorType mem_type)
  {
    switch (mem_type) {
      case (AllocatorType::HostPinned): return &mr_pinned_;
      case (AllocatorType::HostHugePage): return &mr_huge_page_;
      default: return rmm::mr::get_current_device_resource();
    }
  }
};

template <typename T, typename IdxT>
void RaftCagra<T, IdxT>::build(const T* dataset, size_t nrow)
{
  auto dataset_view =
    raft::make_host_matrix_view<const T, int64_t>(dataset, IdxT(nrow), dimension_);

  auto& params = index_params_.cagra_params;

  // Do include the compressed dataset for the CAGRA-Q
  bool shall_include_dataset = params.compression.has_value();

  index_ = std::make_shared<raft::neighbors::cagra::index<T, IdxT>>(
    std::move(raft::neighbors::cagra::detail::build(handle_,
                                                    params,
                                                    dataset_view,
                                                    index_params_.nn_descent_params,
                                                    index_params_.ivf_pq_refine_rate,
                                                    index_params_.ivf_pq_build_params,
                                                    index_params_.ivf_pq_search_params,
                                                    shall_include_dataset)));
}

inline std::string allocator_to_string(AllocatorType mem_type)
{
  if (mem_type == AllocatorType::Device) {
    return "device";
  } else if (mem_type == AllocatorType::HostPinned) {
    return "host_pinned";
  } else if (mem_type == AllocatorType::HostHugePage) {
    return "host_huge_page";
  }
  return "<invalid allocator type>";
}

template <typename T, typename IdxT>
void RaftCagra<T, IdxT>::set_search_param(const AnnSearchParam& param)
{
  auto search_param = dynamic_cast<const SearchParam&>(param);
  search_params_    = search_param.p;
  refine_ratio_     = search_param.refine_ratio;
  if (search_param.graph_mem != graph_mem_) {
    // Move graph to correct memory space
    graph_mem_ = search_param.graph_mem;
    RAFT_LOG_DEBUG("moving graph to new memory space: %s", allocator_to_string(graph_mem_).c_str());
    // We create a new graph and copy to it from existing graph
    auto mr        = get_mr(graph_mem_);
    auto new_graph = make_device_mdarray<IdxT, int64_t>(
      handle_, mr, make_extents<int64_t>(index_->graph().extent(0), index_->graph_degree()));

    raft::copy(new_graph.data_handle(),
               index_->graph().data_handle(),
               index_->graph().size(),
               resource::get_cuda_stream(handle_));

    index_->update_graph(handle_, make_const_mdspan(new_graph.view()));
    // update_graph() only stores a view in the index. We need to keep the graph object alive.
    *graph_ = std::move(new_graph);
  }

  if (search_param.dataset_mem != dataset_mem_ || need_dataset_update_) {
    dataset_mem_ = search_param.dataset_mem;

    // First free up existing memory
    *dataset_ = make_device_matrix<T, int64_t>(handle_, 0, 0);
    index_->update_dataset(handle_, make_const_mdspan(dataset_->view()));

    // Allocate space using the correct memory resource.
    RAFT_LOG_DEBUG("moving dataset to new memory space: %s",
                   allocator_to_string(dataset_mem_).c_str());

    auto mr = get_mr(dataset_mem_);
    raft::neighbors::cagra::detail::copy_with_padding(handle_, *dataset_, *input_dataset_v_, mr);

    auto dataset_view = raft::make_device_strided_matrix_view<const T, int64_t>(
      dataset_->data_handle(), dataset_->extent(0), this->dim_, dataset_->extent(1));
    index_->update_dataset(handle_, dataset_view);

    need_dataset_update_ = false;
  }
}

template <typename T, typename IdxT>
void RaftCagra<T, IdxT>::set_search_dataset(const T* dataset, size_t nrow)
{
  using ds_idx_type = decltype(index_->data().n_rows());
  bool is_vpq =
    dynamic_cast<const raft::neighbors::vpq_dataset<half, ds_idx_type>*>(&index_->data()) ||
    dynamic_cast<const raft::neighbors::vpq_dataset<float, ds_idx_type>*>(&index_->data());
  // It can happen that we are re-using a previous algo object which already has
  // the dataset set. Check if we need update.
  if (static_cast<size_t>(input_dataset_v_->extent(0)) != nrow ||
      input_dataset_v_->data_handle() != dataset) {
    *input_dataset_v_    = make_device_matrix_view<const T, int64_t>(dataset, nrow, this->dim_);
    need_dataset_update_ = !is_vpq;  // ignore update if this is a VPQ dataset.
  }
}

template <typename T, typename IdxT>
void RaftCagra<T, IdxT>::save(const std::string& file) const
{
  raft::neighbors::cagra::serialize<T, IdxT>(handle_, file, *index_);
}

template <typename T, typename IdxT>
void RaftCagra<T, IdxT>::save_to_hnswlib(const std::string& file) const
{
  raft::neighbors::cagra::serialize_to_hnswlib<T, IdxT>(handle_, file, *index_);
}

template <typename T, typename IdxT>
void RaftCagra<T, IdxT>::load(const std::string& file)
{
  index_ = std::make_shared<raft::neighbors::cagra::index<T, IdxT>>(
    std::move(raft::neighbors::cagra::deserialize<T, IdxT>(handle_, file)));
}

template <typename T, typename IdxT>
std::unique_ptr<ANN<T>> RaftCagra<T, IdxT>::copy()
{
  return std::make_unique<RaftCagra<T, IdxT>>(*this);  // use copy constructor
}

template <typename T, typename IdxT>
void RaftCagra<T, IdxT>::search_base(
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
    raft::make_device_matrix_view<const T, int64_t>(queries, batch_size, dimension_);
  auto neighbors_view = raft::make_device_matrix_view<IdxT, int64_t>(neighbors_IdxT, batch_size, k);
  auto distances_view = raft::make_device_matrix_view<float, int64_t>(distances, batch_size, k);

  raft::neighbors::cagra::search(
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
void RaftCagra<T, IdxT>::search(
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
      res, *input_dataset_v_, queries_v, candidate_ixs, k, neighbors, distances, index_->metric());
  }
}
}  // namespace raft::bench::ann
