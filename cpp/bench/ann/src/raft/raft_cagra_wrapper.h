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
#include <optional>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/operators.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/neighbors/cagra.cuh>
#include <raft/neighbors/cagra_serialize.cuh>
#include <raft/neighbors/cagra_types.hpp>
#include <raft/neighbors/detail/cagra/cagra_build.cuh>
#include <raft/neighbors/ivf_pq_types.hpp>
#include <raft/neighbors/nn_descent_types.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "../common/ann_types.hpp"
#include "raft_ann_bench_utils.h"
#include <raft/util/cudart_utils.hpp>

#include "../common/cuda_huge_page_resource.hpp"
#include "../common/cuda_pinned_resource.hpp"

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace raft::bench::ann {

enum class AllocatorType { HostPinned, HostHugePage, Device };
template <typename T, typename IdxT>
class RaftCagra : public ANN<T> {
 public:
  using typename ANN<T>::AnnSearchParam;

  struct SearchParam : public AnnSearchParam {
    raft::neighbors::experimental::cagra::search_params p;
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
      handle_(cudaStreamPerThread),
      need_dataset_update_(true),
      dataset_(make_device_matrix<T, int64_t>(handle_, 0, 0)),
      graph_(make_device_matrix<IdxT, int64_t>(handle_, 0, 0)),
      input_dataset_v_(nullptr, 0, 0),
      graph_mem_(AllocatorType::Device),
      dataset_mem_(AllocatorType::Device)
  {
    index_params_.cagra_params.metric         = parse_metric_type(metric);
    index_params_.ivf_pq_build_params->metric = parse_metric_type(metric);
    RAFT_CUDA_TRY(cudaGetDevice(&device_));
  }

  ~RaftCagra() noexcept {}

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
  void save_to_hnswlib(const std::string& file) const;

 private:
  inline rmm::mr::device_memory_resource* get_mr(AllocatorType mem_type)
  {
    switch (mem_type) {
      case (AllocatorType::HostPinned): return &mr_pinned_;
      case (AllocatorType::HostHugePage): return &mr_huge_page_;
      default: return rmm::mr::get_current_device_resource();
    }
  }
  raft ::mr::cuda_pinned_resource mr_pinned_;
  raft ::mr::cuda_huge_page_resource mr_huge_page_;
  raft::device_resources handle_;
  AllocatorType graph_mem_;
  AllocatorType dataset_mem_;
  BuildParam index_params_;
  bool need_dataset_update_;
  raft::neighbors::cagra::search_params search_params_;
  std::optional<raft::neighbors::cagra::index<T, IdxT>> index_;
  int device_;
  int dimension_;
  raft::device_matrix<IdxT, int64_t, row_major> graph_;
  raft::device_matrix<T, int64_t, row_major> dataset_;
  raft::device_matrix_view<const T, int64_t, row_major> input_dataset_v_;
};

template <typename T, typename IdxT>
void RaftCagra<T, IdxT>::build(const T* dataset, size_t nrow, cudaStream_t)
{
  auto dataset_view =
    raft::make_host_matrix_view<const T, int64_t>(dataset, IdxT(nrow), dimension_);

  auto& params = index_params_.cagra_params;

  index_.emplace(raft::neighbors::cagra::detail::build(handle_,
                                                       params,
                                                       dataset_view,
                                                       index_params_.nn_descent_params,
                                                       index_params_.ivf_pq_refine_rate,
                                                       index_params_.ivf_pq_build_params,
                                                       index_params_.ivf_pq_search_params));
  return;
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
  if (search_param.graph_mem != graph_mem_) {
    // Move graph to correct memory space
    graph_mem_ = search_param.graph_mem;
    RAFT_LOG_INFO("moving graph to new memory space: %s", allocator_to_string(graph_mem_).c_str());
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
    graph_ = std::move(new_graph);
  }

  if (search_param.dataset_mem != dataset_mem_ || need_dataset_update_) {
    dataset_mem_ = search_param.dataset_mem;

    // First free up existing memory
    dataset_ = make_device_matrix<T, int64_t>(handle_, 0, 0);
    index_->update_dataset(handle_, make_const_mdspan(dataset_.view()));

    // Allocate space using the correct memory resource.
    RAFT_LOG_INFO("moving dataset to new memory space: %s",
                  allocator_to_string(dataset_mem_).c_str());

    auto mr = get_mr(dataset_mem_);
    raft::neighbors::cagra::detail::copy_with_padding(handle_, dataset_, input_dataset_v_, mr);

    index_->update_dataset(handle_, make_const_mdspan(dataset_.view()));

    // Ideally, instead of dataset_.view(), we should pass a strided matrix view to update.
    // See Issue https://github.com/rapidsai/raft/issues/1972 for details.
    // auto dataset_view = make_device_strided_matrix_view<const T, int64_t>(
    //   dataset_.data_handle(), dataset_.extent(0), this->dim_, dataset_.extent(1));
    // index_->update_dataset(handle_, dataset_view);
    need_dataset_update_ = false;
  }
}

template <typename T, typename IdxT>
void RaftCagra<T, IdxT>::set_search_dataset(const T* dataset, size_t nrow)
{
  // It can happen that we are re-using a previous algo object which already has
  // the dataset set. Check if we need update.
  if (static_cast<size_t>(input_dataset_v_.extent(0)) != nrow ||
      input_dataset_v_.data_handle() != dataset) {
    input_dataset_v_     = make_device_matrix_view<const T, int64_t>(dataset, nrow, this->dim_);
    need_dataset_update_ = true;
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
                          raft::resource::get_cuda_stream(handle_));
  }

  handle_.sync_stream();
}
}  // namespace raft::bench::ann
