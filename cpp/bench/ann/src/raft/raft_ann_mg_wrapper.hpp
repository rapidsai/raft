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
#include <raft/neighbors/ivf_flat_mg.cuh>
#include <raft/neighbors/ivf_flat_mg_serialize.cuh>

namespace raft::bench::ann {

template <typename T, typename IdxT>
class RaftAnnMG : public ANN<T>, public AnnGPU {
 public:
  using typename ANN<T>::AnnSearchParam;

  struct SearchParam : public AnnSearchParam {
    raft::neighbors::ivf_flat::search_params ivf_flat_params;
  };

  using BuildParam = raft::neighbors::ivf_flat::dist_index_params;

  RaftAnnMG(Metric metric, int dim, const BuildParam& param)
    : ANN<T>(metric, dim), index_params_(param), dimension_(dim)
  {
    index_params_.metric                         = parse_metric_type(metric);
    index_params_.conservative_memory_allocation = true;
    index_params_.mode                           = raft::neighbors::mg::parallel_mode::REPLICATION;
    RAFT_CUDA_TRY(cudaGetDevice(&device_));
  }

  void build(const T* dataset, size_t nrow) final;

  void set_search_param(const AnnSearchParam& param) override;

  // TODO: if the number of results is less than k, the remaining elements of 'neighbors'
  // will be filled with (size_t)-1
  void search(
    const T* queries, int batch_size, int k, size_t* neighbors, float* distances) const override;

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
  std::unique_ptr<ANN<T>> copy() override;

 private:
  // handle_ must go first to make sure it dies last and all memory allocated in pool
  configured_raft_resources handle_{};
  BuildParam index_params_;
  raft::neighbors::ivf_flat::search_params search_params_;
  std::shared_ptr<raft::neighbors::mg::detail::ann_mg_index<raft::neighbors::ivf_flat::index<T, IdxT>, T, IdxT>> index_;
  int device_;
  int dimension_;
};

template <typename T, typename IdxT>
void RaftAnnMG<T, IdxT>::build(const T* dataset, size_t nrow)
{
  std::vector<int> device_ids{0, 1};
  raft::neighbors::mg::nccl_clique clique(device_ids);

  auto dataset_matrix = raft::make_host_matrix_view<const T, IdxT, row_major>(dataset, IdxT(nrow), IdxT(dimension_));
  auto idx = raft::neighbors::mg::build<T, IdxT>(handle_, clique, index_params_, dataset_matrix);
  index_ = std::make_shared<raft::neighbors::mg::detail::ann_mg_index<raft::neighbors::ivf_flat::index<T, IdxT>, T, IdxT>>(std::move(idx));
  return;
}

template <typename T, typename IdxT>
void RaftAnnMG<T, IdxT>::set_search_param(const AnnSearchParam& param)
{
  auto search_param = dynamic_cast<const SearchParam&>(param);
  search_params_    = search_param.ivf_flat_params;
  assert(search_params_.n_probes <= index_params_.n_lists);
}

template <typename T, typename IdxT>
void RaftAnnMG<T, IdxT>::save(const std::string& file) const
{
  std::vector<int> device_ids{0, 1};
  raft::neighbors::mg::nccl_clique clique(device_ids);

  raft::neighbors::mg::serialize<T, IdxT>(handle_, clique, *index_, file);
  return;
}

template <typename T, typename IdxT>
void RaftAnnMG<T, IdxT>::load(const std::string& file)
{
  std::vector<int> device_ids{0, 1};
  raft::neighbors::mg::nccl_clique clique(device_ids);

  index_ = std::make_shared<raft::neighbors::mg::detail::ann_mg_index<raft::neighbors::ivf_flat::index<T, IdxT>, T, IdxT>>(
    std::move(raft::neighbors::mg::deserialize_flat<T, IdxT>(handle_, clique, file)));
}

template <typename T, typename IdxT>
std::unique_ptr<ANN<T>> RaftAnnMG<T, IdxT>::copy()
{
  return std::make_unique<RaftAnnMG<T, IdxT>>(*this);  // use copy constructor
}

template <typename T, typename IdxT>
void RaftAnnMG<T, IdxT>::search(
  const T* queries, int batch_size, int k, size_t* neighbors, float* distances) const
{
  static_assert(sizeof(size_t) == sizeof(IdxT), "IdxT is incompatible with size_t");

  std::vector<int> device_ids{0, 1};
  raft::neighbors::mg::nccl_clique clique(device_ids);

  auto query_matrix = raft::make_host_matrix_view<const T, IdxT, row_major>(queries, IdxT(batch_size), IdxT(dimension_));
  auto neighbors_matrix = raft::make_host_matrix_view<IdxT, IdxT, row_major>((IdxT*)neighbors, IdxT(batch_size), IdxT(k));
  auto distances_matrix = raft::make_host_matrix_view<float, IdxT, row_major>(distances, IdxT(batch_size), IdxT(k));
  raft::neighbors::mg::search<T, IdxT>(handle_, clique, *index_, search_params_, query_matrix, neighbors_matrix, distances_matrix);
  resource::sync_stream(handle_);
  return;
}
}  // namespace raft::bench::ann
