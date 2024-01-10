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

#include "../common/ann_types.hpp"
#include "raft_ann_bench_utils.h"
#include <raft/neighbors/ann_mg.cuh>

namespace raft::bench::ann {

template <typename T, typename IdxT>
class RaftAnnMG : public ANN<T> {
 public:
  using typename ANN<T>::AnnSearchParam;

  struct SearchParam : public AnnSearchParam {
    raft::neighbors::ivf_flat::search_params ivf_flat_params;
  };

  using BuildParam = raft::neighbors::ivf_flat::index_params;

  RaftAnnMG(Metric metric, int dim, const BuildParam& param)
    : ANN<T>(metric, dim), index_params_(param), dimension_(dim)
  {
    index_params_.metric                         = parse_metric_type(metric);
    index_params_.conservative_memory_allocation = true;
    RAFT_CUDA_TRY(cudaGetDevice(&device_));
  }

  ~RaftAnnMG() noexcept {}

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
    property.dataset_memory_type = MemoryType::Host;
    property.query_memory_type   = MemoryType::Host;
    return property;
  }
  void save(const std::string& file) const override;
  void load(const std::string&) override;

 private:
  raft::device_resources handle_;
  BuildParam index_params_;
  raft::neighbors::ivf_flat::search_params search_params_;
  std::optional<raft::neighbors::mg::detail::ann_mg_index<raft::neighbors::ivf_flat::index<T, IdxT>, T, IdxT>> index_;
  int device_;
  int dimension_;
};

template <typename T, typename IdxT>
void RaftAnnMG<T, IdxT>::build(const T* dataset, size_t nrow, cudaStream_t)
{
  std::vector<int> device_ids{0, 1};
  raft::neighbors::mg::dist_mode d_mode = raft::neighbors::mg::dist_mode::INDEX_DUPLICATION;
  auto dataset_matrix = raft::make_host_matrix_view<const T, IdxT, row_major>(dataset, IdxT(nrow), IdxT(dimension_));
  index_ = neighbors::mg::build<T, IdxT>(device_ids, d_mode, index_params_, dataset_matrix);
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
  raft::neighbors::mg::serialize<T, IdxT>(handle_, index_.value(), file);
  return;
}

template <typename T, typename IdxT>
void RaftAnnMG<T, IdxT>::load(const std::string& file)
{
  index_.emplace(raft::neighbors::mg::deserialize<T, IdxT>(handle_, file));
}

template <typename T, typename IdxT>
void RaftAnnMG<T, IdxT>::search(
  const T* queries, int batch_size, int k, size_t* neighbors, float* distances, cudaStream_t) const
{
  static_assert(sizeof(size_t) == sizeof(IdxT), "IdxT is incompatible with size_t");
  auto query_matrix = raft::make_host_matrix_view<const T, IdxT, row_major>(queries, IdxT(batch_size), IdxT(dimension_));
  auto neighbors_matrix = raft::make_host_matrix_view<IdxT, IdxT, row_major>((IdxT*)neighbors, IdxT(batch_size), IdxT(k));
  auto distances_matrix = raft::make_host_matrix_view<float, IdxT, row_major>(distances, IdxT(batch_size), IdxT(k));
  raft::neighbors::mg::search<T, IdxT>(index_.value(), search_params_, query_matrix, neighbors_matrix, distances_matrix);
  resource::sync_stream(handle_);
  return;
}
}  // namespace raft::bench::ann
