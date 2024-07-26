/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "raft_ann_mg_wrapper.hpp"

#include <raft/neighbors/cagra_mg.cuh>
#include <raft/neighbors/cagra_mg_serialize.cuh>
#include <raft/neighbors/nn_descent_types.hpp>

namespace raft::bench::ann {

enum class AllocatorType_t { HostPinned, HostHugePage, Device };
template <typename T, typename IdxT>
class RaftAnnMG_Cagra : public RaftAnnMG<T> {
 public:
  using typename ANN<T>::AnnSearchParam;

  struct SearchParam : public AnnSearchParam {
    raft::neighbors::cagra::search_params p;
    float refine_ratio;
    AllocatorType_t graph_mem   = AllocatorType_t::Device;
    AllocatorType_t dataset_mem = AllocatorType_t::Device;
    auto needs_dataset() const -> bool override { return true; }
  };

  struct BuildParam {
    raft::neighbors::cagra::mg_index_params cagra_params;
    std::optional<raft::neighbors::experimental::nn_descent::index_params> nn_descent_params =
      std::nullopt;
    std::optional<float> ivf_pq_refine_rate                                    = std::nullopt;
    std::optional<raft::neighbors::ivf_pq::index_params> ivf_pq_build_params   = std::nullopt;
    std::optional<raft::neighbors::ivf_pq::search_params> ivf_pq_search_params = std::nullopt;
  };

  RaftAnnMG_Cagra(Metric metric, int dim, const BuildParam& param, int concurrent_searches = 1)
    : RaftAnnMG<T>(metric, dim), index_params_(param), dimension_(dim)
  {
    index_params_.cagra_params.add_data_on_build = true;
    index_params_.cagra_params.mode              = raft::neighbors::mg::parallel_mode::SHARDED;
    index_params_.cagra_params.metric            = parse_metric_type(metric);
    index_params_.ivf_pq_build_params->metric    = parse_metric_type(metric);
  }

  void build(const T* dataset, size_t nrow) final;
  void set_search_param(const AnnSearchParam& param) override;
  void search(
    const T* queries, int batch_size, int k, size_t* neighbors, float* distances) const override;
  void save(const std::string& file) const override;
  void load(const std::string&) override;
  std::unique_ptr<ANN<T>> copy() override;

 private:
  BuildParam index_params_;
  raft::neighbors::cagra::search_params search_params_;
  std::shared_ptr<
    raft::neighbors::mg::detail::ann_mg_index<raft::neighbors::cagra::index<T, IdxT>, T, IdxT>>
    index_;
  float refine_ratio_ = 1.0;
  int dimension_;
};

template <typename T, typename IdxT>
void RaftAnnMG_Cagra<T, IdxT>::build(const T* dataset, size_t nrow)
{
  const auto& handle  = this->clique_->set_current_device_to_root_rank();
  auto dataset_matrix = raft::make_host_matrix_view<const T, IdxT, row_major>(
    dataset, IdxT(nrow), IdxT(this->dimension_));
  auto idx = raft::neighbors::mg::build<T, IdxT>(
    handle, *this->clique_, index_params_.cagra_params, dataset_matrix);
  index_ = std::make_shared<
    raft::neighbors::mg::detail::ann_mg_index<raft::neighbors::cagra::index<T, IdxT>, T, IdxT>>(
    std::move(idx));
  return;
}

template <typename T, typename IdxT>
void RaftAnnMG_Cagra<T, IdxT>::set_search_param(const AnnSearchParam& param)
{
  auto search_param = dynamic_cast<const SearchParam&>(param);
  search_params_    = search_param.p;
  refine_ratio_     = search_param.refine_ratio;
  assert(search_params_.n_probes <= index_params_.n_lists);
}

template <typename T, typename IdxT>
void RaftAnnMG_Cagra<T, IdxT>::save(const std::string& file) const
{
  const auto& handle = this->clique_->set_current_device_to_root_rank();
  raft::neighbors::mg::serialize<T, IdxT>(handle, *this->clique_, *index_, file);
  return;
}

template <typename T, typename IdxT>
void RaftAnnMG_Cagra<T, IdxT>::load(const std::string& file)
{
  const auto& handle = this->clique_->set_current_device_to_root_rank();
  auto idx = raft::neighbors::mg::deserialize_cagra<T, IdxT>(handle, *this->clique_, file);
  index_   = std::make_shared<
    raft::neighbors::mg::detail::ann_mg_index<raft::neighbors::cagra::index<T, IdxT>, T, IdxT>>(
    std::move(idx));
}

template <typename T, typename IdxT>
std::unique_ptr<ANN<T>> RaftAnnMG_Cagra<T, IdxT>::copy()
{
  return std::make_unique<RaftAnnMG_Cagra<T, IdxT>>(*this);  // use copy constructor
}

template <typename T, typename IdxT>
void RaftAnnMG_Cagra<T, IdxT>::search(
  const T* queries, int batch_size, int k, size_t* neighbors, float* distances) const
{
  const auto& handle = this->clique_->set_current_device_to_root_rank();
  auto query_matrix  = raft::make_host_matrix_view<const T, IdxT, row_major>(
    queries, IdxT(batch_size), IdxT(this->dimension_));
  auto neighbors_matrix =
    raft::make_host_matrix_view<IdxT, IdxT, row_major>((IdxT*)neighbors, IdxT(batch_size), IdxT(k));
  auto distances_matrix =
    raft::make_host_matrix_view<float, IdxT, row_major>(distances, IdxT(batch_size), IdxT(k));

  raft::neighbors::mg::search<T, IdxT>(handle,
                                       *this->clique_,
                                       *index_,
                                       search_params_,
                                       query_matrix,
                                       neighbors_matrix,
                                       distances_matrix);
  resource::sync_stream(handle);
  return;
}
}  // namespace raft::bench::ann
