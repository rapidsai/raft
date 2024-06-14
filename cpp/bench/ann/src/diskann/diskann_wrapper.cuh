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

#include <raft/core/host_mdspan.hpp>
#include <raft/neighbors/cagra.cuh>
#include <raft/neighbors/cagra_types.hpp>

#include <index.h>
#include <omp.h>
#include <utils.h>

#include <chrono>
#include <memory>
#include <vector>

#ifndef _WINDOWS
#include <sys/mman.h>
#include <unistd.h>
#else
#include <Windows.h>
#endif

namespace raft::bench::ann {

diskann::Metric parse_metric_type(raft::bench::ann::Metric metric)
{
  if (metric == raft::bench::ann::Metric::kInnerProduct) {
    return diskann::Metric::INNER_PRODUCT;
  } else if (metric == raft::bench::ann::Metric::kEuclidean) {
    return diskann::Metric::L2;
  } else {
    throw std::runtime_error("currently only inner product and L2 supported for benchmarking");
  }
}

template <typename T>
class DiskANNMemory : public ANN<T> {
 public:
  struct BuildParam {
    uint32_t R;
    uint32_t L_build;
    // uint32_t L_load;
    float alpha;
    int num_threads = omp_get_num_procs();
    bool use_cagra_graph;
    bool filtered_index;
    uint32_t cagra_graph_degree;
    uint32_t cagra_intermediate_graph_degree;
  };

  using typename ANN<T>::AnnSearchParam;
  struct SearchParam : public AnnSearchParam {
    uint32_t L_search;
  };

  DiskANNMemory(Metric metric, int dim, const BuildParam& param);

  void build(const T* dataset, size_t nrow) override;

  void set_search_param(const AnnSearchParam& param) override;
  void search(
    const T* queries, int batch_size, int k, size_t* neighbors, float* distances) const override;

  void save(const std::string& path_to_index) const override;
  void load(const std::string& path_to_index) override;
  DiskANNMemory(const DiskANNMemory<T>& other) = default;
  std::unique_ptr<ANN<T>> copy() override { return std::make_unique<DiskANNMemory<T>>(*this); }

  AlgoProperty get_preference() const override
  {
    AlgoProperty property;
    property.dataset_memory_type = MemoryType::Host;
    property.query_memory_type   = MemoryType::Host;
    return property;
  }

 private:
  bool use_cagra_graph_;
  bool use_pq_build_       = false;
  uint32_t build_pq_bytes_ = 0;
  std::shared_ptr<diskann::IndexWriteParameters> diskann_index_write_params_{nullptr};
  std::shared_ptr<diskann::IndexSearchParams> diskann_index_search_params_{nullptr};
  std::shared_ptr<diskann::Index<T>> diskann_index_{nullptr};
  // uint32_t L_load_;
  uint32_t L_search_;
  uint32_t cagra_graph_degree_ = 0;
  uint32_t cagra_intermediate_graph_degree_;
  uint32_t max_points_;
  // std::shared_ptr<FixedThreadPool> thread_pool_;
  Objective metric_objective_;
};

template <typename T>
DiskANNMemory<T>::DiskANNMemory(Metric metric, int dim, const BuildParam& param)
  : ANN<T>(metric, dim)
{
  assert(this->dim_ > 0);
  diskann_index_write_params_ = std::make_shared<diskann::IndexWriteParameters>(
    diskann::IndexWriteParametersBuilder(param.L_build, param.R)
      .with_filter_list_size(0)
      .with_alpha(param.alpha)
      .with_saturate_graph(false)
      .with_num_threads(param.num_threads)
      .build());
  use_cagra_graph_                 = param.use_cagra_graph;
  build_pq_bytes_                  = 0;
  cagra_graph_degree_              = param.cagra_graph_degree;
  cagra_intermediate_graph_degree_ = param.cagra_intermediate_graph_degree;
}

template <typename T>
void DiskANNMemory<T>::build(const T* dataset, size_t nrow)
{
  max_points_ = nrow;
  std::cout << "num_threads" << this->diskann_index_write_params_->num_threads << std::endl;

  this->diskann_index_ = std::make_shared<diskann::Index<T>>(parse_metric_type(this->metric_),
                                                             this->dim_,
                                                             max_points_,
                                                             this->diskann_index_write_params_,
                                                             nullptr,
                                                             0,
                                                             false,
                                                             false,
                                                             false,
                                                             this->use_pq_build_,
                                                             this->build_pq_bytes_,
                                                             false,
                                                             false,
                                                             this->use_cagra_graph_,
                                                             cagra_graph_degree_);

  if (use_cagra_graph_) {
    std::optional<raft::host_matrix<uint32_t, int64_t>> intermediate_graph(
      raft::make_host_matrix<uint32_t, int64_t>(nrow, cagra_intermediate_graph_degree_));

    std::vector<std::vector<uint32_t>> knn_graph(nrow, std::vector<uint32_t>(cagra_graph_degree_));
    auto knn_graph_view = raft::make_host_matrix_view<uint32_t, int64_t>(
      knn_graph[0].data(), nrow, cagra_graph_degree_);
    auto dataset_view = raft::make_host_matrix_view<const T, int64_t>(
      dataset, static_cast<int64_t>(nrow), (int64_t)this->dim_);
    raft::resources res;
    auto start                     = std::chrono::high_resolution_clock::now();
    auto nn_descent_params         = raft::neighbors::experimental::nn_descent::index_params();
    nn_descent_params.graph_degree = cagra_intermediate_graph_degree_;
    nn_descent_params.intermediate_graph_degree = 1.5 * cagra_intermediate_graph_degree_;
    nn_descent_params.max_iterations            = 20;
    raft::neighbors::cagra::build_knn_graph(
      res, dataset_view, intermediate_graph->view(), nn_descent_params);
    raft::neighbors::cagra::optimize(res, intermediate_graph->view(), knn_graph_view);
    // free intermediate graph before trying to create the index
    intermediate_graph.reset();

    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "cagra graph built in" << duration << " seconds" << std::endl;
    diskann_index_->build(dataset, nrow, std::vector<uint32_t>(), knn_graph);
  } else {
    diskann_index_->build(dataset, nrow, std::vector<uint32_t>());
  }
}

template <typename T>
void DiskANNMemory<T>::set_search_param(const AnnSearchParam& param_)
{
  auto param        = dynamic_cast<const SearchParam&>(param_);
  this->L_search_   = param.L_search;
  metric_objective_ = param.metric_objective;
}

template <typename T>
void DiskANNMemory<T>::search(
  const T* queries, int batch_size, int k, size_t* neighbors, float* distances) const
{
  if (this->metric_objective_ == Objective::LATENCY)
    omp_set_num_threads(diskann_index_write_params_->num_threads);
#pragma omp parallel for
  for (int64_t i = 0; i < (int64_t)batch_size; i++) {
    diskann_index_->search(queries + i * this->dim_,
                           static_cast<size_t>(k),
                           L_search_,
                           neighbors + i * k,
                           distances + i * k);
  }
}

template <typename T>
void DiskANNMemory<T>::save(const std::string& path_to_index) const
{
  this->diskann_index_->save(path_to_index.c_str());
}

template <typename T>
void DiskANNMemory<T>::load(const std::string& path_to_index)
{
  this->diskann_index_ = std::make_shared<diskann::Index<T>>(parse_metric_type(this->metric_),
                                                             this->dim_,
                                                             max_points_,
                                                             this->diskann_index_write_params_,
                                                             nullptr,
                                                             0,
                                                             false,
                                                             false,
                                                             false,
                                                             this->use_pq_build_,
                                                             this->build_pq_bytes_,
                                                             false,
                                                             false,
                                                             this->use_cagra_graph_,
                                                             cagra_graph_degree_);
  diskann_index_->load(path_to_index.c_str(), diskann_index_write_params_->num_threads, 100);
}
};  // namespace raft::bench::ann
