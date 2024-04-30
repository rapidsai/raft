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
#include "parameters.h"
#include "raft/core/host_mdspan.hpp"
#include "raft/neighbors/cagra.cuh"
#include "raft/neighbors/cagra_types.hpp"

#include <index.h>
#include <omp.h>
#include <utils.h>

#include <memory>
#include <vector>

#ifndef _WINDOWS
#include <sys/mman.h>
#include <unistd.h>
#else
#include <Windows.h>
#endif

#include "ann_exception.h"
#include "memory_mapper.h"

// #include <diskann/index_factory.h>

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
    uint32_t L_load;
    int num_threads = omp_get_num_procs();
  };

  DiskANNMemory(Metric metric, int dim, const BuildParam& param);

  void build(const T* dataset, size_t nrow) override;

  void set_search_param(const AnnSearchParam& param) override;
  void search(
    const T* queries, int batch_size, int k, size_t* neighbors, float* distances) const override;

  void save(const std::string& path_to_index) const override;
  void load(const std::string& path_to_index) override;
  DiskANNMemory(const DiskANNMemory<T>& other) = default;
  std::unique_ptr<ANN<T>> copy() override { return nullptr; }

  AlgoProperty get_preference() const override
  {
    AlgoProperty property;
    property.dataset_memory_type = MemoryType::Host;
    property.query_memory_type   = MemoryType::Host;
    return property;
  }

 private:
  bool use_cagra_graph_;
  bool use_pq_build_;
  uint32_t build_pq_bytes_;
  std::shared_ptr<diskann::IndexWriteParameters> diskann_index_write_params_{nullptr};
  std::shared_ptr<diskann::IndexSearchParams> diskann_index_search_params_{nullptr};
  std::shared_ptr<diskann::Index<T>> diskann_index_{nullptr};
  int num_threads_search_;
  uint32_t L_build_;
  uint32_t L_search_;
  uint32_t cagra_graph_degree_;
  uint32_t cagra_intermediate_graph_degree_;
};

template <typename T>
DiskANNMemory<T>::DiskANNMemory(Metric metric, int dim, const BuildParam& param)
  : ANN<T>(metric, dim)
{
  assert(dim_ > 0);

  diskann::IndexWriteParameters p = diskann::IndexWriteParametersBuilder(param.L_build, param.R)
                                      .with_filter_list_size(0)
                                      .with_alpha(param.alpha)
                                      .with_saturate_graph(false)
                                      .with_num_threads(param.num_threads)
                                      .build();
  this->diskann_index_write_params_ = std::make_shared<diskann::IndexWriteParameters>(p);
}

template <typename T>
void DiskANNMemory<T>::build(const T* dataset, size_t nrow)
{
  this->diskann_index_ =
    std::make_shared<diskann::Index<T>>(parse_metric_type(this->metric_),
                                                          this->dim_,
                                                          nrow,
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

  if (!this->use_cagra_graph_) {
    auto intermediate_graph = raft::make_host_matrix<uint32_t, int64_t>(nrow, cagra_intermediate_graph_degree_);
    auto knn_graph = raft::make_host_matrix<uint32_t, int64_t>(nrow, cagra_graph_degree_);
    auto dataset_view =
      raft::make_host_matrix_view<const T, int64_t>(dataset, (int64_t)nrow, (int64_t)this->dim_);
    raft::resources res;
    raft::neighbors::cagra::build_knn_graph(res, dataset_view, intermediate_graph.view());
    raft::neighbors::cagra::optimize(res, intermediate_graph.view(), knn_graph.view());
    diskann_index_->build(
      dataset, nrow, std::vector<uint32_t>(), std::shared_ptr<uint32_t>(knn_graph.data_handle()));
  }
  diskann_index_->build(dataset, nrow, std::vector<uint32_t>());
}

template <typename T>
void DiskANNMemory<T>::set_search_param(const AnnSearchParam& param_)
{
  auto param                = dynamic_cast<const SearchParam&>(param_);
  this->num_threads_search_ = param.num_threads;
  L_search_                 = param.L_search;
}

template <typename T>
void DiskANNMemory<T>::search(
  const T* queries, int batch_size, int k, size_t* neighbors, float* distances) const
{
  omp_set_num_threads(num_threads_search_);
#pragma omp parallel for schedule(dynamic, 1)
  for (int64_t i = 0; i < (int64_t)batch_size; i++) {
    diskann_index_->search(queries + i * this->dim_, k, L_search_, neighbors, distances);
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
  this->diskann_index_->load(path_to_index.c_str(), num_threads_search_, L_search_);
}

};  // namespace raft::bench::ann
