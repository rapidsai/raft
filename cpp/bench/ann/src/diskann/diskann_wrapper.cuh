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
#include "../common/thread_pool.hpp"

#include <hnswlib/hnswlib.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <ctime>
#include <future>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "program_options_utils.hpp"
#include "raft/neighbors/cagra_types.hpp"

#include <boost/program_options.hpp>
#include <diskann/index.h>
#include <diskann/utils.h>
#include <omp.h>

#include <cstring>

#ifndef _WINDOWS
#include <sys/mman.h>
#include <unistd.h>
#else
#include <Windows.h>
#endif

#include "ann_exception.h"
#include "memory_mapper.h"

#include <diskann/index_factory.h>

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
  optional_configs.add_options()(
    "num_threads,T",
    po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
    program_options_utils::NUMBER_THREADS_DESCRIPTION);
  optional_configs.add_options()("max_degree,R",
                                 po::value<uint32_t>(&R)->default_value(64),
                                 program_options_utils::MAX_BUILD_DEGREE);
  optional_configs.add_options()("cagra_");
  optional_configs.add_options()("Lbuild,L",
                                 po::value<uint32_t>(&L)->default_value(100),
                                 program_options_utils::GRAPH_BUILD_COMPLEXITY);
  optional_configs.add_options()("alpha",
                                 po::value<float>(&alpha)->default_value(1.2f),
                                 program_options_utils::GRAPH_BUILD_ALPHA);
  optional_configs.add_options()("build_PQ_bytes",
                                 po::value<uint32_t>(&build_PQ_bytes)->default_value(0),
                                 program_options_utils::BUIlD_GRAPH_PQ_BYTES);
  optional_configs.add_options()("use_opq",
                                 po::bool_switch()->default_value(false),
                                 program_options_utils::USE_OPQ);
  optional_configs.add_options()("label_file",
                                 po::value<std::string>(&label_file)->default_value(""),
                                 program_options_utils::LABEL_FILE);
  optional_configs.add_options()("universal_label",
                                 po::value<std::string>(&universal_label)->default_value(""),
                                 program_options_utils::UNIVERSAL_LABEL);

  optional_configs.add_options()("FilteredLbuild",
                                 po::value<uint32_t>(&Lf)->default_value(0),
                                 program_options_utils::FILTERED_LBUILD);
  optional_configs.add_options()("label_type",
                                 po::value<std::string>(&label_type)->default_value("uint"),
                                 program_options_utils::LABEL_TYPE_DESCRIPTION);

  struct BuildParam {
    uint32_t R;
    uint32_t Lb;
    float alpha;
    int num_threads = omp_get_num_procs();
    bool use_raft_cagra;
    bool filtered_index;
    raft::neighbors::cagra::index_params cagra_params;
  };

  using typename ANN<T>::AnnSearchParam;
  struct SearchParam : public AnnSearchParam {
    uint32_t Ls;
    int num_threads = omp_get_num_procs();
  };

  DiskANNMemory(Metric metric, int dim, const BuildParam& param);

  void build(const T* dataset, size_t nrow) override;

  void set_search_param(const AnnSearchParam& param) override;
  void search(
    const T* query, int batch_size, int k, size_t* indices, float* distances) const override;

  void save(const std::string& path_to_index) const override;
  void load(const std::string& path_to_index) override;

  AlgoProperty get_preference() const override
  {
    AlgoProperty property;
    property.dataset_memory_type = MemoryType::Host;
    property.query_memory_type   = MemoryType::Host;
    return property;
  }

 private:
  void get_search_knn_results_(const T* query, int k, size_t* indices, float* distances) const;

  using ANN<T>::metric_;
  using ANN<T>::dim_;
  int num_threads_;
};

template <typename T>
DiskANNMemory<T>::DiskANNMemory(Metric metric, int dim, const BuildParam& param)
  : ANN<T>(metric, dim)
{
  assert(dim_ > 0);

  this->index_build_params = std::make_shared<diskann::IndexWriteParameters>(diskann::IndexWriteParametersBuilder(param.L, param.R)
                              .with_filter_list_size(0)
                              .with_alpha(param.alpha)
                              .with_saturate_graph(false)
                              .with_num_threads(param.num_threads)
                              .build());

  bool use_pq_build = param.build_PQ_bytes > 0;
  this->index_build_config_ = diskann::IndexConfigBuilder()
                                .with_metric(parse_metric_type(metric_))
                                .with_dimension(dim_)
                                .with_max_points(0)
                                .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                                .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                                .with_data_type(diskann::diskann_type_to_name<T>())
                                .with_label_type(diskann::diskann_type_to_name<uint32_t>())
                                .is_dynamic_index(false)
                                .with_index_write_params(this->index_build_params_)
                                .is_enable_tags(false)
                                .is_use_opq(false)
                                .is_pq_dist_build(use_pq_build)
                                .with_num_pq_chunks(this->build_PQ_bytes)
                                .build();
}

template <typename T>
void DiskANNMemory<T>::build(const T* dataset, size_t nrow)
{
  this->index_build_config_.with_max_points(nrow)

  Index<T>::Index(metric_,
                  dim_,
                  nrow,
                  this->index_build_params,
                  nullptr,
                  0,
                  false,
                  false,
                  false,
                  use_pq_build,
                  this->build_PQ_bytes,
                  false);
}

template <typename T>
void HnswLib<T>::set_search_param(const AnnSearchParam& param_)
{
  auto param        = dynamic_cast<const SearchParam&>(param_);
  appr_alg_->ef_    = param.ef;
  metric_objective_ = param.metric_objective;
  num_threads_      = param.num_threads;

  // Create a pool if multiple query threads have been set and the pool hasn't been created already
  bool create_pool = (metric_objective_ == Objective::LATENCY && num_threads_ > 1 && !thread_pool_);
  if (create_pool) { thread_pool_ = std::make_shared<FixedThreadPool>(num_threads_); }
}

template <typename T>
void DiskANNMemory<T>::search(
  const T* query, int batch_size, int k, size_t* indices, float* distances) const
{
  auto f = [&](int i) {
    // hnsw can only handle a single vector at a time.
    get_search_knn_results_(query + i * dim_, k, indices + i * k, distances + i * k);
  };
  if (metric_objective_ == Objective::LATENCY && num_threads_ > 1) {
    thread_pool_->submit(f, batch_size);
  } else {
    for (int i = 0; i < batch_size; i++) {
      f(i);
    }
  }

#pragma omp parallel for schedule(dynamic, 1)
  for (int64_t i = 0; i < (int64_t)query_num; i++) {
    if (filtered_search && !tags) {
      std::string raw_filter = query_filters.size() == 1 ? query_filters[0] : query_filters[i];

      auto retval  = index->search_with_filters(query + i * query_aligned_dim,
                                               raw_filter,
                                               recall_at,
                                               L,
                                               query_result_ids[test_id].data() + i * recall_at,
                                               query_result_dists[test_id].data() + i * recall_at);
      cmp_stats[i] = retval.second;
    } else if (metric == diskann::FAST_L2) {
      index->search_with_optimized_layout(query + i * query_aligned_dim,
                                          recall_at,
                                          L,
                                          query_result_ids[test_id].data() + i * recall_at);
    } else if (tags) {
      if (!filtered_search) {
        index->search_with_tags(query + i * query_aligned_dim,
                                recall_at,
                                L,
                                query_result_tags.data() + i * recall_at,
                                nullptr,
                                res);
      } else {
        std::string raw_filter = query_filters.size() == 1 ? query_filters[0] : query_filters[i];

        index->search_with_tags(query + i * query_aligned_dim,
                                recall_at,
                                L,
                                query_result_tags.data() + i * recall_at,
                                nullptr,
                                res,
                                true,
                                raw_filter);
      }

      for (int64_t r = 0; r < (int64_t)recall_at; r++) {
        query_result_ids[test_id][recall_at * i + r] = query_result_tags[recall_at * i + r];
      }
    } else {
      cmp_stats[i] = index
                       ->search(query + i * query_aligned_dim,
                                recall_at,
                                L,
                                query_result_ids[test_id].data() + i * recall_at)
                       .second;
    }
    auto qe                            = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = qe - qs;
    latency_stats[i]                   = (float)(diff.count() * 1000000);
  }
}

template <typename T>
void DiskANNMemory<T>::save(const std::string& path_to_index) const
{
  index_->save(path_to_index.c_str());
}

template <typename T>
void DiskANNMemory<T>::load(const std::string& path_to_index)
{
  index_->load(path_to_index.c_str(), num_threads, search_l);
}

};  // namespace raft::bench::ann
