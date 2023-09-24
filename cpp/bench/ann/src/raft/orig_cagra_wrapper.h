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

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cagra.h>
// #include "cuann/ann.h"

#include "../common/ann_types.hpp"
#include "raft_ann_bench_utils.h"
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/logger.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/neighbors/cagra_types.hpp>
#include <raft/util/cudart_utils.hpp>

namespace raft::bench::ann {

namespace {
template <typename T>
std::string get_cagra_dtype()
{
  if constexpr (std::is_same_v<T, float>) {
    return "float";
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return "uint8";
  } else if constexpr (std::is_same_v<T, int8_t>) {
    return "int8";
  } else if constexpr (sizeof(T) == 2) {
    return "half";
  } else {
    static_assert(!std::is_same_v<T, T>, "Cagra: type should be float/half/int8/uint8");
  }
  return "";  // stop warning of missing return statement
}

}  // namespace

template <typename T>
class Cagra : public ANN<T> {
 public:
  struct BuildParam {};

  using typename ANN<T>::AnnSearchParam;
  struct SearchParam : public AnnSearchParam {
    raft::neighbors::experimental::cagra::search_params p;
    auto needs_dataset() const -> bool override { return true; }
    std::string search_mode;  // "single-cta", "multi-cta", or "multi-kernel"
    int batch_size;
    int k;
  };

  Cagra(Metric metric, int dim, const BuildParam&) : ANN<T>(metric, dim) {}
  Cagra(const Cagra&)                  = delete;
  const Cagra& operator=(const Cagra&) = delete;
  ~Cagra();

  void build(const T* dataset, size_t nrow, cudaStream_t stream = 0) override;

  void set_search_param(const AnnSearchParam& param) override;

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
    property.dataset_memory_type = MemoryType::Device;
    property.query_memory_type   = MemoryType::Device;
    return property;
  }

  void save(const std::string& file) const override;
  void load(const std::string& file) override;

  void set_search_dataset(const T* dataset, size_t nrow) override
  {
    // std::cout << "Cagra set_search_dataset" << std::endl;
    dataset_ = dataset;
    if (nrow_ == 0) {
      nrow_ = nrow;
    } else if (nrow_ != nrow) {
      throw std::runtime_error("inconsistent nrow between dataset and graph");
    }
  };

 private:
  raft::device_resources handle_;

  void check_search_param_(SearchParam& param);

  using ANN<T>::dim_;
  SearchParam search_param_;
  void* plan_{nullptr};

  const T* dataset_{nullptr};
  size_t nrow_{0};
  INDEX_T* graph_{nullptr};
  size_t degree_{0};

  INDEX_T* tmp_neighbors_{nullptr};
};

template <typename T>
Cagra<T>::~Cagra()
{
  if (plan_) { destroy_plan(plan_); }
  RAFT_CUDA_TRY_NO_THROW(cudaFree(graph_));
  RAFT_CUDA_TRY_NO_THROW(cudaFree(tmp_neighbors_));
}

template <typename T>
void Cagra<T>::build(const T*, size_t, cudaStream_t)
{
  throw std::runtime_error("Cagra's build() is not available now, use its tools to build index");
}

template <typename T>
void Cagra<T>::set_search_param(const AnnSearchParam& param)
{
  // std::cout << "Cagra set_search_param" << std::endl;
  if (!dataset_ || nrow_ == 0) { throw std::runtime_error("Cagra: dataset is not loaded"); }
  if (!graph_ || degree_ == 0) { throw std::runtime_error("Cagra: index is not loaded"); }

  auto new_search_param = dynamic_cast<const SearchParam&>(param);

  if (plan_) { destroy_plan(plan_); }
  if (tmp_neighbors_) RAFT_CUDA_TRY(cudaFree(tmp_neighbors_));
  RAFT_CUDA_TRY(
    cudaMalloc(&tmp_neighbors_, sizeof(size_t) * new_search_param.batch_size * new_search_param.k));

  search_param_ = new_search_param;
  // std::cout << "Cagra creating new plan" << std::endl;
  create_plan(&plan_,
              get_cagra_dtype<T>(),
              0,  // team_size
              search_param_.search_mode,
              search_param_.k,
              search_param_.p.itopk_size,
              search_param_.p.search_width,
              search_param_.p.min_iterations,
              search_param_.p.max_iterations,
              search_param_.batch_size,
              0,                                                           // load_bit_length
              0,                                                           // thread_block_size
              search_param_.search_mode == "multi-cta" ? "hash" : "auto",  // hashmap_mode
              0,                                                           // hashmap_min_bitlen
              0.5,                                                         // hashmap_max_fill_rate
              nrow_,
              dim_,
              degree_,
              dataset_,
              graph_);
}

template <typename T>
void Cagra<T>::search(const T* queries,
                      int batch_size,
                      int k,
                      size_t* neighbors,
                      float* distances,
                      cudaStream_t stream) const
{
  static_assert(std::is_same_v<DISTANCE_T, float>);
  assert(plan_);

  if (k != search_param_.k) {
    throw std::runtime_error("wrong configuration: k (" + std::to_string(k) +
                             ") != search_param.k (" + std::to_string(search_param_.k) + ")");
  }
  if (batch_size > search_param_.batch_size) {
    throw std::runtime_error("wrong configuration: batch_size (" + std::to_string(batch_size) +
                             ") > search_param.batch_size (" +
                             std::to_string(search_param_.batch_size) + ")");
  }

  // uint32_t neighbors_ptr =  std::is_same<INDEX_T, size_t>::value ? tmp_neighbors_

  // std::cout << "Cagra calling search" << std::endl;
  ::search(plan_,
           tmp_neighbors_,
           distances,
           queries,
           batch_size,
           1,
           0x128394,
           nullptr,
           0,
           nullptr,
           stream);

  // std::cout << "Cagra calling unaryop" << std::endl;
  raft::linalg::unaryOp(neighbors, tmp_neighbors_, batch_size * k, raft::cast_op<size_t>(), stream);
  handle_.sync_stream(stream);
}

template <typename T>
void Cagra<T>::save(const std::string& file) const
{
  FILE* fp = fopen(file.c_str(), "w");
  if (!fp) { throw std::runtime_error("fail to open " + file + " for writing"); }

  if (fwrite(&nrow_, sizeof(nrow_), 1, fp) != 1) {
    throw std::runtime_error("fwrite() " + file + " failed");
  }
  if (fwrite(&degree_, sizeof(degree_), 1, fp) != 1) {
    throw std::runtime_error("fwrite() " + file + " failed");
  }

  size_t total = nrow_ * degree_;
  auto h_graph = new INDEX_T[total];
  RAFT_CUDA_TRY(cudaMemcpy(h_graph, graph_, sizeof(*graph_) * total, cudaMemcpyDeviceToHost));
  if (fwrite(h_graph, sizeof(*h_graph), total, fp) != total) {
    throw std::runtime_error("fwrite() " + file + " failed");
  }
  delete[] h_graph;
}

template <typename T>
void Cagra<T>::load(const std::string& file)
{
  // std::cout << "Cagra load graph" << std::endl;
  FILE* fp = fopen(file.c_str(), "r");
  if (!fp) { throw std::runtime_error("fail to open " + file); }

  size_t nrow;
  if (fread(&nrow, sizeof(nrow), 1, fp) != 1) {
    throw std::runtime_error("fread() " + file + " failed");
  }
  if (nrow_ == 0) {
    nrow_ = nrow;
  } else if (nrow_ != nrow) {
    throw std::runtime_error("inconsistent nrow between dataset and graph");
  }

  if (fread(&degree_, sizeof(degree_), 1, fp) != 1) {
    throw std::runtime_error("fread() " + file + " failed");
  }

  size_t total = nrow_ * degree_;
  auto h_graph = new INDEX_T[total];
  if (fread(h_graph, sizeof(*h_graph), total, fp) != total) {
    throw std::runtime_error("fread() " + file + " failed");
  }
  // std::cout << "Cagra alloc device graph" << std::endl;
  RAFT_CUDA_TRY(cudaMalloc(&graph_, sizeof(*graph_) * total));
  RAFT_CUDA_TRY(cudaMemcpy(graph_, h_graph, sizeof(*graph_) * total, cudaMemcpyHostToDevice));
  delete[] h_graph;
}

}  // namespace raft::bench::ann
