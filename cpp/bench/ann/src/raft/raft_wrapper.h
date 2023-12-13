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
#include <memory>
#include <raft/core/device_resources.hpp>
#include <raft/distance/detail/distance.cuh>
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/brute_force.cuh>
#include <raft/neighbors/brute_force_serialize.cuh>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "../common/ann_types.hpp"
#include "raft_ann_bench_utils.h"

namespace raft_temp {

inline raft::distance::DistanceType parse_metric_type(raft::bench::ann::Metric metric)
{
  switch (metric) {
    case raft::bench::ann::Metric::kInnerProduct: return raft::distance::DistanceType::InnerProduct;
    case raft::bench::ann::Metric::kEuclidean: return raft::distance::DistanceType::L2Expanded;
    default: throw std::runtime_error("raft supports only metric type of inner product and L2");
  }
}
}  // namespace raft_temp

namespace raft::bench::ann {

// brute force KNN - RAFT
template <typename T>
class RaftGpu : public ANN<T> {
 public:
  using typename ANN<T>::AnnSearchParam;

  RaftGpu(Metric metric, int dim);

  void build(const T*, size_t, cudaStream_t) final;

  void set_search_param(const AnnSearchParam& param) override;

  // TODO: if the number of results is less than k, the remaining elements of 'neighbors'
  // will be filled with (size_t)-1
  void search(const T* queries,
              int batch_size,
              int k,
              size_t* neighbors,
              float* distances,
              cudaStream_t stream = 0) const final;

  // to enable dataset access from GPU memory
  AlgoProperty get_preference() const override
  {
    AlgoProperty property;
    property.dataset_memory_type = MemoryType::Device;
    property.query_memory_type   = MemoryType::Device;
    return property;
  }
  void set_search_dataset(const T* dataset, size_t nrow) override;
  void save(const std::string& file) const override;
  void load(const std::string&) override;
  std::unique_ptr<ANN<T>> copy() override;

 protected:
  // handle_ must go first to make sure it dies last and all memory allocated in pool
  configured_raft_resources handle_{};
  std::shared_ptr<raft::neighbors::brute_force::index<T>> index_;
  raft::distance::DistanceType metric_type_;
  int device_;
  const T* dataset_;
  size_t nrow_;
};

template <typename T>
RaftGpu<T>::RaftGpu(Metric metric, int dim)
  : ANN<T>(metric, dim), metric_type_(raft_temp::parse_metric_type(metric))
{
  RAFT_CUDA_TRY(cudaGetDevice(&device_));
}

template <typename T>
void RaftGpu<T>::build(const T* dataset, size_t nrow, cudaStream_t stream)
{
  auto dataset_view = raft::make_host_matrix_view<const T, int64_t>(dataset, nrow, this->dim_);
  index_            = std::make_shared<raft::neighbors::brute_force::index<T>>(
    std::move(raft::neighbors::brute_force::build(handle_, dataset_view)));

  handle_.stream_wait(stream);
}

template <typename T>
void RaftGpu<T>::set_search_param(const AnnSearchParam&)
{
  // Nothing to set here as it is brute force implementation
}

template <typename T>
void RaftGpu<T>::set_search_dataset(const T* dataset, size_t nrow)
{
  dataset_ = dataset;
  nrow_    = nrow;
}

template <typename T>
void RaftGpu<T>::save(const std::string& file) const
{
  raft::neighbors::brute_force::serialize<T>(handle_, file, *index_);
}

template <typename T>
void RaftGpu<T>::load(const std::string& file)
{
  index_ = std::make_shared<raft::neighbors::brute_force::index<T>>(
    std::move(raft::neighbors::brute_force::deserialize<T>(handle_, file)));
}

template <typename T>
void RaftGpu<T>::search(const T* queries,
                        int batch_size,
                        int k,
                        size_t* neighbors,
                        float* distances,
                        cudaStream_t stream) const
{
  auto queries_view =
    raft::make_device_matrix_view<const T, int64_t>(queries, batch_size, this->dim_);

  auto neighbors_view = raft::make_device_matrix_view<size_t, int64_t>(neighbors, batch_size, k);
  auto distances_view = raft::make_device_matrix_view<float, int64_t>(distances, batch_size, k);

  raft::neighbors::brute_force::search<T, size_t>(
    handle_, *index_, queries_view, neighbors_view, distances_view);

  handle_.stream_wait(stream);
}

template <typename T>
std::unique_ptr<ANN<T>> RaftGpu<T>::copy()
{
  return std::make_unique<RaftGpu<T>>(*this);  // use copy constructor
}

}  // namespace raft::bench::ann
