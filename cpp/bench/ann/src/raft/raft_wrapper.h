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
#include <raft/distance/detail/distance.cuh>
#include <raft/distance/distance_types.hpp>
#include <raft/spatial/knn/detail/fused_l2_knn.cuh>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "../common/ann_types.hpp"

namespace raft_temp {

inline raft::distance::DistanceType parse_metric_type(raft::bench::ann::Metric metric)
{
  if (metric == raft::bench::ann::Metric::kInnerProduct) {
    return raft::distance::DistanceType::InnerProduct;
  } else if (metric == raft::bench::ann::Metric::kEuclidean) {
    return raft::distance::DistanceType::L2Expanded;
  } else {
    throw std::runtime_error("raft supports only metric type of inner product and L2");
  }
}

}  // namespace raft_temp

namespace raft::bench::ann {

// brute force fused L2 KNN - RAFT
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
  void load(const std::string&) override { return; };

 protected:
  raft::distance::DistanceType metric_type_;
  int device_;
  const T* dataset_;
  size_t nrow_;
};

template <typename T>
RaftGpu<T>::RaftGpu(Metric metric, int dim)
  : ANN<T>(metric, dim), metric_type_(raft_temp::parse_metric_type(metric))
{
  static_assert(std::is_same_v<T, float>, "raft support only float type");
  assert(metric_type_ == raft::distance::DistanceType::L2Expanded);
  RAFT_CUDA_TRY(cudaGetDevice(&device_));
}

template <typename T>
void RaftGpu<T>::build(const T*, size_t, cudaStream_t)
{
  // as this is brute force algo so no index building required
  return;
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
  // create a empty index file as no index to store.
  std::fstream fp;
  fp.open(file.c_str(), std::ios::out);
  if (!fp) {
    printf("Error in creating file!!!\n");
    ;
    return;
  }
  fp.close();
}

template <typename T>
void RaftGpu<T>::search(const T* queries,
                        int batch_size,
                        int k,
                        size_t* neighbors,
                        float* distances,
                        cudaStream_t stream) const
{
  raft::spatial::knn::detail::fusedL2Knn(this->dim_,
                                         reinterpret_cast<int64_t*>(neighbors),
                                         distances,
                                         dataset_,
                                         queries,
                                         nrow_,
                                         static_cast<size_t>(batch_size),
                                         k,
                                         true,
                                         true,
                                         stream,
                                         metric_type_);
}

}  // namespace raft::bench::ann
