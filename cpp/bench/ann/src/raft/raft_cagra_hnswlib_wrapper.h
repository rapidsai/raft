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

#include "../hnswlib/hnswlib_wrapper.h"
#include "raft_cagra_wrapper.h"
#include <memory>

namespace raft::bench::ann {

template <typename T, typename IdxT>
class RaftCagraHnswlib : public ANN<T> {
 public:
  using typename ANN<T>::AnnSearchParam;
  using BuildParam  = typename RaftCagra<T, IdxT>::BuildParam;
  using SearchParam = typename HnswLib<T>::SearchParam;

  RaftCagraHnswlib(Metric metric, int dim, const BuildParam& param, int concurrent_searches = 1)
    : ANN<T>(metric, dim),
      metric_(metric),
      index_params_(param),
      dimension_(dim),
      handle_(cudaStreamPerThread)
  {
  }

  ~RaftCagraHnswlib() noexcept {}

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
    property.dataset_memory_type = MemoryType::HostMmap;
    property.query_memory_type   = MemoryType::Host;
    return property;
  }
  void save(const std::string& file) const override;
  void load(const std::string&) override;
  std::unique_ptr<ANN<T>> copy() override;

 private:
  raft::device_resources handle_;
  Metric metric_;
  BuildParam index_params_;
  int dimension_;

  std::unique_ptr<RaftCagra<T, IdxT>> cagra_build_;
  std::unique_ptr<HnswLib<T>> hnswlib_search_;

  Objective metric_objective_;
};

template <typename T, typename IdxT>
void RaftCagraHnswlib<T, IdxT>::build(const T* dataset, size_t nrow, cudaStream_t stream)
{
  if (not cagra_build_) {
    cagra_build_ = std::make_unique<RaftCagra<T, IdxT>>(metric_, dimension_, index_params_);
  }
  cagra_build_->build(dataset, nrow, stream);
}

template <typename T, typename IdxT>
void RaftCagraHnswlib<T, IdxT>::set_search_param(const AnnSearchParam& param_)
{
  hnswlib_search_->set_search_param(param_);
}

template <typename T, typename IdxT>
void RaftCagraHnswlib<T, IdxT>::save(const std::string& file) const
{
  cagra_build_->save_to_hnswlib(file);
}

template <typename T, typename IdxT>
void RaftCagraHnswlib<T, IdxT>::load(const std::string& file)
{
  typename HnswLib<T>::BuildParam param;
  // these values don't matter since we don't build with HnswLib
  param.M               = 50;
  param.ef_construction = 100;
  if (not hnswlib_search_) {
    hnswlib_search_ = std::make_unique<HnswLib<T>>(metric_, dimension_, param);
  }
  hnswlib_search_->load(file);
  hnswlib_search_->set_base_layer_only();
}

template <typename T, typename IdxT>
void RaftCagraHnswlib<T, IdxT>::search(
  const T* queries, int batch_size, int k, size_t* neighbors, float* distances, cudaStream_t) const
{
  hnswlib_search_->search(queries, batch_size, k, neighbors, distances);
}

}  // namespace raft::bench::ann
