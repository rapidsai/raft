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

#include "../hnswlib/hnswlib_wrapper.h"
#include "raft_cagra_wrapper.h"

#include <memory>

namespace raft::bench::ann {

template <typename T, typename IdxT>
class RaftCagraHnswlib : public ANN<T>, public AnnGPU {
 public:
  using typename ANN<T>::AnnSearchParam;
  using BuildParam  = typename RaftCagra<T, IdxT>::BuildParam;
  using SearchParam = typename HnswLib<T>::SearchParam;

  RaftCagraHnswlib(Metric metric, int dim, const BuildParam& param, int concurrent_searches = 1)
    : ANN<T>(metric, dim),
      cagra_build_{metric, dim, param, concurrent_searches},
      // HnswLib param values don't matter since we don't build with HnswLib
      hnswlib_search_{metric, dim, typename HnswLib<T>::BuildParam{50, 100}}
  {
  }

  void build(const T* dataset, size_t nrow) final;

  void set_search_param(const AnnSearchParam& param) override;

  // TODO: if the number of results is less than k, the remaining elements of 'neighbors'
  // will be filled with (size_t)-1
  void search(const T* queries,
              int batch_size,
              int k,
              AnnBase::index_type* neighbors,
              float* distances) const override;

  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    return cagra_build_.get_sync_stream();
  }

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
  std::unique_ptr<ANN<T>> copy() override
  {
    return std::make_unique<RaftCagraHnswlib<T, IdxT>>(*this);
  }

 private:
  RaftCagra<T, IdxT> cagra_build_;
  HnswLib<T> hnswlib_search_;
};

template <typename T, typename IdxT>
void RaftCagraHnswlib<T, IdxT>::build(const T* dataset, size_t nrow)
{
  cagra_build_.build(dataset, nrow);
}

template <typename T, typename IdxT>
void RaftCagraHnswlib<T, IdxT>::set_search_param(const AnnSearchParam& param_)
{
  hnswlib_search_.set_search_param(param_);
}

template <typename T, typename IdxT>
void RaftCagraHnswlib<T, IdxT>::save(const std::string& file) const
{
  cagra_build_.save_to_hnswlib(file);
}

template <typename T, typename IdxT>
void RaftCagraHnswlib<T, IdxT>::load(const std::string& file)
{
  hnswlib_search_.load(file);
  hnswlib_search_.set_base_layer_only();
}

template <typename T, typename IdxT>
void RaftCagraHnswlib<T, IdxT>::search(
  const T* queries, int batch_size, int k, AnnBase::index_type* neighbors, float* distances) const
{
  hnswlib_search_.search(queries, batch_size, k, neighbors, distances);
}

}  // namespace raft::bench::ann
