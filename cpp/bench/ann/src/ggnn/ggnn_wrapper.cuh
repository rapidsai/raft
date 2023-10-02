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

#include <ggnn/cuda_knn_ggnn_gpu_instance.cuh>
#include <raft/util/cudart_utils.hpp>

#include <memory>
#include <stdexcept>

namespace raft::bench::ann {

template <typename T, DistanceMeasure measure, int D, int KBuild, int KQuery, int S>
class GgnnImpl;

template <typename T>
class Ggnn : public ANN<T> {
 public:
  struct BuildParam {
    int k_build{24};       // KBuild
    int segment_size{32};  // S
    int num_layers{4};     // L
    float tau{0.5};
    int refine_iterations{2};
    int k;  // GGNN requires to know k during building
  };

  using typename ANN<T>::AnnSearchParam;
  struct SearchParam : public AnnSearchParam {
    float tau;
    int block_dim{32};
    int max_iterations{400};
    int cache_size{512};
    int sorted_size{256};
    auto needs_dataset() const -> bool override { return true; }
  };

  Ggnn(Metric metric, int dim, const BuildParam& param);
  ~Ggnn() { delete impl_; }

  void build(const T* dataset, size_t nrow, cudaStream_t stream = 0) override
  {
    impl_->build(dataset, nrow, stream);
  }

  void set_search_param(const AnnSearchParam& param) override { impl_->set_search_param(param); }
  void search(const T* queries,
              int batch_size,
              int k,
              size_t* neighbors,
              float* distances,
              cudaStream_t stream = 0) const override
  {
    impl_->search(queries, batch_size, k, neighbors, distances, stream);
  }

  void save(const std::string& file) const override { impl_->save(file); }
  void load(const std::string& file) override { impl_->load(file); }

  AlgoProperty get_preference() const override { return impl_->get_preference(); }

  void set_search_dataset(const T* dataset, size_t nrow) override
  {
    impl_->set_search_dataset(dataset, nrow);
  };

 private:
  ANN<T>* impl_;
};

template <typename T>
Ggnn<T>::Ggnn(Metric metric, int dim, const BuildParam& param) : ANN<T>(metric, dim)
{
  // ggnn/src/sift1m.cu
  if (metric == Metric::kEuclidean && dim == 128 && param.k_build == 24 && param.k == 10 &&
      param.segment_size == 32) {
    impl_ = new GgnnImpl<T, Euclidean, 128, 24, 10, 32>(metric, dim, param);
  }
  // ggnn/src/deep1b_multi_gpu.cu, and adapt it deep1B
  else if (metric == Metric::kEuclidean && dim == 96 && param.k_build == 24 && param.k == 10 &&
           param.segment_size == 32) {
    impl_ = new GgnnImpl<T, Euclidean, 96, 24, 10, 32>(metric, dim, param);
  } else if (metric == Metric::kInnerProduct && dim == 96 && param.k_build == 24 && param.k == 10 &&
             param.segment_size == 32) {
    impl_ = new GgnnImpl<T, Cosine, 96, 24, 10, 32>(metric, dim, param);
  } else if (metric == Metric::kInnerProduct && dim == 96 && param.k_build == 96 && param.k == 10 &&
             param.segment_size == 64) {
    impl_ = new GgnnImpl<T, Cosine, 96, 96, 10, 64>(metric, dim, param);
  }
  // ggnn/src/glove200.cu, adapt it to glove100
  else if (metric == Metric::kInnerProduct && dim == 100 && param.k_build == 96 && param.k == 10 &&
           param.segment_size == 64) {
    impl_ = new GgnnImpl<T, Cosine, 100, 96, 10, 64>(metric, dim, param);
  } else {
    throw std::runtime_error(
      "ggnn: not supported combination of metric, dim and build param; "
      "see Ggnn's constructor in ggnn_wrapper.cuh for available combinations");
  }
}

template <typename T, DistanceMeasure measure, int D, int KBuild, int KQuery, int S>
class GgnnImpl : public ANN<T> {
 public:
  using typename ANN<T>::AnnSearchParam;

  GgnnImpl(Metric metric, int dim, const typename Ggnn<T>::BuildParam& param);

  void build(const T* dataset, size_t nrow, cudaStream_t stream = 0) override;

  void set_search_param(const AnnSearchParam& param) override;
  void search(const T* queries,
              int batch_size,
              int k,
              size_t* neighbors,
              float* distances,
              cudaStream_t stream = 0) const override;

  void save(const std::string& file) const override;
  void load(const std::string& file) override;

  AlgoProperty get_preference() const override
  {
    AlgoProperty property;
    property.dataset_memory_type = MemoryType::Device;
    property.query_memory_type   = MemoryType::Device;
    return property;
  }

  void set_search_dataset(const T* dataset, size_t nrow) override;

 private:
  using ANN<T>::metric_;
  using ANN<T>::dim_;

  using GGNNGPUInstance = GGNNGPUInstance<measure,
                                          int64_t /* KeyT */,
                                          float /* ValueT */,
                                          size_t /* GAddrT */,
                                          T /* BaseT */,
                                          size_t /* BAddrT */,
                                          D,
                                          KBuild,
                                          KBuild / 2 /* KF */,
                                          KQuery,
                                          S>;
  std::unique_ptr<GGNNGPUInstance> ggnn_;
  typename Ggnn<T>::BuildParam build_param_;
  typename Ggnn<T>::SearchParam search_param_;
};

template <typename T, DistanceMeasure measure, int D, int KBuild, int KQuery, int S>
GgnnImpl<T, measure, D, KBuild, KQuery, S>::GgnnImpl(Metric metric,
                                                     int dim,
                                                     const typename Ggnn<T>::BuildParam& param)
  : ANN<T>(metric, dim), build_param_(param)
{
  if (metric_ == Metric::kInnerProduct) {
    if (measure != Cosine) { throw std::runtime_error("mis-matched metric"); }
  } else if (metric_ == Metric::kEuclidean) {
    if (measure != Euclidean) { throw std::runtime_error("mis-matched metric"); }
  } else {
    throw std::runtime_error(
      "ggnn supports only metric type of InnerProduct, Cosine and Euclidean");
  }

  if (dim != D) { throw std::runtime_error("mis-matched dim"); }
}

template <typename T, DistanceMeasure measure, int D, int KBuild, int KQuery, int S>
void GgnnImpl<T, measure, D, KBuild, KQuery, S>::build(const T* dataset,
                                                       size_t nrow,
                                                       cudaStream_t stream)
{
  int device;
  RAFT_CUDA_TRY(cudaGetDevice(&device));
  ggnn_ = std::make_unique<GGNNGPUInstance>(
    device, nrow, build_param_.num_layers, true, build_param_.tau);

  ggnn_->set_base_data(dataset);
  ggnn_->set_stream(stream);
  ggnn_->build(0);
  for (int i = 0; i < build_param_.refine_iterations; ++i) {
    ggnn_->refine();
  }
}

template <typename T, DistanceMeasure measure, int D, int KBuild, int KQuery, int S>
void GgnnImpl<T, measure, D, KBuild, KQuery, S>::set_search_dataset(const T* dataset, size_t nrow)
{
  ggnn_->set_base_data(dataset);
}

template <typename T, DistanceMeasure measure, int D, int KBuild, int KQuery, int S>
void GgnnImpl<T, measure, D, KBuild, KQuery, S>::set_search_param(const AnnSearchParam& param)
{
  search_param_ = dynamic_cast<const typename Ggnn<T>::SearchParam&>(param);
}

template <typename T, DistanceMeasure measure, int D, int KBuild, int KQuery, int S>
void GgnnImpl<T, measure, D, KBuild, KQuery, S>::search(const T* queries,
                                                        int batch_size,
                                                        int k,
                                                        size_t* neighbors,
                                                        float* distances,
                                                        cudaStream_t stream) const
{
  static_assert(sizeof(size_t) == sizeof(int64_t), "sizes of size_t and GGNN's KeyT are different");
  if (k != KQuery) {
    throw std::runtime_error(
      "k = " + std::to_string(k) +
      ", but this GGNN instance only supports k = " + std::to_string(KQuery));
  }

  ggnn_->set_stream(stream);
  RAFT_CUDA_TRY(cudaMemcpyToSymbol(c_tau_query, &search_param_.tau, sizeof(float)));

  const int block_dim      = search_param_.block_dim;
  const int max_iterations = search_param_.max_iterations;
  const int cache_size     = search_param_.cache_size;
  const int sorted_size    = search_param_.sorted_size;
  // default value
  if (block_dim == 32 && max_iterations == 400 && cache_size == 512 && sorted_size == 256) {
    ggnn_->template queryLayer<32, 400, 512, 256, false>(
      queries, batch_size, reinterpret_cast<int64_t*>(neighbors), distances);
  }
  // ggnn/src/sift1m.cu
  else if (block_dim == 32 && max_iterations == 200 && cache_size == 256 && sorted_size == 64) {
    ggnn_->template queryLayer<32, 200, 256, 64, false>(
      queries, batch_size, reinterpret_cast<int64_t*>(neighbors), distances);
  }
  // ggnn/src/sift1m.cu
  else if (block_dim == 32 && max_iterations == 400 && cache_size == 448 && sorted_size == 64) {
    ggnn_->template queryLayer<32, 400, 448, 64, false>(
      queries, batch_size, reinterpret_cast<int64_t*>(neighbors), distances);
  }
  // ggnn/src/glove200.cu
  else if (block_dim == 128 && max_iterations == 2000 && cache_size == 2048 && sorted_size == 32) {
    ggnn_->template queryLayer<128, 2000, 2048, 32, false>(
      queries, batch_size, reinterpret_cast<int64_t*>(neighbors), distances);
  }
  // for glove100
  else if (block_dim == 64 && max_iterations == 400 && cache_size == 512 && sorted_size == 32) {
    ggnn_->template queryLayer<64, 400, 512, 32, false>(
      queries, batch_size, reinterpret_cast<int64_t*>(neighbors), distances);
  } else if (block_dim == 128 && max_iterations == 2000 && cache_size == 1024 &&
             sorted_size == 32) {
    ggnn_->template queryLayer<128, 2000, 1024, 32, false>(
      queries, batch_size, reinterpret_cast<int64_t*>(neighbors), distances);
  } else {
    throw std::runtime_error("ggnn: not supported search param");
  }
}

template <typename T, DistanceMeasure measure, int D, int KBuild, int KQuery, int S>
void GgnnImpl<T, measure, D, KBuild, KQuery, S>::save(const std::string& file) const
{
  auto& ggnn_host   = ggnn_->ggnn_cpu_buffers.at(0);
  auto& ggnn_device = ggnn_->ggnn_shards.at(0);
  ggnn_->set_stream(0);

  ggnn_host.downloadAsync(ggnn_device);
  RAFT_CUDA_TRY(cudaStreamSynchronize(ggnn_device.stream));
  ggnn_host.store(file);
}

template <typename T, DistanceMeasure measure, int D, int KBuild, int KQuery, int S>
void GgnnImpl<T, measure, D, KBuild, KQuery, S>::load(const std::string& file)
{
  auto& ggnn_host   = ggnn_->ggnn_cpu_buffers.at(0);
  auto& ggnn_device = ggnn_->ggnn_shards.at(0);
  ggnn_->set_stream(0);

  ggnn_host.load(file);
  ggnn_host.uploadAsync(ggnn_device);
  RAFT_CUDA_TRY(cudaStreamSynchronize(ggnn_device.stream));
}

}  // namespace raft::bench::ann
