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
#ifndef FAISS_WRAPPER_H_
#define FAISS_WRAPPER_H_

#include "../common/ann_types.hpp"
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/distance/distance_types.hpp>

#include <raft/core/logger.hpp>
#include <raft/util/cudart_utils.hpp>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexRefine.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/MetricType.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/index_io.h>
#include <omp.h>

#include <raft/core/device_resources.hpp>
#include <raft/core/resource/stream_view.hpp>
#include <raft_runtime/neighbors/refine.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <cassert>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

faiss::MetricType parse_metric_type(raft::bench::ann::Metric metric)
{
  if (metric == raft::bench::ann::Metric::kInnerProduct) {
    return faiss::METRIC_INNER_PRODUCT;
  } else if (metric == raft::bench::ann::Metric::kEuclidean) {
    return faiss::METRIC_L2;
  } else {
    throw std::runtime_error("faiss supports only metric type of inner product and L2");
  }
}

// note BLAS library can still use multi-threading, and
// setting environment variable like OPENBLAS_NUM_THREADS can control it
class OmpSingleThreadScope {
 public:
  OmpSingleThreadScope()
  {
    max_threads_ = omp_get_max_threads();
    omp_set_num_threads(1);
  }
  ~OmpSingleThreadScope()
  {
    // the best we can do
    omp_set_num_threads(max_threads_);
  }

 private:
  int max_threads_;
};

}  // namespace

namespace raft::bench::ann {

struct copyable_event {
  copyable_event() { RAFT_CUDA_TRY(cudaEventCreate(&value_, cudaEventDisableTiming)); }
  ~copyable_event() { RAFT_CUDA_TRY_NO_THROW(cudaEventDestroy(value_)); }
  copyable_event(copyable_event&&)            = default;
  copyable_event& operator=(copyable_event&&) = default;
  copyable_event(const copyable_event& res) : copyable_event{} {}
  copyable_event& operator=(const copyable_event& other) = delete;
  operator cudaEvent_t() const noexcept { return value_; }

 private:
  cudaEvent_t value_{nullptr};
};

template <typename T>
class FaissGpu : public ANN<T> {
 public:
  using typename ANN<T>::AnnSearchParam;
  struct SearchParam : public AnnSearchParam {
    int nprobe;
    float refine_ratio = 1.0;
    auto needs_dataset() const -> bool override { return refine_ratio > 1.0f; }
  };

  struct BuildParam {
    int nlist = 1;
    int ratio = 2;
  };

  FaissGpu(Metric metric, int dim, const BuildParam& param)
    : ANN<T>(metric, dim),
      gpu_resource_{std::make_shared<faiss::gpu::StandardGpuResources>()},
      metric_type_(parse_metric_type(metric)),
      nlist_{param.nlist},
      training_sample_fraction_{1.0 / double(param.ratio)}
  {
    static_assert(std::is_same_v<T, float>, "faiss support only float type");
    RAFT_CUDA_TRY(cudaGetDevice(&device_));
  }

  void build(const T* dataset, size_t nrow, cudaStream_t stream = 0) final;

  virtual void set_search_param(const FaissGpu<T>::AnnSearchParam& param) {}

  void set_search_dataset(const T* dataset, size_t nrow) override { dataset_ = dataset; }

  // TODO: if the number of results is less than k, the remaining elements of 'neighbors'
  // will be filled with (size_t)-1
  void search(const T* queries,
              int batch_size,
              int k,
              size_t* neighbors,
              float* distances,
              cudaStream_t stream = 0) const final;

  AlgoProperty get_preference() const override
  {
    AlgoProperty property;
    // to enable building big dataset which is larger than GPU memory
    property.dataset_memory_type = MemoryType::Host;
    property.query_memory_type   = MemoryType::Device;
    return property;
  }

  auto metric_faiss_to_raft(faiss::MetricType metric) const -> raft::distance::DistanceType;

 protected:
  template <typename GpuIndex, typename CpuIndex>
  void save_(const std::string& file) const;

  template <typename GpuIndex, typename CpuIndex>
  void load_(const std::string& file);

  void stream_wait(cudaStream_t stream) const
  {
    RAFT_CUDA_TRY(cudaEventRecord(sync_, gpu_resource_->getDefaultStream(device_)));
    RAFT_CUDA_TRY(cudaStreamWaitEvent(stream, sync_));
  }

  /** [NOTE Multithreading]
   *
   * `gpu_resource_` is a shared resource:
   *   1. It uses a shared_ptr under the hood, so the copies of it refer to the same
   *      resource implementation instance
   *   2. GpuIndex is probably keeping a reference to it, as it's passed to the constructor
   *
   * To avoid copying the index (database) in each thread, we make both the index and
   * the gpu_resource shared.
   * This means faiss GPU streams are possibly shared among the CPU threads;
   * the throughput search mode may be inaccurate.
   *
   * WARNING: we haven't investigated whether faiss::gpu::GpuIndex or
   * faiss::gpu::StandardGpuResources are thread-safe.
   *
   */
  mutable std::shared_ptr<faiss::gpu::StandardGpuResources> gpu_resource_;
  std::shared_ptr<faiss::gpu::GpuIndex> index_;
  std::shared_ptr<faiss::IndexRefineFlat> index_refine_{nullptr};
  faiss::MetricType metric_type_;
  int nlist_;
  int device_;
  copyable_event sync_{};
  double training_sample_fraction_;
  std::shared_ptr<faiss::SearchParameters> search_params_;
  std::shared_ptr<faiss::IndexRefineSearchParameters> refine_search_params_{nullptr};
  const T* dataset_;
  float refine_ratio_ = 1.0;
};

template <typename T>
auto FaissGpu<T>::metric_faiss_to_raft(faiss::MetricType metric) const
  -> raft::distance::DistanceType
{
  switch (metric) {
    case faiss::MetricType::METRIC_L2: return raft::distance::DistanceType::L2Expanded;
    case faiss::MetricType::METRIC_INNER_PRODUCT:
    default: throw std::runtime_error("FAISS supports only metric type of inner product and L2");
  }
}

template <typename T>
void FaissGpu<T>::build(const T* dataset, size_t nrow, cudaStream_t stream)
{
  // raft::print_host_vector("faiss dataset", dataset, 100, std::cout);
  OmpSingleThreadScope omp_single_thread;
  auto index_ivf = dynamic_cast<faiss::gpu::GpuIndexIVF*>(index_.get());
  if (index_ivf != nullptr) {
    // set the min/max training size for clustering to use the whole provided training set.
    double trainset_size       = training_sample_fraction_ * static_cast<double>(nrow);
    double points_per_centroid = trainset_size / static_cast<double>(nlist_);
    int max_ppc                = std::ceil(points_per_centroid);
    int min_ppc                = std::floor(points_per_centroid);
    if (min_ppc < index_ivf->cp.min_points_per_centroid) {
      RAFT_LOG_WARN(
        "The suggested training set size %zu (data size %zu, training sample ratio %f) yields %d "
        "points per cluster (n_lists = %d). This is smaller than the FAISS default "
        "min_points_per_centroid = %d.",
        static_cast<size_t>(trainset_size),
        nrow,
        training_sample_fraction_,
        min_ppc,
        nlist_,
        index_ivf->cp.min_points_per_centroid);
    }
    index_ivf->cp.max_points_per_centroid = 300;
    index_ivf->cp.min_points_per_centroid = min_ppc;
  }
  index_->train(nrow, dataset);  // faiss::gpu::GpuIndexFlat::train() will do nothing
  assert(index_->is_trained);
  index_->add(nrow, dataset);
  stream_wait(stream);
}

template <typename T>
void FaissGpu<T>::search(const T* queries,
                         int batch_size,
                         int k,
                         size_t* neighbors,
                         float* distances,
                         cudaStream_t stream) const
{
  using IdxT = faiss::idx_t;
  static_assert(sizeof(size_t) == sizeof(faiss::idx_t),
                "sizes of size_t and faiss::idx_t are different");

  if (refine_ratio_ > 1.0) {
    if (raft::get_device_for_address(queries) >= 0) {
      uint32_t k0        = static_cast<uint32_t>(refine_ratio_ * k);
      auto distances_tmp = raft::make_device_matrix<float, IdxT>(
        gpu_resource_->getRaftHandle(device_), batch_size, k0);
      auto candidates =
        raft::make_device_matrix<IdxT, IdxT>(gpu_resource_->getRaftHandle(device_), batch_size, k0);
      index_->search(batch_size,
                     queries,
                     k0,
                     distances_tmp.data_handle(),
                     candidates.data_handle(),
                     this->search_params_.get());

      auto queries_host    = raft::make_host_matrix<T, IdxT>(batch_size, index_->d);
      auto candidates_host = raft::make_host_matrix<IdxT, IdxT>(batch_size, k0);
      auto neighbors_host  = raft::make_host_matrix<IdxT, IdxT>(batch_size, k);
      auto distances_host  = raft::make_host_matrix<float, IdxT>(batch_size, k);
      auto dataset_v       = raft::make_host_matrix_view<const T, faiss::idx_t>(
        this->dataset_, index_->ntotal, index_->d);

      raft::device_resources handle_ = gpu_resource_->getRaftHandle(device_);

      raft::copy(queries_host.data_handle(), queries, queries_host.size(), stream);
      raft::copy(candidates_host.data_handle(),
                 candidates.data_handle(),
                 candidates_host.size(),
                 resource::get_cuda_stream(handle_));

      // wait for the queries to copy to host in 'stream`
      handle_.sync_stream();
      
      raft::runtime::neighbors::refine(handle_,
                                       dataset_v,
                                       queries_host.view(),
                                       candidates_host.view(),
                                       neighbors_host.view(),
                                       distances_host.view(),
                                       metric_faiss_to_raft(index_->metric_type));

      raft::copy(neighbors, (size_t*)neighbors_host.data_handle(), neighbors_host.size(), stream);
      raft::copy(distances, distances_host.data_handle(), distances_host.size(), stream);
    } else {
      index_refine_->search(batch_size,
                            queries,
                            k,
                            distances,
                            reinterpret_cast<faiss::idx_t*>(neighbors),
                            this->refine_search_params_.get());
    }
  } else {
    index_->search(batch_size,
                   queries,
                   k,
                   distances,
                   reinterpret_cast<faiss::idx_t*>(neighbors),
                   this->search_params_.get());
  }
  stream_wait(stream);
}

template <typename T>
template <typename GpuIndex, typename CpuIndex>
void FaissGpu<T>::save_(const std::string& file) const
{
  OmpSingleThreadScope omp_single_thread;

  auto cpu_index = std::make_unique<CpuIndex>();
  dynamic_cast<GpuIndex*>(index_.get())->copyTo(cpu_index.get());
  faiss::write_index(cpu_index.get(), file.c_str());
}

template <typename T>
template <typename GpuIndex, typename CpuIndex>
void FaissGpu<T>::load_(const std::string& file)
{
  OmpSingleThreadScope omp_single_thread;

  std::unique_ptr<CpuIndex> cpu_index(dynamic_cast<CpuIndex*>(faiss::read_index(file.c_str())));
  assert(cpu_index);

  try {
    dynamic_cast<GpuIndex*>(index_.get())->copyFrom(cpu_index.get());

  } catch (const std::exception& e) {
    std::cout << "Error loading index file: " << std::string(e.what()) << std::endl;
  }
}

template <typename T>
class FaissGpuIVFFlat : public FaissGpu<T> {
 public:
  struct BuildParam : public FaissGpu<T>::BuildParam {
    bool use_raft;
  };

  FaissGpuIVFFlat(Metric metric, int dim, const BuildParam& param) : FaissGpu<T>(metric, dim, param)
  {
    faiss::gpu::GpuIndexIVFFlatConfig config;
    config.device   = this->device_;
    config.use_raft = param.use_raft;
    this->index_    = std::make_shared<faiss::gpu::GpuIndexIVFFlat>(
      this->gpu_resource_.get(), dim, param.nlist, this->metric_type_, config);
  }

  void set_search_param(const typename FaissGpu<T>::AnnSearchParam& param) override
  {
    auto search_param = dynamic_cast<const typename FaissGpu<T>::SearchParam&>(param);
    int nprobe        = search_param.nprobe;
    assert(nprobe <= nlist_);

    faiss::IVFSearchParameters faiss_search_params;
    faiss_search_params.nprobe = nprobe;
    this->search_params_       = std::make_shared<faiss::IVFSearchParameters>(faiss_search_params);
    this->refine_ratio_        = search_param.refine_ratio;
  }

  void save(const std::string& file) const override
  {
    this->template save_<faiss::gpu::GpuIndexIVFFlat, faiss::IndexIVFFlat>(file);
  }
  void load(const std::string& file) override
  {
    this->template load_<faiss::gpu::GpuIndexIVFFlat, faiss::IndexIVFFlat>(file);
  }
  std::unique_ptr<ANN<T>> copy() override { return std::make_unique<FaissGpuIVFFlat<T>>(*this); };
};

template <typename T>
class FaissGpuIVFPQ : public FaissGpu<T> {
 public:
  struct BuildParam : public FaissGpu<T>::BuildParam {
    int M;
    bool useFloat16;
    bool usePrecomputed;
    bool use_raft;
    int bitsPerCode;
  };

  FaissGpuIVFPQ(Metric metric, int dim, const BuildParam& param) : FaissGpu<T>(metric, dim, param)
  {
    faiss::gpu::GpuIndexIVFPQConfig config;
    config.useFloat16LookupTables = param.useFloat16;
    config.usePrecomputedTables   = param.usePrecomputed;
    config.use_raft               = param.use_raft;
    config.interleavedLayout      = param.use_raft;
    config.device                 = this->device_;

    this->index_ = std::make_shared<faiss::gpu::GpuIndexIVFPQ>(this->gpu_resource_.get(),
                                                               dim,
                                                               param.nlist,
                                                               param.M,
                                                               param.bitsPerCode,
                                                               this->metric_type_,
                                                               config);
  }

  void set_search_param(const typename FaissGpu<T>::AnnSearchParam& param) override
  {
    auto search_param = dynamic_cast<const typename FaissGpu<T>::SearchParam&>(param);
    int nprobe        = search_param.nprobe;
    assert(nprobe <= nlist_);
    this->refine_ratio_ = search_param.refine_ratio;
    faiss::IVFPQSearchParameters faiss_search_params;
    faiss_search_params.nprobe = nprobe;

    this->search_params_ = std::make_shared<faiss::IVFPQSearchParameters>(faiss_search_params);

    if (search_param.refine_ratio > 1.0) {
      this->index_refine_ =
        std::make_shared<faiss::IndexRefineFlat>(this->index_.get(), this->dataset_);
      this->index_refine_.get()->k_factor = search_param.refine_ratio;
      faiss::IndexRefineSearchParameters faiss_refine_search_params;
      faiss_refine_search_params.k_factor          = this->index_refine_.get()->k_factor;
      faiss_refine_search_params.base_index_params = this->search_params_.get();
      this->refine_search_params_ =
        std::make_unique<faiss::IndexRefineSearchParameters>(faiss_refine_search_params);
    }
  }

  void save(const std::string& file) const override
  {
    this->template save_<faiss::gpu::GpuIndexIVFPQ, faiss::IndexIVFPQ>(file);
  }
  void load(const std::string& file) override
  {
    this->template load_<faiss::gpu::GpuIndexIVFPQ, faiss::IndexIVFPQ>(file);
  }
  std::unique_ptr<ANN<T>> copy() override { return std::make_unique<FaissGpuIVFPQ<T>>(*this); };
};

// TODO: Enable this in cmake
//  ref: https://github.com/rapidsai/raft/issues/1876
template <typename T>
class FaissGpuIVFSQ : public FaissGpu<T> {
 public:
  struct BuildParam : public FaissGpu<T>::BuildParam {
    std::string quantizer_type;
  };

  FaissGpuIVFSQ(Metric metric, int dim, const BuildParam& param) : FaissGpu<T>(metric, dim, param)
  {
    faiss::ScalarQuantizer::QuantizerType qtype;
    if (param.quantizer_type == "fp16") {
      qtype = faiss::ScalarQuantizer::QT_fp16;
    } else if (param.quantizer_type == "int8") {
      qtype = faiss::ScalarQuantizer::QT_8bit;
    } else {
      throw std::runtime_error("FaissGpuIVFSQ supports only fp16 and int8 but got " +
                               param.quantizer_type);
    }

    faiss::gpu::GpuIndexIVFScalarQuantizerConfig config;
    config.device = this->device_;
    this->index_  = std::make_shared<faiss::gpu::GpuIndexIVFScalarQuantizer>(
      this->gpu_resource_.get(), dim, param.nlist, qtype, this->metric_type_, true, config);
  }

  void set_search_param(const typename FaissGpu<T>::AnnSearchParam& param) override
  {
    auto search_param = dynamic_cast<const typename FaissGpu<T>::SearchParam&>(param);
    int nprobe        = search_param.nprobe;
    assert(nprobe <= nlist_);

    faiss::IVFSearchParameters faiss_search_params;
    faiss_search_params.nprobe = nprobe;

    this->search_params_ = std::make_shared<faiss::IVFSearchParameters>(faiss_search_params);
    this->refine_ratio_  = search_param.refine_ratio;
    if (search_param.refine_ratio > 1.0) {
      this->index_refine_ =
        std::make_shared<faiss::IndexRefineFlat>(this->index_.get(), this->dataset_);
      this->index_refine_.get()->k_factor = search_param.refine_ratio;
      faiss::IndexRefineSearchParameters faiss_refine_search_params;
      faiss_refine_search_params.k_factor          = this->index_refine_.get()->k_factor;
      faiss_refine_search_params.base_index_params = this->search_params_.get();
      this->refine_search_params_ =
        std::make_unique<faiss::IndexRefineSearchParameters>(faiss_refine_search_params);
    }
  }

  void save(const std::string& file) const override
  {
    this->template save_<faiss::gpu::GpuIndexIVFScalarQuantizer, faiss::IndexIVFScalarQuantizer>(
      file);
  }
  void load(const std::string& file) override
  {
    this->template load_<faiss::gpu::GpuIndexIVFScalarQuantizer, faiss::IndexIVFScalarQuantizer>(
      file);
  }
  std::unique_ptr<ANN<T>> copy() override { return std::make_unique<FaissGpuIVFSQ<T>>(*this); };
};

template <typename T>
class FaissGpuFlat : public FaissGpu<T> {
 public:
  FaissGpuFlat(Metric metric, int dim)
    : FaissGpu<T>(metric, dim, typename FaissGpu<T>::BuildParam{})
  {
    faiss::gpu::GpuIndexFlatConfig config;
    config.device = this->device_;
    this->index_  = std::make_shared<faiss::gpu::GpuIndexFlat>(
      this->gpu_resource_.get(), dim, this->metric_type_, config);
  }
  void set_search_param(const typename FaissGpu<T>::AnnSearchParam& param) override
  {
    auto search_param = dynamic_cast<const typename FaissGpu<T>::SearchParam&>(param);
    int nprobe        = search_param.nprobe;
    assert(nprobe <= nlist_);

    this->search_params_ = std::make_shared<faiss::SearchParameters>();
  }

  void save(const std::string& file) const override
  {
    this->template save_<faiss::gpu::GpuIndexFlat, faiss::IndexFlat>(file);
  }
  void load(const std::string& file) override
  {
    this->template load_<faiss::gpu::GpuIndexFlat, faiss::IndexFlat>(file);
  }
  std::unique_ptr<ANN<T>> copy() override { return std::make_unique<FaissGpuFlat<T>>(*this); };
};

}  // namespace raft::bench::ann

#endif
