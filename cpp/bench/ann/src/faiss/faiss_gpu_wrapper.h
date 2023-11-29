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
#ifndef FAISS_WRAPPER_H_
#define FAISS_WRAPPER_H_

#include "../common/ann_types.hpp"

#include <raft/core/logger.hpp>
#include <raft/util/cudart_utils.hpp>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexRefine.h>
#include <faiss/IndexScalarQuantizer.h>
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
      metric_type_(parse_metric_type(metric)),
      nlist_{param.nlist},
      training_sample_fraction_{1.0 / double(param.ratio)}
  {
    static_assert(std::is_same_v<T, float>, "faiss support only float type");
    RAFT_CUDA_TRY(cudaGetDevice(&device_));
    RAFT_CUDA_TRY(cudaEventCreate(&sync_, cudaEventDisableTiming));
    faiss_default_stream_ = gpu_resource_.getDefaultStream(device_);
    raft::resource::set_cuda_stream(handle_, faiss_default_stream_);
    RAFT_LOG_INFO("device %d", device_);
    // store the current memory resource in case it is modified by the algorithm
    current_mr_ = rmm::mr::get_per_device_resource(rmm::cuda_device_id{device_});
  }

  virtual ~FaissGpu() noexcept
  {
    // restore the old memory resource
    rmm::mr::set_per_device_resource(rmm::cuda_device_id{device_}, current_mr_);
    RAFT_CUDA_TRY_NO_THROW(cudaEventDestroy(sync_));
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
    property.query_memory_type   = MemoryType::Host;
    return property;
  }

 protected:
  template <typename GpuIndex, typename CpuIndex>
  void save_(const std::string& file) const;

  template <typename GpuIndex, typename CpuIndex>
  void load_(const std::string& file);

  void stream_wait(cudaStream_t stream) const
  {
    RAFT_CUDA_TRY(cudaEventRecord(sync_, faiss_default_stream_));
    RAFT_CUDA_TRY(cudaStreamWaitEvent(stream, sync_));
  }

  mutable faiss::gpu::StandardGpuResources gpu_resource_;
  std::unique_ptr<faiss::gpu::GpuIndex> index_;
  std::unique_ptr<faiss::IndexRefineFlat> index_refine_{nullptr};
  faiss::MetricType metric_type_;
  int nlist_;
  int device_;
  cudaEvent_t sync_{nullptr};
  cudaStream_t faiss_default_stream_{nullptr};
  double training_sample_fraction_;
  std::unique_ptr<faiss::SearchParameters> search_params_;
  const T* dataset_;
  raft::device_resources handle_;
  float refine_ratio_ = 1.0;
  rmm::mr::device_memory_resource* current_mr_{nullptr};
};

template <typename T>
void FaissGpu<T>::build(const T* dataset, size_t nrow, cudaStream_t stream)
{
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
    index_ivf->cp.max_points_per_centroid = max_ppc;
    index_ivf->cp.min_points_per_centroid = min_ppc;
  }
  index_->train(nrow, dataset);  // faiss::gpu::GpuIndexFlat::train() will do nothing
  cudaDeviceSynchronize();
  RAFT_LOG_INFO("faiss index trained");
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
  static_assert(sizeof(size_t) == sizeof(faiss::idx_t),
                "sizes of size_t and faiss::idx_t are different");

  if (this->refine_ratio_ > 1.0) {
    // TODO: FAISS changed their search APIs to accept the search parameters as a struct object
    // but their refine API doesn't allow the struct to be passed in. Once this is fixed, we
    // need to re-enable refinement below
    // index_refine_->search(batch_size, queries, k, distances,
    // reinterpret_cast<faiss::idx_t*>(neighbors), this->search_params_.get()); Related FAISS issue:
    // https://github.com/facebookresearch/faiss/issues/3118
    throw std::runtime_error(
      "FAISS doesn't support refinement in their new APIs so this feature is disabled in the "
      "benchmarks for the time being.");
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
    this->index_    = std::make_unique<faiss::gpu::GpuIndexIVFFlat>(
      &(this->gpu_resource_), dim, param.nlist, this->metric_type_, config);
  }

  void set_search_param(const typename FaissGpu<T>::AnnSearchParam& param) override
  {
    auto search_param = dynamic_cast<const typename FaissGpu<T>::SearchParam&>(param);
    int nprobe        = search_param.nprobe;
    assert(nprobe <= nlist_);

    faiss::IVFSearchParameters faiss_search_params;
    faiss_search_params.nprobe = nprobe;
    this->search_params_       = std::make_unique<faiss::IVFSearchParameters>(faiss_search_params);
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
};

template <typename T>
class FaissGpuIVFPQ : public FaissGpu<T> {
 public:
  struct BuildParam : public FaissGpu<T>::BuildParam {
    int M;
    bool useFloat16;
    bool usePrecomputed;
    bool interleavedLayout;
    bool use_raft;
    int bitsPerCode;
  };

  FaissGpuIVFPQ(Metric metric, int dim, const BuildParam& param) : FaissGpu<T>(metric, dim, param)
  {
    faiss::gpu::GpuIndexIVFPQConfig config;
    config.useFloat16LookupTables = param.useFloat16;
    config.usePrecomputedTables   = param.usePrecomputed;
    config.use_raft               = param.use_raft;
    config.interleavedLayout      = param.interleavedLayout;
    config.device                 = this->device_;

    if (config.use_raft) {
      auto result =
        std::shared_ptr<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>{nullptr};

      auto* upstream = dynamic_cast<rmm::mr::cuda_memory_resource*>(
        rmm::mr::get_per_device_resource(rmm::cuda_device_id(this->device_)));
      if (upstream != nullptr) {
        auto result =
          std::make_shared<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(upstream);
        rmm::mr::set_per_device_resource(rmm::cuda_device_id(this->device_), result.get());
      }
    }
    cudaDeviceSynchronize();
    RAFT_LOG_INFO("set to pool resource");

    int subQuantizers = dim / param.M;
    this->index_      = std::make_unique<faiss::gpu::GpuIndexIVFPQ>(&(this->gpu_resource_),
                                                               dim,
                                                               param.nlist,
                                                               subQuantizers,
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

    this->search_params_ = std::make_unique<faiss::IVFPQSearchParameters>(faiss_search_params);

    if (search_param.refine_ratio > 1.0) {
      this->index_refine_ =
        std::make_unique<faiss::IndexRefineFlat>(this->index_.get(), this->dataset_);
      this->index_refine_.get()->k_factor = search_param.refine_ratio;
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
    this->index_  = std::make_unique<faiss::gpu::GpuIndexIVFScalarQuantizer>(
      &(this->gpu_resource_), dim, param.nlist, qtype, this->metric_type_, true, config);
  }

  void set_search_param(const typename FaissGpu<T>::AnnSearchParam& param) override
  {
    auto search_param = dynamic_cast<const typename FaissGpu<T>::SearchParam&>(param);
    int nprobe        = search_param.nprobe;
    assert(nprobe <= nlist_);

    faiss::IVFSearchParameters faiss_search_params;
    faiss_search_params.nprobe = nprobe;

    this->search_params_ = std::make_unique<faiss::IVFSearchParameters>(faiss_search_params);
    this->refine_ratio_  = search_param.refine_ratio;
    if (search_param.refine_ratio > 1.0) {
      this->index_refine_ =
        std::make_unique<faiss::IndexRefineFlat>(this->index_.get(), this->dataset_);
      this->index_refine_.get()->k_factor = search_param.refine_ratio;
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
};

template <typename T>
class FaissGpuFlat : public FaissGpu<T> {
 public:
  FaissGpuFlat(Metric metric, int dim)
    : FaissGpu<T>(metric, dim, typename FaissGpu<T>::BuildParam{})
  {
    faiss::gpu::GpuIndexFlatConfig config;
    config.device = this->device_;
    this->index_  = std::make_unique<faiss::gpu::GpuIndexFlat>(
      &(this->gpu_resource_), dim, this->metric_type_, config);
  }
  void set_search_param(const typename FaissGpu<T>::AnnSearchParam& param) override
  {
    auto search_param = dynamic_cast<const typename FaissGpu<T>::SearchParam&>(param);
    int nprobe        = search_param.nprobe;
    assert(nprobe <= nlist_);

    this->search_params_ = std::make_unique<faiss::SearchParameters>();
  }

  void save(const std::string& file) const override
  {
    this->template save_<faiss::gpu::GpuIndexFlat, faiss::IndexFlat>(file);
  }
  void load(const std::string& file) override
  {
    this->template load_<faiss::gpu::GpuIndexFlat, faiss::IndexFlat>(file);
  }
};

}  // namespace raft::bench::ann

#endif