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

#include <raft/util/cudart_utils.hpp>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/index_io.h>
#include <omp.h>

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
  };

  FaissGpu(Metric metric, int dim, int nlist);
  virtual ~FaissGpu() noexcept { RAFT_CUDA_TRY_NO_THROW(cudaEventDestroy(sync_)); }

  void build(const T* dataset, size_t nrow, cudaStream_t stream = 0) final;

  void set_search_param(const AnnSearchParam& param) override;

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
  faiss::MetricType metric_type_;
  int nlist_;
  int device_;
  cudaEvent_t sync_{nullptr};
  cudaStream_t faiss_default_stream_{nullptr};
};

template <typename T>
FaissGpu<T>::FaissGpu(Metric metric, int dim, int nlist)
  : ANN<T>(metric, dim), metric_type_(parse_metric_type(metric)), nlist_(nlist)
{
  static_assert(std::is_same_v<T, float>, "faiss support only float type");
  RAFT_CUDA_TRY(cudaGetDevice(&device_));
  RAFT_CUDA_TRY(cudaEventCreate(&sync_, cudaEventDisableTiming));
  faiss_default_stream_ = gpu_resource_.getDefaultStream(device_);
}

template <typename T>
void FaissGpu<T>::build(const T* dataset, size_t nrow, cudaStream_t stream)
{
  OmpSingleThreadScope omp_single_thread;

  index_->train(nrow, dataset);  // faiss::gpu::GpuIndexFlat::train() will do nothing
  assert(index_->is_trained);
  index_->add(nrow, dataset);
  stream_wait(stream);
}

template <typename T>
void FaissGpu<T>::set_search_param(const AnnSearchParam& param)
{
  int nprobe = dynamic_cast<const SearchParam&>(param).nprobe;
  assert(nprobe <= nlist_);
  dynamic_cast<faiss::gpu::GpuIndexIVF*>(index_.get())->setNumProbes(nprobe);
}

template <typename T>
void FaissGpu<T>::search(const T* queries,
                         int batch_size,
                         int k,
                         size_t* neighbors,
                         float* distances,
                         cudaStream_t stream) const
{
  static_assert(sizeof(size_t) == sizeof(faiss::Index::idx_t),
                "sizes of size_t and faiss::Index::idx_t are different");
  index_->search(
    batch_size, queries, k, distances, reinterpret_cast<faiss::Index::idx_t*>(neighbors));
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
  dynamic_cast<GpuIndex*>(index_.get())->copyFrom(cpu_index.get());
}

template <typename T>
class FaissGpuIVFFlat : public FaissGpu<T> {
 public:
  struct BuildParam {
    int nlist;
  };

  FaissGpuIVFFlat(Metric metric, int dim, const BuildParam& param)
    : FaissGpu<T>(metric, dim, param.nlist)
  {
    faiss::gpu::GpuIndexIVFFlatConfig config;
    config.device = this->device_;
    this->index_  = std::make_unique<faiss::gpu::GpuIndexIVFFlat>(
      &(this->gpu_resource_), dim, param.nlist, this->metric_type_, config);
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
  struct BuildParam {
    int nlist;
    int M;
    bool useFloat16;
    bool usePrecomputed;
  };

  FaissGpuIVFPQ(Metric metric, int dim, const BuildParam& param)
    : FaissGpu<T>(metric, dim, param.nlist)
  {
    faiss::gpu::GpuIndexIVFPQConfig config;
    config.useFloat16LookupTables = param.useFloat16;
    config.usePrecomputedTables   = param.usePrecomputed;
    config.device                 = this->device_;
    this->index_ =
      std::make_unique<faiss::gpu::GpuIndexIVFPQ>(&(this->gpu_resource_),
                                                  dim,
                                                  param.nlist,
                                                  param.M,
                                                  8,  // FAISS only supports bitsPerCode=8
                                                  this->metric_type_,
                                                  config);
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

template <typename T>
class FaissGpuIVFSQ : public FaissGpu<T> {
 public:
  struct BuildParam {
    int nlist;
    std::string quantizer_type;
  };

  FaissGpuIVFSQ(Metric metric, int dim, const BuildParam& param)
    : FaissGpu<T>(metric, dim, param.nlist)
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
  FaissGpuFlat(Metric metric, int dim) : FaissGpu<T>(metric, dim, 0)
  {
    faiss::gpu::GpuIndexFlatConfig config;
    config.device = this->device_;
    this->index_  = std::make_unique<faiss::gpu::GpuIndexFlat>(
      &(this->gpu_resource_), dim, this->metric_type_, config);
  }

  // class FaissGpu is more like a IVF class, so need special treating here
  void set_search_param(const typename ANN<T>::AnnSearchParam&) override{};

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
