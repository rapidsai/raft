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
#ifndef FACTORY_H_
#define FACTORY_H_

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "ann.h"
#undef WARP_SIZE
#ifdef RAFT_CUANN_BENCH_USE_FAISS
#include "faiss_wrapper.h"
#endif
#ifdef RAFT_CUANN_BENCH_USE_GGNN
#include "ggnn_wrapper.cuh"
#endif
#ifdef RAFT_CUANN_BENCH_USE_HNSWLIB
#include "hnswlib_wrapper.h"
#endif
#ifdef RAFT_CUANN_BENCH_USE_RAFT_BFKNN
#include "raft_wrapper.h"
#endif
#ifdef RAFT_CUANN_BENCH_USE_RAFT_IVF_FLAT
#include "raft_ivf_flat_wrapper.h"
extern template class cuann::RaftIvfFlatGpu<float, uint64_t>;
extern template class cuann::RaftIvfFlatGpu<uint8_t, uint64_t>;
#endif
#ifdef RAFT_CUANN_BENCH_USE_RAFT_IVF_PQ
#include "raft_ivf_pq_wrapper.h"
extern template class cuann::RaftIvfPQ<float, uint64_t>;
extern template class cuann::RaftIvfPQ<uint8_t, uint64_t>;
#endif
#ifdef RAFT_CUANN_BENCH_USE_MULTI_GPU
#include "multigpu.cuh"
#endif
#define JSON_DIAGNOSTICS 1
#include <nlohmann/json.hpp>

namespace benchmark {

cuann::Metric parse_metric(const std::string& metric_str)
{
  if (metric_str == "inner_product") {
    return cuann::Metric::kInnerProduct;
  } else if (metric_str == "euclidean") {
    return cuann::Metric::kEuclidean;
  } else {
    throw std::runtime_error("invalid metric: '" + metric_str + "'");
  }
}

#ifdef RAFT_CUANN_BENCH_USE_HNSWLIB
template <typename T>
void parse_build_param(const nlohmann::json& conf, typename cuann::HnswLib<T>::BuildParam& param)
{
  param.ef_construction = conf.at("efConstruction");
  param.M               = conf.at("M");
  if (conf.contains("numThreads")) { param.num_threads = conf.at("numThreads"); }
}

template <typename T>
void parse_search_param(const nlohmann::json& conf, typename cuann::HnswLib<T>::SearchParam& param)
{
  param.ef = conf.at("ef");
  if (conf.contains("numThreads")) { param.num_threads = conf.at("numThreads"); }
}
#endif

#ifdef RAFT_CUANN_BENCH_USE_FAISS
template <typename T>
void parse_build_param(const nlohmann::json& conf,
                       typename cuann::FaissGpuIVFFlat<T>::BuildParam& param)
{
  param.nlist = conf.at("nlist");
}

template <typename T>
void parse_build_param(const nlohmann::json& conf,
                       typename cuann::FaissGpuIVFPQ<T>::BuildParam& param)
{
  param.nlist = conf.at("nlist");
  param.M     = conf.at("M");
  if (conf.contains("usePrecomputed")) {
    param.usePrecomputed = conf.at("usePrecomputed");
  } else {
    param.usePrecomputed = false;
  }
  if (conf.contains("useFloat16")) {
    param.useFloat16 = conf.at("useFloat16");
  } else {
    param.useFloat16 = false;
  }
}

template <typename T>
void parse_build_param(const nlohmann::json& conf,
                       typename cuann::FaissGpuIVFSQ<T>::BuildParam& param)
{
  param.nlist          = conf.at("nlist");
  param.quantizer_type = conf.at("quantizer_type");
}

template <typename T>
void parse_search_param(const nlohmann::json& conf, typename cuann::FaissGpu<T>::SearchParam& param)
{
  param.nprobe = conf.at("nprobe");
}
#endif

#ifdef RAFT_CUANN_BENCH_USE_GGNN
template <typename T>
void parse_build_param(const nlohmann::json& conf, typename cuann::Ggnn<T>::BuildParam& param)
{
  param.dataset_size = conf.at("dataset_size");
  param.k            = conf.at("k");

  if (conf.contains("k_build")) { param.k_build = conf.at("k_build"); }
  if (conf.contains("segment_size")) { param.segment_size = conf.at("segment_size"); }
  if (conf.contains("num_layers")) { param.num_layers = conf.at("num_layers"); }
  if (conf.contains("tau")) { param.tau = conf.at("tau"); }
  if (conf.contains("refine_iterations")) {
    param.refine_iterations = conf.at("refine_iterations");
  }
}

template <typename T>
void parse_search_param(const nlohmann::json& conf, typename cuann::Ggnn<T>::SearchParam& param)
{
  param.tau = conf.at("tau");

  if (conf.contains("block_dim")) { param.block_dim = conf.at("block_dim"); }
  if (conf.contains("max_iterations")) { param.max_iterations = conf.at("max_iterations"); }
  if (conf.contains("cache_size")) { param.cache_size = conf.at("cache_size"); }
  if (conf.contains("sorted_size")) { param.sorted_size = conf.at("sorted_size"); }
}
#endif

#ifdef RAFT_CUANN_BENCH_USE_RAFT_IVF_FLAT
template <typename T, typename IdxT>
void parse_build_param(const nlohmann::json& conf,
                       typename cuann::RaftIvfFlatGpu<T, IdxT>::BuildParam& param)
{
  param.n_lists = conf.at("nlist");
  if (conf.contains("niter")) { param.kmeans_n_iters = conf.at("niter"); }
  if (conf.contains("ratio")) {
    param.kmeans_trainset_fraction = 1.0 / (double)conf.at("ratio");
    std::cout << "kmeans_trainset_fraction " << param.kmeans_trainset_fraction;
  }
}

template <typename T, typename IdxT>
void parse_search_param(const nlohmann::json& conf,
                        typename cuann::RaftIvfFlatGpu<T, IdxT>::SearchParam& param)
{
  param.ivf_flat_params.n_probes = conf.at("nprobe");
}
#endif

#ifdef RAFT_CUANN_BENCH_USE_RAFT_IVF_PQ
template <typename T, typename IdxT>
void parse_build_param(const nlohmann::json& conf,
                       typename cuann::RaftIvfPQ<T, IdxT>::BuildParam& param)
{
  param.n_lists = conf.at("nlist");
  if (conf.contains("niter")) { param.kmeans_n_iters = conf.at("niter"); }
  if (conf.contains("ratio")) { param.kmeans_trainset_fraction = 1.0 / (double)conf.at("ratio"); }
  if (conf.contains("pq_bits")) { param.pq_bits = conf.at("pq_bits"); }
  if (conf.contains("pq_dim")) { param.pq_dim = conf.at("pq_dim"); }
}

template <typename T, typename IdxT>
void parse_search_param(const nlohmann::json& conf,
                        typename cuann::RaftIvfPQ<T, IdxT>::SearchParam& param)
{
  param.pq_param.n_probes = conf.at("numProbes");
  if (conf.contains("internalDistanceDtype")) {
    std::string type = conf.at("internalDistanceDtype");
    if (type == "float") {
      param.pq_param.internal_distance_dtype = CUDA_R_32F;
    } else if (type == "half") {
      param.pq_param.internal_distance_dtype = CUDA_R_16F;
    } else {
      throw std::runtime_error("internalDistanceDtype: '" + type +
                               "', should be either 'float' or 'half'");
    }
  } else {
    // set half as default type
    param.pq_param.internal_distance_dtype = CUDA_R_16F;
  }

  if (conf.contains("smemLutDtype")) {
    std::string type = conf.at("smemLutDtype");
    if (type == "float") {
      param.pq_param.lut_dtype = CUDA_R_32F;
    } else if (type == "half") {
      param.pq_param.lut_dtype = CUDA_R_16F;
    } else if (type == "fp8") {
      param.pq_param.lut_dtype = CUDA_R_8U;
    } else {
      throw std::runtime_error("smemLutDtype: '" + type +
                               "', should be either 'float', 'half' or 'fp8'");
    }
  } else {
    // set half as default
    param.pq_param.lut_dtype = CUDA_R_16F;
  }
}
#endif

template <typename T, template <typename> class Algo>
std::unique_ptr<cuann::ANN<T>> make_algo(cuann::Metric metric, int dim, const nlohmann::json& conf)
{
  typename Algo<T>::BuildParam param;
  parse_build_param<T>(conf, param);
  return std::make_unique<Algo<T>>(metric, dim, param);
}

template <typename T, template <typename> class Algo>
std::unique_ptr<cuann::ANN<T>> make_algo(cuann::Metric metric,
                                         int dim,
                                         const nlohmann::json& conf,
                                         const std::vector<int>& dev_list)
{
  typename Algo<T>::BuildParam param;
  parse_build_param<T>(conf, param);

#ifdef RAFT_CUANN_BENCH_USE_MULTI_GPU
  if (dev_list.empty()) {
    return std::make_unique<Algo<T>>(metric, dim, param);
  } else {
    return std::make_unique<cuann::MultiGpuANN<T, Algo<T>>>(metric, dim, param, dev_list);
  }
#else
  (void)dev_list;
  return std::make_unique<Algo<T>>(metric, dim, param);
#endif
}

template <typename T>
std::unique_ptr<cuann::ANN<T>> create_algo(const std::string& algo,
                                           const std::string& distance,
                                           int dim,
                                           float refine_ratio,
                                           const nlohmann::json& conf,
                                           const std::vector<int>& dev_list)
{
  // stop compiler warning; not all algorithms support multi-GPU so it may not be used
  (void)dev_list;
#ifndef RAFT_CUANN_BENCH_USE_MULTI_GPU
  if (!dev_list.empty()) {
    throw std::runtime_error(
      "compiled without RAFT_CUANN_BENCH_USE_MULTI_GPU, but a device list is given");
  }
#endif

  cuann::Metric metric = parse_metric(distance);
  std::unique_ptr<cuann::ANN<T>> ann;

  if constexpr (std::is_same_v<T, float>) {
#ifdef RAFT_CUANN_BENCH_USE_HNSWLIB
    if (algo == "hnswlib") { ann = make_algo<T, cuann::HnswLib>(metric, dim, conf); }
#endif
#ifdef RAFT_CUANN_BENCH_USE_FAISS
    if (algo == "faiss_gpu_ivf_flat") {
      ann = make_algo<T, cuann::FaissGpuIVFFlat>(metric, dim, conf, dev_list);
    } else if (algo == "faiss_gpu_ivf_pq") {
      ann = make_algo<T, cuann::FaissGpuIVFPQ>(metric, dim, conf);
    } else if (algo == "faiss_gpu_ivf_sq") {
      ann = make_algo<T, cuann::FaissGpuIVFSQ>(metric, dim, conf);
    } else if (algo == "faiss_gpu_flat") {
      ann = std::make_unique<cuann::FaissGpuFlat<T>>(metric, dim);
    }
#endif
#ifdef RAFT_CUANN_BENCH_USE_RAFT_BFKNN
    if (algo == "raft_bfknn") { ann = std::make_unique<cuann::RaftGpu<T>>(metric, dim); }
#endif
  }

  if constexpr (std::is_same_v<T, uint8_t>) {
#ifdef RAFT_CUANN_BENCH_USE_HNSWLIB
    if (algo == "hnswlib") { ann = make_algo<T, cuann::HnswLib>(metric, dim, conf); }
#endif
  }

#ifdef RAFT_CUANN_BENCH_USE_GGNN
  if (algo == "ggnn") { ann = make_algo<T, cuann::Ggnn>(metric, dim, conf); }
#endif
#ifdef RAFT_CUANN_BENCH_USE_RAFT_IVF_FLAT
  if (algo == "raft_ivf_flat") {
    typename cuann::RaftIvfFlatGpu<T, uint64_t>::BuildParam param;
    parse_build_param<T, uint64_t>(conf, param);
    ann = std::make_unique<cuann::RaftIvfFlatGpu<T, uint64_t>>(metric, dim, param);
  }
#endif
#ifdef RAFT_CUANN_BENCH_USE_RAFT_IVF_PQ
  if (algo == "raft_ivf_pq") {
    typename cuann::RaftIvfPQ<T, uint64_t>::BuildParam param;
    parse_build_param<T, uint64_t>(conf, param);
    ann = std::make_unique<cuann::RaftIvfPQ<T, uint64_t>>(metric, dim, param, refine_ratio);
  }
#endif
  if (!ann) { throw std::runtime_error("invalid algo: '" + algo + "'"); }

  if (refine_ratio > 1.0) {}
  return ann;
}

template <typename T>
std::unique_ptr<typename cuann::ANN<T>::AnnSearchParam> create_search_param(
  const std::string& algo, const nlohmann::json& conf)
{
#ifdef RAFT_CUANN_BENCH_USE_HNSWLIB
  if (algo == "hnswlib") {
    auto param = std::make_unique<typename cuann::HnswLib<T>::SearchParam>();
    parse_search_param<T>(conf, *param);
    return param;
  }
#endif
#ifdef RAFT_CUANN_BENCH_USE_FAISS
  if (algo == "faiss_gpu_ivf_flat" || algo == "faiss_gpu_ivf_pq" || algo == "faiss_gpu_ivf_sq") {
    auto param = std::make_unique<typename cuann::FaissGpu<T>::SearchParam>();
    parse_search_param<T>(conf, *param);
    return param;
  } else if (algo == "faiss_gpu_flat") {
    auto param = std::make_unique<typename cuann::ANN<T>::AnnSearchParam>();
    return param;
  }
#endif
#ifdef RAFT_CUANN_BENCH_USE_RAFT_BFKNN
  if (algo == "raft_bfknn") {
    auto param = std::make_unique<typename cuann::ANN<T>::AnnSearchParam>();
    return param;
  }
#endif
#ifdef RAFT_CUANN_BENCH_USE_GGNN
  if (algo == "ggnn") {
    auto param = std::make_unique<typename cuann::Ggnn<T>::SearchParam>();
    parse_search_param<T>(conf, *param);
    return param;
  }
#endif
#ifdef RAFT_CUANN_BENCH_USE_RAFT_IVF_FLAT
  if (algo == "raft_ivf_flat") {
    auto param = std::make_unique<typename cuann::RaftIvfFlatGpu<T, uint64_t>::SearchParam>();
    parse_search_param<T, uint64_t>(conf, *param);
    return param;
  }
#endif
#ifdef RAFT_CUANN_BENCH_USE_RAFT_IVF_PQ
  if (algo == "raft_ivf_pq") {
    auto param = std::make_unique<typename cuann::RaftIvfPQ<T, uint64_t>::SearchParam>();
    parse_search_param<T, uint64_t>(conf, *param);
    return param;
  }
#endif
  // else
  throw std::runtime_error("invalid algo: '" + algo + "'");
}

}  // namespace benchmark
#endif
