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

#include "../common/ann_types.hpp"

#undef WARP_SIZE
#include "faiss_gpu_wrapper.h"

#define JSON_DIAGNOSTICS 1
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace raft::bench::ann {

template <typename T>
void parse_base_build_param(const nlohmann::json& conf,
                            typename raft::bench::ann::FaissGpu<T>::BuildParam& param)
{
  param.nlist = conf.at("nlist");
  if (conf.contains("ratio")) { param.ratio = conf.at("ratio"); }
}

template <typename T>
void parse_build_param(const nlohmann::json& conf,
                       typename raft::bench::ann::FaissGpuIVFFlat<T>::BuildParam& param)
{
  parse_base_build_param<T>(conf, param);
}

template <typename T>
void parse_build_param(const nlohmann::json& conf,
                       typename raft::bench::ann::FaissGpuIVFPQ<T>::BuildParam& param)
{
  parse_base_build_param<T>(conf, param);
  param.M = conf.at("M");
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
                       typename raft::bench::ann::FaissGpuIVFSQ<T>::BuildParam& param)
{
  parse_base_build_param<T>(conf, param);
  param.quantizer_type = conf.at("quantizer_type");
}

template <typename T>
void parse_build_param(const nlohmann::json& conf,
                       typename raft::bench::ann::FaissGpuCagra<T>::BuildParam& param)
{
  if (conf.contains("graph_degree")) {
    param.build_params.graph_degree              = conf.at("graph_degree");
    param.build_params.intermediate_graph_degree = param.build_params.graph_degree * 2;
  }
  if (conf.contains("intermediate_graph_degree")) {
    param.build_params.intermediate_graph_degree = conf.at("intermediate_graph_degree");
  }
  if (conf.contains("graph_build_algo")) {
    if (conf.at("graph_build_algo") == "IVF_PQ") {
      param.build_params.build_algo = faiss::gpu::graph_build_algo::IVF_PQ;
    } else if (conf.at("graph_build_algo") == "NN_DESCENT") {
      param.build_params.build_algo = faiss::gpu::graph_build_algo::NN_DESCENT;
    }
  }
}

template <typename T>
void parse_build_param(const nlohmann::json& conf,
                       typename raft::bench::ann::FaissGpuHnswCagra<T>::BuildParam& param)
{
  param.efConstruction = conf.at("efConstruction");
  param.M              = conf.at("M");
}

template <typename T>
void parse_search_param(const nlohmann::json& conf,
                        typename raft::bench::ann::FaissGpu<T>::SearchParam& param)
{
  param.nprobe = conf.at("nprobe");
  if (conf.contains("refine_ratio")) { param.refine_ratio = conf.at("refine_ratio"); }
}

template <typename T>
void parse_search_param(const nlohmann::json& conf,
                        typename raft::bench::ann::FaissGpuCagra<T>::SearchParam& param)
{
  if (conf.contains("itopk")) { param.search_params.itopk_size = conf.at("itopk"); }
  if (conf.contains("search_width")) { param.search_params.search_width = conf.at("search_width"); }
  if (conf.contains("max_iterations")) {
    param.search_params.max_iterations = conf.at("max_iterations");
  }
  if (conf.contains("algo")) {
    if (conf.at("algo") == "single_cta") {
      param.search_params.algo = faiss::gpu::search_algo::SINGLE_CTA;
    } else if (conf.at("algo") == "multi_cta") {
      param.search_params.algo = faiss::gpu::search_algo::MULTI_CTA;
    } else if (conf.at("algo") == "multi_kernel") {
      param.search_params.algo = faiss::gpu::search_algo::MULTI_KERNEL;
    } else if (conf.at("algo") == "auto") {
      param.search_params.algo = faiss::gpu::search_algo::AUTO;
    } else {
      std::string tmp = conf.at("algo");
      THROW("Invalid value for algo: %s", tmp.c_str());
    }
  }
}

template <typename T>
void parse_search_param(const nlohmann::json& conf,
                        typename raft::bench::ann::FaissGpuCagraHnsw<T>::SearchParam& param)
{
  param.search_params.efSearch = conf.at("ef");
  if (conf.contains("numThreads")) { param.num_threads = conf.at("numThreads"); }
}

template <typename T, template <typename> class Algo>
std::unique_ptr<raft::bench::ann::ANN<T>> make_algo(raft::bench::ann::Metric metric,
                                                    int dim,
                                                    const nlohmann::json& conf)
{
  typename Algo<T>::BuildParam param;
  parse_build_param<T>(conf, param);
  return std::make_unique<Algo<T>>(metric, dim, param);
}

template <typename T, template <typename> class Algo>
std::unique_ptr<raft::bench::ann::ANN<T>> make_algo(raft::bench::ann::Metric metric,
                                                    int dim,
                                                    const nlohmann::json& conf,
                                                    const std::vector<int>& dev_list)
{
  typename Algo<T>::BuildParam param;
  parse_build_param<T>(conf, param);

  (void)dev_list;
  return std::make_unique<Algo<T>>(metric, dim, param);
}

template <typename T>
std::unique_ptr<raft::bench::ann::ANN<T>> create_algo(const std::string& algo,
                                                      const std::string& distance,
                                                      int dim,
                                                      const nlohmann::json& conf,
                                                      const std::vector<int>& dev_list)
{
  // stop compiler warning; not all algorithms support multi-GPU so it may not be used
  (void)dev_list;

  std::unique_ptr<raft::bench::ann::ANN<T>> ann;

  if constexpr (std::is_same_v<T, float>) {
    raft::bench::ann::Metric metric = parse_metric(distance);
    if (algo == "faiss_gpu_ivf_flat") {
      ann = make_algo<T, raft::bench::ann::FaissGpuIVFFlat>(metric, dim, conf, dev_list);
    } else if (algo == "faiss_gpu_ivf_pq") {
      ann = make_algo<T, raft::bench::ann::FaissGpuIVFPQ>(metric, dim, conf);
    } else if (algo == "faiss_gpu_ivf_sq") {
      ann = make_algo<T, raft::bench::ann::FaissGpuIVFSQ>(metric, dim, conf);
    } else if (algo == "faiss_gpu_flat") {
      ann = std::make_unique<raft::bench::ann::FaissGpuFlat<T>>(metric, dim);
    } else if (algo == "faiss_gpu_cagra") {
      ann = make_algo<T, raft::bench::ann::FaissGpuCagra>(metric, dim, conf);
    } else if (algo == "faiss_gpu_cagra_hnsw") {
      ann = make_algo<T, raft::bench::ann::FaissGpuCagraHnsw>(metric, dim, conf);
    } else if (algo == "faiss_gpu_hnsw_cagra") {
      ann = make_algo<T, raft::bench::ann::FaissGpuHnswCagra>(metric, dim, conf);
    }
  }

  if constexpr (std::is_same_v<T, uint8_t>) {}

  if (!ann) { throw std::runtime_error("invalid algo: '" + algo + "'"); }

  return ann;
}

template <typename T>
std::unique_ptr<typename raft::bench::ann::ANN<T>::AnnSearchParam> create_search_param(
  const std::string& algo, const nlohmann::json& conf)
{
  if (algo == "faiss_gpu_ivf_flat" || algo == "faiss_gpu_ivf_pq" || algo == "faiss_gpu_ivf_sq") {
    auto param = std::make_unique<typename raft::bench::ann::FaissGpu<T>::SearchParam>();
    parse_search_param<T>(conf, *param);
    return param;
  } else if (algo == "faiss_gpu_flat") {
    auto param = std::make_unique<typename raft::bench::ann::FaissGpu<T>::SearchParam>();
    return param;
  } else if (algo == "faiss_gpu_cagra") {
    auto param = std::make_unique<typename raft::bench::ann::FaissGpuCagra<T>::SearchParam>();
    parse_search_param<T>(conf, *param);
    return param;
  } else if (algo == "faiss_gpu_cagra_hnsw") {
    auto param = std::make_unique<typename raft::bench::ann::FaissGpuCagraHnsw<T>::SearchParam>();
    parse_search_param<T>(conf, *param);
    return param;
  } else if (algo == "faiss_gpu_hnsw_cagra") {
    auto param = std::make_unique<typename raft::bench::ann::FaissGpuHnswCagra<T>::SearchParam>();
    parse_search_param<T>(conf, *param);
    return param;
  }
  // else
  throw std::runtime_error("invalid algo: '" + algo + "'");
}

}  // namespace raft::bench::ann

REGISTER_ALGO_INSTANCE(float);
REGISTER_ALGO_INSTANCE(std::int8_t);
REGISTER_ALGO_INSTANCE(std::uint8_t);

#ifdef ANN_BENCH_BUILD_MAIN
#include "../common/benchmark.hpp"
int main(int argc, char** argv)
{
  rmm::mr::cuda_memory_resource cuda_mr;
  // Construct a resource that uses a coalescing best-fit pool allocator
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{&cuda_mr};
  rmm::mr::set_current_device_resource(
    &pool_mr);  // Updates the current device resource pointer to `pool_mr`
  rmm::mr::device_memory_resource* mr =
    rmm::mr::get_current_device_resource();  // Points to `pool_mr`
  return raft::bench::ann::run_main(argc, argv);
}
#endif
