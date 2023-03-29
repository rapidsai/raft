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

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#ifdef RAFT_COMPILED
#include <raft/neighbors/specializations.cuh>
#endif

#include "../common/ann_types.hpp"
#include "../common/benchmark_util.hpp"
#undef WARP_SIZE
#ifdef RAFT_ANN_BENCH_USE_RAFT_BFKNN
#include "raft_wrapper.h"
#endif
#ifdef RAFT_ANN_BENCH_USE_RAFT_IVF_FLAT
#include "raft_ivf_flat_wrapper.h"
extern template class raft::bench::ann::RaftIvfFlatGpu<float, int64_t>;
extern template class raft::bench::ann::RaftIvfFlatGpu<uint8_t, int64_t>;
extern template class raft::bench::ann::RaftIvfFlatGpu<int8_t, int64_t>;
#endif
#ifdef RAFT_ANN_BENCH_USE_RAFT_IVF_PQ
#include "raft_ivf_pq_wrapper.h"
extern template class raft::bench::ann::RaftIvfPQ<float, int64_t>;
extern template class raft::bench::ann::RaftIvfPQ<uint8_t, int64_t>;
extern template class raft::bench::ann::RaftIvfPQ<int8_t, int64_t>;
#endif
#define JSON_DIAGNOSTICS 1
#include <nlohmann/json.hpp>

namespace raft::bench::ann {

#ifdef RAFT_ANN_BENCH_USE_RAFT_IVF_FLAT
template <typename T, typename IdxT>
void parse_build_param(const nlohmann::json& conf,
                       typename raft::bench::ann::RaftIvfFlatGpu<T, IdxT>::BuildParam& param)
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
                        typename raft::bench::ann::RaftIvfFlatGpu<T, IdxT>::SearchParam& param)
{
  param.ivf_flat_params.n_probes = conf.at("nprobe");
}
#endif

#ifdef RAFT_ANN_BENCH_USE_RAFT_IVF_PQ
template <typename T, typename IdxT>
void parse_build_param(const nlohmann::json& conf,
                       typename raft::bench::ann::RaftIvfPQ<T, IdxT>::BuildParam& param)
{
  param.n_lists = conf.at("nlist");
  if (conf.contains("niter")) { param.kmeans_n_iters = conf.at("niter"); }
  if (conf.contains("ratio")) { param.kmeans_trainset_fraction = 1.0 / (double)conf.at("ratio"); }
  if (conf.contains("pq_bits")) { param.pq_bits = conf.at("pq_bits"); }
  if (conf.contains("pq_dim")) { param.pq_dim = conf.at("pq_dim"); }
}

template <typename T, typename IdxT>
void parse_search_param(const nlohmann::json& conf,
                        typename raft::bench::ann::RaftIvfPQ<T, IdxT>::SearchParam& param)
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
                                                      float refine_ratio,
                                                      const nlohmann::json& conf,
                                                      const std::vector<int>& dev_list)
{
  // stop compiler warning; not all algorithms support multi-GPU so it may not be used
  (void)dev_list;

  raft::bench::ann::Metric metric = parse_metric(distance);
  std::unique_ptr<raft::bench::ann::ANN<T>> ann;

  if constexpr (std::is_same_v<T, float>) {
#ifdef RAFT_ANN_BENCH_USE_RAFT_BFKNN
    if (algo == "raft_bfknn") { ann = std::make_unique<raft::bench::ann::RaftGpu<T>>(metric, dim); }
#endif
  }

  if constexpr (std::is_same_v<T, uint8_t>) {}

#ifdef RAFT_ANN_BENCH_USE_RAFT_IVF_FLAT
  if (algo == "raft_ivf_flat") {
    typename raft::bench::ann::RaftIvfFlatGpu<T, int64_t>::BuildParam param;
    parse_build_param<T, int64_t>(conf, param);
    ann = std::make_unique<raft::bench::ann::RaftIvfFlatGpu<T, int64_t>>(metric, dim, param);
  }
#endif
#ifdef RAFT_ANN_BENCH_USE_RAFT_IVF_PQ
  if (algo == "raft_ivf_pq") {
    typename raft::bench::ann::RaftIvfPQ<T, int64_t>::BuildParam param;
    parse_build_param<T, int64_t>(conf, param);
    ann =
      std::make_unique<raft::bench::ann::RaftIvfPQ<T, int64_t>>(metric, dim, param, refine_ratio);
  }
#endif
  if (!ann) { throw std::runtime_error("invalid algo: '" + algo + "'"); }

  if (refine_ratio > 1.0) {}
  return ann;
}

template <typename T>
std::unique_ptr<typename raft::bench::ann::ANN<T>::AnnSearchParam> create_search_param(
  const std::string& algo, const nlohmann::json& conf)
{
#ifdef RAFT_ANN_BENCH_USE_RAFT_BFKNN
  if (algo == "raft_bfknn") {
    auto param = std::make_unique<typename raft::bench::ann::ANN<T>::AnnSearchParam>();
    return param;
  }
#endif
#ifdef RAFT_ANN_BENCH_USE_RAFT_IVF_FLAT
  if (algo == "raft_ivf_flat") {
    auto param =
      std::make_unique<typename raft::bench::ann::RaftIvfFlatGpu<T, int64_t>::SearchParam>();
    parse_search_param<T, int64_t>(conf, *param);
    return param;
  }
#endif
#ifdef RAFT_ANN_BENCH_USE_RAFT_IVF_PQ
  if (algo == "raft_ivf_pq") {
    auto param = std::make_unique<typename raft::bench::ann::RaftIvfPQ<T, int64_t>::SearchParam>();
    parse_search_param<T, int64_t>(conf, *param);
    return param;
  }
#endif
  // else
  throw std::runtime_error("invalid algo: '" + algo + "'");
}

};  // namespace raft::bench::ann

#include "../common/benchmark.hpp"

int main(int argc, char** argv) { return raft::bench::ann::run_main(argc, argv); }