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
#ifdef RAFT_ANN_BENCH_USE_HNSWLIB
#include "hnswlib_wrapper.h"
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

#ifdef RAFT_ANN_BENCH_USE_HNSWLIB
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

  (void)dev_list;
  return std::make_unique<Algo<T>>(metric, dim, param);
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

  cuann::Metric metric = parse_metric(distance);
  std::unique_ptr<cuann::ANN<T>> ann;

  if constexpr (std::is_same_v<T, float>) {
#ifdef RAFT_ANN_BENCH_USE_HNSWLIB
    if (algo == "hnswlib") { ann = make_algo<T, cuann::HnswLib>(metric, dim, conf); }
#endif
  }

  if constexpr (std::is_same_v<T, uint8_t>) {
#ifdef RAFT_ANN_BENCH_USE_HNSWLIB
    if (algo == "hnswlib") { ann = make_algo<T, cuann::HnswLib>(metric, dim, conf); }
#endif
  }

  if (!ann) { throw std::runtime_error("invalid algo: '" + algo + "'"); }

  if (refine_ratio > 1.0) {}
  return ann;
}

template <typename T>
std::unique_ptr<typename cuann::ANN<T>::AnnSearchParam> create_search_param(
  const std::string& algo, const nlohmann::json& conf)
{
#ifdef RAFT_ANN_BENCH_USE_HNSWLIB
  if (algo == "hnswlib") {
    auto param = std::make_unique<typename cuann::HnswLib<T>::SearchParam>();
    parse_search_param<T>(conf, *param);
    return param;
  }
#endif
  // else
  throw std::runtime_error("invalid algo: '" + algo + "'");
}

}  // namespace benchmark
#endif
