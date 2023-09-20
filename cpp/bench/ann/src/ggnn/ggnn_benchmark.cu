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

#include "../common/ann_types.hpp"
#include "ggnn_wrapper.cuh"
#define JSON_DIAGNOSTICS 1
#include <nlohmann/json.hpp>

namespace raft::bench::ann {

template <typename T>
void parse_build_param(const nlohmann::json& conf,
                       typename raft::bench::ann::Ggnn<T>::BuildParam& param)
{
  param.k = conf.at("k");

  if (conf.contains("k_build")) { param.k_build = conf.at("k_build"); }
  if (conf.contains("segment_size")) { param.segment_size = conf.at("segment_size"); }
  if (conf.contains("num_layers")) { param.num_layers = conf.at("num_layers"); }
  if (conf.contains("tau")) { param.tau = conf.at("tau"); }
  if (conf.contains("refine_iterations")) {
    param.refine_iterations = conf.at("refine_iterations");
  }
}

template <typename T>
void parse_search_param(const nlohmann::json& conf,
                        typename raft::bench::ann::Ggnn<T>::SearchParam& param)
{
  param.tau = conf.at("tau");

  if (conf.contains("block_dim")) { param.block_dim = conf.at("block_dim"); }
  if (conf.contains("max_iterations")) { param.max_iterations = conf.at("max_iterations"); }
  if (conf.contains("cache_size")) { param.cache_size = conf.at("cache_size"); }
  if (conf.contains("sorted_size")) { param.sorted_size = conf.at("sorted_size"); }
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

  raft::bench::ann::Metric metric = parse_metric(distance);
  std::unique_ptr<raft::bench::ann::ANN<T>> ann;

  if constexpr (std::is_same_v<T, float>) {}

  if constexpr (std::is_same_v<T, uint8_t>) {}

  if (algo == "ggnn") { ann = make_algo<T, raft::bench::ann::Ggnn>(metric, dim, conf); }
  if (!ann) { throw std::runtime_error("invalid algo: '" + algo + "'"); }

  return ann;
}

template <typename T>
std::unique_ptr<typename raft::bench::ann::ANN<T>::AnnSearchParam> create_search_param(
  const std::string& algo, const nlohmann::json& conf)
{
  if (algo == "ggnn") {
    auto param = std::make_unique<typename raft::bench::ann::Ggnn<T>::SearchParam>();
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
int main(int argc, char** argv) { return raft::bench::ann::run_main(argc, argv); }
#endif
