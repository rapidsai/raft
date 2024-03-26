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
#include "raft_ann_bench_param_parser.h"
#include "raft_cagra_hnswlib_wrapper.h"

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#define JSON_DIAGNOSTICS 1
#include <nlohmann/json.hpp>

namespace raft::bench::ann {

template <typename T, typename IdxT>
void parse_search_param(const nlohmann::json& conf,
                        typename raft::bench::ann::RaftCagraHnswlib<T, IdxT>::SearchParam& param)
{
  param.ef = conf.at("ef");
  if (conf.contains("numThreads")) { param.num_threads = conf.at("numThreads"); }
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

  [[maybe_unused]] raft::bench::ann::Metric metric = parse_metric(distance);
  std::unique_ptr<raft::bench::ann::ANN<T>> ann;

  if constexpr (std::is_same_v<T, float> or std::is_same_v<T, std::uint8_t>) {
    if (algo == "raft_cagra_hnswlib") {
      typename raft::bench::ann::RaftCagraHnswlib<T, uint32_t>::BuildParam param;
      parse_build_param<T, uint32_t>(conf, param);
      ann = std::make_unique<raft::bench::ann::RaftCagraHnswlib<T, uint32_t>>(metric, dim, param);
    }
  }

  if (!ann) { throw std::runtime_error("invalid algo: '" + algo + "'"); }

  return ann;
}

template <typename T>
std::unique_ptr<typename raft::bench::ann::ANN<T>::AnnSearchParam> create_search_param(
  const std::string& algo, const nlohmann::json& conf)
{
  if (algo == "raft_cagra_hnswlib") {
    auto param =
      std::make_unique<typename raft::bench::ann::RaftCagraHnswlib<T, uint32_t>::SearchParam>();
    parse_search_param<T, uint32_t>(conf, *param);
    return param;
  }

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
  // and is initially sized to half of free device memory.
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{
    &cuda_mr, rmm::percent_of_free_device_memory(50)};
  rmm::mr::set_current_device_resource(
    &pool_mr);  // Updates the current device resource pointer to `pool_mr`
  rmm::mr::device_memory_resource* mr =
    rmm::mr::get_current_device_resource();  // Points to `pool_mr`
  return raft::bench::ann::run_main(argc, argv);
}
#endif
