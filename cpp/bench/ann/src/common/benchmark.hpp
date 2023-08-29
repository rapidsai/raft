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

#include "ann_types.hpp"
#include "conf.hpp"
#include "dataset.hpp"
#include "util.hpp"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <unistd.h>
#include <vector>

#ifdef ANN_BENCH_BUILD_MAIN
#ifdef CPU_ONLY
#define CUDART_FOUND false
#else
#define CUDART_FOUND true
#endif
#else
#define CUDART_FOUND (cudart.found())
#endif

namespace raft::bench::ann {

static inline std::unique_ptr<AnnBase> current_algo{nullptr};

using kv_series = std::vector<std::tuple<std::string, std::vector<nlohmann::json>>>;

inline auto apply_overrides(const std::vector<nlohmann::json>& configs,
                            const kv_series& overrides,
                            std::size_t override_idx = 0) -> std::vector<nlohmann::json>
{
  std::vector<nlohmann::json> results{};
  if (override_idx >= overrides.size()) {
    auto n = configs.size();
    for (size_t i = 0; i < n; i++) {
      auto c               = configs[i];
      c["override_suffix"] = n > 1 ? "/" + std::to_string(i) : "";
      results.push_back(c);
    }
    return results;
  }
  auto rec_configs = apply_overrides(configs, overrides, override_idx + 1);
  auto [key, vals] = overrides[override_idx];
  auto n           = vals.size();
  for (size_t i = 0; i < n; i++) {
    const auto& val = vals[i];
    for (auto rc : rec_configs) {
      if (n > 1) {
        rc["override_suffix"] =
          static_cast<std::string>(rc["override_suffix"]) + "/" + std::to_string(i);
      }
      rc[key] = val;
      results.push_back(rc);
    }
  }
  return results;
}

inline auto apply_overrides(const nlohmann::json& config,
                            const kv_series& overrides,
                            std::size_t override_idx = 0)
{
  return apply_overrides(std::vector{config}, overrides, 0);
}

inline void dump_parameters(::benchmark::State& state, nlohmann::json params)
{
  std::string label = "";
  bool label_empty  = true;
  for (auto& [key, val] : params.items()) {
    if (val.is_number()) {
      state.counters.insert({{key, val}});
    } else if (val.is_boolean()) {
      state.counters.insert({{key, val ? 1.0 : 0.0}});
    } else {
      auto kv = key + "=" + val.dump();
      if (label_empty) {
        label = kv;
      } else {
        label += "#" + kv;
      }
      label_empty = false;
    }
  }
  if (!label_empty) { state.SetLabel(label); }
}

inline auto parse_algo_property(AlgoProperty prop, const nlohmann::json& conf) -> AlgoProperty
{
  if (conf.contains("dataset_memory_type")) {
    prop.dataset_memory_type = parse_memory_type(conf.at("dataset_memory_type"));
  }
  if (conf.contains("query_memory_type")) {
    prop.query_memory_type = parse_memory_type(conf.at("query_memory_type"));
  }
  return prop;
};

template <typename T>
void bench_build(::benchmark::State& state,
                 std::shared_ptr<const Dataset<T>> dataset,
                 Configuration::Index index,
                 bool force_overwrite)
{
  dump_parameters(state, index.build_param);
  if (file_exists(index.file)) {
    if (force_overwrite) {
      log_info("Overwriting file: %s", index.file.c_str());
    } else {
      return state.SkipWithMessage(
        "Index file already exists (use --overwrite to overwrite the index).");
    }
  }

  std::unique_ptr<ANN<T>> algo;
  try {
    algo = ann::create_algo<T>(
      index.algo, dataset->distance(), dataset->dim(), index.build_param, index.dev_list);
  } catch (const std::exception& e) {
    return state.SkipWithError("Failed to create an algo: " + std::string(e.what()));
  }

  const auto algo_property = parse_algo_property(algo->get_preference(), index.build_param);

  const T* base_set      = dataset->base_set(algo_property.dataset_memory_type);
  std::size_t index_size = dataset->base_set_size();

  cuda_timer gpu_timer;
  {
    nvtx_case nvtx{state.name()};
    for (auto _ : state) {
      [[maybe_unused]] auto ntx_lap = nvtx.lap();
      [[maybe_unused]] auto gpu_lap = gpu_timer.lap();
      try {
        algo->build(base_set, index_size, gpu_timer.stream());
      } catch (const std::exception& e) {
        state.SkipWithError(std::string(e.what()));
      }
    }
  }
  state.counters.insert(
    {{"GPU Time", gpu_timer.total_time() / state.iterations()}, {"index_size", index_size}});

  if (state.skipped()) { return; }
  make_sure_parent_dir_exists(index.file);
  algo->save(index.file);
}

template <typename T>
void bench_search(::benchmark::State& state,
                  std::shared_ptr<const Dataset<T>> dataset,
                  Configuration::Index index,
                  std::size_t search_param_ix)
{
  const auto& sp_json = index.search_params[search_param_ix];
  dump_parameters(state, sp_json);

  // NB: `k` and `n_queries` are guaranteed to be populated in conf.cpp
  const std::uint32_t k = sp_json["k"];
  // Amount of data processes in one go
  const std::size_t n_queries = sp_json["n_queries"];
  // Round down the query data to a multiple of the batch size to loop over full batches of data
  const std::size_t query_set_size = (dataset->query_set_size() / n_queries) * n_queries;

  if (!file_exists(index.file)) {
    state.SkipWithError("Index file is missing. Run the benchmark in the build mode first.");
    return;
  }
  // algo is static to cache it between close search runs to save time on index loading
  static std::string index_file = "";
  if (index.file != index_file) {
    current_algo.reset();
    index_file = index.file;
  }
  ANN<T>* algo;
  std::unique_ptr<typename ANN<T>::AnnSearchParam> search_param;
  try {
    if (!current_algo || (algo = dynamic_cast<ANN<T>*>(current_algo.get())) == nullptr) {
      auto ualgo = ann::create_algo<T>(
        index.algo, dataset->distance(), dataset->dim(), index.build_param, index.dev_list);
      algo = ualgo.get();
      algo->load(index_file);
      current_algo = std::move(ualgo);
    }
    search_param = ann::create_search_param<T>(index.algo, sp_json);
  } catch (const std::exception& e) {
    return state.SkipWithError("Failed to create an algo: " + std::string(e.what()));
  }
  algo->set_search_param(*search_param);

  const auto algo_property = parse_algo_property(algo->get_preference(), sp_json);
  const T* query_set       = dataset->query_set(algo_property.query_memory_type);
  buf<float> distances{algo_property.query_memory_type, k * query_set_size};
  buf<std::size_t> neighbors{algo_property.query_memory_type, k * query_set_size};

  if (search_param->needs_dataset()) {
    try {
      algo->set_search_dataset(dataset->base_set(algo_property.dataset_memory_type),
                               dataset->base_set_size());
    } catch (const std::exception&) {
      state.SkipWithError("The algorithm '" + index.name +
                          "' requires the base set, but it's not available.");
      return;
    }
  }

  std::ptrdiff_t batch_offset   = 0;
  std::size_t queries_processed = 0;
  cuda_timer gpu_timer;
  {
    nvtx_case nvtx{state.name()};
    for (auto _ : state) {
      // measure the GPU time using the RAII helper
      [[maybe_unused]] auto ntx_lap = nvtx.lap();
      [[maybe_unused]] auto gpu_lap = gpu_timer.lap();
      // run the search
      try {
        algo->search(query_set + batch_offset * dataset->dim(),
                     n_queries,
                     k,
                     neighbors.data + batch_offset * k,
                     distances.data + batch_offset * k,
                     gpu_timer.stream());
      } catch (const std::exception& e) {
        state.SkipWithError(std::string(e.what()));
      }
      // advance to the next batch
      batch_offset = (batch_offset + n_queries) % query_set_size;
      queries_processed += n_queries;
    }
  }
  state.SetItemsProcessed(queries_processed);
  state.counters.insert({{"k", k}, {"n_queries", n_queries}});
  if (CUDART_FOUND) {
    state.counters.insert({{"GPU Time", gpu_timer.total_time() / state.iterations()},
                           {"GPU QPS", queries_processed / gpu_timer.total_time()}});
  }
  if (state.skipped()) { return; }

  // evaluate recall
  if (dataset->max_k() >= k) {
    const std::int32_t* gt          = dataset->gt_set();
    const std::uint32_t max_k       = dataset->max_k();
    buf<std::size_t> neighbors_host = neighbors.move(MemoryType::Host);

    std::size_t rows        = std::min(queries_processed, query_set_size);
    std::size_t match_count = 0;
    std::size_t total_count = rows * static_cast<size_t>(k);
    for (std::size_t i = 0; i < rows; i++) {
      for (std::uint32_t j = 0; j < k; j++) {
        auto act_idx = std::int32_t(neighbors_host.data[i * k + j]);
        for (std::uint32_t l = 0; l < k; l++) {
          auto exp_idx = gt[i * max_k + l];
          if (act_idx == exp_idx) {
            match_count++;
            break;
          }
        }
      }
    }
    double actual_recall = static_cast<double>(match_count) / static_cast<double>(total_count);
    state.counters.insert({{"Recall", actual_recall}});
  }
}

inline void printf_usage()
{
  ::benchmark::PrintDefaultHelp();
  fprintf(
    stdout,
    "          [--build|--search] \n"
    "          [--overwrite]\n"
    "          [--data_prefix=<prefix>]\n"
    "          [--index_prefix=<prefix>]\n"
    "          [--override_kv=<key:value1:value2:...:valueN>]\n"
    "          <conf>.json\n"
    "\n"
    "Note the non-standard benchmark parameters:\n"
    "  --build: build mode, will build index\n"
    "  --search: search mode, will search using the built index\n"
    "            one and only one of --build and --search should be specified\n"
    "  --overwrite: force overwriting existing index files\n"
    "  --data_prefix=<prefix>:"
    " prepend <prefix> to dataset file paths specified in the <conf>.json (default = 'data/').\n"
    "  --index_prefix=<prefix>:"
    " prepend <prefix> to index file paths specified in the <conf>.json (default = 'index/').\n"
    "  --override_kv=<key:value1:value2:...:valueN>:"
    " override a build/search key one or more times multiplying the number of configurations;"
    " you can use this parameter multiple times to get the Cartesian product of benchmark"
    " configs.\n");
}

template <typename T>
void register_build(std::shared_ptr<const Dataset<T>> dataset,
                    std::vector<Configuration::Index> indices,
                    bool force_overwrite)
{
  for (auto index : indices) {
    auto suf      = static_cast<std::string>(index.build_param["override_suffix"]);
    auto file_suf = suf;
    index.build_param.erase("override_suffix");
    std::replace(file_suf.begin(), file_suf.end(), '/', '-');
    index.file += file_suf;
    auto* b = ::benchmark::RegisterBenchmark(
      index.name + suf, bench_build<T>, dataset, index, force_overwrite);
    b->Unit(benchmark::kSecond);
    b->UseRealTime();
  }
}

template <typename T>
void register_search(std::shared_ptr<const Dataset<T>> dataset,
                     std::vector<Configuration::Index> indices)
{
  for (auto index : indices) {
    for (std::size_t i = 0; i < index.search_params.size(); i++) {
      auto suf = static_cast<std::string>(index.search_params[i]["override_suffix"]);
      index.search_params[i].erase("override_suffix");
      auto* b =
        ::benchmark::RegisterBenchmark(index.name + suf, bench_search<T>, dataset, index, i);
      b->Unit(benchmark::kMillisecond);
      b->UseRealTime();
    }
  }
}

template <typename T>
void dispatch_benchmark(const Configuration& conf,
                        bool force_overwrite,
                        bool build_mode,
                        bool search_mode,
                        std::string data_prefix,
                        std::string index_prefix,
                        kv_series override_kv)
{
  if (CUDART_FOUND) {
    for (auto [key, value] : cuda_info()) {
      ::benchmark::AddCustomContext(key, value);
    }
  }
  const auto dataset_conf = conf.get_dataset_conf();
  auto base_file          = combine_path(data_prefix, dataset_conf.base_file);
  auto query_file         = combine_path(data_prefix, dataset_conf.query_file);
  auto gt_file            = dataset_conf.groundtruth_neighbors_file;
  if (gt_file.has_value()) { gt_file.emplace(combine_path(data_prefix, gt_file.value())); }
  auto dataset = std::make_shared<BinDataset<T>>(dataset_conf.name,
                                                 base_file,
                                                 dataset_conf.subset_first_row,
                                                 dataset_conf.subset_size,
                                                 query_file,
                                                 dataset_conf.distance,
                                                 gt_file);
  ::benchmark::AddCustomContext("dataset", dataset_conf.name);
  ::benchmark::AddCustomContext("distance", dataset_conf.distance);
  std::vector<Configuration::Index> indices = conf.get_indices();
  if (build_mode) {
    if (file_exists(base_file)) {
      log_info("Using the dataset file '%s'", base_file.c_str());
      ::benchmark::AddCustomContext("n_records", std::to_string(dataset->base_set_size()));
      ::benchmark::AddCustomContext("dim", std::to_string(dataset->dim()));
    } else {
      log_warn("Dataset file '%s' does not exist; benchmarking index building is impossible.",
               base_file.c_str());
    }
    std::vector<Configuration::Index> more_indices{};
    for (auto& index : indices) {
      for (auto param : apply_overrides(index.build_param, override_kv)) {
        auto modified_index        = index;
        modified_index.build_param = param;
        modified_index.file        = combine_path(index_prefix, modified_index.file);
        more_indices.push_back(modified_index);
      }
    }
    register_build<T>(dataset, more_indices, force_overwrite);
  } else if (search_mode) {
    if (file_exists(query_file)) {
      log_info("Using the query file '%s'", query_file.c_str());
      ::benchmark::AddCustomContext("max_n_queries", std::to_string(dataset->query_set_size()));
      ::benchmark::AddCustomContext("dim", std::to_string(dataset->dim()));
      if (gt_file.has_value()) {
        if (file_exists(*gt_file)) {
          log_info("Using the ground truth file '%s'", gt_file->c_str());
          ::benchmark::AddCustomContext("max_k", std::to_string(dataset->max_k()));
        } else {
          log_warn("Ground truth file '%s' does not exist; the recall won't be reported.",
                   gt_file->c_str());
        }
      } else {
        log_warn(
          "Ground truth file is not provided; the recall won't be reported. NB: use "
          "the 'groundtruth_neighbors_file' alongside the 'query_file' key to specify the path to "
          "the ground truth in your conf.json.");
      }
    } else {
      log_warn("Query file '%s' does not exist; benchmarking search is impossible.",
               query_file.c_str());
    }
    for (auto& index : indices) {
      index.search_params = apply_overrides(index.search_params, override_kv);
      index.file          = combine_path(index_prefix, index.file);
    }
    register_search<T>(dataset, indices);
  }
}

inline auto parse_bool_flag(const char* arg, const char* pat, bool& result) -> bool
{
  if (strcmp(arg, pat) == 0) {
    result = true;
    return true;
  }
  return false;
}

inline auto parse_string_flag(const char* arg, const char* pat, std::string& result) -> bool
{
  auto n = strlen(pat);
  if (strncmp(pat, arg, strlen(pat)) == 0) {
    result = arg + n + 1;
    return true;
  }
  return false;
}

inline auto run_main(int argc, char** argv) -> int
{
  bool force_overwrite        = false;
  bool build_mode             = false;
  bool search_mode            = false;
  std::string data_prefix     = "data";
  std::string index_prefix    = "index";
  std::string new_override_kv = "";
  kv_series override_kv{};

  char arg0_default[] = "benchmark";  // NOLINT
  char* args_default  = arg0_default;
  if (!argv) {
    argc = 1;
    argv = &args_default;
  }
  if (argc == 1) {
    printf_usage();
    return -1;
  }

  char* conf_path = argv[--argc];
  std::ifstream conf_stream(conf_path);

  for (int i = 1; i < argc; i++) {
    if (parse_bool_flag(argv[i], "--overwrite", force_overwrite) ||
        parse_bool_flag(argv[i], "--build", build_mode) ||
        parse_bool_flag(argv[i], "--search", search_mode) ||
        parse_string_flag(argv[i], "--data_prefix", data_prefix) ||
        parse_string_flag(argv[i], "--index_prefix", index_prefix) ||
        parse_string_flag(argv[i], "--override_kv", new_override_kv)) {
      if (!new_override_kv.empty()) {
        auto kvv = split(new_override_kv, ':');
        auto key = kvv[0];
        std::vector<nlohmann::json> vals{};
        for (std::size_t j = 1; j < kvv.size(); j++) {
          vals.push_back(nlohmann::json::parse(kvv[j]));
        }
        override_kv.emplace_back(key, vals);
        new_override_kv = "";
      }
      for (int j = i; j < argc - 1; j++) {
        argv[j] = argv[j + 1];
      }
      argc--;
      i--;
    }
  }

  if (build_mode == search_mode) {
    log_error("One and only one of --build and --search should be specified");
    printf_usage();
    return -1;
  }

  if (!conf_stream) {
    log_error("Can't open configuration file: %s", conf_path);
    return -1;
  }

  if (!CUDART_FOUND) { log_warn("cudart library is not found, GPU-based indices won't work."); }

  Configuration conf(conf_stream);
  std::string dtype = conf.get_dataset_conf().dtype;

  if (dtype == "float") {
    dispatch_benchmark<float>(
      conf, force_overwrite, build_mode, search_mode, data_prefix, index_prefix, override_kv);
  } else if (dtype == "uint8") {
    dispatch_benchmark<std::uint8_t>(
      conf, force_overwrite, build_mode, search_mode, data_prefix, index_prefix, override_kv);
  } else if (dtype == "int8") {
    dispatch_benchmark<std::int8_t>(
      conf, force_overwrite, build_mode, search_mode, data_prefix, index_prefix, override_kv);
  } else {
    log_error("datatype '%s' is not supported", dtype.c_str());
    return -1;
  }

  ::benchmark::Initialize(&argc, argv, printf_usage);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return -1;
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  // Release a possibly cached ANN object, so that it cannot be alive longer than the handle to a
  // shared library it depends on (dynamic benchmark executable).
  current_algo.reset();
  return 0;
}

};  // namespace raft::bench::ann
