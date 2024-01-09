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
#include <raft/core/logger.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>
namespace raft::bench::ann {

std::mutex init_mutex;
std::condition_variable cond_var;
std::atomic_int processed_threads{0};

static inline std::unique_ptr<AnnBase> current_algo{nullptr};
static inline std::unique_ptr<AlgoProperty> current_algo_props{nullptr};

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
        "Index file already exists (use --force to overwrite the index).");
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
    {{"GPU", gpu_timer.total_time() / state.iterations()}, {"index_size", index_size}});

  if (state.skipped()) { return; }
  make_sure_parent_dir_exists(index.file);
  algo->save(index.file);
}

template <typename T>
void bench_search(::benchmark::State& state,
                  Configuration::Index index,
                  std::size_t search_param_ix,
                  std::shared_ptr<const Dataset<T>> dataset,
                  Objective metric_objective)
{
  std::size_t queries_processed = 0;

  const auto& sp_json = index.search_params[search_param_ix];

  if (state.thread_index() == 0) { dump_parameters(state, sp_json); }

  // NB: `k` and `n_queries` are guaranteed to be populated in conf.cpp
  const std::uint32_t k = sp_json["k"];
  // Amount of data processes in one go
  const std::size_t n_queries = sp_json["n_queries"];
  // Round down the query data to a multiple of the batch size to loop over full batches of data
  const std::size_t query_set_size = (dataset->query_set_size() / n_queries) * n_queries;

  if (dataset->query_set_size() < n_queries) {
    std::stringstream msg;
    msg << "Not enough queries in benchmark set. Expected " << n_queries << ", actual "
        << dataset->query_set_size();
    return state.SkipWithError(msg.str());
  }

  // Each thread start from a different offset, so that the queries that they process do not
  // overlap.
  std::ptrdiff_t batch_offset   = (state.thread_index() * n_queries) % query_set_size;
  std::ptrdiff_t queries_stride = state.threads() * n_queries;
  // Output is saved into a contiguous buffer (separate buffers for each thread).
  std::ptrdiff_t out_offset = 0;

  const T* query_set = nullptr;

  if (!file_exists(index.file)) {
    state.SkipWithError("Index file is missing. Run the benchmark in the build mode first.");
    return;
  }

  /**
   * Make sure the first thread loads the algo and dataset
   */
  if (state.thread_index() == 0) {
    std::unique_lock lk(init_mutex);
    cond_var.wait(lk, [] { return processed_threads.load(std::memory_order_acquire) == 0; });
    // algo is static to cache it between close search runs to save time on index loading
    static std::string index_file = "";
    if (index.file != index_file) {
      current_algo.reset();
      index_file = index.file;
    }

    std::unique_ptr<typename ANN<T>::AnnSearchParam> search_param;
    ANN<T>* algo;
    try {
      if (!current_algo || (algo = dynamic_cast<ANN<T>*>(current_algo.get())) == nullptr) {
        auto ualgo = ann::create_algo<T>(
          index.algo, dataset->distance(), dataset->dim(), index.build_param, index.dev_list);
        algo = ualgo.get();
        algo->load(index_file);
        current_algo = std::move(ualgo);
      }
      search_param                   = ann::create_search_param<T>(index.algo, sp_json);
      search_param->metric_objective = metric_objective;
    } catch (const std::exception& e) {
      state.SkipWithError("Failed to create an algo: " + std::string(e.what()));
      return;
    }

    current_algo_props = std::make_unique<AlgoProperty>(
      std::move(parse_algo_property(algo->get_preference(), sp_json)));

    if (search_param->needs_dataset()) {
      try {
        algo->set_search_dataset(dataset->base_set(current_algo_props->dataset_memory_type),
                                 dataset->base_set_size());
      } catch (const std::exception& ex) {
        state.SkipWithError("The algorithm '" + index.name +
                            "' requires the base set, but it's not available. " +
                            "Exception: " + std::string(ex.what()));
        return;
      }
    }
    try {
      algo->set_search_param(*search_param);

    } catch (const std::exception& ex) {
      state.SkipWithError("An error occurred setting search parameters: " + std::string(ex.what()));
      return;
    }

    query_set = dataset->query_set(current_algo_props->query_memory_type);
    processed_threads.store(state.threads(), std::memory_order_acq_rel);
    cond_var.notify_all();
  } else {
    std::unique_lock lk(init_mutex);
    // All other threads will wait for the first thread to initialize the algo.
    cond_var.wait(lk, [&state] {
      return processed_threads.load(std::memory_order_acquire) == state.threads();
    });
    // gbench ensures that all threads are synchronized at the start of the benchmark loop.
    // We are accessing shared variables (like current_algo, current_algo_probs) before the
    // benchmark loop, therefore the synchronization here is necessary.
  }
  query_set = dataset->query_set(current_algo_props->query_memory_type);

  /**
   * Each thread will manage its own outputs
   */
  std::shared_ptr<buf<float>> distances =
    std::make_shared<buf<float>>(current_algo_props->query_memory_type, k * query_set_size);
  std::shared_ptr<buf<std::size_t>> neighbors =
    std::make_shared<buf<std::size_t>>(current_algo_props->query_memory_type, k * query_set_size);

  cuda_timer gpu_timer;
  {
    nvtx_case nvtx{state.name()};
    [[maybe_unused]] auto ntx_lap = nvtx.lap();
    [[maybe_unused]] auto gpu_lap = gpu_timer.lap();
    auto start                    = std::chrono::high_resolution_clock::now();

    auto algo = dynamic_cast<ANN<T>*>(current_algo.get())->copy();
    for (auto _ : state) {
      // run the search
      try {
        algo->search(query_set + batch_offset * dataset->dim(),
                     n_queries,
                     k,
                     neighbors->data + out_offset * k,
                     distances->data + out_offset * k,
                     gpu_timer.stream());
      } catch (const std::exception& e) {
        state.SkipWithError(std::string(e.what()));
      }

      // advance to the next batch
      batch_offset = (batch_offset + queries_stride) % query_set_size;
      out_offset   = (out_offset + n_queries) % query_set_size;

      queries_processed += n_queries;
    }
    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    if (state.thread_index() == 0) { state.counters.insert({{"end_to_end", duration}}); }
    state.counters.insert(
      {"Latency", {duration / double(state.iterations()), benchmark::Counter::kAvgThreads}});
  }

  state.SetItemsProcessed(queries_processed);
  if (cudart.found()) {
    double gpu_time_per_iteration = gpu_timer.total_time() / (double)state.iterations();
    state.counters.insert({"GPU", {gpu_time_per_iteration, benchmark::Counter::kAvgThreads}});
  }

  // This will be the total number of queries across all threads
  state.counters.insert({{"total_queries", queries_processed}});

  if (state.skipped()) { return; }

  // assume thread has finished processing successfully at this point
  // last thread to finish processing notifies all
  if (processed_threads-- == 0) { cond_var.notify_all(); }

  // Each thread calculates recall on their partition of queries.
  // evaluate recall
  if (dataset->max_k() >= k) {
    const std::int32_t* gt          = dataset->gt_set();
    const std::uint32_t max_k       = dataset->max_k();
    buf<std::size_t> neighbors_host = neighbors->move(MemoryType::Host);
    std::size_t rows                = std::min(queries_processed, query_set_size);
    std::size_t match_count         = 0;
    std::size_t total_count         = rows * static_cast<size_t>(k);

    // We go through the groundtruth with same stride as the benchmark loop.
    size_t out_offset   = 0;
    size_t batch_offset = (state.thread_index() * n_queries) % query_set_size;
    while (out_offset < rows) {
      for (std::size_t i = 0; i < n_queries; i++) {
        size_t i_orig_idx = batch_offset + i;
        size_t i_out_idx  = out_offset + i;
        if (i_out_idx < rows) {
          for (std::uint32_t j = 0; j < k; j++) {
            auto act_idx = std::int32_t(neighbors_host.data[i_out_idx * k + j]);
            for (std::uint32_t l = 0; l < k; l++) {
              auto exp_idx = gt[i_orig_idx * max_k + l];
              if (act_idx == exp_idx) {
                match_count++;
                break;
              }
            }
          }
        }
      }
      out_offset += n_queries;
      batch_offset = (batch_offset + queries_stride) % query_set_size;
    }
    double actual_recall = static_cast<double>(match_count) / static_cast<double>(total_count);
    state.counters.insert({"Recall", {actual_recall, benchmark::Counter::kAvgThreads}});
  }
}

inline void printf_usage()
{
  ::benchmark::PrintDefaultHelp();
  fprintf(stdout,
          "          [--build|--search] \n"
          "          [--force]\n"
          "          [--data_prefix=<prefix>]\n"
          "          [--index_prefix=<prefix>]\n"
          "          [--override_kv=<key:value1:value2:...:valueN>]\n"
          "          [--mode=<latency|throughput>\n"
          "          [--threads=min[:max]]\n"
          "          <conf>.json\n"
          "\n"
          "Note the non-standard benchmark parameters:\n"
          "  --build: build mode, will build index\n"
          "  --search: search mode, will search using the built index\n"
          "            one and only one of --build and --search should be specified\n"
          "  --force: force overwriting existing index files\n"
          "  --data_prefix=<prefix>:"
          " prepend <prefix> to dataset file paths specified in the <conf>.json (default = "
          "'data/').\n"
          "  --index_prefix=<prefix>:"
          " prepend <prefix> to index file paths specified in the <conf>.json (default = "
          "'index/').\n"
          "  --override_kv=<key:value1:value2:...:valueN>:"
          " override a build/search key one or more times multiplying the number of configurations;"
          " you can use this parameter multiple times to get the Cartesian product of benchmark"
          " configs.\n"
          "  --mode=<latency|throughput>"
          " run the benchmarks in latency (accumulate times spent in each batch) or "
          " throughput (pipeline batches and measure end-to-end) mode\n"
          "  --threads=min[:max] specify the number threads to use for throughput benchmark."
          " Power of 2 values between 'min' and 'max' will be used. If only 'min' is specified,"
          " then a single test is run with 'min' threads. By default min=1, max=<num hyper"
          " threads>.\n");
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
    b->MeasureProcessCPUTime();
    b->UseRealTime();
  }
}

template <typename T>
void register_search(std::shared_ptr<const Dataset<T>> dataset,
                     std::vector<Configuration::Index> indices,
                     Objective metric_objective,
                     const std::vector<int>& threads)
{
  for (auto index : indices) {
    for (std::size_t i = 0; i < index.search_params.size(); i++) {
      auto suf = static_cast<std::string>(index.search_params[i]["override_suffix"]);
      index.search_params[i].erase("override_suffix");

      auto* b = ::benchmark::RegisterBenchmark(
                  index.name + suf, bench_search<T>, index, i, dataset, metric_objective)
                  ->Unit(benchmark::kMillisecond)
                  /**
                   * The following are important for getting accuracy QPS measurements on both CPU
                   * and GPU These make sure that
                   *   - `end_to_end` ~ (`Time` * `Iterations`)
                   *   - `items_per_second` ~ (`total_queries` / `end_to_end`)
                   *   - Throughput = `items_per_second`
                   */
                  ->MeasureProcessCPUTime()
                  ->UseRealTime();

      if (metric_objective == Objective::THROUGHPUT) { b->ThreadRange(threads[0], threads[1]); }
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
                        kv_series override_kv,
                        Objective metric_objective,
                        const std::vector<int>& threads)
{
  if (cudart.found()) {
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
          "the 'groundtruth_neighbors_file' alongside the 'query_file' key to specify the "
          "path to "
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
    register_search<T>(dataset, indices, metric_objective, threads);
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
  std::string mode            = "latency";
  std::string threads_arg_txt = "";
  std::vector<int> threads    = {1, -1};  // min_thread, max_thread
  std::string log_level_str   = "";
  int raft_log_level          = raft::logger::get(RAFT_NAME).get_level();
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
    if (parse_bool_flag(argv[i], "--force", force_overwrite) ||
        parse_bool_flag(argv[i], "--build", build_mode) ||
        parse_bool_flag(argv[i], "--search", search_mode) ||
        parse_string_flag(argv[i], "--data_prefix", data_prefix) ||
        parse_string_flag(argv[i], "--index_prefix", index_prefix) ||
        parse_string_flag(argv[i], "--mode", mode) ||
        parse_string_flag(argv[i], "--override_kv", new_override_kv) ||
        parse_string_flag(argv[i], "--threads", threads_arg_txt) ||
        parse_string_flag(argv[i], "--raft_log_level", log_level_str)) {
      if (!log_level_str.empty()) {
        raft_log_level = std::stoi(log_level_str);
        log_level_str  = "";
      }
      if (!threads_arg_txt.empty()) {
        auto threads_arg = split(threads_arg_txt, ':');
        threads[0]       = std::stoi(threads_arg[0]);
        if (threads_arg.size() > 1) {
          threads[1] = std::stoi(threads_arg[1]);
        } else {
          threads[1] = threads[0];
        }
        threads_arg_txt = "";
      }
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

  raft::logger::get(RAFT_NAME).set_level(raft_log_level);

  Objective metric_objective = Objective::LATENCY;
  if (mode == "throughput") { metric_objective = Objective::THROUGHPUT; }

  int max_threads =
    (metric_objective == Objective::THROUGHPUT) ? std::thread::hardware_concurrency() : 1;
  if (threads[1] == -1) threads[1] = max_threads;

  if (metric_objective == Objective::LATENCY) {
    if (threads[0] != 1 || threads[1] != 1) {
      log_warn("Latency mode enabled. Overriding threads arg, running with single thread.");
      threads = {1, 1};
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

  if (cudart.needed() && !cudart.found()) {
    log_warn("cudart library is not found, GPU-based indices won't work.");
  }

  Configuration conf(conf_stream);
  std::string dtype = conf.get_dataset_conf().dtype;

  if (dtype == "float") {
    dispatch_benchmark<float>(conf,
                              force_overwrite,
                              build_mode,
                              search_mode,
                              data_prefix,
                              index_prefix,
                              override_kv,
                              metric_objective,
                              threads);
  } else if (dtype == "uint8") {
    dispatch_benchmark<std::uint8_t>(conf,
                                     force_overwrite,
                                     build_mode,
                                     search_mode,
                                     data_prefix,
                                     index_prefix,
                                     override_kv,
                                     metric_objective,
                                     threads);
  } else if (dtype == "int8") {
    dispatch_benchmark<std::int8_t>(conf,
                                    force_overwrite,
                                    build_mode,
                                    search_mode,
                                    data_prefix,
                                    index_prefix,
                                    override_kv,
                                    metric_objective,
                                    threads);
  } else {
    log_error("datatype '%s' is not supported", dtype.c_str());
    return -1;
  }

  ::benchmark::Initialize(&argc, argv, printf_usage);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return -1;
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  // Release a possibly cached ANN object, so that it cannot be alive longer than the handle
  // to a shared library it depends on (dynamic benchmark executable).
  current_algo.reset();
  return 0;
}
};  // namespace raft::bench::ann
