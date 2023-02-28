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
#ifdef NVTX
#include <nvToolsExt.h>
#endif
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

#include "benchmark_util.hpp"
#include "conf.h"
#include "dataset.h"
#include "util.h"

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::unordered_set;
using std::vector;
using namespace benchmark;
using raft::bench::ann::MemoryType;

namespace raft::bench::ann {

// supported types: float, half (very few implementations support it), uint8_t, int8_t
using data_t = float;

inline bool check_file_exist(const vector<string>& files)
{
  bool ret = true;
  unordered_set<string> processed;
  for (const auto& file : files) {
    if (processed.find(file) == processed.end() && !file_exists(file)) {
      log_error("file '%s' doesn't exist or is not a regular file", file.c_str());
      ret = false;
    }
    processed.insert(file);
  }
  return ret;
}

inline bool check_file_not_exist(const vector<string>& files, bool force_overwrite)
{
  bool ret = true;
  for (const auto& file : files) {
    if (file_exists(file)) {
      if (force_overwrite) {
        log_warn("'%s' already exists, will overwrite it", file.c_str());
      } else {
        log_error("'%s' already exists, use '-f' to force overwriting", file.c_str());
        ret = false;
      }
    }
  }
  return ret;
}

inline bool check_no_duplicate_file(const vector<string>& files)
{
  bool ret = true;
  unordered_set<string> processed;
  for (const auto& file : files) {
    if (processed.find(file) != processed.end()) {
      log_error("'%s' occurs more than once as output file, would be overwritten", file.c_str());
      ret = false;
    }
    processed.insert(file);
  }
  return ret;
}

inline bool mkdir(const vector<string>& dirs)
{
  unordered_set<string> processed;
  for (const auto& dir : dirs) {
    if (processed.find(dir) == processed.end() && !dir_exists(dir)) {
      if (create_dir(dir)) {
        log_info("mkdir '%s'", dir.c_str());
      } else {
        log_error("fail to create output directory '%s'", dir.c_str());
        // won't create any other dir when problem occurs
        return false;
      }
    }
    processed.insert(dir);
  }
  return true;
}

inline bool check(const vector<Configuration::Index>& indices,
                  bool build_mode,
                  bool force_overwrite)
{
  vector<string> files_should_exist;
  vector<string> dirs_should_exist;
  vector<string> output_files;
  for (const auto& index : indices) {
    if (build_mode) {
      output_files.push_back(index.file);
      output_files.push_back(index.file + ".txt");

      auto pos = index.file.rfind('/');
      if (pos != string::npos) { dirs_should_exist.push_back(index.file.substr(0, pos)); }
    } else {
      files_should_exist.push_back(index.file);
      files_should_exist.push_back(index.file + ".txt");

      output_files.push_back(index.search_result_file + ".0.ibin");
      output_files.push_back(index.search_result_file + ".0.txt");

      auto pos = index.search_result_file.rfind('/');
      if (pos != string::npos) {
        dirs_should_exist.push_back(index.search_result_file.substr(0, pos));
      }
    }
  }

  bool ret = true;
  if (!check_file_exist(files_should_exist)) { ret = false; }
  if (!check_file_not_exist(output_files, force_overwrite)) { ret = false; }
  if (!check_no_duplicate_file(output_files)) { ret = false; }
  if (ret && !mkdir(dirs_should_exist)) { ret = false; }
  return ret;
}

inline void write_build_info(const string& file_prefix,
                             const string& dataset,
                             const string& distance,
                             const string& name,
                             const string& algo,
                             const string& build_param,
                             float build_time)
{
  std::ofstream ofs(file_prefix + ".txt");
  if (!ofs) { throw std::runtime_error("can't open build info file: " + file_prefix + ".txt"); }
  ofs << "dataset: " << dataset << "\n"
      << "distance: " << distance << "\n"
      << "\n"
      << "name: " << name << "\n"
      << "algo: " << algo << "\n"
      << "build_param: " << build_param << "\n"
      << "build_time: " << build_time << endl;
  ofs.close();
  if (!ofs) { throw std::runtime_error("can't write to build info file: " + file_prefix + ".txt"); }
}

template <typename T>
void build(const benchmark::Dataset<T>* dataset, const vector<Configuration::Index>& indices)
{
  cudaStream_t stream;
  ANN_CUDA_CHECK(cudaStreamCreate(&stream));

  log_info(
    "base set from dataset '%s', #vector = %zu", dataset->name().c_str(), dataset->base_set_size());

  for (const auto& index : indices) {
    log_info("creating algo '%s', param=%s", index.algo.c_str(), index.build_param.dump().c_str());
    auto algo          = create_algo<T>(index.algo,
                               dataset->distance(),
                               dataset->dim(),
                               index.refine_ratio,
                               index.build_param,
                               index.dev_list);
    auto algo_property = algo->get_property();

    const T* base_set_ptr = nullptr;
    if (algo_property.dataset_memory_type == MemoryType::Host) {
      log_info("%s", "loading base set to memory");
      base_set_ptr = dataset->base_set();
    } else if (algo_property.dataset_memory_type == MemoryType::HostMmap) {
      log_info("%s", "mapping base set to memory");
      base_set_ptr = dataset->mapped_base_set();
    } else if (algo_property.dataset_memory_type == MemoryType::Device) {
      log_info("%s", "loading base set to GPU");
      base_set_ptr = dataset->base_set_on_gpu();
    }

    log_info("building index '%s'", index.name.c_str());
    ANN_CUDA_CHECK(cudaStreamSynchronize(stream));
#ifdef NVTX
    nvtxRangePush("build");
#endif
    Timer timer;
    algo->build(base_set_ptr, dataset->base_set_size(), stream);
    ANN_CUDA_CHECK(cudaStreamSynchronize(stream));
    float elapsed_ms = timer.elapsed_ms();
#ifdef NVTX
    nvtxRangePop();
#endif
    log_info("built index in %.2f seconds", elapsed_ms / 1000.0f);
    ANN_CUDA_CHECK_LAST_ERROR();

    algo->save(index.file);
    write_build_info(index.file,
                     dataset->name(),
                     dataset->distance(),
                     index.name,
                     index.algo,
                     index.build_param.dump(),
                     elapsed_ms / 1000.0f);
    log_info("saved index to %s", index.file.c_str());
  }

  ANN_CUDA_CHECK(cudaStreamDestroy(stream));
}

inline void write_search_result(const string& file_prefix,
                                const string& dataset,
                                const string& distance,
                                const string& name,
                                const string& algo,
                                const string& build_param,
                                const string& search_param,
                                int batch_size,
                                int run_count,
                                int k,
                                float search_time_average,
                                float search_time_p99,
                                float search_time_p999,
                                const int* neighbors,
                                size_t query_set_size)
{
  std::ofstream ofs(file_prefix + ".txt");
  if (!ofs) { throw std::runtime_error("can't open search result file: " + file_prefix + ".txt"); }
  ofs << "dataset: " << dataset << "\n"
      << "distance: " << distance << "\n"
      << "\n"
      << "name: " << name << "\n"
      << "algo: " << algo << "\n"
      << "build_param: " << build_param << "\n"
      << "search_param: " << search_param << "\n"
      << "\n"
      << "batch_size: " << batch_size << "\n"
      << "run_count: " << run_count << "\n"
      << "k: " << k << "\n"
      << "average_search_time: " << search_time_average << endl;
  if (search_time_p99 != std::numeric_limits<float>::max()) {
    ofs << "p99_search_time: " << search_time_p99 << endl;
  }
  if (search_time_p999 != std::numeric_limits<float>::max()) {
    ofs << "p999_search_time: " << search_time_p999 << endl;
  }
  ofs.close();
  if (!ofs) {
    throw std::runtime_error("can't write to search result file: " + file_prefix + ".txt");
  }

  BinFile<int> neighbors_file(file_prefix + ".ibin", "w");
  neighbors_file.write(neighbors, query_set_size, k);
}

template <typename T>
inline void search(const benchmark::Dataset<T>* dataset,
                   const vector<Configuration::Index>& indices)
{
  if (indices.empty()) { return; }
  cudaStream_t stream;
  ANN_CUDA_CHECK(cudaStreamCreate(&stream));

  log_info("loading query set from dataset '%s', #vector = %zu",
           dataset->name().c_str(),
           dataset->query_set_size());
  const T* query_set = dataset->query_set();
  // query set is usually much smaller than base set, so load it eagerly
  const T* d_query_set  = dataset->query_set_on_gpu();
  size_t query_set_size = dataset->query_set_size();

  // currently all indices has same batch_size, k and run_count
  const int batch_size = indices[0].batch_size;
  const int k          = indices[0].k;
  const int run_count  = indices[0].run_count;
  log_info(
    "basic search parameters: batch_size = %d, k = %d, run_count = %d", batch_size, k, run_count);
  if (query_set_size % batch_size != 0) {
    log_warn("query set size (%zu) % batch size (%d) != 0, the size of last batch is %zu",
             query_set_size,
             batch_size,
             query_set_size % batch_size);
  }
  const size_t num_batches = (query_set_size - 1) / batch_size + 1;
  size_t* neighbors        = new size_t[query_set_size * k];
  int* neighbors_buf       = new int[query_set_size * k];
  float* distances         = new float[query_set_size * k];
  vector<float> search_times;
  search_times.reserve(num_batches);
  size_t* d_neighbors;
  float* d_distances;
  ANN_CUDA_CHECK(cudaMalloc((void**)&d_neighbors, query_set_size * k * sizeof(*d_neighbors)));
  ANN_CUDA_CHECK(cudaMalloc((void**)&d_distances, query_set_size * k * sizeof(*d_distances)));

  for (const auto& index : indices) {
    log_info("creating algo '%s', param=%s", index.algo.c_str(), index.build_param.dump().c_str());
    auto algo          = create_algo<T>(index.algo,
                               dataset->distance(),
                               dataset->dim(),
                               index.refine_ratio,
                               index.build_param,
                               index.dev_list);
    auto algo_property = algo->get_property();

    log_info("loading index '%s' from file '%s'", index.name.c_str(), index.file.c_str());
    algo->load(index.file);

    const T* this_query_set = query_set;
    size_t* this_neighbors  = neighbors;
    float* this_distances   = distances;
    if (algo_property.query_memory_type == MemoryType::Device) {
      this_query_set = d_query_set;
      this_neighbors = d_neighbors;
      this_distances = d_distances;
    }

    if (algo_property.need_dataset_when_search) {
      log_info("loading base set from dataset '%s', #vector = %zu",
               dataset->name().c_str(),
               dataset->base_set_size());
      const T* base_set_ptr = nullptr;
      if (algo_property.dataset_memory_type == MemoryType::Host) {
        log_info("%s", "loading base set to memory");
        base_set_ptr = dataset->base_set();
      } else if (algo_property.dataset_memory_type == MemoryType::HostMmap) {
        log_info("%s", "mapping base set to memory");
        base_set_ptr = dataset->mapped_base_set();
      } else if (algo_property.dataset_memory_type == MemoryType::Device) {
        log_info("%s", "loading base set to GPU");
        base_set_ptr = dataset->base_set_on_gpu();
      }
      algo->set_search_dataset(base_set_ptr, dataset->base_set_size());
    }

    for (int i = 0, end_i = index.search_params.size(); i != end_i; ++i) {
      auto p_param = create_search_param<T>(index.algo, index.search_params[i]);
      algo->set_search_param(*p_param);
      log_info("search with param: %s", index.search_params[i].dump().c_str());

      if (algo_property.query_memory_type == MemoryType::Device) {
        ANN_CUDA_CHECK(cudaMemset(d_neighbors, 0, query_set_size * k * sizeof(*d_neighbors)));
        ANN_CUDA_CHECK(cudaMemset(d_distances, 0, query_set_size * k * sizeof(*d_distances)));
      } else {
        memset(neighbors, 0, query_set_size * k * sizeof(*neighbors));
        memset(distances, 0, query_set_size * k * sizeof(*distances));
      }

      float best_search_time_average = std::numeric_limits<float>::max();
      float best_search_time_p99     = std::numeric_limits<float>::max();
      float best_search_time_p999    = std::numeric_limits<float>::max();
      for (int run = 0; run < run_count; ++run) {
        log_info("run %d / %d", run + 1, run_count);
        for (size_t batch_id = 0; batch_id < num_batches; ++batch_id) {
          size_t row            = batch_id * batch_size;
          int actual_batch_size = (batch_id == num_batches - 1) ? query_set_size - row : batch_size;
          ANN_CUDA_CHECK(cudaStreamSynchronize(stream));
#ifdef NVTX
          string nvtx_label = "batch" + to_string(batch_id);
          if (run_count != 1) { nvtx_label = "run" + to_string(run) + "-" + nvtx_label; }
          if (batch_id == 10) {
            run = run_count - 1;
            break;
          }
#endif
          Timer timer;
#ifdef NVTX
          nvtxRangePush(nvtx_label.c_str());
#endif
          algo->search(this_query_set + row * dataset->dim(),
                       actual_batch_size,
                       k,
                       this_neighbors + row * k,
                       this_distances + row * k,
                       stream);
          ANN_CUDA_CHECK(cudaStreamSynchronize(stream));
          float elapsed_ms = timer.elapsed_ms();
#ifdef NVTX
          nvtxRangePop();
#endif
          // If the size of the last batch is less than batch_size, don't count it for
          // search time. But neighbors of the last batch will still be filled, so it's
          // counted for recall calculation.
          if (actual_batch_size == batch_size) {
            search_times.push_back(elapsed_ms / 1000.0f);  // in seconds
          }
        }

        float search_time_average =
          std::accumulate(search_times.cbegin(), search_times.cend(), 0.0f) / search_times.size();
        best_search_time_average = std::min(best_search_time_average, search_time_average);

        if (search_times.size() >= 100) {
          std::sort(search_times.begin(), search_times.end());

          auto calc_percentile_pos = [](float percentile, size_t N) {
            return static_cast<size_t>(std::ceil(percentile / 100.0 * N)) - 1;
          };

          float search_time_p99 = search_times[calc_percentile_pos(99, search_times.size())];
          best_search_time_p99  = std::min(best_search_time_p99, search_time_p99);

          if (search_times.size() >= 1000) {
            float search_time_p999 = search_times[calc_percentile_pos(99.9, search_times.size())];
            best_search_time_p999  = std::min(best_search_time_p999, search_time_p999);
          }
        }
        search_times.clear();
      }
      ANN_CUDA_CHECK_LAST_ERROR();

      if (algo_property.query_memory_type == MemoryType::Device) {
        ANN_CUDA_CHECK(cudaMemcpy(neighbors,
                                  d_neighbors,
                                  query_set_size * k * sizeof(*d_neighbors),
                                  cudaMemcpyDeviceToHost));
        ANN_CUDA_CHECK(cudaMemcpy(distances,
                                  d_distances,
                                  query_set_size * k * sizeof(*d_distances),
                                  cudaMemcpyDeviceToHost));
      }

      for (size_t j = 0; j < query_set_size * k; ++j) {
        neighbors_buf[j] = neighbors[j];
      }
      write_search_result(index.search_result_file + "." + to_string(i),
                          dataset->name(),
                          dataset->distance(),
                          index.name,
                          index.algo,
                          index.build_param.dump(),
                          index.search_params[i].dump(),
                          batch_size,
                          index.run_count,
                          k,
                          best_search_time_average,
                          best_search_time_p99,
                          best_search_time_p999,
                          neighbors_buf,
                          query_set_size);
    }

    log_info("finish searching for index '%s'", index.name.c_str());
  }

  delete[] neighbors;
  delete[] neighbors_buf;
  delete[] distances;
  ANN_CUDA_CHECK(cudaFree(d_neighbors));
  ANN_CUDA_CHECK(cudaFree(d_distances));
  ANN_CUDA_CHECK(cudaStreamDestroy(stream));
}

inline const string usage(const string& argv0)
{
  return "usage: " + argv0 + " -b|s [-c] [-f] [-i index_names] conf.json\n" +
         "   -b: build mode, will build index\n" +
         "   -s: search mode, will search using built index\n" +
         "       one and only one of -b and -s should be specified\n" +
         "   -c: just check command line options and conf.json are sensible\n" +
         "       won't build or search\n" + "   -f: force overwriting existing output files\n" +
         "   -i: by default will build/search all the indices found in conf.json\n" +
         "       '-i' can be used to select a subset of indices\n" +
         "       'index_names' is a list of comma-separated index names\n" +
         "       '*' is allowed as the last character of a name to select all matched indices\n" +
         "       for example, -i \"hnsw1,hnsw2,faiss\" or -i \"hnsw*,faiss\"";
}

inline int run_main(int argc, char** argv)
{
  bool force_overwrite = false;
  bool build_mode      = false;
  bool search_mode     = false;
  bool only_check      = false;
  string index_patterns("*");

  int opt;
  while ((opt = getopt(argc, argv, "bscfi:h")) != -1) {
    switch (opt) {
      case 'b': build_mode = true; break;
      case 's': search_mode = true; break;
      case 'c': only_check = true; break;
      case 'f': force_overwrite = true; break;
      case 'i': index_patterns = optarg; break;
      case 'h': cout << usage(argv[0]) << endl; return -1;
      default: cerr << "\n" << usage(argv[0]) << endl; return -1;
    }
  }
  if (build_mode == search_mode) {
    cerr << "one and only one of -b and -s should be specified\n\n" << usage(argv[0]) << endl;
    return -1;
  }
  if (argc - optind != 1) {
    cerr << usage(argv[0]) << endl;
    return -1;
  }
  string conf_file = argv[optind];

  std::ifstream conf_stream(conf_file.c_str());
  if (!conf_stream) {
    log_error("can't open configuration file: %s", argv[optind]);
    return -1;
  }

  try {
    Configuration conf(conf_stream);

    auto dataset_conf = conf.get_dataset_conf();
    BinDataset<data_t> dataset(dataset_conf.name,
                               dataset_conf.base_file,
                               dataset_conf.subset_first_row,
                               dataset_conf.subset_size,
                               dataset_conf.query_file,
                               dataset_conf.distance);

    vector<Configuration::Index> indices = conf.get_indices(index_patterns);
    if (!check(indices, build_mode, force_overwrite)) { return -1; }

    string message = "will ";
    message += build_mode ? "build:" : "search:";
    for (const auto& index : indices) {
      message += "\n  " + index.name;
    }
    log_info("%s", message.c_str());

    if (only_check) {
      log_info("%s", "all check passed, quit due to option -c");
      return 0;
    }

    if (build_mode) {
      build(&dataset, indices);
    } else if (search_mode) {
      search(&dataset, indices);
    }
  } catch (const std::exception& e) {
    log_error("exception occurs: %s", e.what());
    return -1;
  }
}
}  // namespace raft::bench::ann
