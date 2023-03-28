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
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#define JSON_DIAGNOSTICS 1
#include <nlohmann/json.hpp>

namespace raft::bench::ann {

class Configuration {
 public:
  struct Index {
    std::string name;
    std::string algo;
    nlohmann::json build_param;
    std::string file;
    std::vector<int> dev_list;

    int batch_size;
    int k;
    int run_count;
    std::vector<nlohmann::json> search_params;
    std::string search_result_file;
    float refine_ratio{0.0f};
  };

  struct DatasetConf {
    std::string name;
    std::string base_file;
    // use only a subset of base_file,
    // the range of rows is [subset_first_row, subset_first_row + subset_size)
    // however, subset_size = 0 means using all rows after subset_first_row
    // that is, the subset is [subset_first_row, #rows in base_file)
    size_t subset_first_row{0};
    size_t subset_size{0};
    std::string query_file;
    std::string distance;

    // data type of input dataset, possible values ["float", "int8", "uint8"]
    std::string dtype;
  };

  Configuration(std::istream& conf_stream);

  DatasetConf get_dataset_conf() const { return dataset_conf_; }
  std::vector<Index> get_indices(const std::string& patterns) const;

 private:
  void parse_dataset_(const nlohmann::json& conf);
  void parse_index_(const nlohmann::json& index_conf, const nlohmann::json& search_basic_conf);
  std::unordered_set<std::string> match_(const std::vector<std::string>& candidates,
                                         const std::string& patterns) const;

  DatasetConf dataset_conf_;
  std::vector<Index> indices_;
};

}  // namespace raft::bench::ann
