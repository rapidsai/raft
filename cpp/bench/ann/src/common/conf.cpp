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
#include "conf.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "util.h"

namespace raft::bench::ann {
using std::runtime_error;
using std::string;
using std::unordered_set;
using std::vector;

Configuration::Configuration(std::istream& conf_stream)
{
  // to enable comments in json
  auto conf = nlohmann::json::parse(conf_stream, nullptr, true, true);

  parse_dataset_(conf.at("dataset"));
  parse_index_(conf.at("index"), conf.at("search_basic_param"));
}

vector<Configuration::Index> Configuration::get_indices(const string& patterns) const
{
  vector<string> names;
  for (const auto& index : indices_) {
    names.push_back(index.name);
  }

  auto matched = match_(names, patterns);
  if (matched.empty()) { throw runtime_error("no available index matches '" + patterns + "'"); }

  vector<Index> res;
  for (const auto& index : indices_) {
    if (matched.find(index.name) != matched.end()) { res.push_back(index); }
  }
  return res;
}

void Configuration::parse_dataset_(const nlohmann::json& conf)
{
  dataset_conf_.name       = conf.at("name");
  dataset_conf_.base_file  = conf.at("base_file");
  dataset_conf_.query_file = conf.at("query_file");
  dataset_conf_.distance   = conf.at("distance");

  if (conf.contains("subset_first_row")) {
    dataset_conf_.subset_first_row = conf.at("subset_first_row");
  }
  if (conf.contains("subset_size")) { dataset_conf_.subset_size = conf.at("subset_size"); }

  if (conf.contains("dtype")) {
    dataset_conf_.dtype = conf.at("dtype");
  } else {
    auto filename = dataset_conf_.base_file;
    if (!filename.compare(filename.size() - 4, 4, "fbin")) {
      dataset_conf_.dtype = "float";
    } else if (!filename.compare(filename.size() - 5, 5, "u8bin")) {
      dataset_conf_.dtype = "uint8";
    } else if (!filename.compare(filename.size() - 5, 5, "i8bin")) {
      dataset_conf_.dtype = "int8";
    } else {
      log_error("Could not determine data type of the dataset %s", filename.c_str());
    }
  }
}

void Configuration::parse_index_(const nlohmann::json& index_conf,
                                 const nlohmann::json& search_basic_conf)
{
  const int batch_size = search_basic_conf.at("batch_size");
  const int k          = search_basic_conf.at("k");
  const int run_count  = search_basic_conf.at("run_count");

  for (const auto& conf : index_conf) {
    Index index;
    index.name        = conf.at("name");
    index.algo        = conf.at("algo");
    index.build_param = conf.at("build_param");
    index.file        = conf.at("file");
    index.batch_size  = batch_size;
    index.k           = k;
    index.run_count   = run_count;

    if (conf.contains("multigpu")) {
      for (auto it : conf.at("multigpu")) {
        index.dev_list.push_back(it);
      }
      if (index.dev_list.empty()) { throw std::runtime_error("dev_list shouln't be empty!"); }
      index.dev_list.shrink_to_fit();
      index.build_param["multigpu"] = conf["multigpu"];
    }

    if (conf.contains("refine_ratio")) {
      float refine_ratio = conf.at("refine_ratio");
      if (refine_ratio <= 1.0f) {
        throw runtime_error("'" + index.name + "': refine_ratio should > 1.0");
      }
      index.refine_ratio = refine_ratio;
    }

    for (const auto& param : conf.at("search_params")) {
      index.search_params.push_back(param);
    }
    index.search_result_file = conf.at("search_result_file");

    indices_.push_back(index);
  }
}

unordered_set<string> Configuration::match_(const vector<string>& candidates,
                                            const string& patterns) const
{
  unordered_set<string> matched;
  for (const auto& pat : split(patterns, ',')) {
    if (pat.empty()) { continue; }

    if (pat.back() == '*') {
      auto len = pat.size() - 1;
      for (const auto& item : candidates) {
        if (item.compare(0, len, pat, 0, len) == 0) { matched.insert(item); }
      }
    } else {
      for (const auto& item : candidates) {
        if (item == pat) { matched.insert(item); }
      }
    }
  }

  return matched;
}

}  // namespace raft::bench::ann
