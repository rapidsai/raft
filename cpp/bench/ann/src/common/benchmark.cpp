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
#include "cuda_stub.hpp"  // must go first

#include "ann_types.hpp"

#define JSON_DIAGNOSTICS 1
#include <nlohmann/json.hpp>

#include <memory>
#include <unordered_map>

#include <dlfcn.h>
#include <filesystem>

namespace raft::bench::ann {

struct lib_handle {
  void* handle{nullptr};
  explicit lib_handle(const std::string& name)
  {
    handle = dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (handle == nullptr) {
      auto error_msg = "Failed to load " + name;
      auto err       = dlerror();
      if (err != nullptr && err[0] != '\0') { error_msg += ": " + std::string(err); }
      throw std::runtime_error(error_msg);
    }
  }
  ~lib_handle() noexcept
  {
    if (handle != nullptr) { dlclose(handle); }
  }
};

auto load_lib(const std::string& algo) -> void*
{
  static std::unordered_map<std::string, lib_handle> libs{};
  auto found = libs.find(algo);

  if (found != libs.end()) { return found->second.handle; }
  auto lib_name = "lib" + algo + "_ann_bench.so";
  return libs.emplace(algo, lib_name).first->second.handle;
}

auto get_fun_name(void* addr) -> std::string
{
  Dl_info dl_info;
  if (dladdr(addr, &dl_info) != 0) {
    if (dl_info.dli_sname != nullptr && dl_info.dli_sname[0] != '\0') {
      return std::string{dl_info.dli_sname};
    }
  }
  throw std::logic_error("Failed to find out name of the looked up function");
}

template <typename T>
auto create_algo(const std::string& algo,
                 const std::string& distance,
                 int dim,
                 const nlohmann::json& conf,
                 const std::vector<int>& dev_list) -> std::unique_ptr<raft::bench::ann::ANN<T>>
{
  static auto fname = get_fun_name(reinterpret_cast<void*>(&create_algo<T>));
  auto handle       = load_lib(algo);
  auto fun_addr     = dlsym(handle, fname.c_str());
  if (fun_addr == nullptr) {
    throw std::runtime_error("Couldn't load the create_algo function (" + algo + ")");
  }
  auto fun = reinterpret_cast<decltype(&create_algo<T>)>(fun_addr);
  return fun(algo, distance, dim, conf, dev_list);
}

template <typename T>
std::unique_ptr<typename raft::bench::ann::ANN<T>::AnnSearchParam> create_search_param(
  const std::string& algo, const nlohmann::json& conf)
{
  static auto fname = get_fun_name(reinterpret_cast<void*>(&create_search_param<T>));
  auto handle       = load_lib(algo);
  auto fun_addr     = dlsym(handle, fname.c_str());
  if (fun_addr == nullptr) {
    throw std::runtime_error("Couldn't load the create_search_param function (" + algo + ")");
  }
  auto fun = reinterpret_cast<decltype(&create_search_param<T>)>(fun_addr);
  return fun(algo, conf);
}

};  // namespace raft::bench::ann

REGISTER_ALGO_INSTANCE(float);
REGISTER_ALGO_INSTANCE(std::int8_t);
REGISTER_ALGO_INSTANCE(std::uint8_t);

#include "benchmark.hpp"

int main(int argc, char** argv) { return raft::bench::ann::run_main(argc, argv); }
