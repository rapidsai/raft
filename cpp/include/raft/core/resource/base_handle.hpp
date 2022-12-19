/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "resource.hpp"
#include <string>
#include <unordered_map>

namespace raft::core {

class base_handle_t {
  void add_resource_factory(std::string& key, std::shared_ptr<raft::resource_factory_t> factory)
  {
    if (!factories_.find(key) == factories_.end()) {
      factories_.insert(std::make_pair(key, factory));
    }
  }

  template <typename res_t>
  void* get_resource(std::string& key)
  {
    if (resources_.find(key) == resources_.end()) {
      resource_factory_t factory = factories_.at(key).get();
      resources_.insert(std::make_pair(key, factory->make_resource()))
    }
    reinterpret_cast<res_t>(resources_.at(key).get()->get_resource());
  }

 private:
  // TODO: std::string will be slow!
  std::unordered_map<std::string, std::shared_ptr<raft::resource_t>> resources_;
  std::unordered_map<std::string, std::shared_ptr<raft::resource_factory_t>> factories_;
};
}  // namespace raft::core