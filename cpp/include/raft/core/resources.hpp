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
#pragma once

#include "resource/resource_types.hpp"
#include <mutex>
#include <raft/core/logger.hpp>
#include <string>
#include <unordered_map>

namespace raft {

class resources {
 public:
  resources() {}

  resources(const resources&) = delete;
  resources& operator=(const resources&) = delete;
  resources(resources&&)                 = delete;
  resources& operator=(resources&&) = delete;

  bool has_resource_factory(resource::resource_type resource_type) const
  {
    std::lock_guard<std::mutex> _(mutex_);
    return factories_.find(resource_type) != factories_.end();
  }

  /**
   * This will overwrite any existing resource factories.
   * @param factory
   */
  void add_resource_factory(std::shared_ptr<resource::resource_factory> factory) const
  {
    std::lock_guard<std::mutex> _(mutex_);
    factories_.insert(std::make_pair(factory.get()->get_resource_type(), factory));
  }

  template <typename res_t>
  res_t* get_resource(resource::resource_type resource_type) const
  {
    std::lock_guard<std::mutex> _(mutex_);
    if (resources_.find(resource_type) == resources_.end()) {
      resource::resource_factory* factory = factories_.at(resource_type).get();
      resources_.insert(std::make_pair(resource_type, factory->make_resource()));
    }
    return reinterpret_cast<res_t*>(resources_.at(resource_type).get()->get_resource());
  }

 private:
  mutable std::mutex mutex_;
  mutable std::unordered_map<resource::resource_type, std::shared_ptr<resource::resource_factory>>
    factories_;
  mutable std::unordered_map<resource::resource_type, std::shared_ptr<resource::resource>>
    resources_;
};
}  // namespace raft