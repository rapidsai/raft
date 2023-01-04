/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <algorithm>
#include <mutex>
#include <raft/core/logger.hpp>
#include <string>
#include <vector>

namespace raft {

template <typename T>
using pair_res = std::pair<resource::resource_type, std::shared_ptr<T>>;

using pair_res_factory = pair_res<resource::resource_factory>;
using pair_resource    = pair_res<resource::resource>;

struct pair_first {
  template <typename T1, typename T2>
  bool operator()(T1& p1, T2& p2)
  {
    return p1.first < p2.first;
  }
};

/**
 * @brief Resource container which allows lazy-loading and registration
 * of resource_factory implementations, which in turn generate resource instances.
 *
 * This class is intended to be agnostic of the resources it contains and
 * does not, itself, differentiate between host and device resources. Downstream
 * accessor functions can then register and load resources as needed in order
 * to keep its usage somewhat opaque to end-users.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/resource/cuda_stream.hpp>
 * #include <raft/core/resource/cublas_handle.hpp>
 *
 * raft::resources res;
 * auto stream = raft::resource::get_cuda_stream(res);
 * auto cublas_handle = raft::resource::get_cublas_handle(res);
 * @endcode
 */
class resources {
 public:
  resources() {}

  resources(const resources&) = delete;
  resources& operator=(const resources&) = delete;
  resources(resources&&)                 = delete;
  resources& operator=(resources&&) = delete;

  /**
   * @brief Returns true if a resource_factory has been registered for the
   * given resource_type, false otherwise.
   * @param resource_type resource type to check
   * @return true if resource_factory is registered for the given resource_type
   */
  bool has_resource_factory(resource::resource_type resource_type) const
  {
    std::lock_guard<std::mutex> _(mutex_);
    auto it = std::lower_bound(
      factories_.begin(),
      factories_.end(),
      std::make_pair(resource_type, std::shared_ptr<resource::resource_factory>(nullptr)),
      pair_first());
    return (it != factories_.end() && (*it).first == resource_type);
  }

  /**
   * @brief Register a resource_factory with the current instance.
   * This will overwrite any existing resource factories.
   * @param factory resource factory to register on the current instance
   */
  void add_resource_factory(std::shared_ptr<resource::resource_factory> factory) const
  {
    std::lock_guard<std::mutex> _(mutex_);
    resource::resource_type rtype = factory.get()->get_resource_type();

    auto it = std::lower_bound(
      factories_.begin(), factories_.end(), std::make_pair(rtype, factory), pair_first());
    if (it != factories_.end() && (*it).first == rtype) {
      // Use hashmap semantics- replace values that already exist.
      (*it).second = factory;
    } else {
      factories_.insert(it, std::make_pair(factory.get()->get_resource_type(), factory));
    }
  }

  /**
   * @brief Retrieve a resource for the given resource_type and cast to given pointer type.
   * Note that the resources are loaded lazily on-demand and resources which don't yet
   * exist on the current instance will be created using the corresponding factory, if
   * it exists.
   * @tparam res_t pointer type for which retrieved resource will be casted
   * @param resource_type resource type to retrieve
   * @return the given resource, if it exists.
   */
  template <typename res_t>
  res_t* get_resource(resource::resource_type resource_type) const
  {
    std::lock_guard<std::mutex> _(mutex_);
    auto res_it =
      std::lower_bound(resources_.begin(),
                       resources_.end(),
                       std::make_pair(resource_type, std::shared_ptr<resource::resource>(nullptr)),
                       pair_first());
    if (res_it == resources_.end() || (*res_it).first != resource_type) {
      auto res_fac_it = std::lower_bound(
        factories_.begin(),
        factories_.end(),
        std::make_pair(resource_type, std::shared_ptr<resource::resource_factory>(nullptr)),
        pair_first());
      RAFT_EXPECTS(res_fac_it != factories_.end() && (*res_fac_it).first == resource_type,
                   "No resource factory has been registered for the given resource %d.",
                   resource_type);

      resource::resource_factory* factory = (*res_fac_it).second.get();
      auto created_res                    = factory->make_resource();

      resources_.insert(
        res_it, std::make_pair(resource_type, std::shared_ptr<resource::resource>(created_res)));

      return reinterpret_cast<res_t*>(created_res->get_resource());
    } else {
      // If resource already exists, we just return it.
      return reinterpret_cast<res_t*>((*res_it).second.get()->get_resource());
    }
  }

 private:
  mutable std::mutex mutex_;
  mutable std::vector<pair_res_factory> factories_;
  mutable std::vector<pair_resource> resources_;
};
}  // namespace raft