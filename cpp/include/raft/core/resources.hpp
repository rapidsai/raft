/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <raft/core/error.hpp>  // RAFT_EXPECTS
#include <raft/core/logger.hpp>

#include <algorithm>
#include <mutex>
#include <string>
#include <vector>

namespace raft {

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
  template <typename T>
  using pair_res = std::pair<resource::resource_type, std::shared_ptr<T>>;

  using pair_res_factory = pair_res<resource::resource_factory>;
  using pair_resource    = pair_res<resource::resource>;

  resources()
    : factories_(resource::resource_type::LAST_KEY), resources_(resource::resource_type::LAST_KEY)
  {
    for (int i = 0; i < resource::resource_type::LAST_KEY; ++i) {
      factories_.at(i) = std::make_pair(resource::resource_type::LAST_KEY,
                                        std::make_shared<resource::empty_resource_factory>());
      resources_.at(i) = std::make_pair(resource::resource_type::LAST_KEY,
                                        std::make_shared<resource::empty_resource>());
    }
  }

  /**
   * @brief Shallow copy of underlying resources instance.
   * Note that this does not create any new resources.
   */
  resources(const resources& res) : factories_(res.factories_), resources_(res.resources_) {}
  resources(resources&&)            = delete;
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
    return factories_.at(resource_type).first != resource::resource_type::LAST_KEY;
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
    RAFT_EXPECTS(rtype != resource::resource_type::LAST_KEY,
                 "LAST_KEY is a placeholder and not a valid resource factory type.");
    factories_.at(rtype) = std::make_pair(rtype, factory);
    // Clear the corresponding resource, so that on next `get_resource` the new factory is used
    if (resources_.at(rtype).first != resource::resource_type::LAST_KEY) {
      resources_.at(rtype) = std::make_pair(resource::resource_type::LAST_KEY,
                                            std::make_shared<resource::empty_resource>());
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

    if (resources_.at(resource_type).first == resource::resource_type::LAST_KEY) {
      RAFT_EXPECTS(factories_.at(resource_type).first != resource::resource_type::LAST_KEY,
                   "No resource factory has been registered for the given resource %d.",
                   resource_type);
      resource::resource_factory* factory = factories_.at(resource_type).second.get();
      resources_.at(resource_type)        = std::make_pair(
        resource_type, std::shared_ptr<resource::resource>(factory->make_resource()));
    }

    resource::resource* res = resources_.at(resource_type).second.get();
    return reinterpret_cast<res_t*>(res->get_resource());
  }

 protected:
  mutable std::mutex mutex_;
  mutable std::vector<pair_res_factory> factories_;
  mutable std::vector<pair_resource> resources_;
};
}  // namespace raft
