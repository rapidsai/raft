/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

#include <algorithm>
#include <memory>
#include <typeindex>

namespace raft::resource {

class custom_resource : public resource {
 public:
  custom_resource()                    = default;
  ~custom_resource() noexcept override = default;
  auto get_resource() -> void* override { return this; }

  template <typename ResourceT>
  auto load() -> ResourceT*
  {
    std::lock_guard<std::mutex> _(lock_);
    auto key = std::type_index{typeid(ResourceT)};
    auto pos = std::lower_bound(store_.begin(), store_.end(), kv{key, {nullptr}});
    if ((pos != store_.end()) && std::get<0>(*pos) == key) {
      return reinterpret_cast<ResourceT*>(std::get<1>(*pos).get());
    }
    auto store_ptr = new ResourceT{};
    store_.insert(pos, kv{key, std::shared_ptr<void>(store_ptr, [](void* ptr) {
                            delete reinterpret_cast<ResourceT*>(ptr);
                          })});
    return store_ptr;
  }

 private:
  using kv = std::tuple<std::type_index, std::shared_ptr<void>>;
  std::mutex lock_{};
  std::vector<kv> store_{};
};

/** Factory that knows how to construct a specific raft::resource to populate the res_t. */
class custom_resource_factory : public resource_factory {
 public:
  auto get_resource_type() -> resource_type override { return resource_type::CUSTOM; }
  auto make_resource() -> resource* override { return new custom_resource(); }
};

/**
 * @defgroup resource_custom custom resource functions
 * @{
 */

/**
 * Get the custom default-constructible resource if it exists, create it otherwise.
 *
 * Note: in contrast to the other, hard-coded resources, there's no information about the custom
 * resources at compile time. Hence, custom resources are kept in a hashmap and looked-up at
 * runtime. This leads to slightly slower access times.
 *
 * @tparam ResourceT the type of the resource; it must be complete and default-constructible.
 *
 * @param[in] res the raft resources object
 * @return a pointer to the custom resource.
 */
template <typename ResourceT>
auto get_custom_resource(resources const& res) -> ResourceT*
{
  static_assert(std::is_default_constructible_v<ResourceT>);
  if (!res.has_resource_factory(resource_type::CUSTOM)) {
    res.add_resource_factory(std::make_shared<custom_resource_factory>());
  }
  return res.get_resource<custom_resource>(resource_type::CUSTOM)->load<ResourceT>();
};

/**
 * @}
 */

}  // namespace raft::resource
