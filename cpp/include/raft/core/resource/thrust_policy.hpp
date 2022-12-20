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

#include <raft/core/resource/resource_types.hpp>
#include <rmm/exec_policy.hpp>
namespace raft::core {
class thrust_policy_resource_t : public resource_t {
 public:
  thrust_policy_resource_t(rmm::cuda_stream_view stream_view)
    : thrust_policy_(std::make_unique<rmm::exec_policy>(stream_view))
  {
  }
  void* get_resource() override { return &(*thrust_policy_); }

  ~thrust_policy_resource_t() override {}

 private:
  std::unique_ptr<rmm::exec_policy> thrust_policy_{nullptr};
};

/**
 * Factory that knows how to construct a
 * specific raft::resource_t to populate
 * the handle_t.
 */
class thrust_policy_resource_factory_t : public resource_factory_t {
 public:
  thrust_policy_resource_factory_t(rmm::cuda_stream_view stream_view) : stream_view_(stream_view) {}
  resource_type_t resource_type() override { return resource_type_t::THRUST_POLICY; }
  resource_t* make_resource() override { return new thrust_policy_resource_t(stream_view_); }

 private:
  rmm::cuda_stream_view stream_view_;
};

/**
 * Load a device id from a handle (and populate it on the handle if needed).
 * @param handle raft handle object for managing resources
 * @return
 */
rmm::exec_policy& get_thrust_policy(base_handle_t const& handle)
{
  if (!handle.has_resource_factory(resource_type_t::THRUST_POLICY)) {
    rmm::cuda_stream_view stream = get_cuda_stream(handle);
    handle.add_resource_factory(std::make_shared<thrust_policy_resource_factory_t>(stream));
  }
  return *handle.get_resource<rmm::exec_policy>(resource_type_t::THRUST_POLICY);
};
}  // namespace raft::core