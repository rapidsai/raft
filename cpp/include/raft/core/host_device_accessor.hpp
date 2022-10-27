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

namespace raft {

/**
 * @brief A mixin to distinguish host and device memory. This is the primary
 * accessor used throught RAFT's APIs to denote whether an underlying pointer
 * is accessible from device, host, or both.
 */
template <typename AccessorPolicy, bool is_host, bool is_device>
struct host_device_accessor : public AccessorPolicy {
  using accessor_type   = AccessorPolicy;
  using is_host_type    = std::conditional_t<is_host, std::true_type, std::false_type>;
  using is_device_type  = std::conditional_t<is_device, std::true_type, std::false_type>;
  using is_managed_type = std::conditional_t<is_device && is_host, std::true_type, std::false_type>;
  static constexpr bool is_host_accessible    = is_host;
  static constexpr bool is_device_accessible  = is_device;
  static constexpr bool is_managed_accessible = is_device && is_host;
  // make sure the explicit ctor can fall through
  using AccessorPolicy::AccessorPolicy;
  using offset_policy = host_device_accessor;
  host_device_accessor(AccessorPolicy const& that) : AccessorPolicy{that} {}  // NOLINT
};

}  // namespace raft
