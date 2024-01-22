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

#include <raft/core/logger.hpp>
#include <raft/core/resource/device_memory_resource.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>

#include <mutex>
#include <set>
#include <string>

namespace raft::resource::detail {

/**
 * Warn a user of the calling algorithm if they use the default non-pooled memory allocator,
 * as it may hurt the performance.
 *
 * This helper function is designed to produce the warning once for a given `user_name`.
 *
 * @param[in] res
 * @param[in] user_name the name of the algorithm or any other identification.
 *
 */
inline void warn_non_pool_workspace(resources const& res, std::string user_name)
{
  // Detect if the plain cuda memory resource is used for the workspace
  if (rmm::mr::cuda_memory_resource{}.is_equal(*get_workspace_resource(res)->get_upstream())) {
    static std::set<std::string> notified_names{};
    static std::mutex mutex{};
    std::lock_guard<std::mutex> guard(mutex);
    auto [it, inserted] = notified_names.insert(std::move(user_name));
    if (inserted) {
      RAFT_LOG_WARN(
        "[%s] the default cuda resource is used for the raft workspace allocations. This may lead "
        "to a significant slowdown for this algorithm. Consider using the default pool resource "
        "(`raft::resource::set_workspace_to_pool_resource`) or set your own resource explicitly "
        "(`raft::resource::set_workspace_resource`).",
        it->c_str());
    }
  }
}

}  // namespace raft::resource::detail
