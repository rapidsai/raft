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

#include <raft/core/detail/macros.hpp>
#include <raft/util/device_atomics.cuh>

namespace raft {

/**
 * @defgroup DeviceFunctors Commonly used device-only functors.
 * @{
 */

struct atomic_add_op {
  template <typename Type>
  _RAFT_DEVICE _RAFT_FORCEINLINE Type operator()(Type* address, const Type& val)
  {
    return atomicAdd(address, val);
  }
};

struct atomic_max_op {
  template <typename Type>
  _RAFT_DEVICE _RAFT_FORCEINLINE Type operator()(Type* address, const Type& val)
  {
    return atomicMax(address, val);
  }
};

struct atomic_min_op {
  template <typename Type>
  _RAFT_DEVICE _RAFT_FORCEINLINE Type operator()(Type* address, const Type& val)
  {
    return atomicMin(address, val);
  }
};
/** @} */

}  // namespace raft
