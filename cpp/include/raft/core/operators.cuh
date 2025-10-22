/*
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
