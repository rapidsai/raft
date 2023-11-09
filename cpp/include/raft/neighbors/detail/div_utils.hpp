/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#ifdef _RAFT_HAS_CUDA
#include <raft/util/pow2_utils.cuh>
#else
#include <raft/util/integer_utils.hpp>
#endif

/**
 * @brief A simple wrapper for raft::Pow2 which uses Pow2 utils only when available and regular
 * integer division otherwise. This is done to allow a common interface for division arithmetic for
 * non CUDA headers.
 *
 * @tparam Value_ a compile-time value representable as a power-of-two.
 */
namespace raft::neighbors::detail {
template <auto Value_>
struct div_utils {
  typedef decltype(Value_) Type;
  static constexpr Type Value = Value_;

  template <typename T>
  static constexpr _RAFT_HOST_DEVICE inline auto roundDown(T x)
  {
#if defined(_RAFT_HAS_CUDA)
    return Pow2<Value_>::roundDown(x);
#else
    return raft::round_down_safe(x, Value_);
#endif
  }

  template <typename T>
  static constexpr _RAFT_HOST_DEVICE inline auto mod(T x)
  {
#if defined(_RAFT_HAS_CUDA)
    return Pow2<Value_>::mod(x);
#else
    return x % Value_;
#endif
  }

  template <typename T>
  static constexpr _RAFT_HOST_DEVICE inline auto div(T x)
  {
#if defined(_RAFT_HAS_CUDA)
    return Pow2<Value_>::div(x);
#else
    return x / Value_;
#endif
  }
};
}  // namespace raft::neighbors::detail