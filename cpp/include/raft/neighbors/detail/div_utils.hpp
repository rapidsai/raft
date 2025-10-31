/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
