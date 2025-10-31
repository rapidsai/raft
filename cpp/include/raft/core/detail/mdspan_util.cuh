/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/thirdparty/mdspan/include/experimental/mdspan>

#include <tuple>
#include <utility>

namespace raft::detail {

template <class T, std::size_t N, std::size_t... Idx>
MDSPAN_INLINE_FUNCTION constexpr auto arr_to_tup(T (&arr)[N], std::index_sequence<Idx...>)
{
  return std::make_tuple(arr[Idx]...);
}

template <class T, std::size_t N>
MDSPAN_INLINE_FUNCTION constexpr auto arr_to_tup(T (&arr)[N])
{
  return arr_to_tup(arr, std::make_index_sequence<N>{});
}

template <typename T>
MDSPAN_INLINE_FUNCTION auto native_popc(T v) -> int32_t
{
  int c = 0;
  for (; v != 0; v &= v - 1) {
    c++;
  }
  return c;
}

MDSPAN_INLINE_FUNCTION auto popc(uint32_t v) -> int32_t
{
#if defined(__CUDA_ARCH__)
  return __popc(v);
#elif defined(__GNUC__) || defined(__clang__)
  return __builtin_popcount(v);
#else
  return native_popc(v);
#endif  // compiler
}

MDSPAN_INLINE_FUNCTION auto popc(uint64_t v) -> int32_t
{
#if defined(__CUDA_ARCH__)
  return __popcll(v);
#elif defined(__GNUC__) || defined(__clang__)
  return __builtin_popcountll(v);
#else
  return native_popc(v);
#endif  // compiler
}

}  // end namespace raft::detail
