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

#pragma once

#include <algorithm>
#include <cmath>
#include <type_traits>

#include <raft/core/detail/macros.hpp>

namespace raft {

/**
 * @defgroup Absolute value
 * @{
 */
template <typename T>
RAFT_INLINE_FUNCTION auto abs(T x)
  -> std::enable_if_t<std::is_same_v<float, T> || std::is_same_v<double, T> ||
                        std::is_same_v<int, T> || std::is_same_v<long int, T> ||
                        std::is_same_v<long long int, T>,
                      T>
{
#ifdef __CUDA_ARCH__
  return ::abs(x);
#else
  return std::abs(x);
#endif
}
template <typename T>
constexpr RAFT_INLINE_FUNCTION auto abs(T x)
  -> std::enable_if_t<!std::is_same_v<float, T> && !std::is_same_v<double, T> &&
                        !std::is_same_v<int, T> && !std::is_same_v<long int, T> &&
                        !std::is_same_v<long long int, T>,
                      T>
{
  return x < T{0} ? -x : x;
}
/** @} */

/**
 * Inverse cosine
 */
template <typename T>
RAFT_INLINE_FUNCTION auto acos(T x)
{
#ifdef __CUDA_ARCH__
  return ::acos(x);
#else
  return std::acos(x);
#endif
}

/**
 * Inverse sine
 */
template <typename T>
RAFT_INLINE_FUNCTION auto asin(T x)
{
#ifdef __CUDA_ARCH__
  return ::asin(x);
#else
  return std::asin(x);
#endif
}

/**
 * Inverse hyperbolic tangent
 */
template <typename T>
RAFT_INLINE_FUNCTION auto atanh(T x)
{
#ifdef __CUDA_ARCH__
  return ::atanh(x);
#else
  return std::atanh(x);
#endif
}

/**
 * Cosine
 */
template <typename T>
RAFT_INLINE_FUNCTION auto cos(T x)
{
#ifdef __CUDA_ARCH__
  return ::cos(x);
#else
  return std::cos(x);
#endif
}

/**
 * Exponential function
 */
template <typename T>
RAFT_INLINE_FUNCTION auto exp(T x)
{
#ifdef __CUDA_ARCH__
  return ::exp(x);
#else
  return std::exp(x);
#endif
}

/**
 * Natural logarithm
 */
template <typename T>
RAFT_INLINE_FUNCTION auto log(T x)
{
#ifdef __CUDA_ARCH__
  return ::log(x);
#else
  return std::log(x);
#endif
}

/**
 * @defgroup Maximum of two or more values.
 *
 * The CUDA Math API has overloads for all combinations of float/double. We provide similar
 * functionality while wrapping around std::max, which only supports arguments of the same type.
 * However, though the CUDA Math API supports combinations of unsigned and signed integers, this is
 * very error-prone so we do not support that and require the user to cast instead. (e.g the max of
 * -1 and 1u is 4294967295u...)
 *
 * When no overload matches, we provide a generic implementation but require that both types be the
 * same (and that the less-than operator be defined).
 * @{
 */
template <typename T1, typename T2>
RAFT_INLINE_FUNCTION auto max(const T1& x, const T2& y)
{
#ifdef __CUDA_ARCH__
  // Combinations of types supported by the CUDA Math API
  if constexpr ((std::is_integral_v<T1> && std::is_integral_v<T2> && std::is_same_v<T1, T2>) ||
                ((std::is_same_v<T1, float> || std::is_same_v<T1, double>)&&(
                  std::is_same_v<T2, float> || std::is_same_v<T2, double>))) {
    return ::max(x, y);
  }
  // Else, check that the types are the same and provide a generic implementation
  else {
    static_assert(
      std::is_same_v<T1, T2>,
      "No native max overload for these types. Both argument types must be the same to use "
      "the generic max. Please cast appropriately.");
    return (x < y) ? y : x;
  }
#else
  if constexpr (std::is_same_v<T1, float> && std::is_same_v<T2, double>) {
    return std::max(static_cast<double>(x), y);
  } else if constexpr (std::is_same_v<T1, double> && std::is_same_v<T2, float>) {
    return std::max(x, static_cast<double>(y));
  } else {
    static_assert(
      std::is_same_v<T1, T2>,
      "std::max requires that both argument types be the same. Please cast appropriately.");
    return std::max(x, y);
  }
#endif
}

/** Many-argument overload to avoid verbose nested calls or use with variadic arguments */
template <typename T1, typename T2, typename... Args>
RAFT_INLINE_FUNCTION auto max(const T1& x, const T2& y, Args&&... args)
{
  return raft::max(x, raft::max(y, std::forward<Args>(args)...));
}

/** One-argument overload for convenience when using with variadic arguments */
template <typename T>
constexpr RAFT_INLINE_FUNCTION auto max(const T& x)
{
  return x;
}
/** @} */

/**
 * @defgroup Minimum of two or more values.
 *
 * The CUDA Math API has overloads for all combinations of float/double. We provide similar
 * functionality while wrapping around std::min, which only supports arguments of the same type.
 * However, though the CUDA Math API supports combinations of unsigned and signed integers, this is
 * very error-prone so we do not support that and require the user to cast instead. (e.g the min of
 * -1 and 1u is 1u...)
 *
 * When no overload matches, we provide a generic implementation but require that both types be the
 * same (and that the less-than operator be defined).
 * @{
 */
template <typename T1, typename T2>
RAFT_INLINE_FUNCTION auto min(const T1& x, const T2& y)
{
#ifdef __CUDA_ARCH__
  // Combinations of types supported by the CUDA Math API
  if constexpr ((std::is_integral_v<T1> && std::is_integral_v<T2> && std::is_same_v<T1, T2>) ||
                ((std::is_same_v<T1, float> || std::is_same_v<T1, double>)&&(
                  std::is_same_v<T2, float> || std::is_same_v<T2, double>))) {
    return ::min(x, y);
  }
  // Else, check that the types are the same and provide a generic implementation
  else {
    static_assert(
      std::is_same_v<T1, T2>,
      "No native min overload for these types. Both argument types must be the same to use "
      "the generic min. Please cast appropriately.");
    return (y < x) ? y : x;
  }
#else
  if constexpr (std::is_same_v<T1, float> && std::is_same_v<T2, double>) {
    return std::min(static_cast<double>(x), y);
  } else if constexpr (std::is_same_v<T1, double> && std::is_same_v<T2, float>) {
    return std::min(x, static_cast<double>(y));
  } else {
    static_assert(
      std::is_same_v<T1, T2>,
      "std::min requires that both argument types be the same. Please cast appropriately.");
    return std::min(x, y);
  }
#endif
}

/** Many-argument overload to avoid verbose nested calls or use with variadic arguments */
template <typename T1, typename T2, typename... Args>
RAFT_INLINE_FUNCTION auto min(const T1& x, const T2& y, Args&&... args)
{
  return raft::min(x, raft::min(y, std::forward<Args>(args)...));
}

/** One-argument overload for convenience when using with variadic arguments */
template <typename T>
constexpr RAFT_INLINE_FUNCTION auto min(const T& x)
{
  return x;
}
/** @} */

/**
 * Power
 */
template <typename T1, typename T2>
RAFT_INLINE_FUNCTION auto pow(T1 x, T2 y)
{
#ifdef __CUDA_ARCH__
  return ::pow(x, y);
#else
  return std::pow(x, y);
#endif
}

/**
 * Sign
 */
template <typename T>
RAFT_INLINE_FUNCTION auto sgn(T val) -> int
{
  return (T(0) < val) - (val < T(0));
}

/**
 * Sine
 */
template <typename T>
RAFT_INLINE_FUNCTION auto sin(T x)
{
#ifdef __CUDA_ARCH__
  return ::sin(x);
#else
  return std::sin(x);
#endif
}

/**
 * Sine and cosine
 */
template <typename T>
RAFT_INLINE_FUNCTION std::enable_if_t<std::is_same_v<float, T> || std::is_same_v<double, T>> sincos(
  const T& x, T* s, T* c)
{
#ifdef __CUDA_ARCH__
  ::sincos(x, s, c);
#else
  *s = std::sin(x);
  *c = std::cos(x);
#endif
}

/**
 * Square root
 */
template <typename T>
RAFT_INLINE_FUNCTION auto sqrt(T x)
{
#ifdef __CUDA_ARCH__
  return ::sqrt(x);
#else
  return std::sqrt(x);
#endif
}

/**
 * Hyperbolic tangent
 */
template <typename T>
RAFT_INLINE_FUNCTION auto tanh(T x)
{
#ifdef __CUDA_ARCH__
  return ::tanh(x);
#else
  return std::tanh(x);
#endif
}

}  // namespace raft
