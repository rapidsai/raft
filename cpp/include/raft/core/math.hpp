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

#if _RAFT_HAS_CUDA
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

namespace raft {

/**
 * @defgroup Absolute Absolute value
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
* @defgroup Trigonometry Trigonometry functions
* @{
*/
/** Inverse cosine */
template <typename T>
RAFT_INLINE_FUNCTION auto acos(T x)
{
#ifdef __CUDA_ARCH__
  return ::acos(x);
#else
  return std::acos(x);
#endif
}

/** Inverse sine */
template <typename T>
RAFT_INLINE_FUNCTION auto asin(T x)
{
#ifdef __CUDA_ARCH__
  return ::asin(x);
#else
  return std::asin(x);
#endif
}

/** Inverse hyperbolic tangent */
template <typename T>
RAFT_INLINE_FUNCTION auto atanh(T x)
{
#ifdef __CUDA_ARCH__
  return ::atanh(x);
#else
  return std::atanh(x);
#endif
}

/** Cosine */
template <typename T>
RAFT_INLINE_FUNCTION
#if _RAFT_HAS_CUDA
typename std::enable_if<!std::is_same<T, __half>::value && !std::is_same<T, nv_bfloat16>::value, T>::type
#else
auto
#endif
cos(T x)
{
#ifdef __CUDA_ARCH__
  return ::cos(x);
#else
  return std::cos(x);
#endif
}

#if _RAFT_HAS_CUDA
template <typename T>
RAFT_DEVICE_INLINE_FUNCTION
typename std::enable_if<std::is_same<T, __half>::value, __half>::type
cos(T x)
{
#if (__CUDA_ARCH__ >= 530)
  return ::hcos(x);
#else
  // static_assert(false) would be evaluated during host compilation stage while __CUDA_ARCH__ is at device compilation stage
  // Using this sizeof(T) != sizeof(T) makes it work as it's only triggered during template instantiation and thus at device compilation stage
  static_assert(sizeof(T) != sizeof(T), "__half is only supported on __CUDA_ARCH__ >= 530");
  return T{};
#endif
}

template <typename T>
RAFT_DEVICE_INLINE_FUNCTION
typename std::enable_if<std::is_same<T, nv_bfloat16>::value, nv_bfloat16>::type
cos(T x)
{
#if (__CUDA_ARCH__ >= 800)
  return ::hcos(x);
#else
  // static_assert(false) would be evaluated during host compilation stage while __CUDA_ARCH__ is at device compilation stage
  // Using this sizeof(T) != sizeof(T) makes it work as it's only triggered during template instantiation and thus at device compilation stage
  static_assert(sizeof(T) != sizeof(T), "nv_bfloat16 is only supported on __CUDA_ARCH__ >= 800");
  return T{};
#endif
}
#endif

/** Sine */
template <typename T>
RAFT_INLINE_FUNCTION
#if _RAFT_HAS_CUDA
typename std::enable_if<!std::is_same<T, __half>::value && !std::is_same<T, nv_bfloat16>::value, T>::type
#else
auto
#endif
sin(T x)
{
#ifdef __CUDA_ARCH__
  return ::sin(x);
#else
  return std::sin(x);
#endif
}

#if _RAFT_HAS_CUDA
template <typename T>
RAFT_DEVICE_INLINE_FUNCTION
typename std::enable_if<std::is_same<T, __half>::value, __half>::type
sin(T x)
{
#if (__CUDA_ARCH__ >= 530)
  return ::hsin(x);
#else
  // static_assert(false) would be evaluated during host compilation stage while __CUDA_ARCH__ is at device compilation stage
  // Using this sizeof(T) != sizeof(T) makes it work as it's only triggered during template instantiation and thus at device compilation stage
  static_assert(sizeof(T) != sizeof(T), "__half is only supported on __CUDA_ARCH__ >= 530");
  return T{};
#endif
}

template <typename T>
RAFT_DEVICE_INLINE_FUNCTION
typename std::enable_if<std::is_same<T, nv_bfloat16>::value, nv_bfloat16>::type
sin(T x)
{
#if (__CUDA_ARCH__ >= 800)
  return ::hsin(x);
#else
  // static_assert(false) would be evaluated during host compilation stage while __CUDA_ARCH__ is at device compilation stage
  // Using this sizeof(T) != sizeof(T) makes it work as it's only triggered during template instantiation and thus at device compilation stage
  static_assert(sizeof(T) != sizeof(T), "nv_bfloat16 is only supported on __CUDA_ARCH__ >= 800");
  return T{};
#endif
}
#endif

/** Sine and cosine */
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

/** Hyperbolic tangent */
template <typename T>
RAFT_INLINE_FUNCTION auto tanh(T x)
{
#ifdef __CUDA_ARCH__
return ::tanh(x);
#else
return std::tanh(x);
#endif
}
/** @} */

/**
* @defgroup Exponential Exponential and logarithm
* @{
*/
/** Exponential function */
template <typename T>
RAFT_INLINE_FUNCTION
#if _RAFT_HAS_CUDA
typename std::enable_if<!std::is_same<T, __half>::value && !std::is_same<T, nv_bfloat16>::value, T>::type
#else
auto
#endif
exp(T x)
{
#ifdef __CUDA_ARCH__
  return ::exp(x);
#else
  return std::exp(x);
#endif
}

#if _RAFT_HAS_CUDA
template <typename T>
RAFT_DEVICE_INLINE_FUNCTION
typename std::enable_if<std::is_same<T, __half>::value, __half>::type
exp(T x)
{
#if (__CUDA_ARCH__ >= 530)
  return ::hexp(x);
#else
  // static_assert(false) would be evaluated during host compilation stage while __CUDA_ARCH__ is at device compilation stage
  // Using this sizeof(T) != sizeof(T) makes it work as it's only triggered during template instantiation and thus at device compilation stage
  static_assert(sizeof(T) != sizeof(T), "__half is only supported on __CUDA_ARCH__ >= 530");
  return T{};
#endif
}

template <typename T>
RAFT_DEVICE_INLINE_FUNCTION
typename std::enable_if<std::is_same<T, nv_bfloat16>::value, nv_bfloat16>::type
exp(T x)
{
#if (__CUDA_ARCH__ >= 800)
  return ::hexp(x);
#else
  // static_assert(false) would be evaluated during host compilation stage while __CUDA_ARCH__ is at device compilation stage
  // Using this sizeof(T) != sizeof(T) makes it work as it's only triggered during template instantiation and thus at device compilation stage
  static_assert(sizeof(T) != sizeof(T), "nv_bfloat16 is only supported on __CUDA_ARCH__ >= 800");
  return T{};
#endif
}
#endif

/** Natural logarithm */
template <typename T>
RAFT_INLINE_FUNCTION
#if _RAFT_HAS_CUDA
typename std::enable_if<!std::is_same<T, __half>::value && !std::is_same<T, nv_bfloat16>::value, T>::type
#else
auto
#endif
log(T x)
{
#ifdef __CUDA_ARCH__
return ::log(x);
#else
return std::log(x);
#endif
}

#if _RAFT_HAS_CUDA
template <typename T>
RAFT_DEVICE_INLINE_FUNCTION
typename std::enable_if<std::is_same<T, __half>::value, __half>::type
log(T x)
{
#if (__CUDA_ARCH__ >= 530)
  return ::hlog(x);
#else
  // static_assert(false) would be evaluated during host compilation stage while __CUDA_ARCH__ is at device compilation stage
  // Using this sizeof(T) != sizeof(T) makes it work as it's only triggered during template instantiation and thus at device compilation stage
  static_assert(sizeof(T) != sizeof(T), "__half is only supported on __CUDA_ARCH__ >= 530");
  return T{};
#endif
}

template <typename T>
RAFT_DEVICE_INLINE_FUNCTION
typename std::enable_if<std::is_same<T, nv_bfloat16>::value, nv_bfloat16>::type
log(T x)
{
#if (__CUDA_ARCH__ >= 800)
  return ::hlog(x);
#else
  // static_assert(false) would be evaluated during host compilation stage while __CUDA_ARCH__ is at device compilation stage
  // Using this sizeof(T) != sizeof(T) makes it work as it's only triggered during template instantiation and thus at device compilation stage
  static_assert(sizeof(T) != sizeof(T), "nv_bfloat16 is only supported on __CUDA_ARCH__ >= 800");
  return T{};
#endif
}
#endif
/** @} */

/**
* @defgroup Maximum Maximum of two or more values.
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
template <
    typename T1, 
    typename T2,
    std::enable_if_t<
        CUDA_ONLY_CONDITION(
            RAFT_DEPAREN((
                (!std::is_same_v<T1, __half> && !std::is_same_v<T2, __half>) || 
                (!std::is_same_v<T1, nv_bfloat16> && !std::is_same_v<T2, nv_bfloat16>)
            ))
        ), 
        int
    > = 0
>
RAFT_INLINE_FUNCTION
auto
max(const T1& x, const T2& y)
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

#if _RAFT_HAS_CUDA
template <typename T>
RAFT_DEVICE_INLINE_FUNCTION
typename std::enable_if<std::is_same<T, __half>::value, __half>::type
max(T x, T y)
{
#if (__CUDA_ARCH__ >= 530)
  return ::__hmax(x, y);
#else
  // static_assert(false) would be evaluated during host compilation stage while __CUDA_ARCH__ is at device compilation stage
  // Using this sizeof(T) != sizeof(T) makes it work as it's only triggered during template instantiation and thus at device compilation stage
  static_assert(sizeof(T) != sizeof(T), "__half is only supported on __CUDA_ARCH__ >= 530");
  return T{};
#endif
}

template <typename T>
RAFT_DEVICE_INLINE_FUNCTION
typename std::enable_if<std::is_same<T, nv_bfloat16>::value, nv_bfloat16>::type
max(T x, T y)
{
#if (__CUDA_ARCH__ >= 800)
  return ::__hmax(x, y);
#else
  // static_assert(false) would be evaluated during host compilation stage while __CUDA_ARCH__ is at device compilation stage
  // Using this sizeof(T) != sizeof(T) makes it work as it's only triggered during template instantiation and thus at device compilation stage
  static_assert(sizeof(T) != sizeof(T), "nv_bfloat16 is only supported on __CUDA_ARCH__ >= 800");
  return T{};
#endif
}
#endif

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

#if _RAFT_HAS_CUDA
template <typename T>
RAFT_DEVICE_INLINE_FUNCTION
typename std::enable_if<std::is_same<T, __half>::value, __half>::type
max(T x)
{
#if (__CUDA_ARCH__ >= 530)
  return x;
#else
  // static_assert(false) would be evaluated during host compilation stage while __CUDA_ARCH__ is at device compilation stage
  // Using this sizeof(T) != sizeof(T) makes it work as it's only triggered during template instantiation and thus at device compilation stage
  static_assert(sizeof(T) != sizeof(T), "__half is only supported on __CUDA_ARCH__ >= 530");
  return T{};
#endif
}

template <typename T>
RAFT_DEVICE_INLINE_FUNCTION
typename std::enable_if<std::is_same<T, nv_bfloat16>::value, nv_bfloat16>::type
max(T x)
{
#if (__CUDA_ARCH__ >= 800)
  return x;
#else
  // static_assert(false) would be evaluated during host compilation stage while __CUDA_ARCH__ is at device compilation stage
  // Using this sizeof(T) != sizeof(T) makes it work as it's only triggered during template instantiation and thus at device compilation stage
  static_assert(sizeof(T) != sizeof(T), "nv_bfloat16 is only supported on __CUDA_ARCH__ >= 800");
  return T{};
#endif
}
#endif


/** @} */

/**
* @defgroup Minimum Minimum of two or more values.
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
template <
    typename T1, 
    typename T2,
    std::enable_if_t<
        CUDA_ONLY_CONDITION(
            RAFT_DEPAREN((
                (!std::is_same_v<T1, __half> && !std::is_same_v<T2, __half>) || 
                (!std::is_same_v<T1, nv_bfloat16> && !std::is_same_v<T2, nv_bfloat16>)
            ))
        ), 
        int
    > = 0
>
RAFT_INLINE_FUNCTION
auto
min(const T1& x, const T2& y)
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

#if _RAFT_HAS_CUDA
template <typename T>
RAFT_DEVICE_INLINE_FUNCTION
typename std::enable_if<std::is_same<T, __half>::value, __half>::type
min(T x, T y)
{
#if (__CUDA_ARCH__ >= 530)
  return ::__hmin(x, y);
#else
  // static_assert(false) would be evaluated during host compilation stage while __CUDA_ARCH__ is at device compilation stage
  // Using this sizeof(T) != sizeof(T) makes it work as it's only triggered during template instantiation and thus at device compilation stage
  static_assert(sizeof(T) != sizeof(T), "__half is only supported on __CUDA_ARCH__ >= 530");
  return T{};
#endif
}

template <typename T>
RAFT_DEVICE_INLINE_FUNCTION
typename std::enable_if<std::is_same<T, nv_bfloat16>::value, nv_bfloat16>::type
min(T x, T y)
{
#if (__CUDA_ARCH__ >= 800)
  return ::__hmin(x, y);
#else
  // static_assert(false) would be evaluated during host compilation stage while __CUDA_ARCH__ is at device compilation stage
  // Using this sizeof(T) != sizeof(T) makes it work as it's only triggered during template instantiation and thus at device compilation stage
  static_assert(sizeof(T) != sizeof(T), "nv_bfloat16 is only supported on __CUDA_ARCH__ >= 800");
  return T{};
#endif
}
#endif

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

#if _RAFT_HAS_CUDA
template <typename T>
RAFT_DEVICE_INLINE_FUNCTION
typename std::enable_if<std::is_same<T, __half>::value, __half>::type
min(T x)
{
#if (__CUDA_ARCH__ >= 530)
  return x;
#else
  // static_assert(false) would be evaluated during host compilation stage while __CUDA_ARCH__ is at device compilation stage
  // Using this sizeof(T) != sizeof(T) makes it work as it's only triggered during template instantiation and thus at device compilation stage
  static_assert(sizeof(T) != sizeof(T), "__half is only supported on __CUDA_ARCH__ >= 530");
  return T{};
#endif
}

template <typename T>
RAFT_DEVICE_INLINE_FUNCTION
typename std::enable_if<std::is_same<T, nv_bfloat16>::value, nv_bfloat16>::type
min(T x)
{
#if (__CUDA_ARCH__ >= 800)
  return x;
#else
  // static_assert(false) would be evaluated during host compilation stage while __CUDA_ARCH__ is at device compilation stage
  // Using this sizeof(T) != sizeof(T) makes it work as it's only triggered during template instantiation and thus at device compilation stage
  static_assert(sizeof(T) != sizeof(T), "nv_bfloat16 is only supported on __CUDA_ARCH__ >= 800");
  return T{};
#endif
}
#endif
/** @} */

/**
 * @defgroup Power Power and root functions
 * @{
 */
/** Power */
template <typename T1, typename T2>
RAFT_INLINE_FUNCTION auto pow(T1 x, T2 y)
{
#ifdef __CUDA_ARCH__
  return ::pow(x, y);
#else
  return std::pow(x, y);
#endif
}

/** Square root */
template <typename T>
RAFT_INLINE_FUNCTION
#if _RAFT_HAS_CUDA
typename std::enable_if<!std::is_same<T, __half>::value && !std::is_same<T, nv_bfloat16>::value, T>::type
#else
auto
#endif
sqrt(T x)
{
#ifdef __CUDA_ARCH__
return ::sqrt(x);
#else
return std::sqrt(x);
#endif
}

#if _RAFT_HAS_CUDA
template <typename T>
RAFT_DEVICE_INLINE_FUNCTION
typename std::enable_if<std::is_same<T, __half>::value, __half>::type
sqrt(T x)
{
#if (__CUDA_ARCH__ >= 530)
  return ::hsqrt(x);
#else
  // static_assert(false) would be evaluated during host compilation stage while __CUDA_ARCH__ is at device compilation stage
  // Using this sizeof(T) != sizeof(T) makes it work as it's only triggered during template instantiation and thus at device compilation stage
  static_assert(sizeof(T) != sizeof(T), "__half is only supported on __CUDA_ARCH__ >= 530");
  return T{};
#endif
}

template <typename T>
RAFT_DEVICE_INLINE_FUNCTION
typename std::enable_if<std::is_same<T, nv_bfloat16>::value, nv_bfloat16>::type
sqrt(T x)
{
#if (__CUDA_ARCH__ >= 800)
  return ::hsqrt(x);
#else
  // static_assert(false) would be evaluated during host compilation stage while __CUDA_ARCH__ is at device compilation stage
  // Using this sizeof(T) != sizeof(T) makes it work as it's only triggered during template instantiation and thus at device compilation stage
  static_assert(sizeof(T) != sizeof(T), "nv_bfloat16 is only supported on __CUDA_ARCH__ >= 800");
  return T{};
#endif
}
#endif
/** @} */

/** Sign */
template <typename T>
RAFT_INLINE_FUNCTION auto sgn(T val) -> int
{
return (T(0) < val) - (val < T(0));
}

}  // namespace raft
