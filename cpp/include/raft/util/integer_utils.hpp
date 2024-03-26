/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Eyal Rozenberg <eyalroz@blazingdb.com>
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

/**
 * Utility code involving integer arithmetic
 *
 */

#include <raft/core/detail/macros.hpp>

#include <limits>
#include <stdexcept>
#include <type_traits>

namespace raft {
//! Utility functions
/**
 * Finds the smallest integer not less than `number_to_round` and modulo `S` is
 * zero. This function assumes that `number_to_round` is non-negative and
 * `modulus` is positive.
 */
template <typename S>
constexpr inline S round_up_safe(S number_to_round, S modulus)
{
  auto remainder = number_to_round % modulus;
  if (remainder == 0) { return number_to_round; }
  auto rounded_up = number_to_round - remainder + modulus;
  if (rounded_up < number_to_round) {
    throw std::invalid_argument("Attempt to round up beyond the type's maximum value");
  }
  return rounded_up;
}

/**
 * Finds the largest integer not greater than `number_to_round` and modulo `S` is
 * zero. This function assumes that `number_to_round` is non-negative and
 * `modulus` is positive.
 */
template <typename S>
inline S round_down_safe(S number_to_round, S modulus)
{
  auto remainder    = number_to_round % modulus;
  auto rounded_down = number_to_round - remainder;
  return rounded_down;
}

/**
 * Divides the left-hand-side by the right-hand-side, rounding up
 * to an integral multiple of the right-hand-side, e.g. (9,5) -> 2 , (10,5) -> 2, (11,5) -> 3.
 *
 * @param dividend the number to divide
 * @param divisor the number by which to divide
 * @return The least integer multiple of divisor which is greater than or equal to
 * the non-integral division dividend/divisor.
 *
 * @note sensitive to overflow, i.e. if dividend > std::numeric_limits<S>::max() - divisor,
 * the result will be incorrect
 */
template <typename S, typename T>
constexpr inline S div_rounding_up_unsafe(const S& dividend, const T& divisor) noexcept
{
  return (dividend + divisor - 1) / divisor;
}

namespace detail {
template <typename I>
constexpr inline I div_rounding_up_safe(std::integral_constant<bool, false>,
                                        I dividend,
                                        I divisor) noexcept
{
  // TODO: This could probably be implemented faster
  return (dividend > divisor) ? 1 + div_rounding_up_unsafe(dividend - divisor, divisor)
                              : (dividend > 0);
}

template <typename I>
constexpr inline I div_rounding_up_safe(std::integral_constant<bool, true>,
                                        I dividend,
                                        I divisor) noexcept
{
  auto quotient  = dividend / divisor;
  auto remainder = dividend % divisor;
  return quotient + (remainder != 0);
}

}  // namespace detail

/**
 * Divides the left-hand-side by the right-hand-side, rounding up
 * to an integral multiple of the right-hand-side, e.g. (9,5) -> 2 , (10,5) -> 2, (11,5) -> 3.
 *
 * @param dividend the number to divide
 * @param divisor the number of by which to divide
 * @return The least integer multiple of divisor which is greater than or equal to
 * the non-integral division dividend/divisor.
 *
 * @note will not overflow, and may _or may not_ be slower than the intuitive
 * approach of using (dividend + divisor - 1) / divisor
 */
template <typename I>
constexpr inline auto div_rounding_up_safe(I dividend, I divisor) noexcept
  -> std::enable_if_t<std::is_integral<I>::value, I>
{
  using i_is_a_signed_type = std::integral_constant<bool, std::is_signed<I>::value>;
  return detail::div_rounding_up_safe(i_is_a_signed_type{}, dividend, divisor);
}

template <typename I>
constexpr inline auto is_a_power_of_two(I val) noexcept
  -> std::enable_if_t<std::is_integral<I>::value, bool>
{
  return (val != 0) && (((val - 1) & val) == 0);
}

/**
 * Given an integer `x`, return such `y` that `x <= y` and `is_a_power_of_two(y)`.
 * If such `y` does not exist in `T`, return zero.
 */
template <typename T>
constexpr inline auto bound_by_power_of_two(T x) noexcept
  -> std::enable_if_t<std::is_integral<T>::value, T>
{
  if (is_a_power_of_two(x)) { return x; }
  constexpr T kMaxUnsafe = std::numeric_limits<T>::max();
  constexpr T kMaxSafe   = is_a_power_of_two(kMaxUnsafe) ? kMaxUnsafe : (kMaxUnsafe >> 1);
  const T limited        = std::min(x, kMaxSafe);
  T bound                = T{1};
  while (bound < limited) {
    bound <<= 1;
  }
  return bound < x ? T{0} : bound;
}

/**
 * @brief Return the absolute value of a number.
 *
 * This calls `std::abs()` which performs equivalent: `(value < 0) ? -value : value`.
 *
 * This was created to prevent compile errors calling `std::abs()` with unsigned integers.
 * An example compile error appears as follows:
 * @code{.pseudo}
 * error: more than one instance of overloaded function "std::abs" matches the argument list:
 *          function "abs(int)"
 *          function "std::abs(long)"
 *          function "std::abs(long long)"
 *          function "std::abs(double)"
 *          function "std::abs(float)"
 *          function "std::abs(long double)"
 *          argument types are: (uint64_t)
 * @endcode
 *
 * Not all cases could be if-ed out using std::is_signed<T>::value and satisfy the compiler.
 *
 * @param val Numeric value can be either integer or float type.
 * @return Absolute value if value type is signed.
 */
template <typename T>
constexpr inline auto absolute_value(T val) -> std::enable_if_t<std::is_signed<T>::value, T>
{
  return std::abs(val);
}
// Unsigned type just returns itself.
template <typename T>
constexpr inline auto absolute_value(T val) -> std::enable_if_t<!std::is_signed<T>::value, T>
{
  return val;
}

/**
 * @defgroup Check whether the numeric conversion is narrowing
 *
 * @tparam From source type
 * @tparam To destination type
 * @{
 */
template <typename From, typename To, typename = void>
struct is_narrowing : std::true_type {};

template <typename From, typename To>
struct is_narrowing<From, To, std::void_t<decltype(To{std::declval<From>()})>> : std::false_type {};
/** @} */

/** Check whether the numeric conversion is narrowing */
template <typename From, typename To>
inline constexpr bool is_narrowing_v = is_narrowing<From, To>::value;  // NOLINT

/** Wide multiplication of two unsigned 64-bit integers */
_RAFT_HOST_DEVICE inline void wmul_64bit(uint64_t& res_hi, uint64_t& res_lo, uint64_t a, uint64_t b)
{
#ifdef __CUDA_ARCH__
  asm("mul.hi.u64 %0, %1, %2;" : "=l"(res_hi) : "l"(a), "l"(b));
  asm("mul.lo.u64 %0, %1, %2;" : "=l"(res_lo) : "l"(a), "l"(b));
#else
  uint32_t a_hi, a_lo, b_hi, b_lo;

  a_hi = uint32_t(a >> 32);
  a_lo = uint32_t(a & uint64_t(0x00000000FFFFFFFF));
  b_hi = uint32_t(b >> 32);
  b_lo = uint32_t(b & uint64_t(0x00000000FFFFFFFF));

  uint64_t t0 = uint64_t(a_lo) * uint64_t(b_lo);
  uint64_t t1 = uint64_t(a_hi) * uint64_t(b_lo);
  uint64_t t2 = uint64_t(a_lo) * uint64_t(b_hi);
  uint64_t t3 = uint64_t(a_hi) * uint64_t(b_hi);

  uint64_t carry = 0, trial = 0;

  res_lo = t0;
  trial  = res_lo + (t1 << 32);
  if (trial < res_lo) carry++;
  res_lo = trial;
  trial  = res_lo + (t2 << 32);
  if (trial < res_lo) carry++;
  res_lo = trial;

  // No need to worry about carry in this addition
  res_hi = (t1 >> 32) + (t2 >> 32) + t3 + carry;
#endif
}

}  // namespace raft
