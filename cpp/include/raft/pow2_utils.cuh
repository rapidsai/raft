/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include "cuda_utils.cuh"

namespace raft {

/**
 * @brief Fast arithmetics and alignment checks for power-of-two values known at compile time.
 *
 * @tparam Value_ a compile-time value representable as a power-of-two.
 */
template <auto Value_>
struct Pow2 {
  typedef decltype(Value_) Type;
  static constexpr Type Value = Value_;
  static constexpr Type Log2 = log2(Value);
  static constexpr Type Mask = Value - 1;

  static_assert(std::is_integral<Type>::value, "Value must be integral.");
  static_assert(Value && !(Value & Mask), "Value must be power of two.");

#define Pow2_IsRepresentableAs(I) \
  (std::is_integral<I>::value && Type(I(Value)) == Value)

  /**
   * Integer division by Value truncated toward zero
   * (same as `x / Value` in C++).
   *
   *  Invariant: `x = Value * quot(x) + rem(x)`
   */
  template <typename I>
  static constexpr HDI std::enable_if_t<Pow2_IsRepresentableAs(I), I> quot(
    I x) noexcept {
    if constexpr (std::is_signed<I>::value)
      return (x >> I(Log2)) + (x < 0 && (x & I(Mask)));
    if constexpr (std::is_unsigned<I>::value) return x >> I(Log2);
  }

  /**
   *  Remainder of integer division by Value truncated toward zero
   *  (same as `x % Value` in C++).
   *
   *  Invariant: `x = Value * quot(x) + rem(x)`.
   */
  template <typename I>
  static constexpr HDI std::enable_if_t<Pow2_IsRepresentableAs(I), I> rem(
    I x) noexcept {
    if constexpr (std::is_signed<I>::value)
      return x < 0 ? -((-x) & I(Mask)) : (x & I(Mask));
    if constexpr (std::is_unsigned<I>::value) return x & I(Mask);
  }

  /**
   * Integer division by Value truncated toward negative infinity
   * (same as `x // Value` in Python).
   *
   * Invariant: `x = Value * div(x) + mod(x)`.
   *
   * Note, `div` and `mod` for negative values are slightly faster
   * than `quot` and `rem`, but behave slightly different
   * compared to normal C++ operators `/` and `%`.
   */
  template <typename I>
  static constexpr HDI std::enable_if_t<Pow2_IsRepresentableAs(I), I> div(
    I x) noexcept {
    return x >> I(Log2);
  }

  /**
   * x modulo Value operation (remainder of the `div(x)`)
   * (same as `x % Value` in Python).
   *
   * Invariant: `mod(x) >= 0`
   * Invariant: `x = Value * div(x) + mod(x)`.
   *
   * Note, `div` and `mod` for negative values are slightly faster
   * than `quot` and `rem`, but behave slightly different
   * compared to normal C++ operators `/` and `%`.
   */
  template <typename I>
  static constexpr HDI std::enable_if_t<Pow2_IsRepresentableAs(I), I> mod(
    I x) noexcept {
    return x & I(Mask);
  }

#define Pow2_CHECK_TYPE(T)                                               \
  static_assert(std::is_pointer<T>::value || std::is_integral<T>::value, \
                "Only pointer or integral types make sense here")

  /**
   * Tell whether the pointer or integral is Value-aligned.
   * NB: for pointers, the alignment is checked in bytes, not in elements.
   */
  template <typename PtrT>
  static constexpr HDI bool isAligned(PtrT p) noexcept {
    Pow2_CHECK_TYPE(PtrT);
    if constexpr (Pow2_IsRepresentableAs(PtrT)) return mod(p) == 0;
    if constexpr (!Pow2_IsRepresentableAs(PtrT))
      return mod(reinterpret_cast<Type>(p)) == 0;
  }

  /** Tell whether two pointers have the same address modulo Value. */
  template <typename PtrT, typename PtrS>
  static constexpr HDI bool areSameAlignOffsets(PtrT a, PtrS b) noexcept {
    Pow2_CHECK_TYPE(PtrT);
    Pow2_CHECK_TYPE(PtrS);
    Type x, y;
    if constexpr (Pow2_IsRepresentableAs(PtrT))
      x = Type(mod(a));
    else
      x = mod(reinterpret_cast<Type>(a));
    if constexpr (Pow2_IsRepresentableAs(PtrS))
      y = Type(mod(b));
    else
      y = mod(reinterpret_cast<Type>(b));
    return x == y;
  }

  /** Get this or next Value-aligned address (in bytes) or integral. */
  template <typename PtrT>
  static constexpr HDI PtrT roundUp(PtrT p) noexcept {
    Pow2_CHECK_TYPE(PtrT);
    if constexpr (Pow2_IsRepresentableAs(PtrT))
      return p + PtrT(Mask) - mod(p - PtrT(Mask));
    if constexpr (!Pow2_IsRepresentableAs(PtrT)) {
      auto x = reinterpret_cast<Type>(p);
      return reinterpret_cast<PtrT>(x + Mask - mod(x + Mask));
    }
  }

  /** Get this or previous Value-aligned address (in bytes) or integral. */
  template <typename PtrT>
  static constexpr HDI PtrT roundDown(PtrT p) noexcept {
    Pow2_CHECK_TYPE(PtrT);
    if constexpr (Pow2_IsRepresentableAs(PtrT)) return p - mod(p);
    if constexpr (!Pow2_IsRepresentableAs(PtrT)) {
      auto x = reinterpret_cast<Type>(p);
      return reinterpret_cast<PtrT>(x - mod(x));
    }
  }
#undef Pow2_CHECK_TYPE
#undef Pow2_IsRepresentableAs
};

};  // namespace raft
