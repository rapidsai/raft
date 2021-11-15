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
 * @tparam Value a compile-time value representable as a power-of-two.
 */
template <auto Value>
struct Pow2 {
 public:
  typedef decltype(Value) Type;
  static constexpr Type Log2 = log2(Value);
  static constexpr Type Mask = Value - 1;

  static_assert(std::is_integral<Type>::value, "Value must be integral.");
  static_assert(Value && !(Value & Mask), "Value must be power of two.");

  /** Compute (x % Value). */
  static constexpr HDI Type mod(Type x) noexcept { return x & Mask; }
  /** Compute (x / Value). */
  static constexpr HDI Type div(Type x) noexcept { return x >> Log2; }

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
    return mod(reinterpret_cast<Type>(p)) == 0;
  }

  /** Tell whether two pointers have the same address modulo Value. */
  template <typename PtrT, typename PtrS>
  static constexpr HDI bool areSameAlignOffsets(PtrT a, PtrS b) noexcept {
    Pow2_CHECK_TYPE(PtrT);
    Pow2_CHECK_TYPE(PtrS);
    auto x = reinterpret_cast<Type>(a);
    auto y = reinterpret_cast<Type>(b);
    return mod(x) == mod(y);
  }

  /** Get this or next Value-aligned address (in bytes) or integral. */
  template <typename PtrT>
  static constexpr HDI PtrT roundUp(PtrT p) noexcept {
    Pow2_CHECK_TYPE(PtrT);
    auto x = reinterpret_cast<Type>(p);
    return reinterpret_cast<PtrT>(x + Mask - mod(x - 1));
  }

  /** Get this or previous Value-aligned address (in bytes) or integral. */
  template <typename PtrT>
  static constexpr HDI PtrT roundDown(PtrT p) noexcept {
    Pow2_CHECK_TYPE(PtrT);
    auto x = reinterpret_cast<Type>(p);
    return reinterpret_cast<PtrT>(x - mod(x));
  }
#undef Pow2_CHECK_TYPE
};

};  // namespace raft
