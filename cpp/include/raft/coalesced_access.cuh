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

#include "cuda_utils.cuh"

namespace raft {

/**
 * @brief Check pointers for byte alignment.
 *
 * @tparam VecBytes size of the alignment in bytes.
 */
template <std::size_t VecBytes>
struct CoalescedAccess {
 private:
  static constexpr std::size_t VecMod = VecBytes - 1;

#define CoalescedAccess_CHECK_TYPE(T)                                    \
  static_assert(std::is_pointer<T>::value || std::is_integral<T>::value, \
                "Only pointer or integral types make sense here")

 public:
  static_assert((VecBytes & VecMod) == 0, "VecBytes must be power of two.");

  /** Number of elements fitting in a chunk of memory of size VecBytes. */
  template <typename T>
  static constexpr std::size_t nElems = VecBytes / sizeof(T);

  /** Tell whether the pointer is VecBytes-aligned. */
  template <typename PtrT>
  static constexpr HDI bool isAligned(PtrT p) noexcept {
    CoalescedAccess_CHECK_TYPE(PtrT);
    return (reinterpret_cast<std::size_t>(p) & VecMod) == 0;
  }

  /** Tell whether two pointers have the same address modulo VecBytes. */
  template <typename PtrT, typename PtrS>
  static constexpr HDI bool areSameAlignOffsets(PtrT a, PtrS b) noexcept {
    CoalescedAccess_CHECK_TYPE(PtrT);
    CoalescedAccess_CHECK_TYPE(PtrS);
    auto x = reinterpret_cast<std::size_t>(a);
    auto y = reinterpret_cast<std::size_t>(b);
    return (x & VecMod) == (y & VecMod);
  }

  /** Get this or next VecBytes-aligned address. */
  template <typename PtrT>
  static constexpr HDI PtrT roundUp(PtrT p) noexcept {
    CoalescedAccess_CHECK_TYPE(PtrT);
    auto x = reinterpret_cast<std::size_t>(p);
    return reinterpret_cast<PtrT>(x + VecMod - ((x - 1) & VecMod));
  }

  /** Get this or previous VecBytes-aligned address. */
  template <typename PtrT>
  static constexpr HDI PtrT roundDown(PtrT p) noexcept {
    CoalescedAccess_CHECK_TYPE(PtrT);
    auto x = reinterpret_cast<std::size_t>(p);
    return reinterpret_cast<PtrT>(x - (x & VecMod));
  }
#undef CoalescedAccess_CHECK_TYPE
};

};  // namespace raft
