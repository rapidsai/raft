/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <raft/util/cuda_utils.cuh>
#include <stdint.h>

namespace raft::util {

/**
 * @brief Perform fast integer division and modulo using a known divisor
 * From Hacker's Delight, Second Edition, Chapter 10
 *
 * @note This currently only supports 32b signed integers
 * @todo Extend support for signed divisors
 */
struct FastIntDiv {
  /**
   * @defgroup HostMethods Ctor's that are accessible only from host
   * @{
   * @brief Host-only ctor's
   * @param _d the divisor
   */
  FastIntDiv(int _d) : d(_d) { computeScalars(); }
  FastIntDiv& operator=(int _d)
  {
    d = _d;
    computeScalars();
    return *this;
  }
  /** @} */

  /**
   * @defgroup DeviceMethods Ctor's which even the device-side can access
   * @{
   * @brief host and device ctor's
   * @param other source object to be copied from
   */
  HDI FastIntDiv(const FastIntDiv& other) : d(other.d), m(other.m), p(other.p) {}
  HDI FastIntDiv& operator=(const FastIntDiv& other)
  {
    d = other.d;
    m = other.m;
    p = other.p;
    return *this;
  }
  /** @} */

  /** divisor */
  int d;
  /** the term 'm' as found in the reference chapter */
  unsigned m;
  /** the term 'p' as found in the reference chapter */
  int p;

 private:
  void computeScalars()
  {
    if (d == 1) {
      m = 0;
      p = 1;
      return;
    } else if (d < 0) {
      ASSERT(false, "FastIntDiv: division by negative numbers not supported!");
    } else if (d == 0) {
      ASSERT(false, "FastIntDiv: got division by zero!");
    }
    int64_t nc = ((1LL << 31) / d) * d - 1;
    p          = 31;
    int64_t twoP, rhs;
    do {
      ++p;
      twoP = 1LL << p;
      rhs  = nc * (d - twoP % d);
    } while (twoP <= rhs);
    m = (twoP + d - twoP % d) / d;
  }
};  // struct FastIntDiv

/**
 * @brief Division overload, so that FastIntDiv can be transparently switched
 *        to even on device
 * @param n numerator
 * @param divisor the denominator
 * @return the quotient
 */
HDI int operator/(int n, const FastIntDiv& divisor)
{
  if (divisor.d == 1) return n;
  int ret = (int64_t(divisor.m) * int64_t(n)) >> divisor.p;
  if (n < 0) ++ret;
  return ret;
}

/**
 * @brief Modulo overload, so that FastIntDiv can be transparently switched
 *        to even on device
 * @param n numerator
 * @param divisor the denominator
 * @return the remainder
 */
HDI int operator%(int n, const FastIntDiv& divisor)
{
  int quotient  = n / divisor;
  int remainder = n - quotient * divisor.d;
  return remainder;
}

/** Wide multiplication of two unsigned 64-bit integers */
HDI void wmul_64bit(uint64_t& res_hi, uint64_t& res_lo, uint64_t a, uint64_t b) {
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
  trial = res_lo + (t1 << 32);
  if (trial < res_lo) carry++;
  res_lo = trial;
  trial = res_lo + (t2 << 32);
  if (trial < res_lo) carry++;
  res_lo = trial;

  // No need to worry about carry in this addition
  res_hi = (t1 >> 32) + (t2 >> 32) + t3 + carry;
#endif

}
};  // namespace raft::util
