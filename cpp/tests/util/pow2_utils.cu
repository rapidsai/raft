/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/util/pow2_utils.cuh>

#include <gtest/gtest.h>

namespace raft {

template <auto Val, typename TargetT>
struct Pow2Test : public ::testing::Test {
  typedef Pow2<Val> P;
  std::vector<TargetT> data;

  void SetUp() override
  {
    std::vector<TargetT> pos = {0, 1, 2, 7, 15, 16, 17, 31, 35, 1024, 1623};
    data.insert(data.end(), pos.begin(), pos.end());
    if constexpr (std::is_signed<TargetT>::value) {
      std::vector<TargetT> neg = {-0, -1, -2, -5, -15, -16, -17, -156};
      data.insert(data.end(), neg.begin(), neg.end());
    }
    data.push_back(std::numeric_limits<TargetT>::min());
    data.push_back(std::numeric_limits<TargetT>::max());
  }

  void quotRem()
  {
    for (auto x : data) {
      ASSERT_EQ(P::quot(x), x / P::Value) << "  where x = " << x;
      ASSERT_EQ(P::rem(x), x % P::Value) << "  where x = " << x;
      ASSERT_EQ(x, P::quot(x) * P::Value + P::rem(x));
    }
  }

  void divMod()
  {
    for (auto x : data) {
      ASSERT_GE(P::mod(x), 0) << "  where x = " << x;
      ASSERT_EQ(x, P::div(x) * P::Value + P::mod(x));
    }
  }

  void round()
  {
    for (auto x : data) {
      if (x <= std::numeric_limits<TargetT>::max() - TargetT(P::Value)) ASSERT_GE(P::roundUp(x), x);
      if (x >= std::numeric_limits<TargetT>::min() + TargetT(P::Value))
        ASSERT_LE(P::roundDown(x), x);
      ASSERT_EQ(x - P::roundDown(x), P::mod(x)) << "  where x = " << x;
      ASSERT_EQ(P::mod(P::roundUp(x) + P::mod(x) - x), 0) << "  where x = " << x;
    }
  }

  void alignment()
  {
    for (auto x : data) {
      ASSERT_TRUE(P::areSameAlignOffsets(x, x));
      if (x <= std::numeric_limits<TargetT>::max() - TargetT(P::Value)) {
        ASSERT_TRUE(P::areSameAlignOffsets(x, x + TargetT(P::Value)));
        int aligned_count      = 0;
        int same_aligned_count = 0;
        for (int i = 0; i < int(P::Value); i++) {
          aligned_count += P::isAligned(x + i);
          same_aligned_count += P::areSameAlignOffsets(x, x + i);
        }
        ASSERT_EQ(aligned_count, 1) << "  where x = " << x;
        ASSERT_EQ(same_aligned_count, 1) << "  where x = " << x;
      }
    }
  }
};

#define TEST_IT(T)                 \
  TEST_F(T, quotRem) { divMod(); } \
  TEST_F(T, divMod) { divMod(); }  \
  TEST_F(T, round) { round(); }    \
  TEST_F(T, alignment) { alignment(); }

typedef Pow2Test<16, int> Pow2_i32_i32_16;
typedef Pow2Test<1UL, uint64_t> Pow2_u64_u64_1;
typedef Pow2Test<128UL, int> Pow2_u64_i32_128;
typedef Pow2Test<32LL, uint16_t> Pow2_ll_u16_32;
typedef Pow2Test<16, uint64_t> Pow2_i32_u64_16;
TEST_IT(Pow2_i32_i32_16);
TEST_IT(Pow2_u64_u64_1);
TEST_IT(Pow2_u64_i32_128);
TEST_IT(Pow2_ll_u16_32);
TEST_IT(Pow2_i32_u64_16);

TEST(Pow2, pointers)
{
  typedef Pow2<32UL> P;
  for (ptrdiff_t i = 0; i <= ptrdiff_t(P::Value); i++) {
    auto* p = reinterpret_cast<float*>(16345 + i);
    ASSERT_GE(P::roundUp(p), p);
    ASSERT_LE(P::roundDown(p), p);
  }
}

}  // namespace raft
