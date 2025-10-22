/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/util/seive.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace common {
TEST(Seive, Test)
{
  Seive s1(32);
  ASSERT_TRUE(s1.isPrime(17));
  ASSERT_FALSE(s1.isPrime(28));

  Seive s2(1024 * 1024);
  ASSERT_TRUE(s2.isPrime(107));
  ASSERT_FALSE(s2.isPrime(111));
  ASSERT_TRUE(s2.isPrime(6047));
}

}  // end namespace common
}  // end namespace raft
