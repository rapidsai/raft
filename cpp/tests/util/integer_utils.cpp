/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/util/integer_utils.hpp>

#include <gtest/gtest.h>

#include <iostream>

namespace raft {

static_assert(!is_narrowing_v<uint32_t, uint64_t>);
static_assert(!is_narrowing_v<uint32_t, int64_t>);
static_assert(!is_narrowing_v<uint32_t, uint32_t>);
static_assert(is_narrowing_v<uint32_t, int32_t>);
static_assert(is_narrowing_v<uint32_t, int>);
static_assert(!is_narrowing_v<float, double>);
static_assert(is_narrowing_v<double, float>);

TEST(Raft, rounding_up)
{
  ASSERT_EQ(raft::div_rounding_up_safe(5, 3), 2);
  ASSERT_EQ(raft::div_rounding_up_safe(0, 3), 0);
  ASSERT_EQ(raft::div_rounding_up_safe(7, 8), 1);
  ASSERT_EQ(raft::div_rounding_up_unsafe(5, 3), 2);
  ASSERT_EQ(raft::div_rounding_up_unsafe(0, 3), 0);
  ASSERT_EQ(raft::div_rounding_up_unsafe(7, 8), 1);
}

TEST(Raft, is_a_power_of_two)
{
  ASSERT_EQ(raft::is_a_power_of_two(1 << 5), true);
  ASSERT_EQ(raft::is_a_power_of_two((1 << 5) + 1), false);
}

}  // namespace raft
