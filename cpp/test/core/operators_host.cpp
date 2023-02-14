/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <algorithm>
#include <cmath>

#include <gtest/gtest.h>

#include "../test_utils.h"
#include <raft/core/kvp.hpp>
#include <raft/core/operators.hpp>

TEST(OperatorsHost, IdentityOp)
{
  raft::identity_op op;
  ASSERT_TRUE(raft::match(12.34f, op(12.34f, 0), raft::Compare<float>()));
}

TEST(OperatorsHost, CastOp)
{
  raft::cast_op<float> op;
  ASSERT_TRUE(raft::match(1234.0f, op(1234, 0), raft::CompareApprox<float>(0.00001f)));
}

TEST(OperatorsHost, KeyOp)
{
  raft::key_op op;
  raft::KeyValuePair<int, float> kvp(12, 3.4f);
  ASSERT_TRUE(raft::match(12, op(kvp, 0), raft::Compare<int>()));
}

TEST(OperatorsHost, ValueOp)
{
  raft::value_op op;
  raft::KeyValuePair<int, float> kvp(12, 3.4f);
  ASSERT_TRUE(raft::match(3.4f, op(kvp, 0), raft::CompareApprox<float>(0.00001f)));
}

TEST(OperatorsHost, SqrtOpF)
{
  raft::sqrt_op op;
  ASSERT_TRUE(raft::match(std::sqrt(12.34f), op(12.34f, 0), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(std::sqrt(12.34), op(12.34, 0), raft::CompareApprox<double>(0.000001)));
}

TEST(OperatorsHost, NZOp)
{
  raft::nz_op op;
  ASSERT_TRUE(raft::match(0.0f, op(0.0f, 0), raft::CompareApprox<float>(0.00001f)));
  ASSERT_TRUE(raft::match(1.0f, op(12.34f, 0), raft::CompareApprox<float>(0.00001f)));
}

TEST(OperatorsHost, AbsOp)
{
  raft::abs_op op;
  ASSERT_TRUE(raft::match(12.34f, op(-12.34f, 0), raft::CompareApprox<float>(0.00001f)));
  ASSERT_TRUE(raft::match(12.34, op(-12.34, 0), raft::CompareApprox<double>(0.000001)));
  ASSERT_TRUE(raft::match(1234, op(-1234, 0), raft::Compare<int>()));
}

TEST(OperatorsHost, SqOp)
{
  raft::sq_op op;
  ASSERT_TRUE(raft::match(152.2756f, op(12.34f, 0), raft::CompareApprox<float>(0.00001f)));
  ASSERT_TRUE(raft::match(289, op(-17, 0), raft::Compare<int>()));
}

TEST(OperatorsHost, AddOp)
{
  raft::add_op op;
  ASSERT_TRUE(raft::match(12.34f, op(12.0f, 0.34f), raft::CompareApprox<float>(0.00001f)));
  ASSERT_TRUE(raft::match(1234, op(1200, 34), raft::Compare<int>()));
}

TEST(OperatorsHost, SubOp)
{
  raft::sub_op op;
  ASSERT_TRUE(raft::match(12.34f, op(13.0f, 0.66f), raft::CompareApprox<float>(0.00001f)));
  ASSERT_TRUE(raft::match(1234, op(1300, 66), raft::Compare<int>()));
}

TEST(OperatorsHost, MulOp)
{
  raft::mul_op op;
  ASSERT_TRUE(raft::match(12.34f, op(2.0f, 6.17f), raft::CompareApprox<float>(0.00001f)));
}

TEST(OperatorsHost, DivOp)
{
  raft::div_op op;
  ASSERT_TRUE(raft::match(12.34f, op(37.02f, 3.0f), raft::CompareApprox<float>(0.00001f)));
}

TEST(OperatorsHost, DivCheckZeroOp)
{
  raft::div_checkzero_op op;
  ASSERT_TRUE(raft::match(12.34f, op(37.02f, 3.0f), raft::CompareApprox<float>(0.00001f)));
  ASSERT_TRUE(raft::match(0.0f, op(37.02f, 0.0f), raft::CompareApprox<float>(0.00001f)));
}

TEST(OperatorsHost, PowOp)
{
  raft::pow_op op;
  ASSERT_TRUE(raft::match(1000.0f, op(10.0f, 3.0f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(1000.0, op(10.0, 3.0), raft::CompareApprox<double>(0.000001)));
}

TEST(OperatorsHost, MinOp)
{
  raft::min_op op;
  ASSERT_TRUE(raft::match(3.0f, op(3.0f, 5.0f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(3.0, op(5.0, 3.0), raft::CompareApprox<double>(0.000001)));
  ASSERT_TRUE(raft::match(3, op(3, 5), raft::Compare<int>()));
}

TEST(OperatorsHost, MaxOp)
{
  raft::max_op op;
  ASSERT_TRUE(raft::match(5.0f, op(3.0f, 5.0f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(5.0, op(5.0, 3.0), raft::CompareApprox<double>(0.000001)));
  ASSERT_TRUE(raft::match(5, op(3, 5), raft::Compare<int>()));
}

TEST(OperatorsHost, SqDiffOp)
{
  raft::sqdiff_op op;
  ASSERT_TRUE(raft::match(4.0f, op(3.0f, 5.0f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(4.0, op(5.0, 3.0), raft::CompareApprox<double>(0.000001)));
}

TEST(OperatorsHost, ArgminOp)
{
  raft::argmin_op op;
  raft::KeyValuePair<int, float> kvp_a(0, 1.2f);
  raft::KeyValuePair<int, float> kvp_b(0, 3.4f);
  raft::KeyValuePair<int, float> kvp_c(1, 1.2f);
  ASSERT_TRUE(
    raft::match(kvp_a, op(kvp_a, kvp_b), raft::Compare<raft::KeyValuePair<int, float>>()));
  ASSERT_TRUE(
    raft::match(kvp_a, op(kvp_b, kvp_a), raft::Compare<raft::KeyValuePair<int, float>>()));
  ASSERT_TRUE(
    raft::match(kvp_a, op(kvp_a, kvp_c), raft::Compare<raft::KeyValuePair<int, float>>()));
  ASSERT_TRUE(
    raft::match(kvp_a, op(kvp_c, kvp_a), raft::Compare<raft::KeyValuePair<int, float>>()));
  ASSERT_TRUE(
    raft::match(kvp_c, op(kvp_b, kvp_c), raft::Compare<raft::KeyValuePair<int, float>>()));
  ASSERT_TRUE(
    raft::match(kvp_c, op(kvp_c, kvp_b), raft::Compare<raft::KeyValuePair<int, float>>()));
}

TEST(OperatorsHost, ArgmaxOp)
{
  raft::argmax_op op;
  raft::KeyValuePair<int, float> kvp_a(0, 1.2f);
  raft::KeyValuePair<int, float> kvp_b(0, 3.4f);
  raft::KeyValuePair<int, float> kvp_c(1, 1.2f);
  ASSERT_TRUE(
    raft::match(kvp_b, op(kvp_a, kvp_b), raft::Compare<raft::KeyValuePair<int, float>>()));
  ASSERT_TRUE(
    raft::match(kvp_b, op(kvp_b, kvp_a), raft::Compare<raft::KeyValuePair<int, float>>()));
  ASSERT_TRUE(
    raft::match(kvp_a, op(kvp_a, kvp_c), raft::Compare<raft::KeyValuePair<int, float>>()));
  ASSERT_TRUE(
    raft::match(kvp_a, op(kvp_c, kvp_a), raft::Compare<raft::KeyValuePair<int, float>>()));
  ASSERT_TRUE(
    raft::match(kvp_b, op(kvp_b, kvp_c), raft::Compare<raft::KeyValuePair<int, float>>()));
  ASSERT_TRUE(
    raft::match(kvp_b, op(kvp_c, kvp_b), raft::Compare<raft::KeyValuePair<int, float>>()));
}

TEST(OperatorsHost, ConstOp)
{
  raft::const_op op(12.34f);
  ASSERT_TRUE(raft::match(12.34f, op(), raft::Compare<float>()));
  ASSERT_TRUE(raft::match(12.34f, op(42), raft::Compare<float>()));
  ASSERT_TRUE(raft::match(12.34f, op(13, 37.0f), raft::Compare<float>()));
}

template <typename T>
struct trinary_add {
  const T c;
  constexpr explicit trinary_add(const T& c_) : c{c_} {}
  constexpr RAFT_INLINE_FUNCTION auto operator()(T a, T b) const { return a + b + c; }
};

TEST(OperatorsHost, PlugConstOp)
{
  // First, wrap around a default-constructible op
  {
    raft::plug_const_op<float, raft::add_op> op(0.34f);
    ASSERT_TRUE(raft::match(12.34f, op(12.0f), raft::CompareApprox<float>(0.0001f)));
  }

  // Second, wrap around a non-default-constructible op
  {
    auto op = raft::plug_const_op(10.0f, trinary_add<float>(2.0f));
    ASSERT_TRUE(raft::match(12.34f, op(0.34f), raft::CompareApprox<float>(0.0001f)));
  }
}

TEST(OperatorsHost, AddConstOp)
{
  raft::add_const_op<float> op(0.34f);
  ASSERT_TRUE(raft::match(12.34f, op(12.0f), raft::CompareApprox<float>(0.0001f)));
}

TEST(OperatorsHost, SubConstOp)
{
  raft::sub_const_op<float> op(0.66f);
  ASSERT_TRUE(raft::match(12.34f, op(13.0f), raft::CompareApprox<float>(0.0001f)));
}

TEST(OperatorsHost, MulConstOp)
{
  raft::mul_const_op<float> op(2.0f);
  ASSERT_TRUE(raft::match(12.34f, op(6.17f), raft::CompareApprox<float>(0.0001f)));
}

TEST(OperatorsHost, DivConstOp)
{
  raft::div_const_op<float> op(3.0f);
  ASSERT_TRUE(raft::match(12.34f, op(37.02f), raft::CompareApprox<float>(0.0001f)));
}

TEST(OperatorsHost, DivCheckZeroConstOp)
{
  // Non-zero denominator
  {
    raft::div_checkzero_const_op<float> op(3.0f);
    ASSERT_TRUE(raft::match(12.34f, op(37.02f), raft::CompareApprox<float>(0.0001f)));
  }
  // Zero denominator
  {
    raft::div_checkzero_const_op<float> op(0.0f);
    ASSERT_TRUE(raft::match(0.0f, op(37.02f), raft::CompareApprox<float>(0.0001f)));
  }
}

TEST(OperatorsHost, PowConstOp)
{
  raft::pow_const_op<float> op(3.0f);
  ASSERT_TRUE(raft::match(1000.0f, op(10.0f), raft::CompareApprox<float>(0.0001f)));
}

TEST(OperatorsHost, ComposeOp)
{
  // All ops are default-constructible
  {
    raft::compose_op<raft::sqrt_op, raft::abs_op, raft::cast_op<float>> op;
    ASSERT_TRUE(raft::match(std::sqrt(42.0f), op(-42, 0), raft::CompareApprox<float>(0.0001f)));
  }
  // Some ops are not default-constructible
  {
    auto op = raft::compose_op(
      raft::sqrt_op(), raft::abs_op(), raft::add_const_op<float>(8.0f), raft::cast_op<float>());
    ASSERT_TRUE(raft::match(std::sqrt(42.0f), op(-50, 0), raft::CompareApprox<float>(0.0001f)));
  }
}

TEST(OperatorsHost, MapArgsOp)
{
  // All ops are default-constructible
  {
    raft::map_args_op<raft::add_op, raft::sq_op, raft::abs_op> op;
    ASSERT_TRUE(raft::match(42.0f, op(5.0f, -17.0f), raft::CompareApprox<float>(0.0001f)));
  }
  // Some ops are not default-constructible
  {
    auto op = raft::map_args_op(
      raft::add_op(), raft::pow_const_op<float>(2.0f), raft::mul_const_op<float>(-1.0f));
    ASSERT_TRUE(raft::match(42.0f, op(5.0f, -17.0f), raft::CompareApprox<float>(0.0001f)));
  }
}
