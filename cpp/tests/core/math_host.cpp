/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "../test_utils.h"

#include <raft/core/math.hpp>

#include <gtest/gtest.h>

TEST(MathHost, Abs)
{
  // Integer abs
  ASSERT_TRUE(raft::match(int8_t{123}, raft::abs(int8_t{-123}), raft::Compare<int8_t>()));
  ASSERT_TRUE(raft::match(12345, raft::abs(-12345), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(12345l, raft::abs(-12345l), raft::Compare<long int>()));
  ASSERT_TRUE(
    raft::match(123451234512345ll, raft::abs(-123451234512345ll), raft::Compare<long long int>()));
  // Floating-point abs
  ASSERT_TRUE(raft::match(12.34f, raft::abs(-12.34f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(12.34, raft::abs(-12.34), raft::CompareApprox<double>(0.000001)));
}

TEST(MathHost, Acos)
{
  ASSERT_TRUE(
    raft::match(std::acos(0.123f), raft::acos(0.123f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(
    raft::match(std::acos(0.123), raft::acos(0.123), raft::CompareApprox<double>(0.000001)));
}

TEST(MathHost, Asin)
{
  ASSERT_TRUE(
    raft::match(std::asin(0.123f), raft::asin(0.123f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(
    raft::match(std::asin(0.123), raft::asin(0.123), raft::CompareApprox<double>(0.000001)));
}

TEST(MathHost, Atanh)
{
  ASSERT_TRUE(
    raft::match(std::atanh(0.123f), raft::atanh(0.123f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(
    raft::match(std::atanh(0.123), raft::atanh(0.123), raft::CompareApprox<double>(0.000001)));
}

TEST(MathHost, Cos)
{
  ASSERT_TRUE(
    raft::match(std::cos(12.34f), raft::cos(12.34f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(
    raft::match(std::cos(12.34), raft::cos(12.34), raft::CompareApprox<double>(0.000001)));
}

TEST(MathHost, Exp)
{
  ASSERT_TRUE(
    raft::match(std::exp(12.34f), raft::exp(12.34f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(
    raft::match(std::exp(12.34), raft::exp(12.34), raft::CompareApprox<double>(0.000001)));
}

TEST(MathHost, Log)
{
  ASSERT_TRUE(
    raft::match(std::log(12.34f), raft::log(12.34f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(
    raft::match(std::log(12.34), raft::log(12.34), raft::CompareApprox<double>(0.000001)));
}

TEST(MathHost, Max2)
{
  ASSERT_TRUE(raft::match(1234, raft::max(-1234, 1234), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(1234u, raft::max(1234u, 123u), raft::Compare<unsigned int>()));
  ASSERT_TRUE(raft::match(1234ll, raft::max(-1234ll, 1234ll), raft::Compare<long long int>()));
  ASSERT_TRUE(
    raft::match(1234ull, raft::max(1234ull, 123ull), raft::Compare<unsigned long long int>()));

  ASSERT_TRUE(raft::match(12.34f, raft::max(-12.34f, 12.34f), raft::Compare<float>()));
  ASSERT_TRUE(raft::match(12.34, raft::max(-12.34, 12.34), raft::Compare<double>()));
  ASSERT_TRUE(raft::match(12.34, raft::max(-12.34f, 12.34), raft::CompareApprox<double>(0.000001)));
  ASSERT_TRUE(raft::match(12.34, raft::max(-12.34, 12.34f), raft::CompareApprox<double>(0.000001)));
}

TEST(MathHost, Max3)
{
  ASSERT_TRUE(raft::match(1234, raft::max(1234, 0, -1234), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(1234, raft::max(-1234, 1234, 0), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(1234, raft::max(0, -1234, 1234), raft::Compare<int>()));

  ASSERT_TRUE(
    raft::match(12.34, raft::max(12.34f, 0., -12.34), raft::CompareApprox<double>(0.000001)));
  ASSERT_TRUE(
    raft::match(12.34, raft::max(-12.34, 12.34f, 0.), raft::CompareApprox<double>(0.000001)));
  ASSERT_TRUE(
    raft::match(12.34, raft::max(0., -12.34, 12.34f), raft::CompareApprox<double>(0.000001)));
}

TEST(MathHost, Min2)
{
  ASSERT_TRUE(raft::match(-1234, raft::min(-1234, 1234), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(123u, raft::min(1234u, 123u), raft::Compare<unsigned int>()));
  ASSERT_TRUE(raft::match(-1234ll, raft::min(-1234ll, 1234ll), raft::Compare<long long int>()));
  ASSERT_TRUE(
    raft::match(123ull, raft::min(1234ull, 123ull), raft::Compare<unsigned long long int>()));

  ASSERT_TRUE(raft::match(-12.34f, raft::min(-12.34f, 12.34f), raft::Compare<float>()));
  ASSERT_TRUE(raft::match(-12.34, raft::min(-12.34, 12.34), raft::Compare<double>()));
  ASSERT_TRUE(
    raft::match(-12.34, raft::min(-12.34f, 12.34), raft::CompareApprox<double>(0.000001)));
  ASSERT_TRUE(
    raft::match(-12.34, raft::min(-12.34, 12.34f), raft::CompareApprox<double>(0.000001)));
}

TEST(MathHost, Min3)
{
  ASSERT_TRUE(raft::match(-1234, raft::min(1234, 0, -1234), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(-1234, raft::min(-1234, 1234, 0), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(-1234, raft::min(0, -1234, 1234), raft::Compare<int>()));

  ASSERT_TRUE(
    raft::match(-12.34, raft::min(12.34f, 0., -12.34), raft::CompareApprox<double>(0.000001)));
  ASSERT_TRUE(
    raft::match(-12.34, raft::min(-12.34, 12.34f, 0.), raft::CompareApprox<double>(0.000001)));
  ASSERT_TRUE(
    raft::match(-12.34, raft::min(0., -12.34, 12.34f), raft::CompareApprox<double>(0.000001)));
}

TEST(MathHost, Pow)
{
  ASSERT_TRUE(raft::match(
    std::pow(12.34f, 2.f), raft::pow(12.34f, 2.f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(
    raft::match(std::pow(12.34, 2.), raft::pow(12.34, 2.), raft::CompareApprox<double>(0.000001)));
}

TEST(MathHost, Sgn)
{
  ASSERT_TRUE(raft::match(-1, raft::sgn(-1234), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(0, raft::sgn(0), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(1, raft::sgn(1234), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(-1, raft::sgn(-12.34f), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(0, raft::sgn(0.f), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(1, raft::sgn(12.34f), raft::Compare<int>()));
}

TEST(MathHost, Sin)
{
  ASSERT_TRUE(
    raft::match(std::sin(12.34f), raft::sin(12.34f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(
    raft::match(std::sin(12.34), raft::sin(12.34), raft::CompareApprox<double>(0.000001)));
}

TEST(MathHost, SinCos)
{
  float xf = 12.34f;
  float sf, cf;
  raft::sincos(xf, &sf, &cf);
  ASSERT_TRUE(raft::match(std::sin(12.34f), sf, raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(std::cos(12.34f), cf, raft::CompareApprox<float>(0.0001f)));
  double xd = 12.34f;
  double sd, cd;
  raft::sincos(xd, &sd, &cd);
  ASSERT_TRUE(raft::match(std::sin(12.34), sd, raft::CompareApprox<double>(0.000001)));
  ASSERT_TRUE(raft::match(std::cos(12.34), cd, raft::CompareApprox<double>(0.000001)));
}

TEST(MathHost, Sqrt)
{
  ASSERT_TRUE(
    raft::match(std::sqrt(12.34f), raft::sqrt(12.34f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(
    raft::match(std::sqrt(12.34), raft::sqrt(12.34), raft::CompareApprox<double>(0.000001)));
}

TEST(MathHost, Tanh)
{
  ASSERT_TRUE(
    raft::match(std::tanh(12.34f), raft::tanh(12.34f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(
    raft::match(std::tanh(12.34), raft::tanh(12.34), raft::CompareApprox<double>(0.000001)));
}
