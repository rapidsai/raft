/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>

#include "../test_utils.h"
#include <raft/core/math.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_scalar.hpp>

template <typename OutT, typename OpT, typename... Args>
__global__ void math_eval_kernel(OutT* out, OpT op, Args... args)
{
  out[0] = op(std::forward<Args>(args)...);
}

template <typename OpT, typename... Args>
auto math_eval(OpT op, Args&&... args)
{
  typedef decltype(op(args...)) OutT;
  auto stream = rmm::cuda_stream_default;
  rmm::device_scalar<OutT> result(stream);
  math_eval_kernel<<<1, 1, 0, stream>>>(result.data(), op, std::forward<Args>(args)...);
  return result.value(stream);
}

struct abs_test_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& in) const
  {
    return raft::abs(in);
  }
};

TEST(MathDevice, Abs)
{
  // Integer abs
  ASSERT_TRUE(
    raft::match(int8_t{123}, math_eval(abs_test_op{}, int8_t{-123}), raft::Compare<int8_t>()));
  ASSERT_TRUE(raft::match(12345, math_eval(abs_test_op{}, -12345), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(12345l, math_eval(abs_test_op{}, -12345l), raft::Compare<long int>()));
  ASSERT_TRUE(raft::match(123451234512345ll,
                          math_eval(abs_test_op{}, -123451234512345ll),
                          raft::Compare<long long int>()));
  // Floating-point abs
  ASSERT_TRUE(
    raft::match(12.34f, math_eval(abs_test_op{}, -12.34f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(
    raft::match(12.34, math_eval(abs_test_op{}, -12.34), raft::CompareApprox<double>(0.000001)));
}

struct acos_test_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& in) const
  {
    return raft::acos(in);
  }
};

TEST(MathDevice, Acos)
{
  ASSERT_TRUE(raft::match(
    std::acos(0.123f), math_eval(acos_test_op{}, 0.123f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(
    std::acos(0.123), math_eval(acos_test_op{}, 0.123), raft::CompareApprox<double>(0.000001)));
}

struct asin_test_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& in) const
  {
    return raft::asin(in);
  }
};

TEST(MathDevice, Asin)
{
  ASSERT_TRUE(raft::match(
    std::asin(0.123f), math_eval(asin_test_op{}, 0.123f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(
    std::asin(0.123), math_eval(asin_test_op{}, 0.123), raft::CompareApprox<double>(0.000001)));
}

struct atanh_test_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& in) const
  {
    return raft::atanh(in);
  }
};

TEST(MathDevice, Atanh)
{
  ASSERT_TRUE(raft::match(
    std::atanh(0.123f), math_eval(atanh_test_op{}, 0.123f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(
    std::atanh(0.123), math_eval(atanh_test_op{}, 0.123), raft::CompareApprox<double>(0.000001)));
}

struct cos_test_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& in) const
  {
    return raft::cos(in);
  }
};

TEST(MathDevice, Cos)
{
  ASSERT_TRUE(raft::match(
    std::cos(12.34f), math_eval(cos_test_op{}, 12.34f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(
    std::cos(12.34), math_eval(cos_test_op{}, 12.34), raft::CompareApprox<double>(0.000001)));
}

struct exp_test_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& in) const
  {
    return raft::exp(in);
  }
};

TEST(MathDevice, Exp)
{
  ASSERT_TRUE(raft::match(
    std::exp(12.34f), math_eval(exp_test_op{}, 12.34f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(
    std::exp(12.34), math_eval(exp_test_op{}, 12.34), raft::CompareApprox<double>(0.000001)));
}

struct log_test_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& in) const
  {
    return raft::log(in);
  }
};

TEST(MathDevice, Log)
{
  ASSERT_TRUE(raft::match(
    std::log(12.34f), math_eval(log_test_op{}, 12.34f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(
    std::log(12.34), math_eval(log_test_op{}, 12.34), raft::CompareApprox<double>(0.000001)));
}

struct max_test_op {
  template <typename... Args>
  constexpr RAFT_INLINE_FUNCTION auto operator()(Args&&... args) const
  {
    return raft::max(std::forward<Args>(args)...);
  }
};

TEST(MathDevice, Max2)
{
  ASSERT_TRUE(raft::match(1234, math_eval(max_test_op{}, -1234, 1234), raft::Compare<int>()));
  ASSERT_TRUE(
    raft::match(1234u, math_eval(max_test_op{}, 1234u, 123u), raft::Compare<unsigned int>()));
  ASSERT_TRUE(
    raft::match(1234ll, math_eval(max_test_op{}, -1234ll, 1234ll), raft::Compare<long long int>()));
  ASSERT_TRUE(raft::match(
    1234ull, math_eval(max_test_op{}, 1234ull, 123ull), raft::Compare<unsigned long long int>()));

  ASSERT_TRUE(
    raft::match(12.34f, math_eval(max_test_op{}, -12.34f, 12.34f), raft::Compare<float>()));
  ASSERT_TRUE(raft::match(12.34, math_eval(max_test_op{}, -12.34, 12.34), raft::Compare<double>()));
  ASSERT_TRUE(raft::match(
    12.34, math_eval(max_test_op{}, -12.34f, 12.34), raft::CompareApprox<double>(0.000001)));
  ASSERT_TRUE(raft::match(
    12.34, math_eval(max_test_op{}, -12.34, 12.34f), raft::CompareApprox<double>(0.000001)));
}

TEST(MathDevice, Max3)
{
  ASSERT_TRUE(raft::match(1234, math_eval(max_test_op{}, 1234, 0, -1234), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(1234, math_eval(max_test_op{}, -1234, 1234, 0), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(1234, math_eval(max_test_op{}, 0, -1234, 1234), raft::Compare<int>()));

  ASSERT_TRUE(raft::match(
    12.34, math_eval(max_test_op{}, 12.34f, 0., -12.34), raft::CompareApprox<double>(0.000001)));
  ASSERT_TRUE(raft::match(
    12.34, math_eval(max_test_op{}, -12.34, 12.34f, 0.), raft::CompareApprox<double>(0.000001)));
  ASSERT_TRUE(raft::match(
    12.34, math_eval(max_test_op{}, 0., -12.34, 12.34f), raft::CompareApprox<double>(0.000001)));
}

struct min_test_op {
  template <typename... Args>
  constexpr RAFT_INLINE_FUNCTION auto operator()(Args&&... args) const
  {
    return raft::min(std::forward<Args>(args)...);
  }
};

TEST(MathDevice, Min2)
{
  ASSERT_TRUE(raft::match(-1234, math_eval(min_test_op{}, -1234, 1234), raft::Compare<int>()));
  ASSERT_TRUE(
    raft::match(123u, math_eval(min_test_op{}, 1234u, 123u), raft::Compare<unsigned int>()));
  ASSERT_TRUE(raft::match(
    -1234ll, math_eval(min_test_op{}, -1234ll, 1234ll), raft::Compare<long long int>()));
  ASSERT_TRUE(raft::match(
    123ull, math_eval(min_test_op{}, 1234ull, 123ull), raft::Compare<unsigned long long int>()));

  ASSERT_TRUE(
    raft::match(-12.34f, math_eval(min_test_op{}, -12.34f, 12.34f), raft::Compare<float>()));
  ASSERT_TRUE(
    raft::match(-12.34, math_eval(min_test_op{}, -12.34, 12.34), raft::Compare<double>()));
  ASSERT_TRUE(raft::match(
    -12.34, math_eval(min_test_op{}, -12.34f, 12.34), raft::CompareApprox<double>(0.000001)));
  ASSERT_TRUE(raft::match(
    -12.34, math_eval(min_test_op{}, -12.34, 12.34f), raft::CompareApprox<double>(0.000001)));
}

TEST(MathDevice, Min3)
{
  ASSERT_TRUE(raft::match(-1234, math_eval(min_test_op{}, 1234, 0, -1234), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(-1234, math_eval(min_test_op{}, -1234, 1234, 0), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(-1234, math_eval(min_test_op{}, 0, -1234, 1234), raft::Compare<int>()));

  ASSERT_TRUE(raft::match(
    -12.34, math_eval(min_test_op{}, 12.34f, 0., -12.34), raft::CompareApprox<double>(0.000001)));
  ASSERT_TRUE(raft::match(
    -12.34, math_eval(min_test_op{}, -12.34, 12.34f, 0.), raft::CompareApprox<double>(0.000001)));
  ASSERT_TRUE(raft::match(
    -12.34, math_eval(min_test_op{}, 0., -12.34, 12.34f), raft::CompareApprox<double>(0.000001)));
}

struct pow_test_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& x, const Type& y) const
  {
    return raft::pow(x, y);
  }
};

TEST(MathDevice, Pow)
{
  ASSERT_TRUE(raft::match(std::pow(12.34f, 2.f),
                          math_eval(pow_test_op{}, 12.34f, 2.f),
                          raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(std::pow(12.34, 2.),
                          math_eval(pow_test_op{}, 12.34, 2.),
                          raft::CompareApprox<double>(0.000001)));
}

struct sgn_test_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& in) const
  {
    return raft::sgn(in);
  }
};

TEST(MathDevice, Sgn)
{
  ASSERT_TRUE(raft::match(-1, math_eval(sgn_test_op{}, -1234), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(0, math_eval(sgn_test_op{}, 0), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(1, math_eval(sgn_test_op{}, 1234), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(-1, math_eval(sgn_test_op{}, -12.34f), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(0, math_eval(sgn_test_op{}, 0.f), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(1, math_eval(sgn_test_op{}, 12.34f), raft::Compare<int>()));
}

struct sin_test_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& in) const
  {
    return raft::sin(in);
  }
};

TEST(MathDevice, Sin)
{
  ASSERT_TRUE(raft::match(
    std::sin(12.34f), math_eval(sin_test_op{}, 12.34f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(
    std::sin(12.34), math_eval(sin_test_op{}, 12.34), raft::CompareApprox<double>(0.000001)));
}

struct sincos_test_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& x, Type* s, Type* c) const
  {
    raft::sincos(x, s, c);
    return x;  // unused, just to avoid creating another helper
  }
};

TEST(MathDevice, SinCos)
{
  auto stream = rmm::cuda_stream_default;
  float xf    = 12.34f;
  rmm::device_scalar<float> sf(stream);
  rmm::device_scalar<float> cf(stream);
  math_eval(sincos_test_op{}, xf, sf.data(), cf.data());
  ASSERT_TRUE(raft::match(std::sin(12.34f), sf.value(stream), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(std::cos(12.34f), cf.value(stream), raft::CompareApprox<float>(0.0001f)));
  double xd = 12.34f;
  rmm::device_scalar<double> sd(stream);
  rmm::device_scalar<double> cd(stream);
  math_eval(sincos_test_op{}, xd, sd.data(), cd.data());
  ASSERT_TRUE(raft::match(std::sin(12.34), sd.value(stream), raft::CompareApprox<double>(0.0001f)));
  ASSERT_TRUE(raft::match(std::cos(12.34), cd.value(stream), raft::CompareApprox<double>(0.0001f)));
}

struct sqrt_test_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& in) const
  {
    return raft::sqrt(in);
  }
};

TEST(MathDevice, Sqrt)
{
  ASSERT_TRUE(raft::match(
    std::sqrt(12.34f), math_eval(sqrt_test_op{}, 12.34f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(
    std::sqrt(12.34), math_eval(sqrt_test_op{}, 12.34), raft::CompareApprox<double>(0.000001)));
}

struct tanh_test_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& in) const
  {
    return raft::tanh(in);
  }
};

TEST(MathDevice, Tanh)
{
  ASSERT_TRUE(raft::match(
    std::tanh(12.34f), math_eval(tanh_test_op{}, 12.34f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(
    std::tanh(12.34), math_eval(tanh_test_op{}, 12.34), raft::CompareApprox<double>(0.000001)));
}
