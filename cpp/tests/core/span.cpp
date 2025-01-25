/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#include "test_span.hpp"

#include <raft/core/host_span.hpp>

#include <gtest/gtest.h>

#include <numeric>  // iota

namespace raft {
TEST(Span, DlfConstructors)
{
  // Dynamic extent
  {
    host_span<int> s;
    ASSERT_EQ(s.size(), 0);
    ASSERT_EQ(s.data(), nullptr);

    host_span<int const> cs;
    ASSERT_EQ(cs.size(), 0);
    ASSERT_EQ(cs.data(), nullptr);
  }

  // Static extent
  {
    host_span<int, 0> s;
    ASSERT_EQ(s.size(), 0);
    ASSERT_EQ(s.data(), nullptr);

    host_span<int const, 0> cs;
    ASSERT_EQ(cs.size(), 0);
    ASSERT_EQ(cs.data(), nullptr);
  }

  // Init list.
  {
    host_span<float> s{};
    ASSERT_EQ(s.size(), 0);
    ASSERT_EQ(s.data(), nullptr);

    host_span<int const> cs{};
    ASSERT_EQ(cs.size(), 0);
    ASSERT_EQ(cs.data(), nullptr);
  }
}

TEST(Span, FromNullPtr)
{
  // dynamic extent
  {
    host_span<float> s{nullptr, static_cast<host_span<float>::size_type>(0)};
    ASSERT_EQ(s.size(), 0);
    ASSERT_EQ(s.data(), nullptr);

    host_span<float const> cs{nullptr, static_cast<host_span<float>::size_type>(0)};
    ASSERT_EQ(cs.size(), 0);
    ASSERT_EQ(cs.data(), nullptr);
  }
  // static extent
  {
    host_span<float, 0> s{nullptr, static_cast<host_span<float>::size_type>(0)};
    ASSERT_EQ(s.size(), 0);
    ASSERT_EQ(s.data(), nullptr);

    host_span<float const, 0> cs{nullptr, static_cast<host_span<float>::size_type>(0)};
    ASSERT_EQ(cs.size(), 0);
    ASSERT_EQ(cs.data(), nullptr);
  }
}

TEST(Span, FromPtrLen)
{
  float arr[16];
  std::iota(arr, arr + 16, 0);

  // static extent
  {
    host_span<float> s(arr, 16);
    ASSERT_EQ(s.size(), 16);
    ASSERT_EQ(s.data(), arr);

    for (host_span<float>::size_type i = 0; i < 16; ++i) {
      ASSERT_EQ(s[i], arr[i]);
    }

    host_span<float const> cs(arr, 16);
    ASSERT_EQ(cs.size(), 16);
    ASSERT_EQ(cs.data(), arr);

    for (host_span<float const>::size_type i = 0; i < 16; ++i) {
      ASSERT_EQ(cs[i], arr[i]);
    }
  }

  // dynamic extent
  {
    host_span<float, 16> s(arr, 16);
    ASSERT_EQ(s.size(), 16);
    ASSERT_EQ(s.data(), arr);

    for (size_t i = 0; i < 16; ++i) {
      ASSERT_EQ(s[i], arr[i]);
    }

    host_span<float const, 16> cs(arr, 16);
    ASSERT_EQ(cs.size(), 16);
    ASSERT_EQ(cs.data(), arr);

    for (host_span<float const>::size_type i = 0; i < 16; ++i) {
      ASSERT_EQ(cs[i], arr[i]);
    }
  }
}

TEST(Span, FromFirstLast)
{
  float arr[16];
  initialize_range(arr, arr + 16);

  // dynamic extent
  {
    host_span<float> s(arr, arr + 16);
    ASSERT_EQ(s.size(), 16);
    ASSERT_EQ(s.data(), arr);
    ASSERT_EQ(s.data() + s.size(), arr + 16);

    for (size_t i = 0; i < 16; ++i) {
      ASSERT_EQ(s[i], arr[i]);
    }

    host_span<float const> cs(arr, arr + 16);
    ASSERT_EQ(cs.size(), 16);
    ASSERT_EQ(cs.data(), arr);
    ASSERT_EQ(cs.data() + cs.size(), arr + 16);

    for (size_t i = 0; i < 16; ++i) {
      ASSERT_EQ(cs[i], arr[i]);
    }
  }

  // static extent
  {
    host_span<float, 16> s(arr, arr + 16);
    ASSERT_EQ(s.size(), 16);
    ASSERT_EQ(s.data(), arr);
    ASSERT_EQ(s.data() + s.size(), arr + 16);

    for (size_t i = 0; i < 16; ++i) {
      ASSERT_EQ(s[i], arr[i]);
    }

    host_span<float const> cs(arr, arr + 16);
    ASSERT_EQ(cs.size(), 16);
    ASSERT_EQ(cs.data(), arr);
    ASSERT_EQ(cs.data() + cs.size(), arr + 16);

    for (size_t i = 0; i < 16; ++i) {
      ASSERT_EQ(cs[i], arr[i]);
    }
  }
}

namespace {
struct base_class_t {
  virtual void operator()() {}
};
struct derived_class_t : public base_class_t {
  void operator()() override {}
};
}  // anonymous namespace

TEST(Span, FromOther)
{
  // convert constructor
  {
    host_span<derived_class_t> derived;
    host_span<base_class_t> base{derived};
    ASSERT_EQ(base.size(), derived.size());
    ASSERT_EQ(base.data(), derived.data());
  }

  float arr[16];
  initialize_range(arr, arr + 16);

  // default copy constructor
  {
    host_span<float> s0(arr);
    host_span<float> s1(s0);
    ASSERT_EQ(s0.size(), s1.size());
    ASSERT_EQ(s0.data(), s1.data());
  }
}

TEST(Span, FromArray)
{
  float arr[16];
  initialize_range(arr, arr + 16);

  {
    host_span<float> s(arr);
    ASSERT_EQ(&arr[0], s.data());
    ASSERT_EQ(s.size(), 16);
    for (size_t i = 0; i < 16; ++i) {
      ASSERT_EQ(arr[i], s[i]);
    }
  }

  {
    host_span<float, 16> s(arr);
    ASSERT_EQ(&arr[0], s.data());
    ASSERT_EQ(s.size(), 16);
    for (size_t i = 0; i < 16; ++i) {
      ASSERT_EQ(arr[i], s[i]);
    }
  }
}

TEST(Span, Assignment)
{
  int status = 1;
  test_assignment_t<false>{&status}();
  ASSERT_EQ(status, 1);
}

TEST(Span, BeginEnd)
{
  int status = 1;
  test_beginend_t<false>{&status}();
  ASSERT_EQ(status, 1);
}

TEST(Span, ElementAccess)
{
  float arr[16];
  initialize_range(arr, arr + 16);

  host_span<float> s(arr);
  size_t j = 0;
  for (auto i : s) {
    ASSERT_EQ(i, arr[j]);
    ++j;
  }
}

TEST(Span, Obversers)
{
  int status = 1;
  test_observers_t<false>{&status}();
  ASSERT_EQ(status, 1);
}

TEST(Span, FrontBack)
{
  {
    float arr[4]{0, 1, 2, 3};
    host_span<float, 4> s(arr);
    ASSERT_EQ(s.front(), 0);
    ASSERT_EQ(s.back(), 3);
  }
  {
    std::vector<double> arr{0, 1, 2, 3};
    host_span<double> s(arr.data(), arr.size());
    ASSERT_EQ(s.front(), 0);
    ASSERT_EQ(s.back(), 3);
  }
}

TEST(Span, FirstLast)
{
  // static extent
  {
    float arr[16];
    initialize_range(arr, arr + 16);

    host_span<float> s(arr);
    host_span<float, 4> first = s.first<4>();

    ASSERT_EQ(first.size(), 4);
    ASSERT_EQ(first.data(), arr);

    for (size_t i = 0; i < first.size(); ++i) {
      ASSERT_EQ(first[i], arr[i]);
    }
  }

  {
    float arr[16];
    initialize_range(arr, arr + 16);

    host_span<float> s(arr);
    host_span<float, 4> last = s.last<4>();

    ASSERT_EQ(last.size(), 4);
    ASSERT_EQ(last.data(), arr + 12);

    for (size_t i = 0; i < last.size(); ++i) {
      ASSERT_EQ(last[i], arr[i + 12]);
    }
  }

  // dynamic extent
  {
    float* arr = new float[16];
    initialize_range(arr, arr + 16);
    host_span<float> s(arr, 16);
    host_span<float> first = s.first(4);

    ASSERT_EQ(first.size(), 4);
    ASSERT_EQ(first.data(), s.data());

    for (size_t i = 0; i < first.size(); ++i) {
      ASSERT_EQ(first[i], s[i]);
    }

    delete[] arr;
  }

  {
    float* arr = new float[16];
    initialize_range(arr, arr + 16);
    host_span<float> s(arr, 16);
    host_span<float> last = s.last(4);

    ASSERT_EQ(last.size(), 4);
    ASSERT_EQ(last.data(), s.data() + 12);

    for (size_t i = 0; i < last.size(); ++i) {
      ASSERT_EQ(s[12 + i], last[i]);
    }

    delete[] arr;
  }
}

TEST(Span, Subspan)
{
  int arr[16]{0};
  host_span<int> s1(arr);
  auto s2 = s1.subspan<4>();
  ASSERT_EQ(s1.size() - 4, s2.size());

  auto s3 = s1.subspan(2, 4);
  ASSERT_EQ(s1.data() + 2, s3.data());
  ASSERT_EQ(s3.size(), 4);

  auto s4 = s1.subspan(2, dynamic_extent);
  ASSERT_EQ(s1.data() + 2, s4.data());
  ASSERT_EQ(s4.size(), s1.size() - 2);
}

TEST(Span, Compare)
{
  int status = 1;
  test_compare_t<false>{&status}();
  ASSERT_EQ(status, 1);
}

TEST(Span, AsBytes)
{
  int status = 1;
  test_as_bytes_t<false>{&status}();
  ASSERT_EQ(status, 1);
}

TEST(Span, AsWritableBytes)
{
  int status = 1;
  test_as_writable_bytes_t<false>{&status}();
  ASSERT_EQ(status, 1);
}

TEST(Span, Empty)
{
  {
    host_span<float> s{nullptr, static_cast<host_span<float>::size_type>(0)};
    auto res = s.subspan(0);
    ASSERT_EQ(res.data(), nullptr);
    ASSERT_EQ(res.size(), 0);

    res = s.subspan(0, 0);
    ASSERT_EQ(res.data(), nullptr);
    ASSERT_EQ(res.size(), 0);
  }

  {
    host_span<float, 0> s{nullptr, static_cast<host_span<float>::size_type>(0)};
    auto res = s.subspan(0);
    ASSERT_EQ(res.data(), nullptr);
    ASSERT_EQ(res.size(), 0);

    res = s.subspan(0, 0);
    ASSERT_EQ(res.data(), nullptr);
    ASSERT_EQ(res.size(), 0);
  }
  {
    // Should emit compile error
    // host_span<float, 0> h{nullptr, 0ul};
    // device_span<float, 0> d{h};
  }
}

TEST(Span, RBeginREnd)
{
  int32_t status = 1;
  test_rbeginrend_t<false>{&status}();
  ASSERT_EQ(status, 1);
}
}  // namespace raft
