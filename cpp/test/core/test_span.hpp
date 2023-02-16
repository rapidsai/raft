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
#pragma once
#include <raft/core/span.hpp>

namespace raft {

template <typename Iter>
__host__ __device__ void initialize_range(Iter _begin, Iter _end)
{
  float j = 0;
  for (Iter i = _begin; i != _end; ++i, ++j) {
    *i = j;
  }
}
#define SPAN_ASSERT_TRUE(cond, status) \
  if (!(cond)) { *(status) = -1; }

#define SPAN_ASSERT_FALSE(cond, status) \
  if ((cond)) { *(status) = -1; }

struct test_test_status_t {
  int* status;

  explicit test_test_status_t(int* _status) : status(_status) {}

  __host__ __device__ void operator()() { this->operator()(0); }
  __host__ __device__ void operator()(int _idx) { SPAN_ASSERT_TRUE(false, status); }
};

template <bool is_device>
struct test_assignment_t {
  int* status;

  explicit test_assignment_t(int* _status) : status(_status) {}

  __host__ __device__ void operator()() { this->operator()(0); }
  __host__ __device__ void operator()(int _idx)
  {
    span<float, is_device> s1;

    float arr[] = {3, 4, 5};

    span<const float, is_device> s2 = arr;
    SPAN_ASSERT_TRUE(s2.size() == 3, status);
    SPAN_ASSERT_TRUE(s2.data() == &arr[0], status);

    s2 = s1;
    SPAN_ASSERT_TRUE(s2.empty(), status);
  }
};

template <bool is_device>
struct test_beginend_t {
  int* status;

  explicit test_beginend_t(int* _status) : status(_status) {}

  __host__ __device__ void operator()() { this->operator()(0); }
  __host__ __device__ void operator()(int _idx)
  {
    float arr[16];
    initialize_range(arr, arr + 16);

    span<float, is_device> s(arr);
    typename span<float, is_device>::iterator beg{s.begin()};
    typename span<float, is_device>::iterator end{s.end()};

    SPAN_ASSERT_TRUE(end == beg + 16, status);
    SPAN_ASSERT_TRUE(*beg == arr[0], status);
    SPAN_ASSERT_TRUE(*(end - 1) == arr[15], status);
  }
};

template <bool is_device>
struct test_rbeginrend_t {
  int* status;

  explicit test_rbeginrend_t(int* _status) : status(_status) {}

  __host__ __device__ void operator()() { this->operator()(0); }
  __host__ __device__ void operator()(int _idx)
  {
    float arr[16];
    initialize_range(arr, arr + 16);

    span<float, is_device> s(arr);
    s.rbegin();
    typename span<float, is_device>::reverse_iterator rbeg{s.rbegin()};
    typename span<float, is_device>::reverse_iterator rend{s.rend()};

    SPAN_ASSERT_TRUE(rbeg + 16 == rend, status);
    SPAN_ASSERT_TRUE(*(rbeg) == arr[15], status);
    SPAN_ASSERT_TRUE(*(rend - 1) == arr[0], status);

    typename span<float, is_device>::const_reverse_iterator crbeg{s.crbegin()};
    typename span<float, is_device>::const_reverse_iterator crend{s.crend()};

    SPAN_ASSERT_TRUE(crbeg + 16 == crend, status);
    SPAN_ASSERT_TRUE(*(crbeg) == arr[15], status);
    SPAN_ASSERT_TRUE(*(crend - 1) == arr[0], status);
  }
};

template <bool is_device>
struct test_observers_t {
  int* status;

  explicit test_observers_t(int* _status) : status(_status) {}

  __host__ __device__ void operator()() { this->operator()(0); }
  __host__ __device__ void operator()(int _idx)
  {
    // empty
    {
      float* arr = nullptr;
      span<float, is_device> s(arr, static_cast<typename span<float, is_device>::size_type>(0));
      SPAN_ASSERT_TRUE(s.empty(), status);
    }

    // size, size_types
    {
      float* arr = new float[16];
      span<float, is_device> s(arr, 16);
      SPAN_ASSERT_TRUE(s.size() == 16, status);
      SPAN_ASSERT_TRUE(s.size_bytes() == 16 * sizeof(float), status);
      delete[] arr;
    }
  }
};

template <bool is_device>
struct test_compare_t {
  int* status;

  explicit test_compare_t(int* _status) : status(_status) {}

  __host__ __device__ void operator()() { this->operator()(0); }
  __host__ __device__ void operator()(int _idx)
  {
    float lhs_arr[16], rhs_arr[16];
    initialize_range(lhs_arr, lhs_arr + 16);
    initialize_range(rhs_arr, rhs_arr + 16);

    span<float, is_device> lhs(lhs_arr);
    span<float, is_device> rhs(rhs_arr);

    SPAN_ASSERT_TRUE(lhs == rhs, status);
    SPAN_ASSERT_FALSE(lhs != rhs, status);

    SPAN_ASSERT_TRUE(lhs <= rhs, status);
    SPAN_ASSERT_TRUE(lhs >= rhs, status);

    lhs[2] -= 1;

    SPAN_ASSERT_FALSE(lhs == rhs, status);
    SPAN_ASSERT_TRUE(lhs < rhs, status);
    SPAN_ASSERT_FALSE(lhs > rhs, status);
  }
};

template <bool is_device>
struct test_as_bytes_t {
  int* status;

  explicit test_as_bytes_t(int* _status) : status(_status) {}

  __host__ __device__ void operator()() { this->operator()(0); }
  __host__ __device__ void operator()(int _idx)
  {
    float arr[16];
    initialize_range(arr, arr + 16);

    {
      const span<const float, is_device> s{arr};
      const span<const std::byte, is_device> bs = as_bytes(s);
      SPAN_ASSERT_TRUE(bs.size() == s.size_bytes(), status);
      SPAN_ASSERT_TRUE(static_cast<const void*>(bs.data()) == static_cast<const void*>(s.data()),
                       status);
    }

    {
      span<float, is_device> s;
      const span<const std::byte, is_device> bs = as_bytes(s);
      SPAN_ASSERT_TRUE(bs.size() == s.size(), status);
      SPAN_ASSERT_TRUE(bs.size() == 0, status);
      SPAN_ASSERT_TRUE(bs.size_bytes() == 0, status);
      SPAN_ASSERT_TRUE(static_cast<const void*>(bs.data()) == static_cast<const void*>(s.data()),
                       status);
      SPAN_ASSERT_TRUE(bs.data() == nullptr, status);
    }
  }
};

template <bool is_device>
struct test_as_writable_bytes_t {
  int* status;

  explicit test_as_writable_bytes_t(int* _status) : status(_status) {}

  __host__ __device__ void operator()() { this->operator()(0); }
  __host__ __device__ void operator()(int _idx)
  {
    float arr[16];
    initialize_range(arr, arr + 16);

    {
      span<float, is_device> s;
      span<std::byte, is_device> byte_s = as_writable_bytes(s);
      SPAN_ASSERT_TRUE(byte_s.size() == s.size(), status);
      SPAN_ASSERT_TRUE(byte_s.size_bytes() == s.size_bytes(), status);
      SPAN_ASSERT_TRUE(byte_s.size() == 0, status);
      SPAN_ASSERT_TRUE(byte_s.size_bytes() == 0, status);
      SPAN_ASSERT_TRUE(byte_s.data() == nullptr, status);
      SPAN_ASSERT_TRUE(static_cast<void*>(byte_s.data()) == static_cast<void*>(s.data()), status);
    }

    {
      span<float, is_device> s{arr};
      span<std::byte, is_device> bs{as_writable_bytes(s)};
      SPAN_ASSERT_TRUE(s.size_bytes() == bs.size_bytes(), status);
      SPAN_ASSERT_TRUE(static_cast<void*>(bs.data()) == static_cast<void*>(s.data()), status);
    }
  }
};
}  // namespace raft
