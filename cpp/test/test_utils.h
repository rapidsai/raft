/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include <raft/core/kvp.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>

namespace raft {

template <typename T>
struct Compare {
  bool operator()(const T& a, const T& b) const { return a == b; }
};

template <typename Key, typename Value>
struct Compare<raft::KeyValuePair<Key, Value>> {
  bool operator()(const raft::KeyValuePair<Key, Value>& a,
                  const raft::KeyValuePair<Key, Value>& b) const
  {
    return a.key == b.key && a.value == b.value;
  }
};

template <typename T>
struct CompareApprox {
  CompareApprox(T eps_) : eps(eps_) {}
  bool operator()(const T& a, const T& b) const
  {
    T diff  = std::abs(a - b);
    T m     = std::max(std::abs(a), std::abs(b));
    T ratio = diff > eps ? diff / m : diff;

    return (ratio <= eps);
  }

 private:
  T eps;
};

template <typename T>
struct CompareApproxNaN {
  CompareApproxNaN(T eps_) : eps(eps_) {}
  bool operator()(const T& a, const T& b) const
  {
    T diff  = std::abs(a - b);
    T m     = std::max(std::abs(a), std::abs(b));
    T ratio = diff > eps ? diff / m : diff;

    if (std::isnan(a) && std::isnan(b)) { return true; }
    return (ratio <= eps);
  }

 private:
  T eps;
};

template <typename Key, typename Value>
::std::ostream& operator<<(::std::ostream& os, const raft::KeyValuePair<Key, Value>& kv)
{
  os << "{ " << kv.key << ", " << kv.value << '}';
  return os;
}

template <typename Key, typename Value>
struct CompareApprox<raft::KeyValuePair<Key, Value>> {
  CompareApprox(raft::KeyValuePair<Key, Value> eps)
    : compare_keys(eps.key), compare_values(eps.value)
  {
  }
  bool operator()(const raft::KeyValuePair<Key, Value>& a,
                  const raft::KeyValuePair<Key, Value>& b) const
  {
    return compare_keys(a.key, b.key) && compare_values(a.value, b.value);
  }

 private:
  CompareApprox<Key> compare_keys;
  CompareApprox<Value> compare_values;
};

template <typename T>
struct CompareApproxAbs {
  CompareApproxAbs(T eps_) : eps(eps_) {}
  bool operator()(const T& a, const T& b) const
  {
    T diff  = std::abs(std::abs(a) - std::abs(b));
    T m     = std::max(std::abs(a), std::abs(b));
    T ratio = diff >= eps ? diff / m : diff;
    return (ratio <= eps);
  }

 private:
  T eps;
};

template <typename T>
struct CompareApproxNoScaling {
  CompareApproxNoScaling(T eps_) : eps(eps_) {}
  bool operator()(const T& a, const T& b) const { return (std::abs(a - b) <= eps); }

 private:
  T eps;
};

template <typename T, typename L>
testing::AssertionResult match(const T& expected, const T& actual, L eq_compare)
{
  if (!eq_compare(expected, actual)) {
    return testing::AssertionFailure() << "actual=" << actual << " != expected=" << expected;
  }
  return testing::AssertionSuccess();
}

};  // end namespace raft
