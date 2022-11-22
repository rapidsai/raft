/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <raft/core/kvp.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/for_each.h>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace raft {

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const std::vector<T>& v)
{
  os << "{";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i != 0) os << ",";
    os << v[i];
  }
  os << "}";
  return os;
}

template <typename T>
struct Compare {
  bool operator()(const T& a, const T& b) const { return a == b; }
};

template <typename T>
struct CompareApprox {
  CompareApprox(T eps_) : eps(eps_) {}
  bool operator()(const T& a, const T& b) const
  {
    T diff  = abs(a - b);
    T m     = std::max(abs(a), abs(b));
    T ratio = diff > eps ? diff / m : diff;

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
    T diff  = abs(abs(a) - abs(b));
    T m     = std::max(abs(a), abs(b));
    T ratio = diff >= eps ? diff / m : diff;
    return (ratio <= eps);
  }

 private:
  T eps;
};

template <typename T>
struct CompareApproxNoScaling {
  CompareApproxNoScaling(T eps_) : eps(eps_) {}
  bool operator()(const T& a, const T& b) const { return (abs(a - b) <= eps); }

 private:
  T eps;
};

template <typename T>
__host__ __device__ T abs(const T& a)
{
  return a > T(0) ? a : -a;
}

/*
 * @brief Helper function to compare 2 device n-D arrays with custom comparison
 * @tparam T the data type of the arrays
 * @tparam L the comparator lambda or object function
 * @param expected expected value(s)
 * @param actual actual values
 * @param eq_compare the comparator
 * @param stream cuda stream
 * @return the testing assertion to be later used by ASSERT_TRUE/EXPECT_TRUE
 * @{
 */
template <typename T, typename L>
testing::AssertionResult devArrMatch(
  const T* expected, const T* actual, size_t size, L eq_compare, cudaStream_t stream = 0)
{
  std::unique_ptr<T[]> exp_h(new T[size]);
  std::unique_ptr<T[]> act_h(new T[size]);
  raft::update_host<T>(exp_h.get(), expected, size, stream);
  raft::update_host<T>(act_h.get(), actual, size, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  for (size_t i(0); i < size; ++i) {
    auto exp = exp_h.get()[i];
    auto act = act_h.get()[i];
    if (!eq_compare(exp, act)) {
      return testing::AssertionFailure() << "actual=" << act << " != expected=" << exp << " @" << i;
    }
  }
  return testing::AssertionSuccess();
}

template <typename T, typename L>
testing::AssertionResult devArrMatch(
  T expected, const T* actual, size_t size, L eq_compare, cudaStream_t stream = 0)
{
  std::unique_ptr<T[]> act_h(new T[size]);
  raft::update_host<T>(act_h.get(), actual, size, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  for (size_t i(0); i < size; ++i) {
    auto act = act_h.get()[i];
    if (!eq_compare(expected, act)) {
      return testing::AssertionFailure()
             << "actual=" << act << " != expected=" << expected << " @" << i;
    }
  }
  return testing::AssertionSuccess();
}

template <typename T, typename L>
testing::AssertionResult devArrMatch(const T* expected,
                                     const T* actual,
                                     size_t rows,
                                     size_t cols,
                                     L eq_compare,
                                     cudaStream_t stream = 0)
{
  size_t size = rows * cols;
  std::unique_ptr<T[]> exp_h(new T[size]);
  std::unique_ptr<T[]> act_h(new T[size]);
  raft::update_host<T>(exp_h.get(), expected, size, stream);
  raft::update_host<T>(act_h.get(), actual, size, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  for (size_t i(0); i < rows; ++i) {
    for (size_t j(0); j < cols; ++j) {
      auto idx = i * cols + j;  // row major assumption!
      auto exp = exp_h.get()[idx];
      auto act = act_h.get()[idx];
      if (!eq_compare(exp, act)) {
        return testing::AssertionFailure()
               << "actual=" << act << " != expected=" << exp << " @" << i << "," << j;
      }
    }
  }
  return testing::AssertionSuccess();
}

template <typename T, typename L>
testing::AssertionResult devArrMatch(
  T expected, const T* actual, size_t rows, size_t cols, L eq_compare, cudaStream_t stream = 0)
{
  size_t size = rows * cols;
  std::unique_ptr<T[]> act_h(new T[size]);
  raft::update_host<T>(act_h.get(), actual, size, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  for (size_t i(0); i < rows; ++i) {
    for (size_t j(0); j < cols; ++j) {
      auto idx = i * cols + j;  // row major assumption!
      auto act = act_h.get()[idx];
      if (!eq_compare(expected, act)) {
        return testing::AssertionFailure()
               << "actual=" << act << " != expected=" << expected << " @" << i << "," << j;
      }
    }
  }
  return testing::AssertionSuccess();
}

/*
 * @brief Helper function to compare a device n-D arrays with an expected array
 * on the host, using a custom comparison
 * @tparam T the data type of the arrays
 * @tparam L the comparator lambda or object function
 * @param expected_h host array of expected value(s)
 * @param actual_d device array actual values
 * @param eq_compare the comparator
 * @param stream cuda stream
 * @return the testing assertion to be later used by ASSERT_TRUE/EXPECT_TRUE
 */
template <typename T, typename L>
testing::AssertionResult devArrMatchHost(
  const T* expected_h, const T* actual_d, size_t size, L eq_compare, cudaStream_t stream = 0)
{
  std::unique_ptr<T[]> act_h(new T[size]);
  raft::update_host<T>(act_h.get(), actual_d, size, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  bool ok   = true;
  auto fail = testing::AssertionFailure();
  for (size_t i(0); i < size; ++i) {
    auto exp = expected_h[i];
    auto act = act_h.get()[i];
    if (!eq_compare(exp, act)) {
      ok = false;
      fail << "actual=" << act << " != expected=" << exp << " @" << i << "; ";
    }
  }
  if (!ok) return fail;
  return testing::AssertionSuccess();
}

/**
 * @brief Helper function to compare host vectors using a custom comparison
 * @tparam T the element type
 * @tparam L the comparator lambda or object function
 * @param expected_h host vector of expected value(s)
 * @param actual_h host vector actual values
 * @param eq_compare the comparator
 * @return the testing assertion to be later used by ASSERT_TRUE/EXPECT_TRUE
 */
template <typename T, typename L>
testing::AssertionResult hostVecMatch(const std::vector<T>& expected_h,
                                      const std::vector<T>& actual_h,
                                      L eq_compare)
{
  auto n = actual_h.size();
  if (n != expected_h.size())
    return testing::AssertionFailure()
           << "vector sizez mismatch: "
           << "actual=" << n << " != expected=" << expected_h.size() << "; ";
  for (size_t i = 0; i < n; ++i) {
    auto exp = expected_h[i];
    auto act = actual_h[i];
    if (!eq_compare(exp, act)) {
      return testing::AssertionFailure()
             << "actual=" << act << " != expected=" << exp << " @" << i << "; ";
    }
  }
  return testing::AssertionSuccess();
}

/*
 * @brief Helper function to compare diagonal values of a 2D matrix
 * @tparam T the data type of the arrays
 * @tparam L the comparator lambda or object function
 * @param expected expected value along diagonal
 * @param actual actual matrix
 * @param eq_compare the comparator
 * @param stream cuda stream
 * @return the testing assertion to be later used by ASSERT_TRUE/EXPECT_TRUE
 */
template <typename T, typename L>
testing::AssertionResult diagonalMatch(
  T expected, const T* actual, size_t rows, size_t cols, L eq_compare, cudaStream_t stream = 0)
{
  size_t size = rows * cols;
  std::unique_ptr<T[]> act_h(new T[size]);
  raft::update_host<T>(act_h.get(), actual, size, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  for (size_t i(0); i < rows; ++i) {
    for (size_t j(0); j < cols; ++j) {
      if (i != j) continue;
      auto idx = i * cols + j;  // row major assumption!
      auto act = act_h.get()[idx];
      if (!eq_compare(expected, act)) {
        return testing::AssertionFailure()
               << "actual=" << act << " != expected=" << expected << " @" << i << "," << j;
      }
    }
  }
  return testing::AssertionSuccess();
}

template <typename T, typename L>
testing::AssertionResult match(const T expected, T actual, L eq_compare)
{
  if (!eq_compare(expected, actual)) {
    return testing::AssertionFailure() << "actual=" << actual << " != expected=" << expected;
  }
  return testing::AssertionSuccess();
}

template <typename T, typename IdxT>
typename std::enable_if_t<std::is_floating_point_v<T>> gen_uniform(T* out,
                                                                   raft::random::RngState& rng,
                                                                   IdxT len,
                                                                   cudaStream_t stream,
                                                                   T range_min = T(-1),
                                                                   T range_max = T(1))
{
  raft::random::uniform(rng, out, len, range_min, range_max, stream);
}

template <typename T, typename IdxT>
typename std::enable_if_t<std::is_integral_v<T>> gen_uniform(T* out,
                                                             raft::random::RngState& rng,
                                                             IdxT len,
                                                             cudaStream_t stream,
                                                             T range_min = T(0),
                                                             T range_max = T(100))
{
  raft::random::uniformInt(rng, out, len, range_min, range_max, stream);
}

template <typename T1, typename T2, typename IdxT>
void gen_uniform(raft::KeyValuePair<T1, T2>* out,
                 raft::random::RngState& rng,
                 IdxT len,
                 cudaStream_t stream)
{
  rmm::device_uvector<T1> keys(len, stream);
  rmm::device_uvector<T2> values(len, stream);

  gen_uniform(keys.data(), rng, len, stream);
  gen_uniform(values.data(), rng, len, stream);

  const T1* d_keys   = keys.data();
  const T2* d_values = values.data();
  auto counting      = thrust::make_counting_iterator<IdxT>(0);
  thrust::for_each(rmm::exec_policy(stream),
                   counting,
                   counting + len,
                   [out, d_keys, d_values] __device__(int idx) {
                     out[idx].key   = d_keys[idx];
                     out[idx].value = d_values[idx];
                   });
}

/** @} */

/** time the function call 'func' using cuda events */
#define TIMEIT_LOOP(ms, count, func)                       \
  do {                                                     \
    cudaEvent_t start, stop;                               \
    RAFT_CUDA_TRY(cudaEventCreate(&start));                \
    RAFT_CUDA_TRY(cudaEventCreate(&stop));                 \
    RAFT_CUDA_TRY(cudaEventRecord(start));                 \
    for (int i = 0; i < count; ++i) {                      \
      func;                                                \
    }                                                      \
    RAFT_CUDA_TRY(cudaEventRecord(stop));                  \
    RAFT_CUDA_TRY(cudaEventSynchronize(stop));             \
    ms = 0.f;                                              \
    RAFT_CUDA_TRY(cudaEventElapsedTime(&ms, start, stop)); \
    ms /= args.runs;                                       \
  } while (0)

inline std::vector<float> read_csv(std::string filename, bool skip_first_n_columns = 1)
{
  std::vector<float> result;
  std::ifstream myFile(filename);
  if (!myFile.is_open()) throw std::runtime_error("Could not open file");

  std::string line, colname;
  int val;

  if (myFile.good()) {
    std::getline(myFile, line);
    std::stringstream ss(line);
    while (std::getline(ss, colname, ',')) {}
  }

  int n_lines = 0;
  while (std::getline(myFile, line)) {
    std::stringstream ss(line);
    int colIdx = 0;
    while (ss >> val) {
      if (colIdx >= skip_first_n_columns) {
        result.push_back(val);
        if (ss.peek() == ',') ss.ignore();
      }
      colIdx++;
    }
    n_lines++;
  }

  printf("lines read: %d\n", n_lines);
  myFile.close();
  return result;
}

};  // end namespace raft
