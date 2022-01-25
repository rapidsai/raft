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
#include <experimental/mdspan>
#include <gtest/gtest.h>
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/mdarray.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace {
namespace stdex = std::experimental;
void check_status(int32_t* d_status, rmm::cuda_stream_view stream)
{
  stream.synchronize();
  int32_t h_status{1};
  raft::update_host(&h_status, d_status, 1, stream);
  ASSERT_EQ(h_status, 0);
}

// just simple integration test, main tests are in mdspan ref implementation.
void test_mdspan()
{
  auto stream = rmm::cuda_stream_default;
  rmm::device_uvector<float> a{16ul, stream};
  thrust::sequence(rmm::exec_policy(stream), a.begin(), a.end());
  stdex::mdspan<float, stdex::extents<stdex::dynamic_extent, stdex::dynamic_extent>> span{
    a.data(), 4, 4};
  thrust::device_vector<int32_t> status(1, 0);
  auto p_status = status.data().get();
  thrust::for_each_n(
    rmm::exec_policy(stream), thrust::make_counting_iterator(0ul), 4, [=] __device__(size_t i) {
      auto v = span(0, i);
      if (v != i) { raft::myAtomicAdd(p_status, 1); }
      auto k = stdex::submdspan(span, 0, stdex::full_extent);
      if (k(i) != i) { raft::myAtomicAdd(p_status, 1); }
    });
  check_status(p_status, stream);
}
}  // namespace

TEST(MDSpan, Basic) { test_mdspan(); }

namespace raft {
void test_uvector_policy()
{
  auto s = rmm::cuda_stream{};
  detail::device_uvector<float> dvec(10, s);
  auto a  = dvec[2];
  a       = 3;
  float c = a;
  ASSERT_EQ(c, 3);
}

TEST(MDArray, Policy) { test_uvector_policy(); }

void test_mdarray_basic()
{
  using matrix_extent = stdex::extents<stdex::dynamic_extent, stdex::dynamic_extent>;
  auto s              = rmm::cuda_stream_default;
  {
    /**
     * device policy
     */
    stdex::layout_right::mapping<matrix_extent> layout{matrix_extent{4, 4}};
    using mdarray_t = device_mdarray<float, matrix_extent, stdex::layout_right>;
    auto policy     = mdarray_t::container_policy_type{s};
    static_assert(std::is_same_v<typename decltype(policy)::accessor_type,
                                 detail::device_uvector_policy<float>>);
    device_mdarray<float, matrix_extent, stdex::layout_right> array{layout, policy};

    array(0, 3) = 1;
    ASSERT_EQ(array(0, 3), 1);
    // non-const access
    auto d_view = array.view();
    static_assert(!decltype(d_view)::accessor_type::is_host_type::value);

    thrust::device_vector<int32_t> status(1, 0);
    auto p_status = status.data().get();
    thrust::for_each_n(rmm::exec_policy(s),
                       thrust::make_counting_iterator(0ul),
                       1,
                       [d_view, p_status] __device__(auto i) {
                         if (d_view(0, 3) != 1) { myAtomicAdd(p_status, 1); }
                         d_view(0, 2) = 3;
                         if (d_view(0, 2) != 3) { myAtomicAdd(p_status, 1); }
                       });
    check_status(p_status, s);

    // const ref access
    auto const& arr = array;
    ASSERT_EQ(arr(0, 3), 1);
    auto const_d_view = arr.view();
    thrust::for_each_n(rmm::exec_policy(s),
                       thrust::make_counting_iterator(0ul),
                       1,
                       [const_d_view, p_status] __device__(auto i) {
                         if (const_d_view(0, 3) != 1) { myAtomicAdd(p_status, 1); }
                       });
    check_status(p_status, s);

    // utilities
    static_assert(array.rank_dynamic() == 2);
    static_assert(array.rank() == 2);
    static_assert(array.is_unique());
    static_assert(array.is_contiguous());
    static_assert(array.is_strided());

    static_assert(!std::is_nothrow_default_constructible<mdarray_t>::value);  // cuda stream
    static_assert(std::is_nothrow_move_constructible<mdarray_t>::value);
    static_assert(std::is_nothrow_move_assignable<mdarray_t>::value);
  }
  {
    /**
     * host policy
     */
    using mdarray_t = host_mdarray<float, matrix_extent, stdex::layout_right>;
    mdarray_t::container_policy_type policy;
    static_assert(
      std::is_same_v<typename decltype(policy)::accessor_type, detail::host_vector_policy<float>>);
    stdex::layout_right::mapping<matrix_extent> layout{matrix_extent{4, 4}};
    host_mdarray<float, matrix_extent, stdex::layout_right> array{layout, policy};

    array(0, 3) = 1;
    ASSERT_EQ(array(0, 3), 1);
    auto h_view = array.view();
    static_assert(decltype(h_view)::accessor_type::is_host_type::value);
    thrust::for_each_n(thrust::host, thrust::make_counting_iterator(0ul), 1, [h_view](auto i) {
      ASSERT_EQ(h_view(0, 3), 1);
    });

    static_assert(std::is_nothrow_default_constructible<mdarray_t>::value);
    static_assert(std::is_nothrow_move_constructible<mdarray_t>::value);
    static_assert(std::is_nothrow_move_assignable<mdarray_t>::value);
  }
  {
    /**
     * static extent
     */
    using static_extent = stdex::extents<16, 16>;
    stdex::layout_right::mapping<static_extent> layout{static_extent{}};
    using mdarray_t = device_mdarray<float, static_extent, stdex::layout_right>;
    mdarray_t::container_policy_type policy{s};
    device_mdarray<float, static_extent, stdex::layout_right> array{layout, policy};

    static_assert(array.rank_dynamic() == 0);
    static_assert(array.rank() == 2);
    static_assert(array.is_unique());
    static_assert(array.is_contiguous());
    static_assert(array.is_strided());

    array(0, 3) = 1;
    ASSERT_EQ(array(0, 3), 1);

    auto const& ref = array;
    ASSERT_EQ(ref(0, 3), 1);
  }
}

TEST(MDArray, Basic) { test_mdarray_basic(); }

template <typename BasicMDarray, typename PolicyFn, typename ThrustPolicy>
void test_mdarray_copy_move(ThrustPolicy exec, PolicyFn make_policy)
{
  using matrix_extent = stdex::extents<stdex::dynamic_extent, stdex::dynamic_extent>;
  stdex::layout_right::mapping<matrix_extent> layout{matrix_extent{4, 4}};

  using mdarray_t = BasicMDarray;
  using policy_t  = typename mdarray_t::container_policy_type;
  auto policy     = make_policy();

  mdarray_t arr_origin{layout, policy};
  thrust::sequence(exec, arr_origin.data(), arr_origin.data() + arr_origin.size());

  auto check_eq = [](auto const& l, auto const& r) {
    ASSERT_EQ(l.extents(), r.extents());
    for (size_t i = 0; i < l.view().extent(0); ++i) {
      for (size_t j = 0; j < l.view().extent(1); ++j) {
        ASSERT_EQ(l(i, j), r(i, j));
      }
    }
  };

  {
    // copy ctor
    auto policy = make_policy();
    mdarray_t arr{layout, policy};
    thrust::sequence(exec, arr.data(), arr.data() + arr.size());
    mdarray_t arr_copy_construct{arr};
    check_eq(arr, arr_copy_construct);

    auto const& ref = arr;
    mdarray_t arr_copy_construct_1{ref};
    check_eq(ref, arr_copy_construct_1);
  }

  {
    // copy assign
    auto policy = make_policy();
    mdarray_t arr{layout, policy};
    thrust::sequence(exec, arr.data(), arr.data() + arr.size());
    mdarray_t arr_copy_assign{layout, policy};
    arr_copy_assign = arr;
    check_eq(arr, arr_copy_assign);

    auto const& ref = arr;
    mdarray_t arr_copy_assign_1{layout, policy};
    arr_copy_assign_1 = ref;
    check_eq(ref, arr_copy_assign_1);
  }

  {
    // move ctor
    auto policy = make_policy();
    mdarray_t arr{layout, policy};
    thrust::sequence(exec, arr.data(), arr.data() + arr.size());
    mdarray_t arr_move_construct{std::move(arr)};
    ASSERT_EQ(arr.data(), nullptr);
    check_eq(arr_origin, arr_move_construct);
  }

  {
    // move assign
    auto policy = make_policy();
    mdarray_t arr{layout, policy};
    thrust::sequence(exec, arr.data(), arr.data() + arr.size());
    mdarray_t arr_move_assign{layout, policy};
    arr_move_assign = std::move(arr);
    ASSERT_EQ(arr.data(), nullptr);
    check_eq(arr_origin, arr_move_assign);
  }
}

TEST(MDArray, CopyMove)
{
  using matrix_extent = stdex::extents<stdex::dynamic_extent, stdex::dynamic_extent>;
  using d_matrix_t    = device_mdarray<float, matrix_extent>;
  using policy_t      = typename d_matrix_t::container_policy_type;
  auto s              = rmm::cuda_stream_default;
  test_mdarray_copy_move<d_matrix_t>(rmm::exec_policy(s), [s]() { return policy_t{s}; });

  using h_matrix_t = host_mdarray<float, matrix_extent>;
  test_mdarray_copy_move<h_matrix_t>(thrust::host,
                                     []() { return detail::host_vector_policy<float>{}; });

  {
    d_matrix_t arr;
    auto s = rmm::cuda_stream();
    policy_t policy{s};
    matrix_extent extents{3, 3};
    d_matrix_t::layout_type::mapping<matrix_extent> layout{extents};
    d_matrix_t non_dft{layout, policy};

    arr = non_dft;
    ASSERT_NE(arr.data(), non_dft.data());
    ASSERT_EQ(arr.extent(0), non_dft.extent(0));
  }
  {
    h_matrix_t arr;
    using h_policy_t = typename h_matrix_t::container_policy_type;
    h_policy_t policy{s};
    matrix_extent extents{3, 3};
    h_matrix_t::layout_type::mapping<matrix_extent> layout{extents};
    h_matrix_t non_dft{layout, policy};

    arr = non_dft;
    ASSERT_NE(arr.data(), non_dft.data());
    ASSERT_EQ(arr.extent(0), non_dft.extent(0));
  }
}
}  // namespace raft
