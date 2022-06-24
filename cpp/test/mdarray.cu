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
#include <raft/core/mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

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
  stdex::mdspan<float, stdex::extents<raft::dynamic_extent, raft::dynamic_extent>> span{
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
  using matrix_extent = stdex::extents<dynamic_extent, dynamic_extent>;
  auto s              = rmm::cuda_stream_default;
  {
    /**
     * device policy
     */
    layout_c_contiguous::mapping<matrix_extent> layout{matrix_extent{4, 4}};
    using mdarray_t = device_mdarray<float, matrix_extent, layout_c_contiguous>;
    auto policy     = mdarray_t::container_policy_type{s};
    static_assert(std::is_same_v<typename decltype(policy)::accessor_type,
                                 detail::device_uvector_policy<float>>);
    device_mdarray<float, matrix_extent, layout_c_contiguous> array{layout, policy};

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
    using mdarray_t = host_mdarray<float, matrix_extent, layout_c_contiguous>;
    mdarray_t::container_policy_type policy;
    static_assert(
      std::is_same_v<typename decltype(policy)::accessor_type, detail::host_vector_policy<float>>);
    layout_c_contiguous::mapping<matrix_extent> layout{matrix_extent{4, 4}};
    host_mdarray<float, matrix_extent, layout_c_contiguous> array{layout, policy};

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
    layout_c_contiguous::mapping<static_extent> layout{static_extent{}};
    using mdarray_t = device_mdarray<float, static_extent, layout_c_contiguous>;
    mdarray_t::container_policy_type policy{s};
    device_mdarray<float, static_extent, layout_c_contiguous> array{layout, policy};

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
  using matrix_extent = stdex::extents<dynamic_extent, dynamic_extent>;
  layout_c_contiguous::mapping<matrix_extent> layout{matrix_extent{4, 4}};

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
  using matrix_extent = stdex::extents<dynamic_extent, dynamic_extent>;
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

namespace {
void test_factory_methods()
{
  size_t n{100};
  rmm::device_uvector<float> d_vec(n, rmm::cuda_stream_default);
  {
    auto d_matrix = make_device_matrix_view(d_vec.data(), d_vec.size() / 2, 2);
    ASSERT_EQ(d_matrix.extent(0), n / 2);
    ASSERT_EQ(d_matrix.extent(1), 2);
    ASSERT_EQ(d_matrix.data(), d_vec.data());
  }
  {
    auto const& vec_ref = d_vec;
    auto d_matrix       = make_device_matrix_view(vec_ref.data(), d_vec.size() / 2, 2);
    ASSERT_EQ(d_matrix.extent(0), n / 2);
    ASSERT_EQ(d_matrix.extent(1), 2);
    ASSERT_EQ(d_matrix.data(), d_vec.data());
  }

  std::vector<float> h_vec(n);
  {
    auto h_matrix = make_host_matrix_view(h_vec.data(), h_vec.size() / 2, 2);
    ASSERT_EQ(h_matrix.extent(0), n / 2);
    ASSERT_EQ(h_matrix.extent(1), 2);
    ASSERT_EQ(h_matrix.data(), h_vec.data());
    h_matrix(0, 0) = 13;
    ASSERT_EQ(h_matrix(0, 0), 13);
  }
  {
    auto const& vec_ref = h_vec;
    auto h_matrix       = make_host_matrix_view(vec_ref.data(), d_vec.size() / 2, 2);
    ASSERT_EQ(h_matrix.extent(0), n / 2);
    ASSERT_EQ(h_matrix.extent(1), 2);
    ASSERT_EQ(h_matrix.data(), h_vec.data());
    // const, cannot assign
    // h_matrix(0, 0) = 13;
    ASSERT_EQ(h_matrix(0, 0), 13);
  }

  {
    // host mdarray
    auto h_matrix = make_host_matrix<float>(n, n);
    ASSERT_EQ(h_matrix.extent(0), n);
    ASSERT_EQ(h_matrix.extent(1), n);
    static_assert(h_matrix.rank() == 2);

    auto h_vec = make_host_vector<float>(n);
    static_assert(h_vec.rank() == 1);
    ASSERT_EQ(h_vec.extent(0), n);
  }
  {
    // device mdarray
    auto d_matrix = make_device_matrix<float>(n, n, rmm::cuda_stream_default);
    ASSERT_EQ(d_matrix.extent(0), n);
    ASSERT_EQ(d_matrix.extent(1), n);
    static_assert(d_matrix.rank() == 2);

    auto d_vec = make_device_vector<float>(n, rmm::cuda_stream_default);
    static_assert(d_vec.rank() == 1);
    ASSERT_EQ(d_vec.extent(0), n);
  }

  {
    // device scalar
    auto d_scalar = make_device_scalar<double>(17.0, rmm::cuda_stream_default);
    static_assert(d_scalar.rank() == 1);
    static_assert(d_scalar.rank_dynamic() == 0);
    ASSERT_EQ(d_scalar(0), 17.0);

    auto view = d_scalar.view();
    thrust::device_vector<int32_t> status(1, 0);
    auto p_status = status.data().get();
    thrust::for_each_n(rmm::exec_policy(rmm::cuda_stream_default),
                       thrust::make_counting_iterator(0),
                       1,
                       [=] __device__(auto i) {
                         if (view(i) != 17.0) { myAtomicAdd(p_status, 1); }
                       });
    check_status(p_status, rmm::cuda_stream_default);
  }
  {
    // host scalar
    auto h_scalar = make_host_scalar<double>(17.0);
    static_assert(h_scalar.rank() == 1);
    static_assert(h_scalar.rank_dynamic() == 0);
    ASSERT_EQ(h_scalar(0), 17.0);
    ASSERT_EQ(h_scalar.view()(0), 17.0);

    auto view = make_host_scalar_view(h_scalar.data());
    ASSERT_EQ(view(0), 17.0);
  }
}
}  // anonymous namespace

TEST(MDArray, Factory) { test_factory_methods(); }

namespace {
template <typename T, typename LayoutPolicy>
void check_matrix_layout(device_matrix_view<T, LayoutPolicy> in)
{
  static_assert(in.rank() == 2);
  static_assert(in.is_contiguous());

  bool constexpr kIsCContiguous = std::is_same_v<LayoutPolicy, layout_c_contiguous>;
  bool constexpr kIsFContiguous = std::is_same_v<LayoutPolicy, layout_f_contiguous>;
  // only 1 of them is true
  static_assert(kIsCContiguous || kIsFContiguous);
  static_assert(!(kIsCContiguous && kIsFContiguous));
}
}  // anonymous namespace

TEST(MDArray, FuncArg)
{
  {
    auto d_matrix = make_device_matrix<float>(10, 10, rmm::cuda_stream_default);
    check_matrix_layout(d_matrix.view());
  }
  {
    auto d_matrix =
      make_device_matrix<float, layout_f_contiguous>(10, 10, rmm::cuda_stream_default);
    check_matrix_layout(d_matrix.view());

    auto slice =
      stdex::submdspan(d_matrix.view(), std::make_tuple(2ul, 4ul), std::make_tuple(2ul, 5ul));
    static_assert(slice.is_strided());
    ASSERT_EQ(slice.extent(0), 2);
    ASSERT_EQ(slice.extent(1), 3);
    // is using device_accessor mixin.
    static_assert(
      std::is_same_v<decltype(slice)::accessor_type, device_matrix_view<float>::accessor_type>);
  }
}

namespace {
void test_mdarray_unravel()
{
  {
    uint32_t v{0};
    ASSERT_EQ(detail::native_popc(v), 0);
    ASSERT_EQ(detail::popc(v), 0);
    v = 1;
    ASSERT_EQ(detail::native_popc(v), 1);
    ASSERT_EQ(detail::popc(v), 1);
    v = 0xffffffff;
    ASSERT_EQ(detail::native_popc(v), 32);
    ASSERT_EQ(detail::popc(v), 32);
  }
  {
    uint64_t v{0};
    ASSERT_EQ(detail::native_popc(v), 0);
    ASSERT_EQ(detail::popc(v), 0);
    v = 1;
    ASSERT_EQ(detail::native_popc(v), 1);
    ASSERT_EQ(detail::popc(v), 1);
    v = 0xffffffff;
    ASSERT_EQ(detail::native_popc(v), 32);
    ASSERT_EQ(detail::popc(v), 32);
    v = 0xffffffffffffffff;
    ASSERT_EQ(detail::native_popc(v), 64);
    ASSERT_EQ(detail::popc(v), 64);
  }

  // examples from numpy unravel_index
  {
    auto coord = unravel_index(22, detail::matrix_extent{7, 6}, stdex::layout_right{});
    static_assert(std::tuple_size<decltype(coord)>::value == 2);
    ASSERT_EQ(std::get<0>(coord), 3);
    ASSERT_EQ(std::get<1>(coord), 4);
  }
  {
    auto coord = unravel_index(41, detail::matrix_extent{7, 6}, stdex::layout_right{});
    static_assert(std::tuple_size<decltype(coord)>::value == 2);
    ASSERT_EQ(std::get<0>(coord), 6);
    ASSERT_EQ(std::get<1>(coord), 5);
  }
  {
    auto coord = unravel_index(37, detail::matrix_extent{7, 6}, stdex::layout_right{});
    static_assert(std::tuple_size<decltype(coord)>::value == 2);
    ASSERT_EQ(std::get<0>(coord), 6);
    ASSERT_EQ(std::get<1>(coord), 1);
  }
  // assignment
  {
    auto m   = make_host_matrix<float>(7, 6);
    auto m_v = m.view();
    for (size_t i = 0; i < m.size(); ++i) {
      auto coord             = unravel_index(i, m.extents(), typename decltype(m)::layout_type{});
      std::apply(m_v, coord) = i;
    }
    for (size_t i = 0; i < m.size(); ++i) {
      auto coord = unravel_index(i, m.extents(), typename decltype(m)::layout_type{});
      ASSERT_EQ(std::apply(m_v, coord), i);
    }
  }

  {
    handle_t handle;
    auto m   = make_device_matrix<float>(handle, 7, 6);
    auto m_v = m.view();
    thrust::for_each_n(handle.get_thrust_policy(),
                       thrust::make_counting_iterator(0ul),
                       m_v.size(),
                       [=] HD(size_t i) {
                         auto coord =
                           unravel_index(i, m_v.extents(), typename decltype(m_v)::layout_type{});
                         std::apply(m_v, coord) = static_cast<float>(i);
                       });
    thrust::device_vector<int32_t> status(1, 0);
    auto p_status = status.data().get();
    thrust::for_each_n(handle.get_thrust_policy(),
                       thrust::make_counting_iterator(0ul),
                       m_v.size(),
                       [=] __device__(size_t i) {
                         auto coord =
                           unravel_index(i, m_v.extents(), typename decltype(m_v)::layout_type{});
                         auto v = std::apply(m_v, coord);
                         if (v != static_cast<float>(i)) { raft::myAtomicAdd(p_status, 1); }
                       });
    check_status(p_status, handle.get_stream());
  }
}
}  // anonymous namespace

TEST(MDArray, Unravel) { test_mdarray_unravel(); }
}  // namespace raft
