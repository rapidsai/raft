/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include "ctest_common.hpp"

#include <experimental/mdspan>

namespace stdex = ::std::experimental;

// Only works with newer constexpr
#if defined(_MDSPAN_USE_CONSTEXPR_14) && _MDSPAN_USE_CONSTEXPR_14

//==============================================================================
// <editor-fold desc="1D dynamic extent ptrdiff_t submdspan"> {{{1

template<class Layout>
constexpr bool
dynamic_extent_1d() {
  int data[] = {1, 2, 3, 4, 5};
  auto s = stdex::mdspan<int, stdex::dextents<size_t,1>, Layout>(data, 5);
  int result = 0;
  for (size_t i = 0; i < s.extent(0); ++i) {
    auto ss = stdex::submdspan(s, i);
    result += __MDSPAN_OP0(ss);
  }
  // 1 + 2 + 3 + 4 + 5
  constexpr_assert_equal(15, result);
  return result == 15;
}

MDSPAN_STATIC_TEST(dynamic_extent_1d<stdex::layout_left>());
MDSPAN_STATIC_TEST(dynamic_extent_1d<stdex::layout_right>());


// </editor-fold> end 1D dynamic extent ptrdiff_t submdspan }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="1D dynamic extent all submdspan"> {{{1

template<class Layout>
constexpr bool
dynamic_extent_1d_all_slice() {
  int data[] = {1, 2, 3, 4, 5};
  auto s = stdex::mdspan<
    int, stdex::extents<size_t,stdex::dynamic_extent>, Layout>(data, 5);
  int result = 0;
  auto ss = stdex::submdspan(s, stdex::full_extent);
  for (size_t i = 0; i < s.extent(0); ++i) {
    result += __MDSPAN_OP(ss, i);
  }
  // 1 + 2 + 3 + 4 + 5
  constexpr_assert_equal(15, result);
  return result == 15;
}

MDSPAN_STATIC_TEST(dynamic_extent_1d_all_slice<stdex::layout_left>());
MDSPAN_STATIC_TEST(dynamic_extent_1d_all_slice<stdex::layout_right>());

// </editor-fold> end 1D dynamic extent all submdspan }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="1D dynamic extent pair slice"> {{{1

template<class Layout>
constexpr bool
dynamic_extent_1d_pair_full() {
  int data[] = {1, 2, 3, 4, 5};
  auto s = stdex::mdspan<
    int, stdex::extents<size_t,stdex::dynamic_extent>, Layout>(data, 5);
  int result = 0;
  auto ss = stdex::submdspan(s, std::pair<std::ptrdiff_t, std::ptrdiff_t>{0, 5});
  for (size_t i = 0; i < s.extent(0); ++i) {
    result += __MDSPAN_OP(ss, i);
  }
  constexpr_assert_equal(15, result);
  return result == 15;
}

MDSPAN_STATIC_TEST(dynamic_extent_1d_pair_full<stdex::layout_left>());
MDSPAN_STATIC_TEST(dynamic_extent_1d_pair_full<stdex::layout_right>());

template<class Layout>
constexpr bool
dynamic_extent_1d_pair_each() {
  int data[] = {1, 2, 3, 4, 5};
  auto s = stdex::mdspan<
    int, stdex::extents<size_t,stdex::dynamic_extent>, Layout>(data, 5);
  int result = 0;
  for (size_t i = 0; i < s.extent(0); ++i) {
    auto ss = stdex::submdspan(s,
      std::pair<std::ptrdiff_t, std::ptrdiff_t>{i, i+1});
    result += __MDSPAN_OP(ss, 0);
  }
  constexpr_assert_equal(15, result);
  return result == 15;
}

// MSVC ICE
#ifndef _MDSPAN_COMPILER_MSVC
MDSPAN_STATIC_TEST(dynamic_extent_1d_pair_each<stdex::layout_left>());
MDSPAN_STATIC_TEST(dynamic_extent_1d_pair_each<stdex::layout_right>());
#endif

// </editor-fold> end 1D dynamic extent pair slice submdspan }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="1D dynamic extent pair, all, ptrdiff_t slice"> {{{1

template<class Layout>
constexpr bool
dynamic_extent_1d_all_three() {
  int data[] = {1, 2, 3, 4, 5};
  auto s = stdex::mdspan<
    int, stdex::extents<size_t,stdex::dynamic_extent>, Layout>(data, 5);
  auto s1 = stdex::submdspan(s, std::pair<std::ptrdiff_t, std::ptrdiff_t>{0, 5});
  auto s2 = stdex::submdspan(s1, stdex::full_extent);
  int result = 0;
  for (size_t i = 0; i < s.extent(0); ++i) {
    auto ss = stdex::submdspan(s2, i);
    result += __MDSPAN_OP0(ss);
  }
  constexpr_assert_equal(15, result);
  return result == 15;
}

// MSVC ICE
#ifndef _MDSPAN_COMPILER_MSVC
MDSPAN_STATIC_TEST(dynamic_extent_1d_all_three<stdex::layout_left>());
MDSPAN_STATIC_TEST(dynamic_extent_1d_all_three<stdex::layout_right>());
#endif

// </editor-fold> end 1D dynamic extent pair, all, ptrdifft slice }}}1
//==============================================================================

template<class Layout>
constexpr bool
dynamic_extent_2d_idx_idx() {
  int data[] = { 1, 2, 3, 4, 5, 6 };
  auto s = stdex::mdspan<
    int, stdex::extents<size_t,stdex::dynamic_extent, stdex::dynamic_extent>, Layout>(
      data, 2, 3);
  int result = 0;
  for(size_t row = 0; row < s.extent(0); ++row) {
    for(size_t col = 0; col < s.extent(1); ++col) {
      auto ss = stdex::submdspan(s, row, col);
      result += __MDSPAN_OP0(ss);
    }
  }
  constexpr_assert_equal(21, result);
  return result == 21;
}
MDSPAN_STATIC_TEST(dynamic_extent_2d_idx_idx<stdex::layout_left>());
MDSPAN_STATIC_TEST(dynamic_extent_2d_idx_idx<stdex::layout_right>());

template<class Layout>
constexpr bool
dynamic_extent_2d_idx_all_idx() {
  int data[] = { 1, 2, 3, 4, 5, 6 };
  auto s = stdex::mdspan<
    int, stdex::extents<size_t,stdex::dynamic_extent, stdex::dynamic_extent>, Layout>(
      data, 2, 3);
  int result = 0;
  for(size_t row = 0; row < s.extent(0); ++row) {
    auto srow = stdex::submdspan(s, row, stdex::full_extent);
    for(size_t col = 0; col < s.extent(1); ++col) {
      auto scol = stdex::submdspan(srow, col);
      constexpr_assert_equal(__MDSPAN_OP0(scol), __MDSPAN_OP(srow, col));
      result += __MDSPAN_OP0(scol);
    }
  }
  constexpr_assert_equal(21, result);
  return result == 21;
}

// MSVC ICE
#ifndef _MDSPAN_COMPILER_MSVC
MDSPAN_STATIC_TEST(dynamic_extent_2d_idx_all_idx<stdex::layout_left>());
MDSPAN_STATIC_TEST(dynamic_extent_2d_idx_all_idx<stdex::layout_right>());
#endif

//==============================================================================

constexpr int
simple_static_submdspan_test_1(int add_to_row) {
  int data[] = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  };
  auto s = stdex::mdspan<int, stdex::extents<size_t,3, 3>>(data);
  int result = 0;
  for(int col = 0; col < 3; ++col) {
    auto scol = stdex::submdspan(s, stdex::full_extent, col);
    for(int row = 0; row < 3; ++row) {
      auto srow = stdex::submdspan(scol, row);
      result += __MDSPAN_OP0(srow) * (row + add_to_row);
    }
  }
  return result;
}

// MSVC ICE
#if !defined(_MDSPAN_COMPILER_MSVC) && (!defined(__GNUC__) || (__GNUC__>=6 && __GNUC_MINOR__>=4))
MDSPAN_STATIC_TEST(
  // 1 + 2 + 3 + 2*(4 + 5 + 6) + 3*(7 + 8 + 9) = 108
  simple_static_submdspan_test_1(1) == 108
);

MDSPAN_STATIC_TEST(
  // -1 - 2 - 3 + 7 + 8 + 9 = 18
  simple_static_submdspan_test_1(-1) == 18
);

MDSPAN_STATIC_TEST(
  // -1 - 2 - 3 + 7 + 8 + 9 = 18
  stdex::mdspan<double, stdex::extents<size_t,simple_static_submdspan_test_1(-1)>>{nullptr}.extent(0) == 18
);
#endif

//==============================================================================

constexpr bool
mixed_submdspan_left_test_2() {
  int data[] = {
    1, 4, 7,
    2, 5, 8,
    3, 6, 9,
    0, 0, 0,
    0, 0, 0
  };
  auto s = stdex::mdspan<int,
    stdex::extents<size_t,3, stdex::dynamic_extent>, stdex::layout_left>(data, 5);
  int result = 0;
  for(int col = 0; col < 5; ++col) {
    auto scol = stdex::submdspan(s, stdex::full_extent, col);
    for(int row = 0; row < 3; ++row) {
      auto srow = stdex::submdspan(scol, row);
      result += __MDSPAN_OP0(srow) * (row + 1);
    }
  }
  // 1 + 2 + 3 + 2*(4 + 5 + 6) + 3*(7 + 8 + 9)= 108
  constexpr_assert_equal(108, result);
  for(int row = 0; row < 3; ++row) {
    auto srow = stdex::submdspan(s, row, stdex::full_extent);
    for(int col = 0; col < 5; ++col) {
      auto scol = stdex::submdspan(srow, col);
      result += __MDSPAN_OP0(scol) * (row + 1);
    }
  }
  result /= 2;
  // 2 * (1 + 2 + 3 + 2*(4 + 5 + 6) + 3*(7 + 8 + 9)) / 2 = 108
  constexpr_assert_equal(108, result);
  return result == 108;
}

// MSVC ICE
#if !defined(_MDSPAN_COMPILER_MSVC) && (!defined(__GNUC__) || (__GNUC__>=6 && __GNUC_MINOR__>=4))
MDSPAN_STATIC_TEST(
  // 2 * (1 + 2 + 3 + 2*(4 + 5 + 6) + 3*(7 + 8 + 9)) / 2 = 108
  mixed_submdspan_left_test_2()
);
#endif

//==============================================================================

template <class Layout>
constexpr bool
mixed_submdspan_test_3() {
  int data[] = {
    1, 4, 7, 2, 5,
    8, 3, 6, 9, 0,
    0, 0, 0, 0, 0
  };
  auto s = stdex::mdspan<
    int, stdex::extents<size_t,3, stdex::dynamic_extent>, Layout>(data, 5);
  int result = 0;
  for(int col = 0; col < 5; ++col) {
    auto scol = stdex::submdspan(s, stdex::full_extent, col);
    for(int row = 0; row < 3; ++row) {
      auto srow = stdex::submdspan(scol, row);
      result += __MDSPAN_OP0(srow) * (row + 1);
    }
  }
  constexpr_assert_equal(71, result);
  for(int row = 0; row < 3; ++row) {
    auto srow = stdex::submdspan(s, row, stdex::full_extent);
    for(int col = 0; col < 5; ++col) {
      auto scol = stdex::submdspan(srow, col);
      result += __MDSPAN_OP0(scol) * (row + 1);
    }
  }
  result /= 2;
  // 2 * (1 + 4 + 7 + 2 + 5 + 2*(8 + 3 + 6 + 9)) / 2 = 71
  constexpr_assert_equal(71, result);
  return result == 71;
}

// MSVC ICE
#ifndef _MDSPAN_COMPILER_MSVC
MDSPAN_STATIC_TEST(
  mixed_submdspan_test_3<stdex::layout_right>()
);
#endif

//==============================================================================

#if defined(MDSPAN_ENABLE_EXPENSIVE_COMPILATION_TESTS) && MDSPAN_ENABLE_EXPENSIVE_COMPILATION_TESTS

template <ptrdiff_t Val, size_t Idx>
constexpr auto _repeated_ptrdiff_t = Val;
template <class T, size_t Idx>
using _repeated_with_idxs_t = T;

template <class Layout, size_t... Idxs>
constexpr bool
submdspan_single_element_stress_test_impl_2(
  std::integer_sequence<size_t, Idxs...>
) {
  using mdspan_t = stdex::mdspan<
    int, stdex::extents<size_t,_repeated_ptrdiff_t<1, Idxs>...>, Layout>;
  using dyn_mdspan_t = stdex::mdspan<
    int, stdex::extents<size_t,_repeated_ptrdiff_t<stdex::dynamic_extent, Idxs>...>, Layout>;
  int data[] = { 42 };
  auto s = mdspan_t(data);
  auto s_dyn = dyn_mdspan_t(data, _repeated_ptrdiff_t<1, Idxs>...);
  auto ss = stdex::submdspan(s, _repeated_ptrdiff_t<0, Idxs>...);
  auto ss_dyn = stdex::submdspan(s_dyn, _repeated_ptrdiff_t<0, Idxs>...);
  auto ss_all = stdex::submdspan(s, _repeated_with_idxs_t<stdex::full_extent_t, Idxs>{}...);
  auto ss_all_dyn = stdex::submdspan(s_dyn, _repeated_with_idxs_t<stdex::full_extent_t, Idxs>{}...);
  auto val = __MDSPAN_OP(ss_all, (_repeated_ptrdiff_t<0, Idxs>...));
  auto val_dyn = __MDSPAN_OP(ss_all_dyn, (_repeated_ptrdiff_t<0, Idxs>...));
  auto ss_pair = stdex::submdspan(s, _repeated_with_idxs_t<std::pair<ptrdiff_t, ptrdiff_t>, Idxs>{0, 1}...);
  auto ss_pair_dyn = stdex::submdspan(s_dyn, _repeated_with_idxs_t<std::pair<ptrdiff_t, ptrdiff_t>, Idxs>{0, 1}...);
  auto val_pair = __MDSPAN_OP(ss_pair, (_repeated_ptrdiff_t<0, Idxs>...));
  auto val_pair_dyn = __MDSPAN_OP(ss_pair_dyn, (_repeated_ptrdiff_t<0, Idxs>...));
  constexpr_assert_equal(42, ss());
  constexpr_assert_equal(42, ss_dyn());
  constexpr_assert_equal(42, val);
  constexpr_assert_equal(42, val_dyn);
  constexpr_assert_equal(42, val_pair);
  constexpr_assert_equal(42, val_pair_dyn);
  return __MDSPAN_OP0(ss) == 42 && __MDSPAN_OP0(ss_dyn) == 42 && val == 42 && val_dyn == 42 && val_pair == 42 && val_pair_dyn == 42;
}

template <class Layout, size_t... Sizes>
constexpr bool
submdspan_single_element_stress_test_impl_1(
  std::integer_sequence<size_t, Sizes...>
) {
  return _MDSPAN_FOLD_AND(
    submdspan_single_element_stress_test_impl_2<Layout>(
      std::make_index_sequence<Sizes>{}
    ) /* && ... */
  );
}

template <class Layout, size_t N>
constexpr bool
submdspan_single_element_stress_test() {
  return submdspan_single_element_stress_test_impl_1<Layout>(
    std::make_index_sequence<N+2>{}
  );
}

MDSPAN_STATIC_TEST(
  submdspan_single_element_stress_test<stdex::layout_left, 15>()
);
MDSPAN_STATIC_TEST(
  submdspan_single_element_stress_test<stdex::layout_right, 15>()
);

#endif // MDSPAN_DISABLE_EXPENSIVE_COMPILATION_TESTS

#endif // defined(_MDSPAN_USE_CONSTEXPR_14) && _MDSPAN_USE_CONSTEXPR_14
