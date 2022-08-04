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

namespace stdex = std::experimental;

// Only works with newer constexpr
#if defined(_MDSPAN_USE_CONSTEXPR_14) && _MDSPAN_USE_CONSTEXPR_14

//==============================================================================

constexpr int
simple_static_sum_test_1(int add_to_row) {
  int data[] = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  };
  auto s = stdex::mdspan<int, stdex::extents<size_t,3, 3>>(data);
  int result = 0;
  for(int col = 0; col < 3; ++col) {
    for(int row = 0; row < 3; ++row) {
      result += __MDSPAN_OP(s, row, col) * (row + add_to_row);
    }
  }
  return result;
}

MDSPAN_STATIC_TEST(
  // 1 + 2 + 3 + 2*(4 + 5 + 6) + 3*(7 + 8 + 9) = 108
  simple_static_sum_test_1(1) == 108
);

MDSPAN_STATIC_TEST(
  // -1 - 2 - 3 + 7 + 8 + 9 = 18
  simple_static_sum_test_1(-1) == 18
);

#if !defined(__INTEL_COMPILER) || (__INTEL_COMPILER>=1800)
MDSPAN_STATIC_TEST(
  // -1 - 2 - 3 + 7 + 8 + 9 = 18
  stdex::mdspan<double, stdex::extents<size_t,simple_static_sum_test_1(-1)>>{nullptr}.extent(0) == 18
);
#endif

//==============================================================================

constexpr int
simple_test_1d_constexpr_in_type() {
  int data[] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15, 16, 17, 18
  };
  auto s = stdex::mdspan<int, stdex::extents<size_t,simple_static_sum_test_1(-1)>>(data);
  // 4 + 14 + 18 + 1 = 37
  return s[3] + s[13] + s[17] + s[0];
}

MDSPAN_STATIC_TEST(
  simple_test_1d_constexpr_in_type() == 37
);

//==============================================================================

constexpr int
simple_dynamic_sum_test_2(int add_to_row) {
  int data[] = {
    1, 2, 3, 0,
    4, 5, 6, 0,
    7, 8, 9, 0
  };
  auto s = stdex::mdspan<int, stdex::dextents<size_t,2>>(data, 3, 4);
  int result = 0;
  for(int col = 0; col < 3; ++col) {
    for(int row = 0; row < 3; ++row) {
      result += __MDSPAN_OP(s, row, col) * (row + add_to_row);
    }
  }
  return result;
}

MDSPAN_STATIC_TEST(
  // 1 + 2 + 3 + 2*(4 + 5 + 6) + 3*(7 + 8 + 9) = 108
  simple_dynamic_sum_test_2(1) == 108
);

MDSPAN_STATIC_TEST(
  // -1 - 2 - 3 + 7 + 8 + 9 = 18
  simple_dynamic_sum_test_2(-1) == 18
);

//==============================================================================

constexpr int
simple_mixed_layout_left_sum_test_3(int add_to_row) {
  int data[] = {
    1, 4, 7,
    2, 5, 8,
    3, 6, 9,
    0, 0, 0
  };
  auto s = stdex::mdspan<
    int, stdex::extents<size_t,stdex::dynamic_extent, stdex::dynamic_extent>,
    stdex::layout_left
  >(data, 3, 4);
  int result = 0;
  for(int col = 0; col < 3; ++col) {
    for(int row = 0; row < 3; ++row) {
      result += __MDSPAN_OP(s, row, col) * (row + add_to_row);
    }
  }
  return result;
}

MDSPAN_STATIC_TEST(
  // 1 + 2 + 3 + 2*(4 + 5 + 6) + 3*(7 + 8 + 9) = 108
  simple_mixed_layout_left_sum_test_3(1) == 108
);

//==============================================================================

#if defined(MDSPAN_ENABLE_EXPENSIVE_COMPILATION_TESTS) && MDSPAN_ENABLE_EXPENSIVE_COMPILATION_TESTS

template <ptrdiff_t Val, size_t Idx>
constexpr auto _repeated_ptrdiff_t = Val;

template <class Layout, size_t... Idxs>
constexpr bool
multidimensional_single_element_stress_test_impl_2(
  std::integer_sequence<size_t, Idxs...>
) {
  using mdspan_t = stdex::mdspan<
    int, stdex::extents<size_t,_repeated_ptrdiff_t<1, Idxs>...>, Layout>;
  using dyn_mdspan_t = stdex::mdspan<
    int, stdex::extents<size_t,_repeated_ptrdiff_t<stdex::dynamic_extent, Idxs>...>, Layout>;
  int data[] = { 42 };
  auto s = mdspan_t(data);
  auto s_dyn = dyn_mdspan_t(data, _repeated_ptrdiff_t<1, Idxs>...);
  auto val = __MDSPAN_OP(s, _repeated_ptrdiff_t<0, Idxs>...);
  auto val_dyn = __MDSPAN_OP(s_dyn, _repeated_ptrdiff_t<0, Idxs>...);
  constexpr_assert_equal(42, val);
  constexpr_assert_equal(42, val_dyn);
  return val == 42 && val_dyn == 42;
}

template <class Layout, size_t... Sizes>
constexpr bool
multidimensional_single_element_stress_test_impl_1(
  std::integer_sequence<size_t, Sizes...>
) {
  return _MDSPAN_FOLD_AND(
    multidimensional_single_element_stress_test_impl_2<Layout>(
      std::make_index_sequence<Sizes>{}
    ) /* && ... */
  );
}

template <class Layout, size_t N>
constexpr bool
multidimensional_single_element_stress_test() {
  return multidimensional_single_element_stress_test_impl_1<Layout>(
    std::make_index_sequence<N+2>{}
  );
}

MDSPAN_STATIC_TEST(
  multidimensional_single_element_stress_test<stdex::layout_left, 20>()
);
MDSPAN_STATIC_TEST(
  multidimensional_single_element_stress_test<stdex::layout_right, 20>()
);

template <class Layout, size_t Idx1, size_t Idx2>
constexpr bool
stress_test_2d_single_element_stress_test_impl_2(
  std::integral_constant<size_t, Idx1>,
  std::integral_constant<size_t, Idx2>
) {
  using mdspan_t = stdex::mdspan<
    int, stdex::extents<size_t,Idx1, Idx2>, Layout>;
  using dyn_mdspan_1_t = stdex::mdspan<
    int, stdex::extents<size_t,stdex::dynamic_extent, Idx2>, Layout>;
  using dyn_mdspan_2_t = stdex::mdspan<
    int, stdex::extents<size_t,Idx1, stdex::dynamic_extent>, Layout>;
  using dyn_mdspan_t = stdex::mdspan<
    int, stdex::extents<size_t,stdex::dynamic_extent, stdex::dynamic_extent>, Layout>;
  int data[Idx1*Idx2] = { };
  auto s = mdspan_t(data);
  auto s1 = dyn_mdspan_1_t(data, Idx1);
  auto s2 = dyn_mdspan_2_t(data, Idx2);
  auto s12 = dyn_mdspan_t(data, Idx1, Idx2);
  for(ptrdiff_t i = 0; i < Idx1; ++i) {
    for(ptrdiff_t j = 0; j < Idx2; ++j) {
      __MDSPAN_OP(s, i, j) = __MDSPAN_OP(s1, i, j) = __MDSPAN_OP(s2, i, j) = __MDSPAN_OP(s12, i, j) = 1;
    }
  }
  int result = 0;
  for(ptrdiff_t i = 0; i < Idx1; ++i) {
    for(ptrdiff_t j = 0; j < Idx2; ++j) {
      result += __MDSPAN_OP(s, i, j) + __MDSPAN_OP(s1, i, j) + __MDSPAN_OP(s2, i, j) + __MDSPAN_OP(s12, i, j);
    }
  }
  result /= 4;
  constexpr_assert_equal(Idx1*Idx2, result);
  return result == Idx1 * Idx2;
}

template <class Layout, size_t Idx1, size_t... Idxs2>
constexpr bool
stress_test_2d_single_element_stress_test_impl_1(
  std::integral_constant<size_t, Idx1> idx1,
  std::integer_sequence<size_t, Idxs2...>
)
{
  return _MDSPAN_FOLD_AND(
    stress_test_2d_single_element_stress_test_impl_2<Layout>(
      idx1, std::integral_constant<size_t, Idxs2+1>{}
    ) /* && ... */
  );
}

template <class Layout, size_t... Idxs1, size_t... Idxs2>
constexpr bool
stress_test_2d_single_element_stress_test_impl_0(
  std::integer_sequence<size_t, Idxs1...>,
  std::integer_sequence<size_t, Idxs2...> idxs2
)
{
  return _MDSPAN_FOLD_AND(
    stress_test_2d_single_element_stress_test_impl_1<Layout>(
      std::integral_constant<size_t, Idxs1+1>{}, idxs2
    ) /* && ... */
  );
}

template <class Layout, size_t N, size_t M = N>
constexpr bool
stress_test_2d_single_element_stress_test() {
  return stress_test_2d_single_element_stress_test_impl_0<Layout>(
    std::make_index_sequence<N>{},
    std::make_index_sequence<M>{}
  );
}

MDSPAN_STATIC_TEST(
  stress_test_2d_single_element_stress_test<stdex::layout_left, 8, 8>()
);
MDSPAN_STATIC_TEST(
  stress_test_2d_single_element_stress_test<stdex::layout_right, 8, 8>()
);
// Not evaluated at constexpr time to get around limits, but still compiled
static bool _stress_2d_result =
  stress_test_2d_single_element_stress_test<stdex::layout_left, 6, 15>()
  && stress_test_2d_single_element_stress_test<stdex::layout_right, 15, 6>();

#endif // MDSPAN_DISABLE_EXPENSIVE_COMPILATION_TESTS


#endif //defined(_MDSPAN_USE_CONSTEXPR_14) && _MDSPAN_USE_CONSTEXPR_14
