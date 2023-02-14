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

#include <experimental/mdspan>

#include <gtest/gtest.h>

namespace stdex = std::experimental;
_MDSPAN_INLINE_VARIABLE constexpr auto dyn = stdex::dynamic_extent;

template <class> struct TestLayoutCtors;
template <class Mapping, size_t... DynamicSizes>
struct TestLayoutCtors<std::tuple<
  Mapping,
  std::integer_sequence<size_t, DynamicSizes...>
>> : public ::testing::Test {
  using mapping_type = Mapping;
  using extents_type = typename mapping_type::extents_type;
  Mapping map = { extents_type{ DynamicSizes... } };
};

template <class Extents, size_t... DynamicSizes>
using test_left_type = std::tuple<
  typename stdex::layout_left::template mapping<Extents>,
  std::integer_sequence<size_t, DynamicSizes...>
>;

template <class Extents, size_t... DynamicSizes>
using test_right_type = std::tuple<
  typename stdex::layout_right::template mapping<Extents>,
  std::integer_sequence<size_t, DynamicSizes...>
>;

using layout_test_types =
  ::testing::Types<
    test_left_type<stdex::extents<size_t,10>>,
    test_right_type<stdex::extents<size_t,10>>,
    //----------
    test_left_type<stdex::extents<size_t,dyn>, 10>,
    test_right_type<stdex::extents<size_t,dyn>, 10>,
    //----------
    test_left_type<stdex::extents<size_t,dyn, 10>, 5>,
    test_left_type<stdex::extents<size_t,5, dyn>, 10>,
    test_left_type<stdex::extents<size_t,5, 10>>,
    test_right_type<stdex::extents<size_t,dyn, 10>, 5>,
    test_right_type<stdex::extents<size_t,5, dyn>, 10>,
    test_right_type<stdex::extents<size_t,5, 10>>
  >;

TYPED_TEST_SUITE(TestLayoutCtors, layout_test_types);

TYPED_TEST(TestLayoutCtors, default_ctor) {
  // Default constructor ensures extents() == Extents() is true.
  auto m = typename TestFixture::mapping_type();
  ASSERT_EQ(m.extents(), typename TestFixture::extents_type());
  auto m2 = typename TestFixture::mapping_type{};
  ASSERT_EQ(m2.extents(), typename TestFixture::extents_type{});
  ASSERT_EQ(m, m2);
}

template <class> struct TestLayoutCompatCtors;
template <class Mapping, size_t... DynamicSizes, class Mapping2, size_t... DynamicSizes2>
struct TestLayoutCompatCtors<std::tuple<
  Mapping,
  std::integer_sequence<size_t, DynamicSizes...>,
  Mapping2,
  std::integer_sequence<size_t, DynamicSizes2...>
>> : public ::testing::Test {
  using mapping_type1 = Mapping;
  using mapping_type2 = Mapping2;
  using extents_type1 = std::remove_reference_t<decltype(std::declval<mapping_type1>().extents())>;
  using extents_type2 = std::remove_reference_t<decltype(std::declval<mapping_type2>().extents())>;
  Mapping map1 = { extents_type1{ DynamicSizes... } };
  Mapping2 map2 = { extents_type2{ DynamicSizes2... } };
};

template <class E1, class S1, class E2, class S2>
using test_left_type_compatible = std::tuple<
  typename stdex::layout_left::template mapping<E1>, S1,
  typename stdex::layout_left::template mapping<E2>, S2
>;
template <class E1, class S1, class E2, class S2>
using test_right_type_compatible = std::tuple<
  typename stdex::layout_right::template mapping<E1>, S1,
  typename stdex::layout_right::template mapping<E2>, S2
>;
template <size_t... Ds>
using _sizes = std::integer_sequence<size_t, Ds...>;
template <size_t... Ds>
using _exts = stdex::extents<size_t,Ds...>;

template <template <class, class, class, class> class _test_case_type>
using compatible_layout_test_types =
  ::testing::Types<
    _test_case_type<_exts<dyn>, _sizes<10>, _exts<10>, _sizes<>>,
    //--------------------
    _test_case_type<_exts<dyn, 10>, _sizes<5>, _exts<5, dyn>, _sizes<10>>,
    _test_case_type<_exts<dyn, dyn>, _sizes<5, 10>, _exts<5, dyn>, _sizes<10>>,
    _test_case_type<_exts<dyn, dyn>, _sizes<5, 10>, _exts<dyn, 10>, _sizes<5>>,
    _test_case_type<_exts<dyn, dyn>, _sizes<5, 10>, _exts<5, 10>, _sizes<>>,
    _test_case_type<_exts<5, 10>, _sizes<>, _exts<5, dyn>, _sizes<10>>,
    _test_case_type<_exts<5, 10>, _sizes<>, _exts<dyn, 10>, _sizes<5>>,
    //--------------------
    _test_case_type<_exts<dyn, dyn, 15>, _sizes<5, 10>, _exts<5, dyn, 15>, _sizes<10>>,
    _test_case_type<_exts<5, 10, 15>, _sizes<>, _exts<5, dyn, 15>, _sizes<10>>,
    _test_case_type<_exts<5, 10, 15>, _sizes<>, _exts<dyn, dyn, dyn>, _sizes<5, 10, 15>>
  >;

using left_compatible_test_types = compatible_layout_test_types<test_left_type_compatible>;
using right_compatible_test_types = compatible_layout_test_types<test_right_type_compatible>;

template <class T> struct TestLayoutLeftCompatCtors : TestLayoutCompatCtors<T> { };
template <class T> struct TestLayoutRightCompatCtors : TestLayoutCompatCtors<T> { };

TYPED_TEST_SUITE(TestLayoutLeftCompatCtors, left_compatible_test_types);
TYPED_TEST_SUITE(TestLayoutRightCompatCtors, right_compatible_test_types);

TYPED_TEST(TestLayoutLeftCompatCtors, compatible_construct_1) {
  // The compatible mapping constructor ensures extents() == other.extents() is true.
  auto m1 = typename TestFixture::mapping_type1(this->map2);
  ASSERT_EQ(m1.extents(), this->map2.extents());
}

TYPED_TEST(TestLayoutRightCompatCtors, compatible_construct_1) {
  // The compatible mapping constructor ensures extents() == other.extents() is true.
  auto m1 = typename TestFixture::mapping_type1(this->map2);
  ASSERT_EQ(m1.extents(), this->map2.extents());
}

TYPED_TEST(TestLayoutLeftCompatCtors, compatible_construct_2) {
  // The compatible mapping constructor ensures extents() == other.extents() is true.
  auto m2 = typename TestFixture::mapping_type2(this->map1);
  ASSERT_EQ(m2.extents(), this->map1.extents());
}

TYPED_TEST(TestLayoutRightCompatCtors, compatible_construct_2) {
  // The compatible mapping constructor ensures extents() == other.extents() is true.
  auto m2 = typename TestFixture::mapping_type2(this->map1);
  ASSERT_EQ(m2.extents(), this->map1.extents());
}

TYPED_TEST(TestLayoutLeftCompatCtors, compatible_assign_1) {
  #if MDSPAN_HAS_CXX_17
  if constexpr (std::is_convertible_v<typename TestFixture::mapping_type2, typename TestFixture::mapping_type1>)
    this->map1 = this->map2;
  else
  #endif
    this->map1 = typename TestFixture::mapping_type1(this->map2);
  ASSERT_EQ(this->map1.extents(), this->map2.extents());
}

TYPED_TEST(TestLayoutRightCompatCtors, compatible_assign_1) {
  #if MDSPAN_HAS_CXX_17
  if constexpr (std::is_convertible_v<typename TestFixture::mapping_type2, typename TestFixture::mapping_type1>)
    this->map1 = this->map2;
  else
  #endif
    this->map1 = typename TestFixture::mapping_type1(this->map2);
  ASSERT_EQ(this->map1.extents(), this->map2.extents());
}

TYPED_TEST(TestLayoutLeftCompatCtors, compatible_assign_2) {
  #if MDSPAN_HAS_CXX_17
  if constexpr (std::is_convertible_v<typename TestFixture::mapping_type1, typename TestFixture::mapping_type2>)
    this->map2 = this->map1;
  else
  #endif
    this->map2 = typename TestFixture::mapping_type2(this->map1);
  ASSERT_EQ(this->map1.extents(), this->map2.extents());
}

TYPED_TEST(TestLayoutRightCompatCtors, compatible_assign_2) {
  #if MDSPAN_HAS_CXX_17
  if constexpr (std::is_convertible_v<typename TestFixture::mapping_type1, typename TestFixture::mapping_type2>)
    this->map2 = this->map1;
  else
  #endif
    this->map2 = typename TestFixture::mapping_type2(this->map1);
  ASSERT_EQ(this->map1.extents(), this->map2.extents());
}

TEST(TestLayoutLeftListInitialization, test_layout_left_extent_initialization) {
  stdex::layout_left::mapping<stdex::extents<size_t,dyn, dyn>> m{stdex::dextents<size_t,2>{16, 32}};
  ASSERT_EQ(m.extents().rank(), 2);
  ASSERT_EQ(m.extents().rank_dynamic(), 2);
  ASSERT_EQ(m.extents().extent(0), 16);
  ASSERT_EQ(m.extents().extent(1), 32);
  ASSERT_EQ(m.stride(0), 1);
  ASSERT_EQ(m.stride(1), 16);
  ASSERT_TRUE(m.is_exhaustive());
}

#if defined(_MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
TEST(TestLayoutLeftCTAD, test_layout_left_ctad) {
  stdex::layout_left::mapping m{stdex::extents{16, 32}};
  ASSERT_EQ(m.extents().rank(), 2);
  ASSERT_EQ(m.extents().rank_dynamic(), 2);
  ASSERT_EQ(m.extents().extent(0), 16);
  ASSERT_EQ(m.extents().extent(1), 32);
  ASSERT_EQ(m.stride(0), 1);
  ASSERT_EQ(m.stride(1), 16);
  ASSERT_TRUE(m.is_exhaustive());
}
#endif

TEST(TestLayoutRightListInitialization, test_layout_right_extent_initialization) {
  stdex::layout_right::mapping<stdex::extents<size_t,dyn, dyn>> m{stdex::dextents<size_t,2>{16, 32}};
  ASSERT_EQ(m.extents().rank(), 2);
  ASSERT_EQ(m.extents().rank_dynamic(), 2);
  ASSERT_EQ(m.extents().extent(0), 16);
  ASSERT_EQ(m.extents().extent(1), 32);
  ASSERT_EQ(m.stride(0), 32);
  ASSERT_EQ(m.stride(1), 1);
  ASSERT_TRUE(m.is_exhaustive());
}

#if defined(_MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
TEST(TestLayoutRightCTAD, test_layout_right_ctad) {
  stdex::layout_right::mapping m{stdex::extents{16, 32}};
  ASSERT_EQ(m.extents().rank(), 2);
  ASSERT_EQ(m.extents().rank_dynamic(), 2);
  ASSERT_EQ(m.extents().extent(0), 16);
  ASSERT_EQ(m.extents().extent(1), 32);
  ASSERT_EQ(m.stride(0), 32);
  ASSERT_EQ(m.stride(1), 1);
  ASSERT_TRUE(m.is_exhaustive());
}
#endif
