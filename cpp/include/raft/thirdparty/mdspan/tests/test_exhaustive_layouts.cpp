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

#include <tuple>
#include <utility>

namespace stdex = std::experimental;
_MDSPAN_INLINE_VARIABLE constexpr auto dyn = stdex::dynamic_extent;

template <class Extents>
size_t get_expected_mapping(
  typename stdex::layout_left::template mapping<Extents> const& map,
  size_t i, size_t j, size_t k
)
{
  auto const& extents = map.extents();
  return i + j*extents.extent(0) + k*extents.extent(0)*extents.extent(1);
}

template <class Extents>
size_t get_expected_mapping(
  typename stdex::layout_right::template mapping<Extents> const& map,
  size_t i, size_t j, size_t k
)
{
  auto const& extents = map.extents();
  return k + j*extents.extent(2) + i*extents.extent(1)*extents.extent(2);
}

template <class> struct TestLayout;

template <class Mapping, size_t... DynamicSizes>
struct TestLayout<std::tuple<
  Mapping,
  std::integer_sequence<size_t, DynamicSizes...>
>> : public ::testing::Test
{
  Mapping map;

  void initialize_mapping() {
    using extents_type = std::remove_cv_t<std::remove_reference_t<decltype(map.extents())>>;
    map = Mapping(extents_type(DynamicSizes...));
  }

  void SetUp() override {
    initialize_mapping();
  }

  template <class... Index>
  size_t expected_mapping(Index... idxs) const {
    return get_expected_mapping(map, idxs...);
  }
};

template <class Extents, size_t... DynamicSizes>
using test_3d_left_types = std::tuple<
  typename stdex::layout_left::template mapping<Extents>,
  std::integer_sequence<size_t, DynamicSizes...>
>;

template <class Extents, size_t... DynamicSizes>
using test_3d_right_types = std::tuple<
  typename stdex::layout_right::template mapping<Extents>,
  std::integer_sequence<size_t, DynamicSizes...>
>;

using layout_test_types_3d =
  ::testing::Types<
    test_3d_left_types<stdex::extents<size_t,3, 4, 5>>,
    test_3d_left_types<stdex::extents<size_t,5, 4, 3>>,
    test_3d_left_types<stdex::extents<size_t,3, 4, dyn>, 5>,
    test_3d_left_types<stdex::extents<size_t,5, 4, dyn>, 3>,
    test_3d_left_types<stdex::extents<size_t,3, dyn, 5>, 4>,
    test_3d_left_types<stdex::extents<size_t,5, dyn, 3>, 4>,
    test_3d_left_types<stdex::extents<size_t,dyn, 4, 5>, 3>,
    test_3d_left_types<stdex::extents<size_t,dyn, 4, 3>, 5>,
    test_3d_left_types<stdex::extents<size_t,dyn, dyn, 5>, 3, 4>,
    test_3d_left_types<stdex::extents<size_t,dyn, dyn, 3>, 5, 4>,
    test_3d_left_types<stdex::extents<size_t,dyn, 4, dyn>, 3, 5>,
    test_3d_left_types<stdex::extents<size_t,dyn, 4, dyn>, 5, 3>,
    test_3d_left_types<stdex::extents<size_t,3, dyn, dyn>, 4, 5>,
    test_3d_left_types<stdex::extents<size_t,5, dyn, dyn>, 4, 3>,
    test_3d_left_types<stdex::extents<size_t,dyn, dyn, dyn>, 3, 4, 5>,
    test_3d_left_types<stdex::extents<size_t,dyn, dyn, dyn>, 5, 4, 3>,
    test_3d_right_types<stdex::extents<size_t,3, 4, 5>>,
    test_3d_right_types<stdex::extents<size_t,5, 4, 3>>,
    test_3d_right_types<stdex::extents<size_t,3, 4, dyn>, 5>,
    test_3d_right_types<stdex::extents<size_t,5, 4, dyn>, 3>,
    test_3d_right_types<stdex::extents<size_t,3, dyn, 5>, 4>,
    test_3d_right_types<stdex::extents<size_t,5, dyn, 3>, 4>,
    test_3d_right_types<stdex::extents<size_t,dyn, 4, 5>, 3>,
    test_3d_right_types<stdex::extents<size_t,dyn, 4, 3>, 5>,
    test_3d_right_types<stdex::extents<size_t,dyn, dyn, 5>, 3, 4>,
    test_3d_right_types<stdex::extents<size_t,dyn, dyn, 3>, 5, 4>,
    test_3d_right_types<stdex::extents<size_t,dyn, 4, dyn>, 3, 5>,
    test_3d_right_types<stdex::extents<size_t,dyn, 4, dyn>, 5, 3>,
    test_3d_right_types<stdex::extents<size_t,3, dyn, dyn>, 4, 5>,
    test_3d_right_types<stdex::extents<size_t,5, dyn, dyn>, 4, 3>,
    test_3d_right_types<stdex::extents<size_t,dyn, dyn, dyn>, 3, 4, 5>,
    test_3d_right_types<stdex::extents<size_t,dyn, dyn, dyn>, 5, 4, 3>
  >;

template <class T> struct TestLayout3D : TestLayout<T> { };

TYPED_TEST_SUITE(TestLayout3D, layout_test_types_3d);

TYPED_TEST(TestLayout3D, mapping_works) {
  for(size_t i = 0; i < this->map.extents().extent(0); ++i) {
    for(size_t j = 0; j < this->map.extents().extent(1); ++j) {
      for(size_t k = 0; k < this->map.extents().extent(2); ++k) {
        EXPECT_EQ(this->map(i, j, k), this->expected_mapping(i, j, k))
          << "Incorrect mapping for index " << i << ", " << j << ", " << k
          << " with extents "
          << this->map.extents().extent(0) << ", "
          << this->map.extents().extent(1) << ", "
          << this->map.extents().extent(2) << ".";
      }
    }
  }
}

TYPED_TEST(TestLayout3D, required_span_size_works) {
  ASSERT_EQ(this->map.required_span_size(), 3*4*5);
}

template<class Layout1, class Layout2, class Extents1, class Extents2, bool Implicit, size_t ... AllSizes>
using test_layout_conversion =
  std::tuple<Layout1, Layout2, Extents1, Extents2,
             std::integer_sequence<size_t, AllSizes...>,
             std::integral_constant<bool, Implicit>>;

// We separate layout_conversion_test_types into multiple type lists
// in order to avoid an nvc++ compiler error (recursion too deep).

// Left <=> Left conversion
using layout_conversion_test_types_left_to_left =
  ::testing::Types<
     test_layout_conversion<stdex::layout_left , stdex::layout_left , stdex::extents<size_t,dyn>,                 stdex::extents<size_t,5>,               true, 5>
    ,test_layout_conversion<stdex::layout_left,  stdex::layout_left , stdex::extents<size_t>,                    stdex::extents<size_t>,                true>
    ,test_layout_conversion<stdex::layout_left , stdex::layout_left , stdex::extents<size_t,5>,                   stdex::extents<size_t,dyn>,             false, 5>
    ,test_layout_conversion<stdex::layout_left , stdex::layout_left , stdex::extents<size_t,dyn,dyn>,             stdex::extents<size_t,5,10>,            true, 5,10>
    ,test_layout_conversion<stdex::layout_left , stdex::layout_left , stdex::extents<size_t,dyn,dyn>,             stdex::extents<size_t,5,dyn>,           true, 5,0> // intentional 0 here to test degenerated
    ,test_layout_conversion<stdex::layout_left , stdex::layout_left , stdex::extents<size_t,dyn,10>,              stdex::extents<size_t,5,10>,            true, 5,10>
    ,test_layout_conversion<stdex::layout_left , stdex::layout_left , stdex::extents<size_t,5,dyn>,               stdex::extents<size_t,5,10>,            true, 5,10>
    ,test_layout_conversion<stdex::layout_left , stdex::layout_left , stdex::extents<size_t,5,dyn>,               stdex::extents<size_t,5,dyn>,           true, 5,10>
    ,test_layout_conversion<stdex::layout_left , stdex::layout_left , stdex::extents<size_t,dyn,10,15,dyn,25>,    stdex::extents<size_t,5,10,dyn,20,dyn>, false, 5, 10, 15, 20, 25>
    ,test_layout_conversion<stdex::layout_left , stdex::layout_left , stdex::extents<size_t,5,10,15,20,25>,       stdex::extents<size_t,5,10,dyn,20,dyn>, false, 5, 10, 15, 20, 25>
    ,test_layout_conversion<stdex::layout_left , stdex::layout_left , stdex::extents<size_t,dyn,dyn,dyn,dyn,dyn>, stdex::extents<size_t,5,10,dyn,20,dyn>, true, 5, 10, 15, 20, 25>
    ,test_layout_conversion<stdex::layout_left , stdex::layout_left , stdex::extents<size_t,5,10,15,20,25>,       stdex::extents<size_t,5,10,15,20,25>,   true, 5, 10, 15, 20, 25>
  >;

// Right <=> Right conversion
using layout_conversion_test_types_right_to_right =
  ::testing::Types<
    test_layout_conversion<stdex::layout_right, stdex::layout_right, stdex::extents<size_t,dyn>,                 stdex::extents<size_t,5>,               true, 5>
    ,test_layout_conversion<stdex::layout_right, stdex::layout_right, stdex::extents<size_t>,                    stdex::extents<size_t>,                true>
    ,test_layout_conversion<stdex::layout_right, stdex::layout_right, stdex::extents<size_t,5>,                   stdex::extents<size_t,dyn>,             false, 5>
    ,test_layout_conversion<stdex::layout_right, stdex::layout_right, stdex::extents<size_t,dyn,dyn>,             stdex::extents<size_t,5,10>,            true, 5,10>
    ,test_layout_conversion<stdex::layout_right, stdex::layout_right, stdex::extents<size_t,dyn,dyn>,             stdex::extents<size_t,5,dyn>,           true, 5,0> //intentional 0 to test degenerate
    ,test_layout_conversion<stdex::layout_right, stdex::layout_right, stdex::extents<size_t,dyn,10>,              stdex::extents<size_t,5,10>,            true, 5,10>
    ,test_layout_conversion<stdex::layout_right, stdex::layout_right, stdex::extents<size_t,5,dyn>,               stdex::extents<size_t,5,10>,            true, 5,10>
    ,test_layout_conversion<stdex::layout_right, stdex::layout_right, stdex::extents<size_t,5,dyn>,               stdex::extents<size_t,5,dyn>,           true, 5,10>
    ,test_layout_conversion<stdex::layout_right, stdex::layout_right, stdex::extents<size_t,dyn,10,15,dyn,25>,    stdex::extents<size_t,5,10,dyn,20,dyn>, false, 5, 10, 15, 20, 25>
    ,test_layout_conversion<stdex::layout_right, stdex::layout_right, stdex::extents<size_t,5,10,15,20,25>,       stdex::extents<size_t,5,10,dyn,20,dyn>, false, 5, 10, 15, 20, 25>
    ,test_layout_conversion<stdex::layout_right, stdex::layout_right, stdex::extents<size_t,dyn,dyn,dyn,dyn,dyn>, stdex::extents<size_t,5,10,dyn,20,dyn>, true, 5, 10, 15, 20, 25>
    ,test_layout_conversion<stdex::layout_right, stdex::layout_right, stdex::extents<size_t,5,10,15,20,25>,       stdex::extents<size_t,5,10,15,20,25>,   true, 5, 10, 15, 20, 25>
  >;

// Stride <=> Stride conversion
using layout_conversion_test_types_stride_to_stride =
  ::testing::Types<
    test_layout_conversion<stdex::layout_stride, stdex::layout_stride, stdex::extents<size_t,dyn>,                 stdex::extents<size_t,5>,               true, 5>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_stride, stdex::extents<size_t>,                    stdex::extents<size_t>,                true>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_stride, stdex::extents<size_t,5>,                   stdex::extents<size_t,dyn>,             false, 5>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_stride, stdex::extents<size_t,dyn,dyn>,             stdex::extents<size_t,5,10>,            true, 5,10>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_stride, stdex::extents<size_t,dyn,dyn>,             stdex::extents<size_t,5,dyn>,           true, 5,0> //intentional 0 to test degenerate
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_stride, stdex::extents<size_t,dyn,10>,              stdex::extents<size_t,5,10>,            true, 5,10>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_stride, stdex::extents<size_t,5,dyn>,               stdex::extents<size_t,5,10>,            true, 5,10>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_stride, stdex::extents<size_t,5,dyn>,               stdex::extents<size_t,5,dyn>,           true, 5,10>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_stride, stdex::extents<size_t,dyn,10,15,dyn,25>,    stdex::extents<size_t,5,10,dyn,20,dyn>, false, 5, 10, 15, 20, 25>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_stride, stdex::extents<size_t,5,10,15,20,25>,       stdex::extents<size_t,5,10,dyn,20,dyn>, false, 5, 10, 15, 20, 25>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_stride, stdex::extents<size_t,dyn,dyn,dyn,dyn,dyn>, stdex::extents<size_t,5,10,dyn,20,dyn>, true, 5, 10, 15, 20, 25>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_stride, stdex::extents<size_t,5,10,15,20,25>,       stdex::extents<size_t,5,10,15,20,25>,   true, 5, 10, 15, 20, 25>
  >;

// Right => Stride conversion
using layout_conversion_test_types_right_to_stride =
  ::testing::Types<
    test_layout_conversion<stdex::layout_stride, stdex::layout_right, stdex::extents<size_t,dyn>,                 stdex::extents<size_t,5>,               true, 5>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_right, stdex::extents<size_t>,                    stdex::extents<size_t>,                true>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_right, stdex::extents<size_t,5>,                   stdex::extents<size_t,dyn>,             false, 5>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_right, stdex::extents<size_t,dyn,dyn>,             stdex::extents<size_t,5,10>,            true, 5,10>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_right, stdex::extents<size_t,dyn,dyn>,             stdex::extents<size_t,5,dyn>,           true, 5,0> //intentional 0 to test degenerate
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_right, stdex::extents<size_t,dyn,10>,              stdex::extents<size_t,5,10>,            true, 5,10>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_right, stdex::extents<size_t,5,dyn>,               stdex::extents<size_t,5,10>,            true, 5,10>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_right, stdex::extents<size_t,5,dyn>,               stdex::extents<size_t,5,dyn>,           true, 5,10>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_right, stdex::extents<size_t,dyn,10,15,dyn,25>,    stdex::extents<size_t,5,10,dyn,20,dyn>, false, 5, 10, 15, 20, 25>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_right, stdex::extents<size_t,5,10,15,20,25>,       stdex::extents<size_t,5,10,dyn,20,dyn>, false, 5, 10, 15, 20, 25>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_right, stdex::extents<size_t,dyn,dyn,dyn,dyn,dyn>, stdex::extents<size_t,5,10,dyn,20,dyn>, true, 5, 10, 15, 20, 25>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_right, stdex::extents<size_t,5,10,15,20,25>,       stdex::extents<size_t,5,10,15,20,25>,   true, 5, 10, 15, 20, 25>
  >;

// Left => Stride conversion
using layout_conversion_test_types_left_to_stride =
  ::testing::Types<
    test_layout_conversion<stdex::layout_stride, stdex::layout_left , stdex::extents<size_t,dyn>,                 stdex::extents<size_t,5>,               true, 5>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_left , stdex::extents<size_t>,                    stdex::extents<size_t>,                true>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_left , stdex::extents<size_t,5>,                   stdex::extents<size_t,dyn>,             false, 5>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_left , stdex::extents<size_t,dyn,dyn>,             stdex::extents<size_t,5,10>,            true, 5,10>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_left , stdex::extents<size_t,dyn,dyn>,             stdex::extents<size_t,5,dyn>,           true, 5,0> //intentional 0 to test degenerate
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_left , stdex::extents<size_t,dyn,10>,              stdex::extents<size_t,5,10>,            true, 5,10>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_left , stdex::extents<size_t,5,dyn>,               stdex::extents<size_t,5,10>,            true, 5,10>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_left , stdex::extents<size_t,5,dyn>,               stdex::extents<size_t,5,dyn>,           true, 5,10>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_left , stdex::extents<size_t,dyn,10,15,dyn,25>,    stdex::extents<size_t,5,10,dyn,20,dyn>, false, 5, 10, 15, 20, 25>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_left , stdex::extents<size_t,5,10,15,20,25>,       stdex::extents<size_t,5,10,dyn,20,dyn>, false, 5, 10, 15, 20, 25>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_left , stdex::extents<size_t,dyn,dyn,dyn,dyn,dyn>, stdex::extents<size_t,5,10,dyn,20,dyn>, true, 5, 10, 15, 20, 25>
    ,test_layout_conversion<stdex::layout_stride, stdex::layout_left , stdex::extents<size_t,5,10,15,20,25>,       stdex::extents<size_t,5,10,15,20,25>,   true, 5, 10, 15, 20, 25>
  >;

// Stride => Left conversion
using layout_conversion_test_types_stride_to_left =
  ::testing::Types<
    test_layout_conversion<stdex::layout_left  , stdex::layout_stride, stdex::extents<size_t,dyn>,                 stdex::extents<size_t,5>,               false, 5>
    ,test_layout_conversion<stdex::layout_left  , stdex::layout_stride, stdex::extents<size_t>,                    stdex::extents<size_t>,                true>
    ,test_layout_conversion<stdex::layout_left  , stdex::layout_stride, stdex::extents<size_t,5>,                   stdex::extents<size_t,dyn>,             false, 5>
    ,test_layout_conversion<stdex::layout_left  , stdex::layout_stride, stdex::extents<size_t,dyn,dyn>,             stdex::extents<size_t,5,10>,            false, 5,10>
    ,test_layout_conversion<stdex::layout_left  , stdex::layout_stride, stdex::extents<size_t,dyn,dyn>,             stdex::extents<size_t,5,dyn>,           false, 5,0> //intentional 0 to test degenerate
    ,test_layout_conversion<stdex::layout_left  , stdex::layout_stride, stdex::extents<size_t,dyn,10>,              stdex::extents<size_t,5,10>,            false, 5,10>
    ,test_layout_conversion<stdex::layout_left  , stdex::layout_stride, stdex::extents<size_t,5,dyn>,               stdex::extents<size_t,5,10>,            false, 5,10>
    ,test_layout_conversion<stdex::layout_left  , stdex::layout_stride, stdex::extents<size_t,5,dyn>,               stdex::extents<size_t,5,dyn>,           false, 5,10>
    ,test_layout_conversion<stdex::layout_left  , stdex::layout_stride, stdex::extents<size_t,dyn,10,15,dyn,25>,    stdex::extents<size_t,5,10,dyn,20,dyn>, false, 5, 10, 15, 20, 25>
    ,test_layout_conversion<stdex::layout_left  , stdex::layout_stride, stdex::extents<size_t,5,10,15,20,25>,       stdex::extents<size_t,5,10,dyn,20,dyn>, false, 5, 10, 15, 20, 25>
    ,test_layout_conversion<stdex::layout_left  , stdex::layout_stride, stdex::extents<size_t,dyn,dyn,dyn,dyn,dyn>, stdex::extents<size_t,5,10,dyn,20,dyn>, false, 5, 10, 15, 20, 25>
    ,test_layout_conversion<stdex::layout_left  , stdex::layout_stride, stdex::extents<size_t,5,10,15,20,25>,       stdex::extents<size_t,5,10,15,20,25>,   false, 5, 10, 15, 20, 25>
  >;

// Stride => Right conversion
using layout_conversion_test_types_stride_to_right =
  ::testing::Types<
    test_layout_conversion<stdex::layout_right , stdex::layout_stride, stdex::extents<size_t,dyn>,                 stdex::extents<size_t,5>,               false, 5>
    ,test_layout_conversion<stdex::layout_right , stdex::layout_stride, stdex::extents<size_t>,                    stdex::extents<size_t>,                true>
    ,test_layout_conversion<stdex::layout_right , stdex::layout_stride, stdex::extents<size_t,5>,                   stdex::extents<size_t,dyn>,             false, 5>
    ,test_layout_conversion<stdex::layout_right , stdex::layout_stride, stdex::extents<size_t,dyn,dyn>,             stdex::extents<size_t,5,10>,            false, 5,10>
    ,test_layout_conversion<stdex::layout_right , stdex::layout_stride, stdex::extents<size_t,dyn,dyn>,             stdex::extents<size_t,5,dyn>,           false, 5,0> //intentional 0 to test degenerate
    ,test_layout_conversion<stdex::layout_right , stdex::layout_stride, stdex::extents<size_t,dyn,10>,              stdex::extents<size_t,5,10>,            false, 5,10>
    ,test_layout_conversion<stdex::layout_right , stdex::layout_stride, stdex::extents<size_t,5,dyn>,               stdex::extents<size_t,5,10>,            false, 5,10>
    ,test_layout_conversion<stdex::layout_right , stdex::layout_stride, stdex::extents<size_t,5,dyn>,               stdex::extents<size_t,5,dyn>,           false, 5,10>
    ,test_layout_conversion<stdex::layout_right , stdex::layout_stride, stdex::extents<size_t,dyn,10,15,dyn,25>,    stdex::extents<size_t,5,10,dyn,20,dyn>, false, 5, 10, 15, 20, 25>
    ,test_layout_conversion<stdex::layout_right , stdex::layout_stride, stdex::extents<size_t,5,10,15,20,25>,       stdex::extents<size_t,5,10,dyn,20,dyn>, false, 5, 10, 15, 20, 25>
    ,test_layout_conversion<stdex::layout_right , stdex::layout_stride, stdex::extents<size_t,dyn,dyn,dyn,dyn,dyn>, stdex::extents<size_t,5,10,dyn,20,dyn>, false, 5, 10, 15, 20, 25>
    ,test_layout_conversion<stdex::layout_right , stdex::layout_stride, stdex::extents<size_t,5,10,15,20,25>,       stdex::extents<size_t,5,10,15,20,25>,   false, 5, 10, 15, 20, 25>
  >;

// Left <=> Right conversion
using layout_conversion_test_types_left_to_right =
  ::testing::Types<
    test_layout_conversion<stdex::layout_right, stdex::layout_left , stdex::extents<size_t,dyn>,                 stdex::extents<size_t,5>,               true, 5>
    ,test_layout_conversion<stdex::layout_right, stdex::layout_left , stdex::extents<size_t,5>,                   stdex::extents<size_t,dyn>,             false, 5>
    ,test_layout_conversion<stdex::layout_left,  stdex::layout_right, stdex::extents<size_t,dyn>,                 stdex::extents<size_t,5>,               true, 5>
    ,test_layout_conversion<stdex::layout_left,  stdex::layout_right, stdex::extents<size_t,5>,                   stdex::extents<size_t,dyn>,             false, 5>
    ,test_layout_conversion<stdex::layout_right, stdex::layout_left , stdex::extents<size_t>,                    stdex::extents<size_t>,                true>
    ,test_layout_conversion<stdex::layout_left,  stdex::layout_right, stdex::extents<size_t>,                    stdex::extents<size_t>,                true>
  >;

template<class T> struct TestLayoutConversion;

template<class Layout1, class Layout2, class Extents1, class Extents2, bool Implicit, size_t ... AllSizes>
struct TestLayoutConversion<
  std::tuple<Layout1,
             Layout2,
             Extents1,
             Extents2,
             std::integer_sequence<size_t, AllSizes...>,
             std::integral_constant<bool, Implicit>>>  : public ::testing::Test {
  static constexpr bool implicit = Implicit;
  using layout_1_t = Layout1;
  using exts_1_t = Extents1;
  using map_1_t = typename Layout1::template mapping<exts_1_t>;
  using layout_2_t = Layout2;
  using exts_2_t = Extents2;
  using map_2_t = typename Layout2::template mapping<exts_2_t>;
  exts_1_t exts1 { AllSizes... };
  exts_2_t exts2 { AllSizes... };

  map_1_t implicit_conv(map_1_t map) const {
    return map;
  }

  template<class Extents>
  static constexpr std::enable_if_t<
    std::is_same<Extents, Extents2>::value &&
   !std::is_same<Layout1, stdex::layout_stride>::value &&
    std::is_same<Layout2, stdex::layout_stride>::value,
    map_2_t> create_map2(Extents exts) {
      return map_2_t(map_1_t(Extents1(exts)));
  }

  template<class Extents>
  static constexpr std::enable_if_t<
    std::is_same<Extents, Extents2>::value &&
    std::is_same<Layout1, stdex::layout_stride>::value &&
    std::is_same<Layout2, stdex::layout_stride>::value,
    map_2_t> create_map2(Extents exts) {
      return map_2_t(typename stdex::layout_left::template mapping<Extents>(exts));
  }

  template<class Extents>
  static constexpr std::enable_if_t<
     std::is_same<Extents,Extents2>::value &&
    !std::is_same<Layout2,stdex::layout_stride>::value,
    map_2_t> create_map2(Extents exts) {
      return map_2_t(exts);
  }
};

// We can't reuse TestLayoutConversion directly for multiple suites,
// because this results in duplicate test names.
// However, creating an alias for each suite works around this.

TYPED_TEST_SUITE(TestLayoutConversion, layout_conversion_test_types_left_to_left);

template<class T>
using TestLayoutConversion_R2R = TestLayoutConversion<T>;
TYPED_TEST_SUITE(TestLayoutConversion_R2R, layout_conversion_test_types_right_to_right);

template<class T>
using TestLayoutConversion_S2S = TestLayoutConversion<T>;
TYPED_TEST_SUITE(TestLayoutConversion_S2S, layout_conversion_test_types_stride_to_stride);

template<class T>
using TestLayoutConversion_R2S = TestLayoutConversion<T>;
TYPED_TEST_SUITE(TestLayoutConversion_R2S, layout_conversion_test_types_right_to_stride);

template<class T>
using TestLayoutConversion_L2S = TestLayoutConversion<T>;
TYPED_TEST_SUITE(TestLayoutConversion_L2S, layout_conversion_test_types_left_to_stride);

template<class T>
using TestLayoutConversion_S2L = TestLayoutConversion<T>;
TYPED_TEST_SUITE(TestLayoutConversion_S2L, layout_conversion_test_types_stride_to_left);

template<class T>
using TestLayoutConversion_S2R = TestLayoutConversion<T>;
TYPED_TEST_SUITE(TestLayoutConversion_S2R, layout_conversion_test_types_stride_to_right);

template<class T>
using TestLayoutConversion_L2R = TestLayoutConversion<T>;
TYPED_TEST_SUITE(TestLayoutConversion_L2R, layout_conversion_test_types_left_to_right);

TYPED_TEST(TestLayoutConversion, implicit_conversion) {
  typename TestFixture::map_2_t map2 = this->create_map2(this->exts2);
  typename TestFixture::map_1_t map1;
  #if MDSPAN_HAS_CXX_20 && !defined(_MDSPAN_COMPILER_MSVC)
  static_assert(TestFixture::implicit ==
   (
    (
     !std::is_same<typename TestFixture::layout_2_t, stdex::layout_stride>::value &&
     std::is_convertible<typename TestFixture::exts_2_t, typename TestFixture::exts_1_t>::value
    ) || (
     std::is_same<typename TestFixture::layout_2_t, stdex::layout_stride>::value &&
     std::is_same<typename TestFixture::layout_1_t, stdex::layout_stride>::value &&
     std::is_convertible<typename TestFixture::exts_2_t, typename TestFixture::exts_1_t>::value
    )|| (
     std::is_same<typename TestFixture::layout_2_t, stdex::layout_stride>::value &&
     !std::is_same<typename TestFixture::layout_1_t, stdex::layout_stride>::value &&
     TestFixture::exts_2_t::rank()==0
    )
   )
  );
  static_assert(std::is_convertible<typename TestFixture::map_2_t, typename TestFixture::map_1_t>::value ==
                TestFixture::implicit);
  if constexpr(TestFixture::implicit)
    map1 = this->implicit_conv(map2);
  else
  #endif
    map1 = typename TestFixture::map_1_t(map2);

  for(size_t r=0; r != this->exts1.rank(); ++r) {
    ASSERT_EQ(map1.extents().extent(r), map2.extents().extent(r));
    ASSERT_EQ(map1.stride(r), map2.stride(r));
  }
}

