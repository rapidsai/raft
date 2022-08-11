/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2021) Sandia Corporation
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
#include "offload_utils.hpp"
#include <gtest/gtest.h>

namespace stdex = std::experimental;

// Making actual test implementations part of the class,
// can't create CUDA lambdas in the primary test functions, since they are private
// making a non-member function would require replicating all the type info

template <class> struct TestExtents;
template <size_t... Extents, size_t... DynamicSizes>
struct TestExtents<
  std::tuple<
    stdex::extents<size_t, Extents...>,
    std::integer_sequence<size_t, DynamicSizes...>
  >
> : public ::testing::Test {
  using extents_type = stdex::extents<size_t,Extents...>;
  // Double Braces here to make it work with GCC 5
  // Otherwise: "error: array must be initialized with a brace-enclosed initializer"
  const std::array<size_t, sizeof...(Extents)> static_sizes {{ Extents... }};
  const std::array<size_t, sizeof...(DynamicSizes)> dyn_sizes {{ DynamicSizes... }};
  extents_type exts { DynamicSizes... };
  using Fixture = TestExtents< std::tuple<
                    stdex::extents<size_t,Extents...>,
                    std::integer_sequence<size_t, DynamicSizes...>
                  >>;

  void test_rank() {
    size_t* result = allocate_array<size_t>(2);

    dispatch([=] _MDSPAN_HOST_DEVICE () {
      extents_type _exts(DynamicSizes...);
      // Silencing an unused warning in nvc++ the condition will never be true
      size_t dyn_val = exts.rank()>0?static_cast<size_t>(_exts.extent(0)):1;
      result[0] = dyn_val > 1e9 ? dyn_val : _exts.rank();
      result[1] = _exts.rank_dynamic();
      // Some compilers warn about unused _exts since the functions are all static constexpr
      (void) _exts;
    });
    EXPECT_EQ(result[0], static_sizes.size());
    EXPECT_EQ(result[1], dyn_sizes.size());

    free_array(result);
  }

  void test_static_extent() {
    size_t* result = allocate_array<size_t>(extents_type::rank());

    dispatch([=] _MDSPAN_HOST_DEVICE () {
      extents_type _exts(DynamicSizes...);
      for(size_t r=0; r<_exts.rank(); r++) {
        // Silencing an unused warning in nvc++ the condition will never be true
        size_t dyn_val = static_cast<size_t>(_exts.extent(r));
        result[r] = dyn_val > 1e9 ? dyn_val : _exts.static_extent(r);
      }
      // Some compilers warn about unused _exts since the functions are all static constexpr
      (void) _exts;
    });
    for(size_t r=0; r<extents_type::rank(); r++) {
      EXPECT_EQ(result[r], static_sizes[r]);
    }

    free_array(result);
  }

  void test_extent() {
    size_t* result = allocate_array<size_t>(extents_type::rank());

    dispatch([=] _MDSPAN_HOST_DEVICE () {
      extents_type _exts(DynamicSizes...);
      for(size_t r=0; r<_exts.rank(); r++ )
        result[r] = _exts.extent(r);
      // Some compilers warn about unused _exts since the functions are all static constexpr
      (void) _exts;
    });
    int dyn_count = 0;
    for(size_t r=0; r<extents_type::rank(); r++) {
      bool is_dynamic = static_sizes[r] == stdex::dynamic_extent;
      auto expected = is_dynamic ? dyn_sizes[dyn_count++] : static_sizes[r];
      EXPECT_EQ(result[r], expected);
    }

    free_array(result);
  }
};

template <size_t... Ds>
using _sizes = std::integer_sequence<size_t, Ds...>;
template <size_t... Ds>
using _exts = stdex::extents<size_t,Ds...>;

using extents_test_types =
  ::testing::Types<
    std::tuple<_exts<10>, _sizes<>>,
    std::tuple<_exts<stdex::dynamic_extent>, _sizes<10>>,
    std::tuple<_exts<10, 3>, _sizes<>>,
    std::tuple<_exts<stdex::dynamic_extent, 3>, _sizes<10>>,
    std::tuple<_exts<10, stdex::dynamic_extent>, _sizes<3>>,
    std::tuple<_exts<stdex::dynamic_extent, stdex::dynamic_extent>, _sizes<10, 3>>
  >;

TYPED_TEST_SUITE(TestExtents, extents_test_types);

TYPED_TEST(TestExtents, rank) {
  __MDSPAN_TESTS_RUN_TEST(this->test_rank())
}

TYPED_TEST(TestExtents, static_extent) {
  __MDSPAN_TESTS_RUN_TEST(this->test_static_extent())
}

TYPED_TEST(TestExtents, extent) {
  __MDSPAN_TESTS_RUN_TEST(this->test_extent())
}

TYPED_TEST(TestExtents, default_ctor) {
  auto e = typename TestFixture::extents_type();
  auto e2 = typename TestFixture::extents_type{};
  EXPECT_EQ(e, e2);
  for (size_t r = 0; r < e.rank(); ++r) {
    bool is_dynamic = (e.static_extent(r) == stdex::dynamic_extent);
    EXPECT_EQ(e.extent(r), is_dynamic ? 0 : e.static_extent(r));
  }
}

TYPED_TEST(TestExtents, array_ctor) {
  auto e = typename TestFixture::extents_type(this->dyn_sizes);
  EXPECT_EQ(e, this->exts);
}

TYPED_TEST(TestExtents, copy_ctor) {
  typename TestFixture::extents_type e { this->exts };
  EXPECT_EQ(e, this->exts);
}

TYPED_TEST(TestExtents, copy_assign) {
  typename TestFixture::extents_type e;
  e = this->exts;
  EXPECT_EQ(e, this->exts);
}

// Only types can be part of the gtest type iteration
// so i need this wrapper to wrap around true/false values
template<bool Val1, bool Val2>
struct _BoolPairDeducer {
  static constexpr bool val1 = Val1;
  static constexpr bool val2 = Val2;
};


template <class> struct TestExtentsCompatCtors;
template <size_t... Extents, size_t... DynamicSizes, size_t... Extents2, size_t... DynamicSizes2, bool ImplicitExts1ToExts2, bool ImplicitExts2ToExts1>
struct TestExtentsCompatCtors<std::tuple<
  stdex::extents<size_t,Extents...>,
  std::integer_sequence<size_t, DynamicSizes...>,
  stdex::extents<size_t,Extents2...>,
  std::integer_sequence<size_t, DynamicSizes2...>,
  _BoolPairDeducer<ImplicitExts1ToExts2,ImplicitExts2ToExts1>
>> : public ::testing::Test {
  using extents_type1 = stdex::extents<size_t,Extents...>;
  using extents_type2 = stdex::extents<size_t,Extents2...>;
  extents_type1 exts1 { DynamicSizes... };
  extents_type2 exts2 { DynamicSizes2... };
  static constexpr bool implicit_exts1_to_exts2 = ImplicitExts1ToExts2;
  static constexpr bool implicit_exts2_to_exts1 = ImplicitExts2ToExts1;

  // Idx starts at rank()
  template<class Exts, size_t Idx, bool construct_all, bool Static>
  struct make_extents;

  // Construct all extents - doesn't matter if idx is static or not
  template<class Exts, size_t Idx, bool Static>
  struct make_extents<Exts,Idx,true,Static>
  {
    template<class ... SizeTypes>
    static Exts construct(SizeTypes...sizes) {
      return make_extents<Exts, Idx-1,true,Static>::construct(Idx*5, sizes...);
    }
  };

  // Construct dynamic extents - opnly add a parameter if the current index is not static
  template<class Exts, size_t Idx>
  struct make_extents<Exts,Idx,false,false>
  {
    template<class ... SizeTypes>
    static Exts construct(SizeTypes...sizes) {
      return make_extents<Exts, Idx-1,false,(Idx>1?Exts::static_extent(Idx-2)!=stdex::dynamic_extent:true)>::construct(Idx*5, sizes...);
    }
  };
  template<class Exts, size_t Idx>
  struct make_extents<Exts,Idx,false,true>
  {
    template<class ... SizeTypes>
    static Exts construct(SizeTypes...sizes) {
      return make_extents<Exts, Idx-1,false,(Idx>1?Exts::static_extent(Idx-2)!=stdex::dynamic_extent:true)>::construct(sizes...);
    }
  };

  // End case
  template<class Exts>
  struct make_extents<Exts,0,false,true> {
    template<class ... SizeTypes>
    static Exts construct(SizeTypes...sizes) {
       return Exts{sizes...};
    }
  };
  template<class Exts>
  struct make_extents<Exts,0,true,true> {
    template<class ... SizeTypes>
    static Exts construct(SizeTypes...sizes) {
       return Exts{sizes...};
    }
  };
};

using compatible_extents_test_types =
  ::testing::Types<
    std::tuple<_exts<stdex::dynamic_extent>, _sizes<5>, _exts<5>, _sizes<>, _BoolPairDeducer<false,true>>,
    std::tuple<_exts<5>, _sizes<>, _exts<stdex::dynamic_extent>, _sizes<5>, _BoolPairDeducer<true,false>>,
    //--------------------
    std::tuple<_exts<stdex::dynamic_extent, 10>, _sizes<5>, _exts<5, stdex::dynamic_extent>, _sizes<10>,  _BoolPairDeducer<false, false>>,
    std::tuple<_exts<stdex::dynamic_extent, stdex::dynamic_extent>, _sizes<5, 10>, _exts<5, stdex::dynamic_extent>, _sizes<10>,  _BoolPairDeducer<false, true>>,
    std::tuple<_exts<stdex::dynamic_extent, stdex::dynamic_extent>, _sizes<5, 10>, _exts<stdex::dynamic_extent, 10>, _sizes<5>,  _BoolPairDeducer<false, true>>,
    std::tuple<_exts<stdex::dynamic_extent, stdex::dynamic_extent>, _sizes<5, 10>, _exts<5, 10>, _sizes<>,  _BoolPairDeducer<false, true>>,
    std::tuple<_exts<5, 10>, _sizes<>, _exts<5, stdex::dynamic_extent>, _sizes<10>,  _BoolPairDeducer<true, false>>,
    std::tuple<_exts<5, 10>, _sizes<>, _exts<stdex::dynamic_extent, 10>, _sizes<5>,  _BoolPairDeducer<true, false>>,
    //--------------------
    std::tuple<_exts<stdex::dynamic_extent, stdex::dynamic_extent, 15>, _sizes<5, 10>, _exts<5, stdex::dynamic_extent, 15>, _sizes<10>,  _BoolPairDeducer<false, true>>,
    std::tuple<_exts<5, 10, 15>, _sizes<>, _exts<5, stdex::dynamic_extent, 15>, _sizes<10>,  _BoolPairDeducer<true, false>>,
    std::tuple<_exts<5, 10, 15>, _sizes<>, _exts<stdex::dynamic_extent, stdex::dynamic_extent, stdex::dynamic_extent>, _sizes<5, 10, 15>,  _BoolPairDeducer<true, false>>
  >;

TYPED_TEST_SUITE(TestExtentsCompatCtors, compatible_extents_test_types);

TYPED_TEST(TestExtentsCompatCtors, compatible_construct_1) {
  auto e1 = typename TestFixture::extents_type1(this->exts2);
  EXPECT_EQ(e1, this->exts2);
}

TYPED_TEST(TestExtentsCompatCtors, compatible_construct_2) {
  auto e2 = typename TestFixture::extents_type2(this->exts1);
  EXPECT_EQ(e2, this->exts1);
}

TYPED_TEST(TestExtentsCompatCtors, compatible_assign_1) {
#if MDSPAN_HAS_CXX_17
  if constexpr(std::is_convertible_v<typename TestFixture::extents_type2,
                           typename TestFixture::extents_type1>) {
    this->exts1 = this->exts2;
  } else {
    this->exts1 = typename TestFixture::extents_type1(this->exts2);
  }
#else
  this->exts1 = typename TestFixture::extents_type1(this->exts2);
#endif
  EXPECT_EQ(this->exts1, this->exts2);

}

TYPED_TEST(TestExtentsCompatCtors, compatible_assign_2) {
#if MDSPAN_HAS_CXX_17
  if constexpr(std::is_convertible_v<typename TestFixture::extents_type1,
                           typename TestFixture::extents_type2>) {
    this->exts2 = this->exts1;
  } else {
    this->exts2 = typename TestFixture::extents_type2(this->exts1);
  }
#else
  this->exts2 = typename TestFixture::extents_type2(this->exts1);
#endif
  EXPECT_EQ(this->exts1, this->exts2);
}


template<class Extents, bool>
struct ImplicitConversionToExts;
template<class Extents>
struct ImplicitConversionToExts<Extents,true> {
  static  bool convert(Extents) {
    return true;
  }
};
template<class Extents>
struct ImplicitConversionToExts<Extents,false> {
  template<class E>
  static  bool convert(E) {
    return false;
  }
};


#ifdef _MDSPAN_USE_CONDITIONAL_EXPLICIT
TYPED_TEST(TestExtentsCompatCtors, implicit_construct_1) {
  bool exts1_convertible_exts2 =
    std::is_convertible<typename TestFixture::extents_type1,
                        typename TestFixture::extents_type2>::value;
  bool exts2_convertible_exts1 =
    std::is_convertible<typename TestFixture::extents_type2,
                        typename TestFixture::extents_type1>::value;

  bool exts1_implicit_exts2 = ImplicitConversionToExts<
                               typename TestFixture::extents_type2,
                                        TestFixture::implicit_exts1_to_exts2>::convert(this->exts1);

  bool exts2_implicit_exts1 = ImplicitConversionToExts<
                               typename TestFixture::extents_type1,
                                        TestFixture::implicit_exts2_to_exts1>::convert(this->exts2);

  EXPECT_EQ(exts1_convertible_exts2,exts1_implicit_exts2);
  EXPECT_EQ(exts2_convertible_exts1,exts2_implicit_exts1);
  EXPECT_EQ(exts1_convertible_exts2,TestFixture::implicit_exts1_to_exts2);
  EXPECT_EQ(exts2_convertible_exts1,TestFixture::implicit_exts2_to_exts1);
}
#endif

TEST(TestExtentsCtorStdArrayConvertibleToSizeT, test_extents_ctor_std_array_convertible_to_size_t) {
  std::array<int, 2> i{2, 2};
  stdex::dextents<size_t,2> e{i};
  ASSERT_EQ(e.rank(), 2);
  ASSERT_EQ(e.rank_dynamic(), 2);
  ASSERT_EQ(e.extent(0), 2);
  ASSERT_EQ(e.extent(1), 2);
}

#ifdef __cpp_lib_span
TEST(TestExtentsCtorStdArrayConvertibleToSizeT, test_extents_ctor_std_span_convertible_to_size_t) {
  std::array<int, 2> i{2, 2};
  std::span<int ,2> s(i.data(),2);
  stdex::dextents<size_t,2> e{s};
  ASSERT_EQ(e.rank(), 2);
  ASSERT_EQ(e.rank_dynamic(), 2);
  ASSERT_EQ(e.extent(0), 2);
  ASSERT_EQ(e.extent(1), 2);
}
#endif

TYPED_TEST(TestExtentsCompatCtors, construct_from_dynamic_sizes) {
  using e1_t = typename TestFixture::extents_type1;
  constexpr bool e1_last_ext_static = e1_t::rank()>0?e1_t::static_extent(e1_t::rank()-1)!=stdex::dynamic_extent:true;
  e1_t e1 = TestFixture::template make_extents<e1_t,e1_t::rank(),false,e1_last_ext_static>::construct();
  for(size_t r=0; r<e1.rank(); r++)
    ASSERT_EQ(e1.extent(r), (r+1)*5);

  using e2_t = typename TestFixture::extents_type2;
  constexpr bool e2_last_ext_static = e2_t::rank()>0?e2_t::static_extent(e2_t::rank()-1)!=stdex::dynamic_extent:true;
  e2_t e2 = TestFixture::template make_extents<e2_t,e2_t::rank(),false,e2_last_ext_static>::construct();
  for(size_t r=0; r<e2.rank(); r++)
    ASSERT_EQ(e2.extent(r), (r+1)*5);
}

TYPED_TEST(TestExtentsCompatCtors, construct_from_all_sizes) {
  using e1_t = typename TestFixture::extents_type1;
  e1_t e1 = TestFixture::template make_extents<e1_t,e1_t::rank(),true,true>::construct();
  for(size_t r=0; r<e1.rank(); r++)
    ASSERT_EQ(e1.extent(r), (r+1)*5);

  using e2_t = typename TestFixture::extents_type2;
  e2_t e2 = TestFixture::template make_extents<e2_t,e1_t::rank(),true,true>::construct();
  for(size_t r=0; r<e2.rank(); r++)
    ASSERT_EQ(e2.extent(r), (r+1)*5);
}

TYPED_TEST(TestExtentsCompatCtors, construct_from_dynamic_array) {
  using e1_t = typename TestFixture::extents_type1;
  std::array<int, e1_t::rank_dynamic()> ext1_array_dynamic;

  int dyn_idx = 0;
  for(size_t r=0; r<e1_t::rank(); r++) {
    if(e1_t::static_extent(r)==stdex::dynamic_extent) {
      ext1_array_dynamic[dyn_idx] = static_cast<int>((r+1)*5);
      dyn_idx++;
    }
  }
  e1_t e1(ext1_array_dynamic);
  for(size_t r=0; r<e1.rank(); r++)
    ASSERT_EQ(e1.extent(r), (r+1)*5);

  using e2_t = typename TestFixture::extents_type2;
  std::array<int, e2_t::rank_dynamic()> ext2_array_dynamic;

  dyn_idx = 0;
  for(size_t r=0; r<e2_t::rank(); r++) {
    if(e2_t::static_extent(r)==stdex::dynamic_extent) {
      ext2_array_dynamic[dyn_idx] = static_cast<int>((r+1)*5);
      dyn_idx++;
    }
  }
  e2_t e2(ext2_array_dynamic);
  for(size_t r=0; r<e2.rank(); r++)
    ASSERT_EQ(e2.extent(r), (r+1)*5);
}

TYPED_TEST(TestExtentsCompatCtors, construct_from_all_array) {
  using e1_t = typename TestFixture::extents_type1;
  std::array<int, e1_t::rank()> ext1_array_all;

  for(size_t r=0; r<e1_t::rank(); r++) ext1_array_all[r] = static_cast<int>((r+1)*5);
  e1_t e1(ext1_array_all);
  for(size_t r=0; r<e1.rank(); r++)
    ASSERT_EQ(e1.extent(r), (r+1)*5);

  using e2_t = typename TestFixture::extents_type2;
  std::array<int, e2_t::rank()> ext2_array_all;

  for(size_t r=0; r<e2_t::rank(); r++) ext2_array_all[r] = static_cast<int>((r+1)*5);
  e2_t e2(ext2_array_all);
  for(size_t r=0; r<e2.rank(); r++)
    ASSERT_EQ(e2.extent(r), (r+1)*5);
}

#if defined(_MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
TEST(TestExtentsCTADPack, test_extents_ctad_pack) {
  stdex::extents m0;
  ASSERT_EQ(m0.rank(), 0);
  ASSERT_EQ(m0.rank_dynamic(), 0);

  stdex::extents m1(64);
  ASSERT_EQ(m1.rank(), 1);
  ASSERT_EQ(m1.rank_dynamic(), 1);
  ASSERT_EQ(m1.extent(0), 64);

  stdex::extents m2(64, 128);
  ASSERT_EQ(m2.rank(), 2);
  ASSERT_EQ(m2.rank_dynamic(), 2);
  ASSERT_EQ(m2.extent(0), 64);
  ASSERT_EQ(m2.extent(1), 128);

  stdex::extents m3(64, 128, 256);
  ASSERT_EQ(m3.rank(), 3);
  ASSERT_EQ(m3.rank_dynamic(), 3);
  ASSERT_EQ(m3.extent(0), 64);
  ASSERT_EQ(m3.extent(1), 128);
  ASSERT_EQ(m3.extent(2), 256);
}

// TODO: It appears to currently be impossible to write a deduction guide that
// makes this work.
/*
TEST(TestExtentsCTADStdArray, test_extents_ctad_std_array) {
  stdex::extents m0{std::array<size_t, 0>{}};
  ASSERT_EQ(m0.rank(), 0);
  ASSERT_EQ(m0.rank_dynamic(), 0);

  // TODO: `extents` should accept an array of any type convertible to `size_t`.
  stdex::extents m1{std::array{64UL}};
  ASSERT_EQ(m1.rank(), 1);
  ASSERT_EQ(m1.rank_dynamic(), 1);
  ASSERT_EQ(m1.extent(0), 64);

  // TODO: `extents` should accept an array of any type convertible to `size_t`.
  stdex::extents m2{std::array{64UL, 128UL}};
  ASSERT_EQ(m2.rank(), 2);
  ASSERT_EQ(m2.rank_dynamic(), 2);
  ASSERT_EQ(m2.extent(0), 64);
  ASSERT_EQ(m2.extent(1), 128);

  // TODO: `extents` should accept an array of any type convertible to `size_t`.
  stdex::extents m3{std::array{64UL, 128UL, 256UL}};
  ASSERT_EQ(m3.rank(), 3);
  ASSERT_EQ(m3.rank_dynamic(), 3);
  ASSERT_EQ(m3.extent(0), 64);
  ASSERT_EQ(m3.extent(1), 128);
  ASSERT_EQ(m3.extent(2), 256);
}
*/
#endif
