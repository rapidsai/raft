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
#include <vector>

#include <gtest/gtest.h>

namespace stdex = std::experimental;
_MDSPAN_INLINE_VARIABLE constexpr auto dyn = stdex::dynamic_extent;

TEST(TestMdspanCtorDataCArray, test_mdspan_ctor_data_carray) {
  int data[1] = {42};
  stdex::mdspan<int, stdex::extents<1>> m(data);
  ASSERT_EQ(m.data(), data);
  ASSERT_EQ(m.rank(), 1);
  ASSERT_EQ(m.rank_dynamic(), 0);
  ASSERT_EQ(m.extent(0), 1);
  ASSERT_EQ(m.stride(0), 1);
  ASSERT_EQ(__MDSPAN_OP(m, 0), 42);
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdspanCtorDataStdArray, test_mdspan_ctor_data_carray) {
  std::array<int, 1> d = {42};
  stdex::mdspan<int, stdex::extents<1>> m(d.data());
  ASSERT_EQ(m.data(), d.data());
  ASSERT_EQ(m.rank(), 1);
  ASSERT_EQ(m.rank_dynamic(), 0);
  ASSERT_EQ(m.extent(0), 1);
  ASSERT_EQ(m.stride(0), 1);
  ASSERT_EQ(__MDSPAN_OP(m, 0), 42);
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdspanCtorDataVector, test_mdspan_ctor_data_carray) {
  std::vector<int> d = {42};
  stdex::mdspan<int, stdex::extents<1>> m(d.data());
  ASSERT_EQ(m.data(), d.data());
  ASSERT_EQ(m.rank(), 1);
  ASSERT_EQ(m.rank_dynamic(), 0);
  ASSERT_EQ(m.extent(0), 1);
  ASSERT_EQ(m.stride(0), 1);
  ASSERT_EQ(__MDSPAN_OP(m, 0), 42);
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdspanCtorExtentsStdArrayConvertibleToSizeT, test_mdspan_ctor_extents_std_array_convertible_to_size_t) {
  std::array<int, 4> d{42, 17, 71, 24};
  std::array<int, 2> e{2, 2};
  stdex::mdspan<int, stdex::dextents<2>> m(d.data(), e);
  ASSERT_EQ(m.data(), d.data());
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 2);
  ASSERT_EQ(m.extent(0), 2);
  ASSERT_EQ(m.extent(1), 2);
  ASSERT_EQ(m.stride(0), 2);
  ASSERT_EQ(m.stride(1), 1);
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdspanListInitializationLayoutLeft, test_mdspan_list_initialization_layout_left) {
  std::array<int, 1> d{42};
  stdex::mdspan<int, stdex::extents<dyn, dyn>, stdex::layout_left> m{d.data(), 16, 32};
  ASSERT_EQ(m.data(), d.data());
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 2);
  ASSERT_EQ(m.extent(0), 16);
  ASSERT_EQ(m.extent(1), 32);
  ASSERT_EQ(m.stride(0), 1);
  ASSERT_EQ(m.stride(1), 16);
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdspanListInitializationLayoutRight, test_mdspan_list_initialization_layout_right) {
  std::array<int, 1> d{42};
  stdex::mdspan<int, stdex::extents<dyn, dyn>, stdex::layout_right> m{d.data(), 16, 32};
  ASSERT_EQ(m.data(), d.data());
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 2);
  ASSERT_EQ(m.extent(0), 16);
  ASSERT_EQ(m.extent(1), 32);
  ASSERT_EQ(m.stride(0), 32);
  ASSERT_EQ(m.stride(1), 1);
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdspanListInitializationLayoutStride, test_mdspan_list_initialization_layout_stride) {
  std::array<int, 1> d{42};
  stdex::mdspan<int, stdex::extents<dyn, dyn>, stdex::layout_stride> m{d.data(), {stdex::dextents<2>{16, 32}, std::array<std::size_t, 2>{1, 128}}};
  ASSERT_EQ(m.data(), d.data());
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 2);
  ASSERT_EQ(m.extent(0), 16);
  ASSERT_EQ(m.extent(1), 32);
  ASSERT_EQ(m.stride(0), 1);
  ASSERT_EQ(m.stride(1), 128);
  ASSERT_FALSE(m.is_contiguous());
}

#if defined(_MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
TEST(TestMdspanCTADExtentsPack, test_mdspan_ctad_extents_pack) {
  std::array<int, 1> d{42};
  stdex::mdspan m(d.data(), 64, 128);
  ASSERT_EQ(m.data(), d.data());
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 2);
  ASSERT_EQ(m.extent(0), 64);
  ASSERT_EQ(m.extent(1), 128);
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdspanCTADExtentsObject, test_mdspan_ctad_extents_object) {
  std::array<int, 1> d{42};
  stdex::mdspan m{d.data(), stdex::extents{64, 128}};
  ASSERT_EQ(m.data(), d.data());
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 2);
  ASSERT_EQ(m.extent(0), 64);
  ASSERT_EQ(m.extent(1), 128);
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdspanCTADExtentsStdArray, test_mdspan_ctad_extents_std_array) {
  std::array<int, 1> d{42};
  stdex::mdspan m{d.data(), std::array{64, 128}};
  ASSERT_EQ(m.data(), d.data());
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 2);
  ASSERT_EQ(m.extent(0), 64);
  ASSERT_EQ(m.extent(1), 128);
  ASSERT_TRUE(m.is_contiguous());
}

TEST(TestMdspanCTADLayoutLeft, test_mdspan_ctad_layout_left) {
  std::array<int, 1> d{42};

  stdex::mdspan m0{d.data(), stdex::layout_left::mapping{stdex::extents{16, 32}}};
  ASSERT_EQ(m0.data(), d.data());
  ASSERT_EQ(m0.rank(), 2);
  ASSERT_EQ(m0.rank_dynamic(), 2);
  ASSERT_EQ(m0.extent(0), 16);
  ASSERT_EQ(m0.extent(1), 32);
  ASSERT_EQ(m0.stride(0), 1);
  ASSERT_EQ(m0.stride(1), 16);
  ASSERT_TRUE(m0.is_contiguous());

// TODO: Perhaps one day I'll get this to work.
/*
  stdex::mdspan m1{d.data(), stdex::layout_left::mapping{{16, 32}}};
  ASSERT_EQ(m1.data(), d.data());
  ASSERT_EQ(m1.rank(), 2);
  ASSERT_EQ(m1.rank_dynamic(), 2);
  ASSERT_EQ(m1.extent(0), 16);
  ASSERT_EQ(m1.extent(1), 32);
  ASSERT_EQ(m1.stride(0), 1);
  ASSERT_EQ(m1.stride(1), 16);
  ASSERT_TRUE(m1.is_contiguous());
*/
}

TEST(TestMdspanCTADLayoutRight, test_mdspan_ctad_layout_right) {
  std::array<int, 1> d{42};

  stdex::mdspan m0{d.data(), stdex::layout_right::mapping{stdex::extents{16, 32}}};
  ASSERT_EQ(m0.data(), d.data());
  ASSERT_EQ(m0.rank(), 2);
  ASSERT_EQ(m0.rank_dynamic(), 2);
  ASSERT_EQ(m0.extent(0), 16);
  ASSERT_EQ(m0.extent(1), 32);
  ASSERT_EQ(m0.stride(0), 32);
  ASSERT_EQ(m0.stride(1), 1);
  ASSERT_TRUE(m0.is_contiguous());

// TODO: Perhaps one day I'll get this to work.
/*
  stdex::mdspan m1{d.data(), stdex::layout_right::mapping{{16, 32}}};
  ASSERT_EQ(m1.data(), d.data());
  ASSERT_EQ(m1.rank(), 2);
  ASSERT_EQ(m1.rank_dynamic(), 2);
  ASSERT_EQ(m1.extent(0), 16);
  ASSERT_EQ(m1.extent(1), 32);
  ASSERT_EQ(m1.stride(0), 32);
  ASSERT_EQ(m1.stride(1), 1);
  ASSERT_TRUE(m1.is_contiguous());
*/
}

TEST(TestMdspanCTADLayoutStride, test_mdspan_ctad_layout_stride) {
  std::array<int, 1> d{42};

  stdex::mdspan m0{d.data(), stdex::layout_stride::mapping{stdex::extents{16, 32}, std::array{1, 128}}};
  ASSERT_EQ(m0.data(), d.data());
  ASSERT_EQ(m0.rank(), 2);
  ASSERT_EQ(m0.rank_dynamic(), 2);
  ASSERT_EQ(m0.extent(0), 16);
  ASSERT_EQ(m0.extent(1), 32);
  ASSERT_EQ(m0.stride(0), 1);
  ASSERT_EQ(m0.stride(1), 128);
  ASSERT_FALSE(m0.is_contiguous());

  /* 
  stdex::mdspan m1{d.data(), stdex::layout_stride::mapping{stdex::extents{16, 32}, stdex::extents{1, 128}}};
  ASSERT_EQ(m1.data(), d.data());
  ASSERT_EQ(m1.rank(), 2);
  ASSERT_EQ(m1.rank_dynamic(), 2);
  ASSERT_EQ(m1.extent(0), 16);
  ASSERT_EQ(m1.extent(1), 32);
  ASSERT_EQ(m1.stride(0), 1);
  ASSERT_EQ(m1.stride(1), 128);
  ASSERT_FALSE(m1.is_contiguous());
  */

// TODO: Perhaps one day I'll get this to work.
/*
  stdex::mdspan m2{d.data(), stdex::layout_stride::mapping{{16, 32}, {1, 128}}};
  ASSERT_EQ(m2.data(), d.data());
  ASSERT_EQ(m2.rank(), 2);
  ASSERT_EQ(m2.rank_dynamic(), 2);
  ASSERT_EQ(m2.extent(0), 16);
  ASSERT_EQ(m2.extent(1), 32);
  ASSERT_EQ(m2.stride(0), 1);
  ASSERT_EQ(m2.stride(1), 128);
  ASSERT_FALSE(m2.is_contiguous());
*/
}
#endif
