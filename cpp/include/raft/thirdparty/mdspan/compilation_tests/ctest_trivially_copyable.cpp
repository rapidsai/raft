/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2020) Sandia Corporation
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

#include <type_traits>

namespace stdex = std::experimental;

//==============================================================================
// <editor-fold desc="helper utilities"> {{{1

MDSPAN_STATIC_TEST(
  !std::is_base_of<stdex::extents<int, 1, 2, 3>, stdex::detail::__partially_static_sizes<int, size_t, 1, 2, 3>>::value
);

MDSPAN_STATIC_TEST(
  !std::is_base_of<stdex::detail::__partially_static_sizes<int, size_t, 1, 2, 3>, stdex::extents<int, 1, 2, 3>>::value
);

MDSPAN_STATIC_TEST(
  std::is_trivially_copyable<
    stdex::detail::__partially_static_sizes<int, size_t, 1, 2, 3>
  >::value
);

// </editor-fold> end helper utilities }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="extents"> {{{1

MDSPAN_STATIC_TEST(
  std::is_trivially_copyable<
    stdex::extents<size_t,1, 2, stdex::dynamic_extent>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_trivially_copyable<
    stdex::extents<size_t,stdex::dynamic_extent>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_trivially_copyable<
    stdex::extents<size_t,stdex::dynamic_extent, stdex::dynamic_extent>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_trivially_copyable<
    stdex::extents<size_t,stdex::dynamic_extent, 1, 2, 45>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_trivially_copyable<
    stdex::extents<size_t,45, stdex::dynamic_extent, 1>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_trivially_copyable<
    stdex::extents<size_t,1, 2, 3>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_trivially_copyable<
    stdex::extents<size_t,42>
  >::value
);

// </editor-fold> end extents }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="layouts"> {{{1

MDSPAN_STATIC_TEST(
  std::is_trivially_copyable<
    stdex::layout_left::template mapping<
      stdex::extents<size_t,42, stdex::dynamic_extent, 73>
    >
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_trivially_copyable<
    stdex::layout_right::template mapping<
      stdex::extents<size_t,42, stdex::dynamic_extent, 73>
    >
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_trivially_copyable<
    stdex::layout_right::template mapping<
      stdex::extents<size_t,stdex::dynamic_extent, stdex::dynamic_extent>
    >
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_trivially_copyable<
    stdex::layout_stride::template mapping<
      stdex::extents<size_t,42, stdex::dynamic_extent, 73>
    >
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_trivially_copyable<
    stdex::layout_stride::template mapping<
      stdex::extents<size_t,42, 27, 73>
    >
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_trivially_copyable<
    stdex::layout_stride::template mapping<
      stdex::extents<size_t,stdex::dynamic_extent, stdex::dynamic_extent>
    >
  >::value
);

struct layout_stride_as_member_should_be_standard_layout :
  stdex::layout_stride::template mapping<
    stdex::extents<size_t,1, 2, 3>
  >
{
  int foo;
};

// Fails with MSVC which adds some padding
#ifndef _MDSPAN_COMPILER_MSVC
MDSPAN_STATIC_TEST(
  std::is_trivially_copyable<layout_stride_as_member_should_be_standard_layout>::value
);
#endif

// </editor-fold> end layouts }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="mdspan"> {{{1

MDSPAN_STATIC_TEST(
  std::is_trivially_copyable<
    stdex::mdspan<double, stdex::extents<size_t,1, 2, 3>>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_trivially_copyable<
    stdex::mdspan<int, stdex::dextents<size_t,2>>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_trivially_copyable<
    stdex::mdspan<
      double, stdex::extents<size_t,stdex::dynamic_extent, stdex::dynamic_extent>,
      stdex::layout_left, stdex::default_accessor<double>
    >
  >::value
);

// </editor-fold> end mdspan }}}1
//==============================================================================



