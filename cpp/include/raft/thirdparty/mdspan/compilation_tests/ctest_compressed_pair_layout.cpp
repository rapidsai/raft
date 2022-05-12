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

enum emptyness {
  non_empty = false,
  empty     = true
};

enum standard_layoutness {
  non_standard_layout = false,
  standard_layout     = true
};

enum trivially_copyableness {
  non_trivially_copyable = false,
  trivially_copyable     = true
};

template <class T, size_t Size,
          emptyness Empty = non_empty,
          standard_layoutness StandardLayout = standard_layout,
          trivially_copyableness TriviallyCopyable = trivially_copyable>
void test() {
  MDSPAN_STATIC_TEST(sizeof(T) == Size);
  MDSPAN_STATIC_TEST(std::is_empty<T>::value == Empty);
#if !defined(__INTEL_COMPILER) || (__INTEL_COMPILER>=1900)
  MDSPAN_STATIC_TEST(std::is_standard_layout<T>::value == StandardLayout);
#endif
  MDSPAN_STATIC_TEST(std::is_trivially_copyable<T>::value == TriviallyCopyable);
}

template <class T, class U>
using CP = stdex::detail::__compressed_pair<T, U>;

struct E0 {};
struct E1 {};
struct E2 {};
struct E3 {};

void instantiate_tests() {
//==============================================================================
// <editor-fold desc="compressed pair layout: 2 leaf elements"> {{{1
#ifdef _MDSPAN_COMPILER_MSVC
test<CP<E0, E0>, 1, empty, non_standard_layout>();
test<CP<E0, E1>, 1, empty, standard_layout>();
#else
test<CP<E0, E0>,     2,                empty>();
test<CP<E0, E1>,     1,                empty>();
#endif
test<CP<int*, E1>,   sizeof(int*),     non_empty>();
test<CP<E0, int*>,   sizeof(int*),     non_empty>();
test<CP<int*, int*>, 2 * sizeof(int*), non_empty>();
// </editor-fold> end compressed pair layout: 2 leaf elements }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="compressed pair layout: 1 nested pair, 3 leaf element"> {{{1
#if defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
test<CP<E0,   CP<E0,   E0>>,   3,                empty>();     // Emulation can't handle this correctly.
#endif
#ifdef _MDSPAN_COMPILER_MSVC
test<CP<E0,   CP<E1,   E2>>,   2,                empty>();
#else
test<CP<E0,   CP<E1,   E2>>,   1,                empty>();
#endif
test<CP<E0,   CP<E1,   int*>>, sizeof(int*),     non_empty>();
test<CP<E0,   CP<int*, E2>>,   sizeof(int*),     non_empty>();
test<CP<E0,   CP<int*, int*>>, 2 * sizeof(int*), non_empty>();
#ifdef _MDSPAN_COMPILER_MSVC
test<CP<int*, CP<E1,   E2>>,   2 * sizeof(int*), non_empty>();
#else
test<CP<int*, CP<E1,   E2>>,   sizeof(int*),     non_empty>();
#endif
test<CP<int*, CP<E1,   int*>>, 2 * sizeof(int*), non_empty>();
test<CP<int*, CP<int*, E2>>,   2 * sizeof(int*), non_empty>();
test<CP<int*, CP<int*, int*>>, 3 * sizeof(int*), non_empty>();
#if defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
test<CP<CP<E0,   E0>,   E0>,   3,                empty>();     // Emulation can't handle this correctly.
#endif
#ifdef _MDSPAN_COMPILER_MSVC
test<CP<CP<E0,   E1>,   E2>,   2,                empty>();
test<CP<CP<E0,   E1>,   int*>, 2 * sizeof(int*), non_empty>();
#else
test<CP<CP<E0,   E1>,   E2>,   1,                empty>();
test<CP<CP<E0,   E1>,   int*>, sizeof(int*),     non_empty>();
#endif
test<CP<CP<E0,   int*>, E2>,   sizeof(int*),     non_empty>();
test<CP<CP<E0,   int*>, int*>, 2 * sizeof(int*), non_empty>();
test<CP<CP<int*, E1>,   E2>,   sizeof(int*),     non_empty>();
test<CP<CP<int*, E1>,   int*>, 2 * sizeof(int*), non_empty>();
test<CP<CP<int*, int*>, E2>,   2 * sizeof(int*), non_empty>();
test<CP<CP<int*, int*>, int*>, 3 * sizeof(int*), non_empty>();
// </editor-fold> end compressed pair layout: 1 nested pair, 3 leaf element }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="compressed pair layout: 2 nested pairs, 4 leaf element"> {{{1
#if defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
test<CP<CP<E0,   E0>,   CP<E0,  E0>>,    4,                empty>(); // Emulation can't handle this correctly.
#endif
#ifdef _MDSPAN_COMPILER_MSVC
test<CP<CP<E0,   E1>,   CP<E2,   E3>>,   3,                empty>();
test<CP<CP<E0,   E1>,   CP<E2,   int*>>, 2 * sizeof(int*), non_empty>();
test<CP<CP<E0,   E1>,   CP<int*, E3>>,   2 * sizeof(int*), non_empty>();
test<CP<CP<E0,   E1>,   CP<int*, int*>>, 3 * sizeof(int*), non_empty>();
#else
test<CP<CP<E0,   E1>,   CP<E2,   E3>>,   1,                empty>();
test<CP<CP<E0,   E1>,   CP<E2,   int*>>, sizeof(int*),     non_empty>();
test<CP<CP<E0,   E1>,   CP<int*, E3>>,   sizeof(int*),     non_empty>();
test<CP<CP<E0,   E1>,   CP<int*, int*>>, 2 * sizeof(int*), non_empty>();
#endif
test<CP<CP<E0,   int*>, CP<E2,   int*>>, 2 * sizeof(int*), non_empty>();
test<CP<CP<E0,   int*>, CP<int*, E3>>,   2 * sizeof(int*), non_empty>();
test<CP<CP<E0,   int*>, CP<int*, int*>>, 3 * sizeof(int*), non_empty>();
test<CP<CP<int*, E1>,   CP<E2,   int*>>, 2 * sizeof(int*), non_empty>();
test<CP<CP<int*, E1>,   CP<int*, E3>>,   2 * sizeof(int*), non_empty>();
test<CP<CP<int*, E1>,   CP<int*, int*>>, 3 * sizeof(int*), non_empty>();
test<CP<CP<int*, int*>, CP<E2,   int*>>, 3 * sizeof(int*), non_empty>();
test<CP<CP<int*, int*>, CP<int*, E3>>,   3 * sizeof(int*), non_empty>();
test<CP<CP<int*, int*>, CP<int*, int*>>, 4 * sizeof(int*), non_empty>();
// </editor-fold> end compressed pair layout: 2 nested pairs, 4 leaf elements }}}1
//==============================================================================
}

