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

using E1 = stdex::extents<int32_t, stdex::dynamic_extent, 3>;

MDSPAN_STATIC_TEST(
  std::is_same<typename E1::index_type, int32_t>::value &&
  std::is_same<typename E1::size_type, uint32_t>::value &&
  std::is_same<typename E1::rank_type, size_t>::value &&
  std::is_same<decltype(E1::rank()), typename E1::rank_type>::value &&
  std::is_same<decltype(E1::rank_dynamic()), typename E1::rank_type>::value &&
  std::is_same<decltype(E1::static_extent(0)), size_t>::value &&
  std::is_same<decltype(E1::static_extent(1)), size_t>::value &&
  std::is_same<decltype(std::declval<E1>().extent(0)), typename E1::index_type>::value &&
  std::is_same<decltype(std::declval<E1>().extent(1)), typename E1::index_type>::value &&
  (E1::rank()==2) &&
  (E1::rank_dynamic()==1) &&
  (E1::static_extent(0) == stdex::dynamic_extent) &&
  (E1::static_extent(1) == 3)
);

using E2 = stdex::extents<int64_t, stdex::dynamic_extent, 3, stdex::dynamic_extent>;

MDSPAN_STATIC_TEST(
  std::is_same<typename E2::index_type, int64_t>::value &&
  std::is_same<typename E2::size_type, uint64_t>::value &&
  std::is_same<typename E2::rank_type, size_t>::value &&
  std::is_same<decltype(E2::rank()), typename E2::rank_type>::value &&
  std::is_same<decltype(E2::rank_dynamic()), typename E2::rank_type>::value &&
  std::is_same<decltype(E2::static_extent(0)), size_t>::value &&
  std::is_same<decltype(E2::static_extent(1)), size_t>::value &&
  std::is_same<decltype(E2::static_extent(2)), size_t>::value &&
  std::is_same<decltype(std::declval<E2>().extent(0)), typename E2::index_type>::value &&
  std::is_same<decltype(std::declval<E2>().extent(1)), typename E2::index_type>::value &&
  std::is_same<decltype(std::declval<E2>().extent(2)), typename E2::index_type>::value &&
  (E2::rank()==3) &&
  (E2::rank_dynamic()==2) &&
  (E2::static_extent(0) == stdex::dynamic_extent) &&
  (E2::static_extent(1) == 3) &&
  (E2::static_extent(2) == stdex::dynamic_extent)
);
