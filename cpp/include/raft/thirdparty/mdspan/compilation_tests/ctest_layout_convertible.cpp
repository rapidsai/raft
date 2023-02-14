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

struct NotARealLayout {
  template<class Extents>
  struct mapping {
    using extents_type = Extents;
    using rank_type = typename extents_type::rank_type;
    using index_type = typename extents_type::index_type;
    using layout_type = NotARealLayout;

    constexpr extents_type& extents() const { return ext; }

    template<class ... Idx>
    index_type operator()(Idx ...) const { return 0; }

    index_type required_span_size() const { return 0; }

    index_type stride(rank_type) const { return 1; }

    private:
      extents_type ext;
  };
};

template<bool unique>
struct AStridedLayout {
  template<class Extents>
  struct mapping {
    using extents_type = Extents;
    using rank_type = typename extents_type::rank_type;
    using index_type = typename extents_type::index_type;
    using layout_type = AStridedLayout;

    constexpr extents_type& extents() const { return ext; }

    template<class ... Idx>
    index_type operator()(Idx ...) const { return 0; }

    index_type required_span_size() const { return 0; }

    index_type stride(rank_type) const { return 1; }

    constexpr static bool is_always_strided() { return true; }
    constexpr static bool is_always_unique() { return unique; }
    constexpr static bool is_always_exhaustive() { return true; }
    constexpr bool is_strided() { return true; }
    constexpr bool is_unique() { return unique; }
    constexpr bool is_exhaustive() { return true; }

    private:
      extents_type ext;
  };
};

using E1 = stdex::extents<int32_t, 2,2>;
using E2 = stdex::extents<int64_t, 2,2>;
using LS1 = stdex::layout_stride::mapping<E1>;
using LS2 = stdex::layout_stride::mapping<E2>;

MDSPAN_STATIC_TEST(
  !std::is_constructible<LS1, AStridedLayout<false>::mapping<E2>>::value &&
  !std::is_convertible<AStridedLayout<false>::mapping<E2>, LS1>::value
);

MDSPAN_STATIC_TEST(
  std::is_constructible<LS2, AStridedLayout<true>::mapping<E1>>::value &&
  std::is_convertible<AStridedLayout<true>::mapping<E1>, LS2>::value
);

MDSPAN_STATIC_TEST(
  !std::is_constructible<LS1, NotARealLayout::mapping<E2>>::value
);


