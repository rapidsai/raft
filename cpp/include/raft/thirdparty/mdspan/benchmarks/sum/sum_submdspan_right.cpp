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

#include <memory>
#include <random>
#include <iostream>

#include "sum_3d_common.hpp"

//================================================================================

using index_type = int;

template <class T, size_t... Es>
using lmdspan = stdex::mdspan<T, std::experimental::extents<index_type, Es...>, stdex::layout_left>;
template <class T, size_t... Es>
using rmdspan = stdex::mdspan<T, std::experimental::extents<index_type, Es...>, stdex::layout_right>;

//================================================================================

template <class MDSpan, class... DynSizes>
void BM_MDSpan_Sum_Subspan_3D_right(benchmark::State& state, MDSpan, DynSizes... dyn) {

  using value_type = typename MDSpan::value_type;
  auto buffer = std::make_unique<value_type[]>(
    MDSpan{nullptr, dyn...}.mapping().required_span_size()
  );

  auto s = MDSpan{buffer.get(), dyn...};
  mdspan_benchmark::fill_random(s);

  for (auto _ : state) {
    benchmark::DoNotOptimize(s);
    benchmark::DoNotOptimize(s.data_handle());
    value_type sum = 0;
    using index_type = typename MDSpan::index_type;
    for(index_type i = 0; i < s.extent(0); ++i) {
      auto sub_i = stdex::submdspan(s, i, stdex::full_extent, stdex::full_extent);
      for (index_type j = 0; j < s.extent(1); ++j) {
        auto sub_i_j = stdex::submdspan(sub_i, j, stdex::full_extent);
        for (index_type k = 0; k < s.extent(2); ++k) {
          sum += sub_i_j(k);
        }
      }
    }
    benchmark::DoNotOptimize(sum);
    benchmark::DoNotOptimize(s.data_handle());
  }
  state.SetBytesProcessed(s.size() * sizeof(value_type) * state.iterations());
}
MDSPAN_BENCHMARK_ALL_3D(BM_MDSpan_Sum_Subspan_3D_right, right_, rmdspan, 20, 20, 20);
MDSPAN_BENCHMARK_ALL_3D(BM_MDSpan_Sum_Subspan_3D_right, left_, lmdspan, 20, 20, 20);
MDSPAN_BENCHMARK_ALL_3D(BM_MDSpan_Sum_Subspan_3D_right, right_, rmdspan, 200, 200, 200);
MDSPAN_BENCHMARK_ALL_3D(BM_MDSpan_Sum_Subspan_3D_right, left_, lmdspan, 200, 200, 200);

//================================================================================

BENCHMARK_CAPTURE(
  BM_Raw_Sum_3D_right, size_20_20_20, int(), size_t(20), size_t(20), size_t(20)
);
BENCHMARK_CAPTURE(
  BM_Raw_Sum_3D_right, size_200_200_200, int(), size_t(200), size_t(200), size_t(200)
);

BENCHMARK_CAPTURE(
  BM_Raw_Static_Sum_3D_right, size_20_20_20, int(),
  std::integral_constant<size_t, 20>{},
  std::integral_constant<size_t, 20>{},
  std::integral_constant<size_t, 20>{}
);
BENCHMARK_CAPTURE(
  BM_Raw_Static_Sum_3D_right, size_200_200_200, int(),
  std::integral_constant<size_t, 200>{},
  std::integral_constant<size_t, 200>{},
  std::integral_constant<size_t, 200>{}
);

BENCHMARK_CAPTURE(
  BM_Raw_Sum_3D_right_iter_left, size_20_20_20, int(), 20, 20, 20
);
BENCHMARK_CAPTURE(
  BM_Raw_Sum_3D_right_iter_left, size_200_200_200, int(), 200, 200, 200
);
BENCHMARK_CAPTURE(
  BM_Raw_Static_Sum_3D_right_iter_left, size_20_20_20, int(),
  std::integral_constant<size_t, 20>{},
  std::integral_constant<size_t, 20>{},
  std::integral_constant<size_t, 20>{}
);
BENCHMARK_CAPTURE(
  BM_Raw_Static_Sum_3D_right_iter_left, size_200_200_200, int(),
  std::integral_constant<size_t, 200>{},
  std::integral_constant<size_t, 200>{},
  std::integral_constant<size_t, 200>{}
);

//================================================================================

namespace _impl {

template <class, class T>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr T&& _repeated_with(T&& v) noexcept { return std::forward<T>(v); }

template <class T, class... Rest>
MDSPAN_FORCE_INLINE_FUNCTION
_MDSPAN_CONSTEXPR_14 void _do_sum_submdspan(
  T& sum,
  stdex::mdspan<T, std::experimental::extents<index_type>, Rest...> s
)
{
  sum += s();
}

template <class T, size_t E, size_t... Es, class... Rest>
MDSPAN_FORCE_INLINE_FUNCTION
_MDSPAN_CONSTEXPR_14 void _do_sum_submdspan(
  T& sum,
  stdex::mdspan<T, std::experimental::extents<index_type, E, Es...>, Rest...> s
)
{
  for(index_type i = 0; i < s.extent(0); ++i) {
    _impl::_do_sum_submdspan(sum, stdex::submdspan(
      s, i, _repeated_with<decltype(Es)>(stdex::full_extent)...)
    );
  }
}

} // end namespace _impl

template <class MDSpan, class... DynSizes>
void BM_MDSpan_Sum_Subspan_MD_right(benchmark::State& state, MDSpan, DynSizes... dyn) {
  using value_type = typename MDSpan::value_type;
  auto buffer = std::make_unique<value_type[]>(
    MDSpan{nullptr, dyn...}.mapping().required_span_size()
  );
  auto s = MDSpan{buffer.get(), dyn...};
  mdspan_benchmark::fill_random(s);
  for (auto _ : state) {
    value_type sum = 0;
    _impl::_do_sum_submdspan(sum, s);
    benchmark::DoNotOptimize(sum);
    benchmark::DoNotOptimize(s.data_handle());
  }
  state.SetBytesProcessed(s.size() * sizeof(value_type) * state.iterations());
}

BENCHMARK_CAPTURE(
  BM_MDSpan_Sum_Subspan_MD_right, fixed_10D_size_1024,
  stdex::mdspan<int, stdex::extents<index_type, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2>>{nullptr}
);
BENCHMARK_CAPTURE(
  BM_MDSpan_Sum_Subspan_MD_right, dyn_10D_size_1024,
  stdex::mdspan<int, stdex::dextents<index_type, 10>>{},
  2, 2, 2, 2, 2,
  2, 2, 2, 2, 2
);
BENCHMARK_CAPTURE(
  BM_MDSpan_Sum_Subspan_MD_right, fixed5_dyn5_10D_alternate_size_1024,
  stdex::mdspan<int, stdex::extents<index_type, 2, stdex::dynamic_extent, 2, stdex::dynamic_extent, 2, stdex::dynamic_extent, 2, stdex::dynamic_extent, 2, stdex::dynamic_extent>>{},
  2, 2, 2, 2, 2
);
BENCHMARK_CAPTURE(
  BM_MDSpan_Sum_Subspan_MD_right, fixed8_dyn2_10D_alternate_size_1024,
  stdex::mdspan<int, stdex::extents<index_type, 2, 2, 2, 2, stdex::dynamic_extent, 2, 2, 2, 2, stdex::dynamic_extent>>{},
  2, 2
);
BENCHMARK_CAPTURE(
  BM_MDSpan_Sum_Subspan_MD_right, fixed_5D_size_1024,
  stdex::mdspan<int, stdex::extents<index_type, 4, 4, 4, 4, 4>>{nullptr}
);
BENCHMARK_CAPTURE(
  BM_MDSpan_Sum_Subspan_MD_right, dyn_5D_size_1024,
  stdex::mdspan<int, stdex::dextents<index_type, 5>>{},
  4, 4, 4, 4, 4
);
BENCHMARK_CAPTURE(
  BM_Raw_Sum_1D, size_1024, int(), 1024
);

//================================================================================

BENCHMARK_MAIN();

