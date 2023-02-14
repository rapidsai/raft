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
#include <sstream>
#include <stdexcept>

#include "fill.hpp"

//================================================================================

static constexpr int global_repeat = 1;

//================================================================================

using index_type = int;
template <class T, size_t... Es>
using lmdspan = stdex::mdspan<T, stdex::extents<index_type, Es...>, stdex::layout_left>;
template <class T, size_t... Es>
using rmdspan = stdex::mdspan<T, stdex::extents<index_type, Es...>, stdex::layout_right>;

void throw_runtime_exception(const std::string &msg) {
  std::ostringstream o;
  o << msg;
  throw std::runtime_error(o.str());
}

template<class MDSpan>
void OpenMP_first_touch_2D(MDSpan s) {
  #pragma omp parallel for
  for(index_type i = 0; i < s.extent(0); i ++) {
    for(index_type j = 0; j < s.extent(1); j ++) {
      s(i,j) = 0;
    }
  }
}

template<class MDSpan>
void OpenMP_first_touch_1D(MDSpan s) {
  #pragma omp parallel for
  for(index_type i = 0; i < s.extent(0); i ++) {
    s(i) = 0;
  }
}

//================================================================================

template <class MDSpanMatrix, class... DynSizes>
void BM_MDSpan_OpenMP_MatVec(benchmark::State& state, MDSpanMatrix, DynSizes... dyn) {

  using value_type = typename MDSpanMatrix::value_type;
  using MDSpanVector = lmdspan<value_type,stdex::dynamic_extent>;

  auto buffer_size_A = MDSpanMatrix{nullptr, dyn...}.mapping().required_span_size();
  auto buffer_A = std::make_unique<value_type[]>(buffer_size_A);
  auto A = MDSpanMatrix{buffer_A.get(), dyn...};
  OpenMP_first_touch_2D(A);
  mdspan_benchmark::fill_random(A);

  auto buffer_size_x = MDSpanVector{nullptr, A.extent(1)}.mapping().required_span_size();
  auto buffer_x = std::make_unique<value_type[]>(buffer_size_x);
  auto x = MDSpanVector{buffer_x.get(), A.extent(1)};
  OpenMP_first_touch_1D(x);
  mdspan_benchmark::fill_random(x);

  auto buffer_size_y = MDSpanVector{nullptr, A.extent(0)}.mapping().required_span_size();
  auto buffer_y = std::make_unique<value_type[]>(buffer_size_y);
  auto y = MDSpanVector{buffer_y.get(), A.extent(0)};
  OpenMP_first_touch_1D(y);
  mdspan_benchmark::fill_random(y);

  #pragma omp parallel for
  for(index_type i = 0; i < A.extent(0); i ++) {
    value_type y_i = 0;
    for(index_type j = 0; j < A.extent(1); j ++) {
      y_i += A(i,j) * x(j);
    }
    y(i) = y_i;
  }

  int R = 10;
  for (auto _ : state) {
    benchmark::DoNotOptimize(A.data_handle());
    benchmark::DoNotOptimize(y.data_handle());
    benchmark::DoNotOptimize(x.data_handle());
    for(int r=0; r<R; r++) {
    #pragma omp parallel for
    for(index_type i = 0; i < A.extent(0); i ++) {
      value_type y_i = 0;
      for(index_type j = 0; j < A.extent(1); j ++) {
        y_i += A(i,j) * x(j);
      }
      y(i) += y_i;
    }
    }
    benchmark::ClobberMemory();
  }
  size_t num_elements = 2 * A.extent(0) * A.extent(1) + 2 * A.extent(0);
  state.SetBytesProcessed( R * num_elements * sizeof(value_type) * state.iterations() * global_repeat);
  state.counters["repeats"] = global_repeat;
}

BENCHMARK_CAPTURE(BM_MDSpan_OpenMP_MatVec, left, lmdspan<double,stdex::dynamic_extent,stdex::dynamic_extent>(), 100000, 5000);
BENCHMARK_CAPTURE(BM_MDSpan_OpenMP_MatVec, right, rmdspan<double,stdex::dynamic_extent,stdex::dynamic_extent>(), 100000, 5000);


template <class MDSpanMatrix, class... DynSizes>
void BM_MDSpan_OpenMP_MatVec_Raw_Left(benchmark::State& state, MDSpanMatrix, DynSizes... dyn) {

  using value_type = typename MDSpanMatrix::value_type;
  using MDSpanVector = lmdspan<value_type,stdex::dynamic_extent>;

  auto buffer_size_A = MDSpanMatrix{nullptr, dyn...}.mapping().required_span_size();
  auto buffer_A = std::make_unique<value_type[]>(buffer_size_A);
  auto A = MDSpanMatrix{buffer_A.get(), dyn...};
  OpenMP_first_touch_2D(A);
  mdspan_benchmark::fill_random(A);

  auto buffer_size_x = MDSpanVector{nullptr, A.extent(1)}.mapping().required_span_size();
  auto buffer_x = std::make_unique<value_type[]>(buffer_size_x);
  auto x = MDSpanVector{buffer_x.get(), A.extent(1)};
  OpenMP_first_touch_1D(x);
  mdspan_benchmark::fill_random(x);

  auto buffer_size_y = MDSpanVector{nullptr, A.extent(0)}.mapping().required_span_size();
  auto buffer_y = std::make_unique<value_type[]>(buffer_size_y);
  auto y = MDSpanVector{buffer_y.get(), A.extent(0)};
  OpenMP_first_touch_1D(y);
  mdspan_benchmark::fill_random(y);

  index_type N = A.extent(0);
  index_type M = A.extent(1);

  value_type* p_A = A.data_handle();
  value_type* p_x = x.data_handle();
  value_type* p_y = y.data_handle();

  #pragma omp parallel for
  for(index_type i = 0; i < N; i ++) {
    value_type y_i = 0;
    for(index_type j = 0; j < M; j ++) {
      y_i += p_A[i + j * N] * x[j];
    }
    y[i] = y_i;
  }

  int R = 10;
  for (auto _ : state) {
    benchmark::DoNotOptimize(A.data_handle());
    benchmark::DoNotOptimize(y.data_handle());
    benchmark::DoNotOptimize(x.data_handle());
    for(int r=0; r<R; r++) {
    #pragma omp parallel for
    for(index_type i = 0; i < A.extent(0); i ++) {
      value_type y_i = 0;
      for(index_type j = 0; j < A.extent(1); j ++) {
        y_i += p_A[i + j * N] * p_x[j];
      }
      p_y[i] += y_i;
    }
    }
    benchmark::ClobberMemory();
  }
  size_t num_elements = 2 * A.extent(0) * A.extent(1) + 2 * A.extent(0);
  state.SetBytesProcessed( R * num_elements * sizeof(value_type) * state.iterations() * global_repeat);
  state.counters["repeats"] = global_repeat;
}

BENCHMARK_CAPTURE(BM_MDSpan_OpenMP_MatVec_Raw_Left, left, lmdspan<double,stdex::dynamic_extent,stdex::dynamic_extent>(), 100000, 5000);

template <class MDSpanMatrix, class... DynSizes>
void BM_MDSpan_OpenMP_MatVec_Raw_Right(benchmark::State& state, MDSpanMatrix, DynSizes... dyn) {

  using value_type = typename MDSpanMatrix::value_type;
  using MDSpanVector = lmdspan<value_type,stdex::dynamic_extent>;

  auto buffer_size_A = MDSpanMatrix{nullptr, dyn...}.mapping().required_span_size();
  auto buffer_A = std::make_unique<value_type[]>(buffer_size_A);
  auto A = MDSpanMatrix{buffer_A.get(), dyn...};
  OpenMP_first_touch_2D(A);
  mdspan_benchmark::fill_random(A);

  auto buffer_size_x = MDSpanVector{nullptr, A.extent(1)}.mapping().required_span_size();
  auto buffer_x = std::make_unique<value_type[]>(buffer_size_x);
  auto x = MDSpanVector{buffer_x.get(), A.extent(1)};
  OpenMP_first_touch_1D(x);
  mdspan_benchmark::fill_random(x);

  auto buffer_size_y = MDSpanVector{nullptr, A.extent(0)}.mapping().required_span_size();
  auto buffer_y = std::make_unique<value_type[]>(buffer_size_y);
  auto y = MDSpanVector{buffer_y.get(), A.extent(0)};
  OpenMP_first_touch_1D(y);
  mdspan_benchmark::fill_random(y);

  index_type N = A.extent(0);
  index_type M = A.extent(1);

  value_type* p_A = A.data_handle();
  value_type* p_x = x.data_handle();
  value_type* p_y = y.data_handle();

  #pragma omp parallel for
  for(index_type i = 0; i < N; i ++) {
    value_type y_i = 0;
    for(index_type j = 0; j < M; j ++) {
      y_i += p_A[i * M + j] * x[j];
    }
    y[i] = y_i;
  }

  int R = 10;
  for (auto _ : state) {
    benchmark::DoNotOptimize(A.data_handle());
    benchmark::DoNotOptimize(y.data_handle());
    benchmark::DoNotOptimize(x.data_handle());
    for(int r=0; r<R; r++) {
    #pragma omp parallel for
    for(index_type i = 0; i < A.extent(0); i ++) {
      value_type y_i = 0;
      for(index_type j = 0; j < A.extent(1); j ++) {
        y_i += p_A[i * M + j] * p_x[j];
      }
      p_y[i] += y_i;
    }
    }
    benchmark::ClobberMemory();
  }
  size_t num_elements = 2 * A.extent(0) * A.extent(1) + 2 * A.extent(0);
  state.SetBytesProcessed( R * num_elements * sizeof(value_type) * state.iterations() * global_repeat);
  state.counters["repeats"] = global_repeat;
}

BENCHMARK_CAPTURE(BM_MDSpan_OpenMP_MatVec_Raw_Right, right, rmdspan<double,stdex::dynamic_extent,stdex::dynamic_extent>(), 100000, 5000);

//================================================================================

BENCHMARK_MAIN();
