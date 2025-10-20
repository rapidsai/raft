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

#ifndef MDSPAN_BENCHMARKS_SUM_SUM_3D_COMMON_HPP
#define MDSPAN_BENCHMARKS_SUM_SUM_3D_COMMON_HPP

#include <benchmark/benchmark.h>

#include "fill.hpp"

namespace stdex = std::experimental;

template <class T, class Size>
void BM_Raw_Sum_1D(benchmark::State& state, T, Size size) {
  auto buffer = std::make_unique<T[]>(size);
  {
    // just for setup...
    auto wrapped = stdex::mdspan<T, stdex::dextents<size_t, 1>>{buffer.get(), size};
    mdspan_benchmark::fill_random(wrapped);
  }
  T* data = buffer.get();
  for (auto _ : state) {
    T sum = 0;
    for(Size i = 0; i < size; ++i) {
      sum += data[i];
    }
    benchmark::DoNotOptimize(sum);
    benchmark::DoNotOptimize(data);
  }
  state.SetBytesProcessed(size * sizeof(T) * state.iterations());
}

//==============================================================================

template <class T, class SizeX, class SizeY, class SizeZ>
void BM_Raw_Sum_3D_right(benchmark::State& state, T, SizeX x, SizeY y, SizeZ z) {

  benchmark::DoNotOptimize(x);
  benchmark::DoNotOptimize(y);
  benchmark::DoNotOptimize(z);

  auto buffer = std::make_unique<T[]>(x * y * z);
  {
    // just for setup...
    auto wrapped = stdex::mdspan<T, stdex::dextents<size_t, 1>>{buffer.get(), x*y*z};
    mdspan_benchmark::fill_random(wrapped);
  }
  T* data = buffer.get();


  for (auto _ : state) {
    benchmark::DoNotOptimize(data);
    T sum = 0;
    for(SizeX i = 0; i < x; ++i) {
      for(SizeY j = 0; j < y; ++j) {
        for(SizeZ k = 0; k < z; ++k) {
          sum += data[k + j*z + i*z*y];
        }
      }
    }
    benchmark::DoNotOptimize(sum);
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(x * y * z * sizeof(T) * state.iterations());
}

//================================================================================

template <class T, class SizeX, class SizeY, class SizeZ>
void BM_Raw_Sum_3D_left(benchmark::State& state, T, SizeX x, SizeY y, SizeZ z) {
  auto buffer = std::make_unique<T[]>(x * y * z);
  {
    // just for setup...
    auto wrapped = stdex::mdspan<T, stdex::dextents<size_t, 1>>{buffer.get(), x*y*z};
    mdspan_benchmark::fill_random(wrapped);
  }
  T* data = buffer.get();
  for (auto _ : state) {
    benchmark::DoNotOptimize(data);
    T sum = 0;
    for(SizeZ k = 0; k < z; ++k) {
      for(SizeY j = 0; j < y; ++j) {
        for(SizeX i = 0; i < x; ++i) {
          sum += data[i + j*x + k*x*y];
        }
      }
    }
    benchmark::DoNotOptimize(sum);
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(x * y * z * sizeof(T) * state.iterations());
}

//================================================================================

template <class T, class SizeX, class SizeY, class SizeZ>
void BM_Raw_Sum_3D_right_iter_left(benchmark::State& state, T, SizeX x, SizeY y, SizeZ z) {
  auto buffer = std::make_unique<T[]>(x * y * z);
  {
    // just for setup...
    auto wrapped = stdex::mdspan<T, stdex::dextents<size_t, 1>>{buffer.get(), x*y*z};
    mdspan_benchmark::fill_random(wrapped);
  }

  benchmark::DoNotOptimize(x);
  benchmark::DoNotOptimize(y);
  benchmark::DoNotOptimize(z);
  benchmark::ClobberMemory();

  T* data = buffer.get();
  for (auto _ : state) {
    benchmark::DoNotOptimize(data);
    T sum = 0;
    for(SizeZ k = 0; k < z; ++k) {
      for(SizeY j = 0; j < y; ++j) {
        for(SizeX i = 0; i < x; ++i) {
          sum += data[k + j*z + i*z*y];
        }
      }
    }
    benchmark::DoNotOptimize(sum);
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(x * y * z * sizeof(T) * state.iterations());
}

//================================================================================

//================================================================================

template <class T, size_t x, size_t y, size_t z>
void BM_Raw_Static_Sum_3D_right(benchmark::State& state, T,
  std::integral_constant<size_t, x>,
  std::integral_constant<size_t, y>,
  std::integral_constant<size_t, z>
) {
  auto buffer = std::make_unique<T[]>(x * y * z);
  {
    // just for setup...
    auto wrapped = stdex::mdspan<T, stdex::dextents<size_t, 1>>{buffer.get(), x*y*z};
    mdspan_benchmark::fill_random(wrapped);
  }
  T* data = buffer.get();
  for (auto _ : state) {
    benchmark::DoNotOptimize(data);
    T sum = 0;
    for(size_t i = 0; i < x; ++i) {
      for(size_t j = 0; j < y; ++j) {
        for(size_t k = 0; k < z; ++k) {
          sum += data[k + j*z + i*z*y];
        }
      }
    }
    benchmark::DoNotOptimize(sum);
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(x * y * z * sizeof(T) * state.iterations());
}

//================================================================================

template <class T, size_t x, size_t y, size_t z>
void BM_Raw_Static_Sum_3D_left(benchmark::State& state, T,
  std::integral_constant<size_t, x>,
  std::integral_constant<size_t, y>,
  std::integral_constant<size_t, z>
) {
  auto buffer = std::make_unique<T[]>(x * y * z);
  {
    // just for setup...
    auto wrapped = stdex::mdspan<T, stdex::dextents<size_t, 1>>{buffer.get(), x*y*z};
    mdspan_benchmark::fill_random(wrapped);
  }
  T* data = buffer.get();
  for (auto _ : state) {
    benchmark::DoNotOptimize(data);
    T sum = 0;
    for(size_t k = 0; k < z; ++k) {
      for(size_t j = 0; j < y; ++j) {
        for(size_t i = 0; i < x; ++i) {
          sum += data[i + j*x + k*x*y];
        }
      }
    }
    benchmark::DoNotOptimize(sum);
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(x * y * z * sizeof(T) * state.iterations());
}

//================================================================================

template <class T, size_t x, size_t y, size_t z>
void BM_Raw_Static_Sum_3D_right_iter_left(benchmark::State& state, T,
  std::integral_constant<size_t, x>,
  std::integral_constant<size_t, y>,
  std::integral_constant<size_t, z>
) {
  auto buffer = std::make_unique<T[]>(x * y * z);
  {
    // just for setup...
    auto wrapped = stdex::mdspan<T, stdex::dextents<size_t, 1>>{buffer.get(), x*y*z};
    mdspan_benchmark::fill_random(wrapped);
  }
  T* data = buffer.get();
  for (auto _ : state) {
    benchmark::DoNotOptimize(data);
    T sum = 0;
    for(size_t k = 0; k < z; ++k) {
      for(size_t j = 0; j < y; ++j) {
        for(size_t i = 0; i < x; ++i) {
          sum += data[k + j*z + i*z*y];
        }
      }
    }
    benchmark::DoNotOptimize(sum);
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(x * y * z * sizeof(T) * state.iterations());
}


#endif // MDSPAN_BENCHMARKS_SUM_SUM_3D_COMMON_HPP
