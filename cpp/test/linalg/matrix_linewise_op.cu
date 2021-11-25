/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_profiler_api.h>
#include <gtest/gtest.h>
#include <nvToolsExt.h>
#include <raft/cudart_utils.h>
#include <raft/linalg/matrix_linewise_op.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/random/rng.hpp>
#include <rmm/device_uvector.hpp>
#include "../test_utils.h"
#include "matrix_vector_op.cuh"

namespace raft {
namespace linalg {

constexpr std::size_t PTR_PADDING = 128;

template <typename... Args>
void PUSH_RANGE(rmm::cuda_stream_view stream, const char* name, Args... args) {
  int length = std::snprintf(nullptr, 0, name, args...);
  assert(length >= 0);
  auto buf = std::make_unique<char[]>(length + 1);
  std::snprintf(buf.get(), length + 1, name, args...);
  stream.synchronize();
  nvtxRangePushA(buf.get());
}
template <>
void PUSH_RANGE(rmm::cuda_stream_view stream, const char* name) {
  stream.synchronize();
  nvtxRangePushA(name);
}

void POP_RANGE(rmm::cuda_stream_view stream) {
  stream.synchronize();
  nvtxRangePop();
}

struct LinewiseTestParams {
  double tolerance;
  std::size_t workSizeBytes;
  uint64_t seed;
  bool useVanillaMatrixVectorOp;
  bool checkCorrectness;
  int inAlignOffset;
  int outAlignOffset;
};

template <typename T, typename I, typename ParamsReader>
struct LinewiseTest
  : public ::testing::TestWithParam<typename ParamsReader::Params> {
  const LinewiseTestParams params;
  const raft::handle_t handle;
  rmm::cuda_stream_view stream;

  LinewiseTest()
    : testing::TestWithParam<typename ParamsReader::Params>(),
      params(ParamsReader::read(
        ::testing::TestWithParam<typename ParamsReader::Params>::GetParam())),
      handle(),
      stream(handle.get_stream_view()) {}

  void runLinewiseSum(T* out, const T* in, const I lineLen, const I nLines,
                      const bool alongLines, const T* vec) {
    auto f = [] __device__(T a, T b) -> T { return a + b; };
    if (params.useVanillaMatrixVectorOp)
      matrixVectorOp(out, in, vec, lineLen, nLines, true, alongLines, f,
                     stream);
    else
      matrixLinewiseOp(out, in, lineLen, nLines, alongLines, f, stream, vec);
  }

  void runLinewiseSum(T* out, const T* in, const I lineLen, const I nLines,
                      const bool alongLines, const T* vec1, const T* vec2) {
    auto f = [] __device__(T a, T b, T c) -> T { return a + b + c; };
    if (params.useVanillaMatrixVectorOp)
      matrixVectorOp(out, in, vec1, vec2, lineLen, nLines, true, alongLines, f,
                     stream);
    else
      matrixLinewiseOp(out, in, lineLen, nLines, alongLines, f, stream, vec1,
                       vec2);
  }

  rmm::device_uvector<T> genData() {
    raft::random::Rng r(params.seed);
    const std::size_t workSizeElems = params.workSizeBytes / sizeof(T);
    rmm::device_uvector<T> blob(workSizeElems, stream);
    r.uniform(blob.data(), workSizeElems, T(-1.0), T(1.0), stream);
    return blob;
  }

  /**
   * Suggest multiple versions of matrix dimensions (n, m), such that
   *
   * (2 * n * m + numVectors * m + minUnused) * sizeof(T) <= workSize.
   *
   * This way I know I can create two matrices and numVectors vectors of size m,
   * such that they fit into the allocated workSet.
   */
  std::vector<std::tuple<I, I>> suggestDimensions(I numVectors) {
    const std::size_t workSizeElems = params.workSizeBytes / sizeof(T);
    std::vector<std::tuple<I, I>> out;
    const double b = double(numVectors);
    const double s = double(workSizeElems) - double(PTR_PADDING * 2 * (2 + b));
    double squareN = 0.25 * (sqrt(8.0 * s + b * b) - b);

    auto solveForN = [s, b](I m) -> double {
      return (s - b * double(m)) / double(2 * m);
    };
    auto solveForM = [s, b](I n) -> double { return s / double(2 * n + b); };
    auto addIfMakesSense = [&out](double x, double y) {
      if (x <= 0 || y <= 0) return;
      I n = I(floor(x));
      I m = I(floor(y));
      if (n > 0 && m > 0) out.push_back(std::make_tuple(n, m));
    };
    std::vector<double> sizes = {15, 16, 17, 256, 257, 263, 1024};
    addIfMakesSense(squareN, squareN);
    for (I k : sizes) {
      addIfMakesSense(solveForN(k), k);
      addIfMakesSense(k, solveForM(k));
    }

    return out;
  }

  std::tuple<T*, const T*, const T*, const T*> assignSafePtrs(
    rmm::device_uvector<T>& blob, I n, I m) {
    typedef raft::Pow2<PTR_PADDING> Align;
    T* out = Align::roundUp(blob.data()) + params.outAlignOffset;
    const T* in =
      const_cast<const T*>(Align::roundUp(out + n * m + PTR_PADDING)) +
      params.inAlignOffset;
    const T* vec1 = Align::roundUp(in + n * m + PTR_PADDING);
    const T* vec2 = Align::roundUp(vec1 + m + PTR_PADDING);
    ASSERT(blob.data() + blob.size() >= vec2 + PTR_PADDING,
           "Failed to allocate pointers: the workset is not big enough.");
    return std::make_tuple(out, in, vec1, vec2);
  }

  testing::AssertionResult run() {
    rmm::device_uvector<T> blob = genData();
    rmm::device_uvector<T> blob_val(
      params.checkCorrectness ? blob.size() / 2 : 0, stream);

    auto dims = suggestDimensions(2);

    stream.synchronize();
    cudaProfilerStart();
    PUSH_RANGE(stream, params.useVanillaMatrixVectorOp ? "method: original"
                                                       : "method: linewise");
    for (auto [n, m] : dims) {
      auto [out, in, vec1, vec2] = assignSafePtrs(blob, n, m);
      PUSH_RANGE(stream, "Dims-%zu-%zu", std::size_t(n), std::size_t(m));
      for (auto alongRows : ::testing::Bool()) {
        PUSH_RANGE(stream, alongRows ? "alongRows" : "acrossRows");
        auto lineLen = alongRows ? m : n;
        auto nLines = alongRows ? n : m;
        {
          PUSH_RANGE(stream, "one vec");
          runLinewiseSum(out, in, lineLen, nLines, alongRows, vec1);
          POP_RANGE(stream);
          if (params.checkCorrectness) {
            naiveMatVec(blob_val.data(), in, vec1, lineLen, nLines, true,
                        alongRows, T(1));
            EXPECT_NO_FATAL_FAILURE(
              devArrMatch(blob_val.data(), out, n * m,
                          CompareApprox<float>(params.tolerance)))
              << "with one vec";
          }
          PUSH_RANGE(stream, "two vecs");
          runLinewiseSum(out, in, lineLen, nLines, alongRows, vec1, vec2);
          POP_RANGE(stream);
          if (params.checkCorrectness) {
            naiveMatVec(blob_val.data(), in, vec1, vec2, lineLen, nLines, true,
                        alongRows, T(1));
            EXPECT_NO_FATAL_FAILURE(
              devArrMatch(blob_val.data(), out, n * m,
                          CompareApprox<float>(params.tolerance)))
              << "with two vecs";
          }
        }
        POP_RANGE(stream);
      }
      POP_RANGE(stream);
    }
    POP_RANGE(stream);
    cudaProfilerStop();

    return testing::AssertionSuccess();
  }
};

#define TEST_IT(fun, TestClass, ElemType, IndexType)                         \
  typedef LinewiseTest<ElemType, IndexType, TestClass>                       \
    TestClass##_##ElemType##_##IndexType;                                    \
  TEST_P(TestClass##_##ElemType##_##IndexType, fun) { ASSERT_TRUE(fun()); }  \
  INSTANTIATE_TEST_SUITE_P(LinewiseOp, TestClass##_##ElemType##_##IndexType, \
                           TestClass##Params)

auto MegabyteParams =
  ::testing::Combine(::testing::Bool(), ::testing::Values(0, 1, 2, 4),
                     ::testing::Values(0, 1, 2, 3));

struct Megabyte {
  typedef std::tuple<bool, int, int> Params;
  static LinewiseTestParams read(Params ps) {
    return {/** .tolerance */ 0.00001,
            /** .workSizeBytes */ 1024 * 1024,
            /** .seed */ 42ULL,
            /** .useVanillaMatrixVectorOp */ std::get<0>(ps),
            /** .checkCorrectness */ true,
            /** .inAlignOffset */ std::get<1>(ps),
            /** .outAlignOffset */ std::get<2>(ps)};
  }
};

auto GigabyteParams = ::testing::Combine(
  ::testing::Bool(), ::testing::Values(0, 1, 2), ::testing::Values(0, 1, 2));

struct Gigabyte {
  typedef std::tuple<bool, int, int> Params;
  static LinewiseTestParams read(Params ps) {
    return {/** .tolerance */ 0.00001,
            /** .workSizeBytes */ 1024 * 1024 * 1024,
            /** .seed */ 42ULL,
            /** .useVanillaMatrixVectorOp */ std::get<0>(ps),
            /** .checkCorrectness */ false,
            /** .inAlignOffset */ std::get<1>(ps),
            /** .outAlignOffset */ std::get<2>(ps)};
  }
};

auto TenGigsParams = GigabyteParams;

struct TenGigs {
  typedef std::tuple<bool, int, int> Params;
  static LinewiseTestParams read(Params ps) {
    return {/** .tolerance */ 0.00001,
            /** .workSizeBytes */ 10ULL * 1024ULL * 1024ULL * 1024ULL,
            /** .seed */ 42ULL,
            /** .useVanillaMatrixVectorOp */ std::get<0>(ps),
            /** .checkCorrectness */ false,
            /** .inAlignOffset */ std::get<1>(ps),
            /** .outAlignOffset */ std::get<2>(ps)};
  }
};

TEST_IT(run, Megabyte, float, int);
TEST_IT(run, Megabyte, double, int);
TEST_IT(run, Gigabyte, float, int);
TEST_IT(run, Gigabyte, double, int);
TEST_IT(run, TenGigs, float, uint64_t);
TEST_IT(run, TenGigs, double, uint64_t);

}  // end namespace linalg
}  // end namespace raft
