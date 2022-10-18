/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

#include "../test_utils.h"
#include "matrix_vector_op.cuh"
#include <gtest/gtest.h>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

namespace raft {
namespace linalg {

template <typename T, typename IdxType = int>
struct MatVecOpInputs {
  T tolerance;
  IdxType rows, cols;
  bool rowMajor, bcastAlongRows;
  unsigned long long int seed;
};

template <typename T, typename IdxType>
::std::ostream& operator<<(::std::ostream& os, const MatVecOpInputs<T, IdxType>& dims)
{
  return os;
}

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access
// within its class
template <typename OpT,
          typename OutT,
          typename IdxType,
          typename MatT,
          typename Vec1T,
          typename Vec2T>
void matrixVectorOpLaunch(const raft::handle_t& handle,
                          OutT* out,
                          const MatT* in,
                          const Vec1T* vec1,
                          const Vec2T* vec2,
                          IdxType D,
                          IdxType N,
                          bool rowMajor,
                          bool bcastAlongRows)
{
  auto out_row_major = raft::make_device_matrix_view<OutT, IdxType, raft::row_major>(out, N, D);
  auto in_row_major = raft::make_device_matrix_view<const MatT, IdxType, raft::row_major>(in, N, D);

  auto out_col_major = raft::make_device_matrix_view<OutT, IdxType, raft::col_major>(out, N, D);
  auto in_col_major = raft::make_device_matrix_view<const MatT, IdxType, raft::col_major>(in, N, D);

  auto apply     = bcastAlongRows ? Apply::ALONG_ROWS : Apply::ALONG_COLUMNS;
  auto len       = bcastAlongRows ? D : N;
  auto vec1_view = raft::make_device_vector_view<const Vec1T, IdxType>(vec1, len);
  
  if constexpr (OpT::useTwoVectors) {
    auto vec2_view = raft::make_device_vector_view<const Vec2T, IdxType>(vec2, len);
    if (rowMajor) {
      matrix_vector_op(handle, in_row_major, vec1_view, vec2_view, out_row_major, apply, OpT{});
    } else {
      matrix_vector_op(handle, in_col_major, vec1_view, vec2_view, out_col_major, apply, OpT{});
    }
  } else {
    if (rowMajor) {
      matrix_vector_op(handle, in_row_major, vec1_view, out_row_major, apply, OpT{});
    } else {
      matrix_vector_op(handle, in_col_major, vec1_view, out_col_major, apply, OpT{});
    }
  }
}

template <typename OpT,
          typename OutT,
          typename IdxType,
          typename MatT  = OutT,
          typename Vec1T = MatT,
          typename Vec2T = Vec1T>
class MatVecOpTest : public ::testing::TestWithParam<MatVecOpInputs<OutT, IdxType>> {
 public:
  MatVecOpTest()
    : params(::testing::TestWithParam<MatVecOpInputs<OutT, IdxType>>::GetParam()),
      stream(handle.get_stream()),
      in(params.rows * params.cols, stream),
      out_ref(params.rows * params.cols, stream),
      out(params.rows * params.cols, stream),
      vec1(params.bcastAlongRows ? params.cols : params.rows, stream),
      vec2(params.bcastAlongRows ? params.cols : params.rows, stream)
  {
  }

 protected:
  void SetUp() override
  {
    raft::random::RngState r(params.seed);
    IdxType N = params.rows, D = params.cols;
    IdxType len    = N * D;
    IdxType vecLen = params.bcastAlongRows ? D : N;
    uniform(handle, r, in.data(), len, (MatT)-1.0, (MatT)1.0);
    uniform(handle, r, vec1.data(), vecLen, (Vec1T)-1.0, (Vec1T)1.0);
    uniform(handle, r, vec2.data(), vecLen, (Vec2T)-1.0, (Vec2T)1.0);
    if constexpr (OpT::useTwoVectors) {
      naiveMatVec(out_ref.data(),
                  in.data(),
                  vec1.data(),
                  vec2.data(),
                  D,
                  N,
                  params.rowMajor,
                  params.bcastAlongRows,
                  (OutT)1.0,
                  stream);
    } else {
      naiveMatVec(out_ref.data(),
                  in.data(),
                  vec1.data(),
                  D,
                  N,
                  params.rowMajor,
                  params.bcastAlongRows,
                  (OutT)1.0,
                  stream);
    }
    matrixVectorOpLaunch<OpT>(handle,
                              out.data(),
                              in.data(),
                              vec1.data(),
                              vec2.data(),
                              D,
                              N,
                              params.rowMajor,
                              params.bcastAlongRows);
    handle.sync_stream();
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;

  MatVecOpInputs<OutT, IdxType> params;
  rmm::device_uvector<MatT> in;
  rmm::device_uvector<OutT> out;
  rmm::device_uvector<MatT> out_ref;
  rmm::device_uvector<Vec1T> vec1;
  rmm::device_uvector<Vec2T> vec2;
};

#define MVTEST(TestClass, inputs)                                     \
  TEST_P(TestClass, Result)                                           \
  {                                                                   \
    ASSERT_TRUE(devArrMatch(out_ref.data(),                           \
                            out.data(),                               \
                            params.rows* params.cols,                 \
                            CompareApprox<float>(params.tolerance))); \
  }                                                                   \
  INSTANTIATE_TEST_SUITE_P(MatVecOpTests, TestClass, ::testing::ValuesIn(inputs))

const std::vector<MatVecOpInputs<float, int>> inputsf_i32 = {
  {0.00001f, 1024, 32, true, true, 1234ULL},
  {0.00001f, 1024, 64, true, true, 1234ULL},
  {0.00001f, 1024, 32, true, false, 1234ULL},
  {0.00001f, 1024, 64, true, false, 1234ULL},
  {0.00001f, 1024, 32, false, true, 1234ULL},
  {0.00001f, 1024, 64, false, true, 1234ULL},
  {0.00001f, 1024, 32, false, false, 1234ULL},
  {0.00001f, 1024, 64, false, false, 1234ULL}};
const std::vector<MatVecOpInputs<float, size_t>> inputsf_i64 = {
  {0.00001f, 2500, 250, false, false, 1234ULL}, {0.00001f, 2500, 250, false, false, 1234ULL}};
const std::vector<MatVecOpInputs<double, int>> inputsd_i32 = {
  {0.0000001, 1024, 32, true, true, 1234ULL},
  {0.0000001, 1024, 64, true, true, 1234ULL},
  {0.0000001, 1024, 32, true, false, 1234ULL},
  {0.0000001, 1024, 64, true, false, 1234ULL},
  {0.0000001, 1024, 32, false, true, 1234ULL},
  {0.0000001, 1024, 64, false, true, 1234ULL},
  {0.0000001, 1024, 32, false, false, 1234ULL},
  {0.0000001, 1024, 64, false, false, 1234ULL}};
const std::vector<MatVecOpInputs<double, size_t>> inputsd_i64 = {
  {0.0000001, 2500, 250, false, false, 1234ULL}, {0.0000001, 2500, 250, false, false, 1234ULL}};

template <typename T>
struct Add1Vec {
  static constexpr bool useTwoVectors = false;
  HDI float operator()(T a, T b) const { return a + b; };
};
template <typename T>
struct Add2Vec {
  static constexpr bool useTwoVectors = true;
  HDI float operator()(T a, T b, T c) const { return a + b + c; };
};

typedef MatVecOpTest<Add1Vec<float>, float, int> MatVecOpTestF_i32_add1vec;
typedef MatVecOpTest<Add2Vec<float>, float, int> MatVecOpTestF_i32_add2vec;
typedef MatVecOpTest<Add1Vec<float>, float, size_t> MatVecOpTestF_i64_add1vec;
typedef MatVecOpTest<Add2Vec<float>, float, size_t> MatVecOpTestF_i64_add2vec;
typedef MatVecOpTest<Add1Vec<double>, double, int> MatVecOpTestD_i32_add1vec;
typedef MatVecOpTest<Add2Vec<double>, double, int> MatVecOpTestD_i32_add2vec;
typedef MatVecOpTest<Add1Vec<double>, double, size_t> MatVecOpTestD_i64_add1vec;
typedef MatVecOpTest<Add2Vec<double>, double, size_t> MatVecOpTestD_i64_add2vec;

MVTEST(MatVecOpTestF_i32_add1vec, inputsf_i32);
MVTEST(MatVecOpTestF_i32_add2vec, inputsf_i32);
MVTEST(MatVecOpTestF_i64_add1vec, inputsf_i64);
MVTEST(MatVecOpTestF_i64_add2vec, inputsf_i64);
MVTEST(MatVecOpTestD_i32_add1vec, inputsd_i32);
MVTEST(MatVecOpTestD_i32_add2vec, inputsd_i32);
MVTEST(MatVecOpTestD_i64_add1vec, inputsd_i64);
MVTEST(MatVecOpTestD_i64_add2vec, inputsd_i64);

}  // end namespace linalg
}  // end namespace raft
