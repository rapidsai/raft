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

template <typename IdxType = int>
struct MatVecOpInputs {
  IdxType rows, cols;
  bool rowMajor, bcastAlongRows;
  unsigned long long int seed;
};

template <typename IdxType>
::std::ostream& operator<<(::std::ostream& os, const MatVecOpInputs<IdxType>& dims)
{
  return os;
}

template <typename T, typename LenT>
inline void gen_uniform(const raft::handle_t& handle, raft::random::RngState& rng, T* ptr, LenT len)
{
  if constexpr (std::is_integral_v<T>) {
    raft::random::uniformInt(handle, rng, ptr, len, (T)0, (T)100);
  } else {
    raft::random::uniform(handle, rng, ptr, len, (T)-10.0, (T)10.0);
  }
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
class MatVecOpTest : public ::testing::TestWithParam<MatVecOpInputs<IdxType>> {
 public:
  MatVecOpTest()
    : params(::testing::TestWithParam<MatVecOpInputs<IdxType>>::GetParam()),
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
    gen_uniform<MatT>(handle, r, in.data(), len);
    gen_uniform<Vec1T>(handle, r, vec1.data(), vecLen);
    gen_uniform<Vec2T>(handle, r, vec2.data(), vecLen);
    if constexpr (OpT::useTwoVectors) {
      naiveMatVec(out_ref.data(),
                  in.data(),
                  vec1.data(),
                  vec2.data(),
                  D,
                  N,
                  params.rowMajor,
                  params.bcastAlongRows,
                  OpT{},
                  stream);
    } else {
      naiveMatVec(out_ref.data(),
                  in.data(),
                  vec1.data(),
                  D,
                  N,
                  params.rowMajor,
                  params.bcastAlongRows,
                  OpT{},
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

  MatVecOpInputs<IdxType> params;
  rmm::device_uvector<MatT> in;
  rmm::device_uvector<OutT> out;
  rmm::device_uvector<OutT> out_ref;
  rmm::device_uvector<Vec1T> vec1;
  rmm::device_uvector<Vec2T> vec2;
};

#define MVTEST(TestClass, inputs, tolerance)                                                   \
  TEST_P(TestClass, Result)                                                                    \
  {                                                                                            \
    ASSERT_TRUE(devArrMatch(                                                                   \
      out_ref.data(), out.data(), params.rows* params.cols, CompareApprox<float>(tolerance))); \
  }                                                                                            \
  INSTANTIATE_TEST_SUITE_P(MatVecOpTests, TestClass, ::testing::ValuesIn(inputs))

#define MV_EPS_F 0.00001f
#define MV_EPS_D 0.0000001

/*
 * This set of tests covers cases where all the types are the same.
 */

const std::vector<MatVecOpInputs<int>> inputs_i32    = {{1024, 32, true, true, 1234ULL},
                                                     {1024, 64, true, true, 1234ULL},
                                                     {1024, 32, true, false, 1234ULL},
                                                     {1024, 64, true, false, 1234ULL},
                                                     {1024, 32, false, true, 1234ULL},
                                                     {1024, 64, false, true, 1234ULL},
                                                     {1024, 32, false, false, 1234ULL},
                                                     {1024, 64, false, false, 1234ULL}};
const std::vector<MatVecOpInputs<size_t>> inputs_i64 = {{2500, 250, false, false, 1234ULL},
                                                        {2500, 250, false, false, 1234ULL}};

template <typename T>
struct Add1Vec {
  static constexpr bool useTwoVectors = false;
  HDI T operator()(T a, T b) const { return a + b; };
};
template <typename T>
struct Add2Vec {
  static constexpr bool useTwoVectors = true;
  HDI T operator()(T a, T b, T c) const { return a + b + c; };
};

typedef MatVecOpTest<Add1Vec<float>, float, int> MatVecOpTestF_i32_add1vec;
typedef MatVecOpTest<Add2Vec<float>, float, int> MatVecOpTestF_i32_add2vec;
typedef MatVecOpTest<Add1Vec<float>, float, size_t> MatVecOpTestF_i64_add1vec;
typedef MatVecOpTest<Add2Vec<float>, float, size_t> MatVecOpTestF_i64_add2vec;
typedef MatVecOpTest<Add1Vec<double>, double, int> MatVecOpTestD_i32_add1vec;
typedef MatVecOpTest<Add2Vec<double>, double, int> MatVecOpTestD_i32_add2vec;
typedef MatVecOpTest<Add1Vec<double>, double, size_t> MatVecOpTestD_i64_add1vec;
typedef MatVecOpTest<Add2Vec<double>, double, size_t> MatVecOpTestD_i64_add2vec;

MVTEST(MatVecOpTestF_i32_add1vec, inputs_i32, MV_EPS_F);
MVTEST(MatVecOpTestF_i32_add2vec, inputs_i32, MV_EPS_F);
MVTEST(MatVecOpTestF_i64_add1vec, inputs_i64, MV_EPS_F);
MVTEST(MatVecOpTestF_i64_add2vec, inputs_i64, MV_EPS_F);
MVTEST(MatVecOpTestD_i32_add1vec, inputs_i32, MV_EPS_D);
MVTEST(MatVecOpTestD_i32_add2vec, inputs_i32, MV_EPS_D);
MVTEST(MatVecOpTestD_i64_add1vec, inputs_i64, MV_EPS_D);
MVTEST(MatVecOpTestD_i64_add2vec, inputs_i64, MV_EPS_D);

/*
 * This set of tests covers cases with different types.
 */

template <typename OutT, typename MatT, typename Vec1T, typename Vec2T>
struct MulAndAdd {
  static constexpr bool useTwoVectors = true;
  HDI OutT operator()(MatT a, Vec1T b, Vec2T c) const { return a * b + c; };
};

typedef MatVecOpTest<MulAndAdd<float, float, int32_t, float>, float, int, float, int32_t, float>
  MatVecOpTestF_i32_MulAndAdd_f_i32_f;
typedef MatVecOpTest<MulAndAdd<float, float, int32_t, double>, float, int, float, int32_t, double>
  MatVecOpTestF_i32_MulAndAdd_f_i32_d;
typedef MatVecOpTest<MulAndAdd<float, float, int64_t, float>, float, int, float, int64_t, float>
  MatVecOpTestF_i32_MulAndAdd_f_i64_f;
typedef MatVecOpTest<MulAndAdd<double, double, int32_t, float>, double, int, double, int32_t, float>
  MatVecOpTestD_i32_MulAndAdd_d_i32_f;

MVTEST(MatVecOpTestF_i32_MulAndAdd_f_i32_f, inputs_i32, MV_EPS_F);
MVTEST(MatVecOpTestF_i32_MulAndAdd_f_i32_d, inputs_i32, MV_EPS_F);
MVTEST(MatVecOpTestF_i32_MulAndAdd_f_i64_f, inputs_i32, MV_EPS_F);
MVTEST(MatVecOpTestD_i32_MulAndAdd_d_i32_f, inputs_i32, (double)MV_EPS_F);

}  // end namespace linalg
}  // end namespace raft
