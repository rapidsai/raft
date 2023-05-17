/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

#include "../test_utils.cuh"
#include "matrix_vector_op.cuh"
#include <gtest/gtest.h>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/itertools.hpp>
#include <type_traits>

namespace raft {
namespace linalg {

template <typename IdxType = int>
struct MatVecOpInputs {
  IdxType rows, cols;
  bool rowMajor, bcastAlongRows;
  IdxType inAlignOffset, outAlignOffset;
  unsigned long long int seed;
};

template <typename IdxType>
::std::ostream& operator<<(::std::ostream& os, const MatVecOpInputs<IdxType>& dims)
{
  return os;
}

template <typename T, typename LenT>
inline void gen_uniform(const raft::resources& handle,
                        raft::random::RngState& rng,
                        T* ptr,
                        LenT len)
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
template <typename OpT, typename MatT, typename IdxType, typename Vec1T, typename Vec2T>
void matrixVectorOpLaunch(const raft::resources& handle,
                          MatT* out,
                          const MatT* in,
                          const Vec1T* vec1,
                          const Vec2T* vec2,
                          IdxType D,
                          IdxType N,
                          bool rowMajor,
                          bool bcastAlongRows)
{
  auto out_row_major = raft::make_device_matrix_view<MatT, IdxType, raft::row_major>(out, N, D);
  auto in_row_major = raft::make_device_matrix_view<const MatT, IdxType, raft::row_major>(in, N, D);

  auto out_col_major = raft::make_device_matrix_view<MatT, IdxType, raft::col_major>(out, N, D);
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
          typename MatT,
          typename IdxType,
          typename Vec1T = MatT,
          typename Vec2T = Vec1T>
class MatVecOpTest : public ::testing::TestWithParam<MatVecOpInputs<IdxType>> {
 public:
  MatVecOpTest()
    : stream(resource::get_cuda_stream(handle)),
      params(::testing::TestWithParam<MatVecOpInputs<IdxType>>::GetParam()),
      vec_size(params.bcastAlongRows ? params.cols : params.rows),
      in(params.rows * params.cols + params.inAlignOffset, stream),
      out_ref(params.rows * params.cols + params.outAlignOffset, stream),
      out(params.rows * params.cols + params.outAlignOffset, stream),
      vec1(vec_size, stream),
      vec2(vec_size, stream)
  {
  }

 protected:
  void SetUp() override
  {
    MatT* in_ptr      = in.data() + params.inAlignOffset;
    MatT* out_ptr     = out.data() + params.outAlignOffset;
    MatT* out_ref_ptr = out_ref.data() + params.outAlignOffset;

    raft::random::RngState r(params.seed);
    IdxType len = params.rows * params.cols;
    gen_uniform<MatT>(handle, r, in_ptr, len);
    gen_uniform<Vec1T>(handle, r, vec1.data(), vec_size);
    gen_uniform<Vec2T>(handle, r, vec2.data(), vec_size);
    if constexpr (OpT::useTwoVectors) {
      naiveMatVec(out_ref_ptr,
                  in_ptr,
                  vec1.data(),
                  vec2.data(),
                  params.cols,
                  params.rows,
                  params.rowMajor,
                  params.bcastAlongRows,
                  OpT{},
                  stream);
    } else {
      naiveMatVec(out_ref_ptr,
                  in_ptr,
                  vec1.data(),
                  params.cols,
                  params.rows,
                  params.rowMajor,
                  params.bcastAlongRows,
                  OpT{},
                  stream);
    }
    matrixVectorOpLaunch<OpT>(handle,
                              out_ptr,
                              in_ptr,
                              vec1.data(),
                              vec2.data(),
                              params.cols,
                              params.rows,
                              params.rowMajor,
                              params.bcastAlongRows);
    resource::sync_stream(handle);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  MatVecOpInputs<IdxType> params;
  IdxType vec_size;
  rmm::device_uvector<MatT> in;
  rmm::device_uvector<MatT> out;
  rmm::device_uvector<MatT> out_ref;
  rmm::device_uvector<Vec1T> vec1;
  rmm::device_uvector<Vec2T> vec2;
};

#define MVTEST(TestClass, OutType, inputs, tolerance)                 \
  TEST_P(TestClass, Result)                                           \
  {                                                                   \
    if constexpr (std::is_floating_point_v<OutType>) {                \
      ASSERT_TRUE(devArrMatch(out_ref.data() + params.outAlignOffset, \
                              out.data() + params.outAlignOffset,     \
                              params.rows * params.cols,              \
                              CompareApprox<OutType>(tolerance)));    \
    } else {                                                          \
      ASSERT_TRUE(devArrMatch(out_ref.data() + params.outAlignOffset, \
                              out.data() + params.outAlignOffset,     \
                              params.rows * params.cols,              \
                              Compare<OutType>()));                   \
    }                                                                 \
  }                                                                   \
  INSTANTIATE_TEST_SUITE_P(MatVecOpTests, TestClass, ::testing::ValuesIn(inputs))

#define MV_EPS_F 0.00001f
#define MV_EPS_D 0.0000001

/*
 * This set of tests covers cases where all the types are the same.
 */

const std::vector<MatVecOpInputs<int>> inputs_i32 =
  raft::util::itertools::product<MatVecOpInputs<int>>(
    {1024}, {32, 64}, {true, false}, {true, false}, {0, 1, 2}, {0, 1, 2}, {1234ULL});
const std::vector<MatVecOpInputs<int64_t>> inputs_i64 =
  raft::util::itertools::product<MatVecOpInputs<int64_t>>(
    {2500}, {250}, {false}, {false}, {0, 1}, {0, 1}, {1234ULL});

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
typedef MatVecOpTest<Add1Vec<float>, float, int64_t> MatVecOpTestF_i64_add1vec;
typedef MatVecOpTest<Add2Vec<float>, float, int64_t> MatVecOpTestF_i64_add2vec;
typedef MatVecOpTest<Add1Vec<double>, double, int> MatVecOpTestD_i32_add1vec;
typedef MatVecOpTest<Add2Vec<double>, double, int> MatVecOpTestD_i32_add2vec;
typedef MatVecOpTest<Add1Vec<double>, double, int64_t> MatVecOpTestD_i64_add1vec;
typedef MatVecOpTest<Add2Vec<double>, double, int64_t> MatVecOpTestD_i64_add2vec;

MVTEST(MatVecOpTestF_i32_add1vec, float, inputs_i32, MV_EPS_F);
MVTEST(MatVecOpTestF_i32_add2vec, float, inputs_i32, MV_EPS_F);
MVTEST(MatVecOpTestF_i64_add1vec, float, inputs_i64, MV_EPS_F);
MVTEST(MatVecOpTestF_i64_add2vec, float, inputs_i64, MV_EPS_F);
MVTEST(MatVecOpTestD_i32_add1vec, double, inputs_i32, MV_EPS_D);
MVTEST(MatVecOpTestD_i32_add2vec, double, inputs_i32, MV_EPS_D);
MVTEST(MatVecOpTestD_i64_add1vec, double, inputs_i64, MV_EPS_D);
MVTEST(MatVecOpTestD_i64_add2vec, double, inputs_i64, MV_EPS_D);

/*
 * This set of tests covers cases with different types.
 */

template <typename MatT, typename Vec1T, typename Vec2T>
struct MulAndAdd {
  static constexpr bool useTwoVectors = true;
  HDI MatT operator()(MatT a, Vec1T b, Vec2T c) const { return a * b + c; };
};

typedef MatVecOpTest<MulAndAdd<float, int32_t, float>, float, int, int32_t, float>
  MatVecOpTestF_i32_MulAndAdd_i32_f;
typedef MatVecOpTest<MulAndAdd<float, int32_t, double>, float, int, int32_t, double>
  MatVecOpTestF_i32_MulAndAdd_i32_d;
typedef MatVecOpTest<MulAndAdd<float, int64_t, float>, float, int, int64_t, float>
  MatVecOpTestF_i32_MulAndAdd_i64_f;
typedef MatVecOpTest<MulAndAdd<double, int32_t, float>, double, int, int32_t, float>
  MatVecOpTestD_i32_MulAndAdd_i32_f;

MVTEST(MatVecOpTestF_i32_MulAndAdd_i32_f, float, inputs_i32, MV_EPS_F);
MVTEST(MatVecOpTestF_i32_MulAndAdd_i32_d, float, inputs_i32, MV_EPS_F);
MVTEST(MatVecOpTestF_i32_MulAndAdd_i64_f, float, inputs_i32, MV_EPS_F);
MVTEST(MatVecOpTestD_i32_MulAndAdd_i32_f, double, inputs_i32, (double)MV_EPS_F);

struct DQMultiply {
  static constexpr bool useTwoVectors = true;
  HDI int8_t operator()(int8_t a, float b, float c) const
  {
    return static_cast<int8_t>((static_cast<float>(a) / 100.0f * (b + c) / 20.0f) * 100.0f);
  };
};

typedef MatVecOpTest<DQMultiply, int8_t, int, float, float> MatVecOpTestI8_i32_DQMultiply_f_f;

MVTEST(MatVecOpTestI8_i32_DQMultiply_f_f, int8_t, inputs_i32, 0);

}  // end namespace linalg
}  // end namespace raft
