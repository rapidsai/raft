/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>
#include <raft/cudart_utils.h>
#include <raft/random/rng.cuh>
#include "../fixture.hpp"
#include "../test_utils.h"
#include "matrix_vector_op.cuh"

namespace raft {
namespace linalg {

template <typename T, typename IdxType = int>
struct mat_vec_op_inputs {
  T tolerance;
  IdxType rows, cols;
  bool row_major, bcast_along_rows, use_two_vectors;
  uint64_t seed;
};

template <typename T, typename IdxType>
::std::ostream &operator<<(::std::ostream &os,
                           const mat_vec_op_inputs<T, IdxType> &dims) {
  return os;
}

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access
// within its class
template <typename T, typename IdxType>
void matrix_vector_op_launch(T *out, const T *in, const T *vec1, const T *vec2,
                             IdxType D, IdxType N, bool row_major,
                             bool bcast_along_rows, bool use_two_vectors,
                             cudaStream_t stream) {
  if (use_two_vectors) {
    matrixVectorOp(
      out, in, vec1, vec2, D, N, row_major, bcast_along_rows,
      [] __device__(T a, T b, T c) { return a + b + c; }, stream);
  } else {
    matrixVectorOp(
      out, in, vec1, D, N, row_major, bcast_along_rows,
      [] __device__(T a, T b) { return a + b; }, stream);
  }
}

template <typename T, typename IdxType>
class mat_vec_op_test : public raft::fixture<mat_vec_op_inputs<T, IdxType>> {
 protected:
  void initialize() override {
    params_ =
      ::testing::TestWithParam<mat_vec_op_inputs<T, IdxType>>::GetParam();
    raft::random::Rng r(params_.seed);
    auto n = params_.rows, d = params_.cols;
    auto len = n * d;
    auto stream = this->handle().get_stream();
    allocate(in_, len);
    allocate(out_ref_, len);
    allocate(out_, len);
    auto vec_len = params_.bcast_along_rows ? d : n;
    allocate(vec1_, vec_len);
    allocate(vec2_, vec_len);
    constexpr auto kOne = static_cast<T>(1.0);
    r.uniform(in_, len, -kOne, kOne, stream);
    r.uniform(vec1_, vec_len, -kOne, kOne, stream);
    r.uniform(vec2_, vec_len, -kOne, kOne, stream);
    if (params_.use_two_vectors) {
      naive_mat_vec(out_ref_, in_, vec1_, vec2_, d, n, params_.row_major,
                    params_.bcast_along_rows, kOne);
    } else {
      naive_mat_vec(out_ref_, in_, vec1_, d, n, params_.row_major,
                    params_.bcast_along_rows, kOne);
    }
    matrix_vector_op_launch(out_, in_, vec1_, vec2_, d, n, params_.row_major,
                            params_.bcast_along_rows, params_.use_two_vectors,
                            stream);
  }

  void finalize() override {
    CUDA_CHECK(cudaFree(vec1_));
    CUDA_CHECK(cudaFree(vec2_));
    CUDA_CHECK(cudaFree(out_));
    CUDA_CHECK(cudaFree(out_ref_));
    CUDA_CHECK(cudaFree(in_));
  }

  void check() override {
    ASSERT_TRUE(devArrMatch(out_ref_, out_, params_.rows * params_.cols,
                            compare_approx<T>(params_.tolerance)));
  }

  mat_vec_op_inputs<T, IdxType> params_;
  T *in_, *out_, *out_ref_, *vec1_, *vec2_;
};

const std::vector<mat_vec_op_inputs<float, int>> kInputsFI32 = {
  {0.00001f, 1024, 32, true, true, false, 1234ULL},
  {0.00001f, 1024, 64, true, true, false, 1234ULL},
  {0.00001f, 1024, 32, true, false, false, 1234ULL},
  {0.00001f, 1024, 64, true, false, false, 1234ULL},
  {0.00001f, 1024, 32, false, true, false, 1234ULL},
  {0.00001f, 1024, 64, false, true, false, 1234ULL},
  {0.00001f, 1024, 32, false, false, false, 1234ULL},
  {0.00001f, 1024, 64, false, false, false, 1234ULL},

  {0.00001f, 1024, 32, true, true, true, 1234ULL},
  {0.00001f, 1024, 64, true, true, true, 1234ULL},
  {0.00001f, 1024, 32, true, false, true, 1234ULL},
  {0.00001f, 1024, 64, true, false, true, 1234ULL},
  {0.00001f, 1024, 32, false, true, true, 1234ULL},
  {0.00001f, 1024, 64, false, true, true, 1234ULL},
  {0.00001f, 1024, 32, false, false, true, 1234ULL},
  {0.00001f, 1024, 64, false, false, true, 1234ULL}};
using mat_vec_op_test_f_i32 = mat_vec_op_test<float, int>;
RUN_TEST_BASE(mat_vec_op, mat_vec_op_test_f_i32, kInputsFI32);

const std::vector<mat_vec_op_inputs<float, size_t>> kInputsFI64 = {
  {0.00001f, 2500, 250, false, false, false, 1234ULL},
  {0.00001f, 2500, 250, false, false, true, 1234ULL}};
using mat_vec_op_test_f_i64 = mat_vec_op_test<float, size_t>;
RUN_TEST_BASE(mat_vec_op, mat_vec_op_test_f_i64, kInputsFI64);

const std::vector<mat_vec_op_inputs<double, int>> kInputsDI32 = {
  {0.0000001, 1024, 32, true, true, false, 1234ULL},
  {0.0000001, 1024, 64, true, true, false, 1234ULL},
  {0.0000001, 1024, 32, true, false, false, 1234ULL},
  {0.0000001, 1024, 64, true, false, false, 1234ULL},
  {0.0000001, 1024, 32, false, true, false, 1234ULL},
  {0.0000001, 1024, 64, false, true, false, 1234ULL},
  {0.0000001, 1024, 32, false, false, false, 1234ULL},
  {0.0000001, 1024, 64, false, false, false, 1234ULL},

  {0.0000001, 1024, 32, true, true, true, 1234ULL},
  {0.0000001, 1024, 64, true, true, true, 1234ULL},
  {0.0000001, 1024, 32, true, false, true, 1234ULL},
  {0.0000001, 1024, 64, true, false, true, 1234ULL},
  {0.0000001, 1024, 32, false, true, true, 1234ULL},
  {0.0000001, 1024, 64, false, true, true, 1234ULL},
  {0.0000001, 1024, 32, false, false, true, 1234ULL},
  {0.0000001, 1024, 64, false, false, true, 1234ULL}};
using mat_vec_op_test_d_i32 = mat_vec_op_test<double, int>;
RUN_TEST_BASE(mat_vec_op, mat_vec_op_test_d_i32, kInputsDI32);

const std::vector<mat_vec_op_inputs<double, size_t>> kInputsDI64 = {
  {0.0000001, 2500, 250, false, false, false, 1234ULL},
  {0.0000001, 2500, 250, false, false, true, 1234ULL}};
using mat_vec_op_test_d_i64 = mat_vec_op_test<double, size_t>;
RUN_TEST_BASE(mat_vec_op, mat_vec_op_test_d_i64, kInputsDI64);

}  // end namespace linalg
}  // end namespace raft
