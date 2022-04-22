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
#include <gtest/gtest.h>
#include <raft/cudart_utils.h>
#include <raft/matrix/math.cuh>
#include <raft/random/rng.cuh>

namespace raft {
namespace matrix {

template <typename Type>
__global__ void nativePowerKernel(Type* in, Type* out, int len)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) { out[idx] = in[idx] * in[idx]; }
}

template <typename Type>
void naivePower(Type* in, Type* out, int len, cudaStream_t stream)
{
  static const int TPB = 64;
  int nblks            = raft::ceildiv(len, TPB);
  nativePowerKernel<Type><<<nblks, TPB, 0, stream>>>(in, out, len);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename Type>
__global__ void nativeSqrtKernel(Type* in, Type* out, int len)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) { out[idx] = sqrt(in[idx]); }
}

template <typename Type>
void naiveSqrt(Type* in, Type* out, int len, cudaStream_t stream)
{
  static const int TPB = 64;
  int nblks            = raft::ceildiv(len, TPB);
  nativeSqrtKernel<Type><<<nblks, TPB, 0, stream>>>(in, out, len);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename Type>
__global__ void naiveSignFlipKernel(Type* in, Type* out, int rowCount, int colCount)
{
  int d_i = blockIdx.x * rowCount;
  int end = d_i + rowCount;

  if (blockIdx.x < colCount) {
    Type max      = 0.0;
    int max_index = 0;
    for (int i = d_i; i < end; i++) {
      Type val = in[i];
      if (val < 0.0) { val = -val; }
      if (val > max) {
        max       = val;
        max_index = i;
      }
    }

    for (int i = d_i; i < end; i++) {
      if (in[max_index] < 0.0) {
        out[i] = -in[i];
      } else {
        out[i] = in[i];
      }
    }
  }

  __syncthreads();
}

template <typename Type>
void naiveSignFlip(Type* in, Type* out, int rowCount, int colCount, cudaStream_t stream)
{
  naiveSignFlipKernel<Type><<<colCount, 1, 0, stream>>>(in, out, rowCount, colCount);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename T>
struct MathInputs {
  T tolerance;
  int n_row;
  int n_col;
  int len;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const MathInputs<T>& dims)
{
  return os;
}

template <typename T>
class MathTest : public ::testing::TestWithParam<MathInputs<T>> {
 public:
  MathTest()
    : params(::testing::TestWithParam<MathInputs<T>>::GetParam()),
      stream(handle.get_stream()),
      in_power(params.len, stream),
      out_power_ref(params.len, stream),
      in_sqrt(params.len, stream),
      out_sqrt_ref(params.len, stream),
      in_sign_flip(params.len, stream),
      out_sign_flip_ref(params.len, stream),
      in_ratio(4, stream),
      out_ratio_ref(4, stream),
      in_recip(4, stream),
      in_recip_ref(4, stream),
      out_recip(4, stream),
      in_smallzero(4, stream),
      out_smallzero(4, stream),
      out_smallzero_ref(4, stream)
  {
  }

 protected:
  void SetUp() override
  {
    random::RngState r(params.seed);
    int len         = params.len;
    T in_ratio_h[4] = {1.0, 2.0, 2.0, 3.0};
    update_device(in_ratio.data(), in_ratio_h, 4, stream);

    T out_ratio_ref_h[4] = {0.125, 0.25, 0.25, 0.375};
    update_device(out_ratio_ref.data(), out_ratio_ref_h, 4, stream);

    uniform(r, in_power.data(), len, T(-1.0), T(1.0), stream);
    uniform(r, in_sqrt.data(), len, T(0.0), T(1.0), stream);
    // uniform(r, in_ratio, len, T(0.0), T(1.0));
    uniform(r, in_sign_flip.data(), len, T(-100.0), T(100.0), stream);

    naivePower(in_power.data(), out_power_ref.data(), len, stream);
    power(in_power.data(), len, stream);

    naiveSqrt(in_sqrt.data(), out_sqrt_ref.data(), len, stream);
    seqRoot(in_sqrt.data(), len, stream);

    ratio(handle, in_ratio.data(), in_ratio.data(), 4, stream);

    naiveSignFlip(
      in_sign_flip.data(), out_sign_flip_ref.data(), params.n_row, params.n_col, stream);
    signFlip(in_sign_flip.data(), params.n_row, params.n_col, stream);

    // default threshold is 1e-15
    std::vector<T> in_recip_h     = {0.1, 0.01, -0.01, 0.1e-16};
    std::vector<T> in_recip_ref_h = {10.0, 100.0, -100.0, 0.0};
    update_device(in_recip.data(), in_recip_h.data(), 4, stream);
    update_device(in_recip_ref.data(), in_recip_ref_h.data(), 4, stream);
    T recip_scalar = T(1.0);

    // this `reciprocal()` has to go first bc next one modifies its input
    reciprocal(in_recip.data(), out_recip.data(), recip_scalar, 4, stream);

    reciprocal(in_recip.data(), recip_scalar, 4, stream, true);

    std::vector<T> in_small_val_zero_h     = {0.1, 1e-16, -1e-16, -0.1};
    std::vector<T> in_small_val_zero_ref_h = {0.1, 0.0, 0.0, -0.1};

    update_device(in_smallzero.data(), in_small_val_zero_h.data(), 4, stream);
    update_device(out_smallzero_ref.data(), in_small_val_zero_ref_h.data(), 4, stream);
    setSmallValuesZero(out_smallzero.data(), in_smallzero.data(), 4, stream);
    setSmallValuesZero(in_smallzero.data(), 4, stream);
    handle.sync_stream(stream);
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;

  MathInputs<T> params;
  rmm::device_uvector<T> in_power, out_power_ref, in_sqrt, out_sqrt_ref, in_ratio, out_ratio_ref,
    in_sign_flip, out_sign_flip_ref, in_recip, in_recip_ref, out_recip, in_smallzero, out_smallzero,
    out_smallzero_ref;
};

const std::vector<MathInputs<float>> inputsf = {{0.00001f, 1024, 1024, 1024 * 1024, 1234ULL}};

const std::vector<MathInputs<double>> inputsd = {{0.00001, 1024, 1024, 1024 * 1024, 1234ULL}};

typedef MathTest<float> MathPowerTestF;
TEST_P(MathPowerTestF, Result)
{
  ASSERT_TRUE(devArrMatch(in_power.data(),
                          out_power_ref.data(),
                          params.len,
                          CompareApprox<float>(params.tolerance),
                          stream));
}

typedef MathTest<double> MathPowerTestD;
TEST_P(MathPowerTestD, Result)
{
  ASSERT_TRUE(devArrMatch(in_power.data(),
                          out_power_ref.data(),
                          params.len,
                          CompareApprox<double>(params.tolerance),
                          stream));
}

typedef MathTest<float> MathSqrtTestF;
TEST_P(MathSqrtTestF, Result)
{
  ASSERT_TRUE(devArrMatch(in_sqrt.data(),
                          out_sqrt_ref.data(),
                          params.len,
                          CompareApprox<float>(params.tolerance),
                          stream));
}

typedef MathTest<double> MathSqrtTestD;
TEST_P(MathSqrtTestD, Result)
{
  ASSERT_TRUE(devArrMatch(in_sqrt.data(),
                          out_sqrt_ref.data(),
                          params.len,
                          CompareApprox<double>(params.tolerance),
                          stream));
}

typedef MathTest<float> MathRatioTestF;
TEST_P(MathRatioTestF, Result)
{
  ASSERT_TRUE(devArrMatch(
    in_ratio.data(), out_ratio_ref.data(), 4, CompareApprox<float>(params.tolerance), stream));
}

typedef MathTest<double> MathRatioTestD;
TEST_P(MathRatioTestD, Result)
{
  ASSERT_TRUE(devArrMatch(
    in_ratio.data(), out_ratio_ref.data(), 4, CompareApprox<double>(params.tolerance), stream));
}

typedef MathTest<float> MathSignFlipTestF;
TEST_P(MathSignFlipTestF, Result)
{
  ASSERT_TRUE(devArrMatch(in_sign_flip.data(),
                          out_sign_flip_ref.data(),
                          params.len,
                          CompareApprox<float>(params.tolerance),
                          stream));
}

typedef MathTest<double> MathSignFlipTestD;
TEST_P(MathSignFlipTestD, Result)
{
  ASSERT_TRUE(devArrMatch(in_sign_flip.data(),
                          out_sign_flip_ref.data(),
                          params.len,
                          CompareApprox<double>(params.tolerance),
                          stream));
}

typedef MathTest<float> MathReciprocalTestF;
TEST_P(MathReciprocalTestF, Result)
{
  ASSERT_TRUE(devArrMatch(
    in_recip.data(), in_recip_ref.data(), 4, CompareApprox<float>(params.tolerance), stream));

  // 4-th term tests `setzero=true` functionality, not present in this version of `reciprocal`.
  ASSERT_TRUE(devArrMatch(
    out_recip.data(), in_recip_ref.data(), 3, CompareApprox<float>(params.tolerance), stream));
}

typedef MathTest<double> MathReciprocalTestD;
TEST_P(MathReciprocalTestD, Result)
{
  ASSERT_TRUE(devArrMatch(
    in_recip.data(), in_recip_ref.data(), 4, CompareApprox<double>(params.tolerance), stream));

  // 4-th term tests `setzero=true` functionality, not present in this version of `reciprocal`.
  ASSERT_TRUE(devArrMatch(
    out_recip.data(), in_recip_ref.data(), 3, CompareApprox<double>(params.tolerance), stream));
}

typedef MathTest<float> MathSetSmallZeroTestF;
TEST_P(MathSetSmallZeroTestF, Result)
{
  ASSERT_TRUE(devArrMatch(in_smallzero.data(),
                          out_smallzero_ref.data(),
                          4,
                          CompareApprox<float>(params.tolerance),
                          stream));

  ASSERT_TRUE(devArrMatch(out_smallzero.data(),
                          out_smallzero_ref.data(),
                          4,
                          CompareApprox<float>(params.tolerance),
                          stream));
}

typedef MathTest<double> MathSetSmallZeroTestD;
TEST_P(MathSetSmallZeroTestD, Result)
{
  ASSERT_TRUE(devArrMatch(in_smallzero.data(),
                          out_smallzero_ref.data(),
                          4,
                          CompareApprox<double>(params.tolerance),
                          stream));

  ASSERT_TRUE(devArrMatch(out_smallzero.data(),
                          out_smallzero_ref.data(),
                          4,
                          CompareApprox<double>(params.tolerance),
                          stream));
}

INSTANTIATE_TEST_SUITE_P(MathTests, MathPowerTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(MathTests, MathPowerTestD, ::testing::ValuesIn(inputsd));

INSTANTIATE_TEST_SUITE_P(MathTests, MathSqrtTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(MathTests, MathSqrtTestD, ::testing::ValuesIn(inputsd));

INSTANTIATE_TEST_SUITE_P(MathTests, MathRatioTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(MathTests, MathRatioTestD, ::testing::ValuesIn(inputsd));

INSTANTIATE_TEST_SUITE_P(MathTests, MathSignFlipTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(MathTests, MathSignFlipTestD, ::testing::ValuesIn(inputsd));

INSTANTIATE_TEST_SUITE_P(MathTests, MathReciprocalTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(MathTests, MathReciprocalTestD, ::testing::ValuesIn(inputsd));

INSTANTIATE_TEST_SUITE_P(MathTests, MathSetSmallZeroTestF, ::testing::ValuesIn(inputsf));
INSTANTIATE_TEST_SUITE_P(MathTests, MathSetSmallZeroTestD, ::testing::ValuesIn(inputsd));

}  // namespace matrix
}  // namespace raft
