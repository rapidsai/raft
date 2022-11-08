/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <raft/linalg/dot.cuh>

#include "../test_utils.h"
#include <gtest/gtest.h>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <rmm/device_scalar.hpp>

namespace raft {
namespace linalg {
// Reference dot implementation.
template <typename T>
__global__ void naiveDot(const int n, const T* x, int incx, const T* y, int incy, T* out)
{
  T sum = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    sum += x[i * incx] * y[i * incy];
  }
  atomicAdd(out, sum);
}

template <typename InType, typename IndexType = int, typename OutType = InType>
struct DotInputs {
  OutType tolerance;
  IndexType len;
  IndexType incx;
  IndexType incy;
  unsigned long long int seed;
};

template <typename T, typename IndexType = int>
class DotTest : public ::testing::TestWithParam<DotInputs<T>> {
 protected:
  raft::handle_t handle;
  DotInputs<T, IndexType> params;
  rmm::device_scalar<T> output;
  rmm::device_scalar<T> refoutput;

 public:
  DotTest()
    : testing::TestWithParam<DotInputs<T>>(),
      output(0, handle.get_stream()),
      refoutput(0, handle.get_stream())
  {
    handle.sync_stream();
  }

 protected:
  void SetUp() override
  {
    params = ::testing::TestWithParam<DotInputs<T>>::GetParam();

    cudaStream_t stream = handle.get_stream();

    raft::random::RngState r(params.seed);

    IndexType x_len = params.len * params.incx;
    IndexType y_len = params.len * params.incy;

    rmm::device_uvector<T> x(x_len, stream);
    rmm::device_uvector<T> y(y_len, stream);
    uniform(handle, r, x.data(), x_len, T(-1.0), T(1.0));
    uniform(handle, r, y.data(), y_len, T(-1.0), T(1.0));

    naiveDot<<<256, 256, 0, stream>>>(
      params.len, x.data(), params.incx, y.data(), params.incy, refoutput.data());

    auto out_view = make_device_scalar_view<T, IndexType>(output.data());

    if ((params.incx > 1) && (params.incy > 1)) {
      dot(handle,
          make_device_vector_view<const T, IndexType, layout_stride>(
            x.data(), make_vector_strided_layout(params.len, params.incx)),
          make_device_vector_view<const T, IndexType, layout_stride>(
            y.data(), make_vector_strided_layout(params.len, params.incy)),
          out_view);
    } else if (params.incx > 1) {
      dot(handle,
          make_device_vector_view<const T, IndexType, layout_stride>(
            x.data(), make_vector_strided_layout(params.len, params.incx)),
          make_device_vector_view<const T>(y.data(), params.len),
          out_view);
    } else if (params.incy > 1) {
      dot(handle,
          make_device_vector_view<const T>(x.data(), params.len),
          make_device_vector_view<const T, IndexType, layout_stride>(
            y.data(), make_vector_strided_layout(params.len, params.incy)),
          out_view);
    } else {
      dot(handle,
          make_device_vector_view<const T>(x.data(), params.len),
          make_device_vector_view<const T>(y.data(), params.len),
          out_view);
    }
    handle.sync_stream();
  }

  void TearDown() override {}
};

const std::vector<DotInputs<float>> inputsf = {
  {0.0001f, 1024 * 1024, 1, 1, 1234ULL},
  {0.0001f, 16 * 1024 * 1024, 1, 1, 1234ULL},
  {0.0001f, 98689, 1, 1, 1234ULL},
  {0.0001f, 4 * 1024 * 1024, 1, 1, 1234ULL},
  {0.0001f, 1024 * 1024, 4, 1, 1234ULL},
  {0.0001f, 1024 * 1024, 1, 3, 1234ULL},
  {0.0001f, 1024 * 1024, 4, 3, 1234ULL},
};

const std::vector<DotInputs<double>> inputsd = {
  {0.000001f, 1024 * 1024, 1, 1, 1234ULL},
  {0.000001f, 16 * 1024 * 1024, 1, 1, 1234ULL},
  {0.000001f, 98689, 1, 1, 1234ULL},
  {0.000001f, 4 * 1024 * 1024, 1, 1, 1234ULL},
  {0.000001f, 1024 * 1024, 4, 1, 1234ULL},
  {0.000001f, 1024 * 1024, 1, 3, 1234ULL},
  {0.000001f, 1024 * 1024, 4, 3, 1234ULL},
};

typedef DotTest<float> DotTestF;
TEST_P(DotTestF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(
    refoutput.data(), output.data(), 1, raft::CompareApprox<float>(params.tolerance)));
}

typedef DotTest<double> DotTestD;
TEST_P(DotTestD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(
    refoutput.data(), output.data(), 1, raft::CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_SUITE_P(DotTests, DotTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(DotTests, DotTestD, ::testing::ValuesIn(inputsd));

}  // end namespace linalg
}  // end namespace raft
