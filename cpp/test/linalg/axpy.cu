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
#include <raft/linalg/axpy.cuh>

#include "../test_utils.h"
#include <gtest/gtest.h>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>

namespace raft {
namespace linalg {

// Reference axpy implementation.
template <typename T>
__global__ void naiveAxpy(const int n, const T alpha, const T* x, T* y, int incx, int incy)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) { y[idx * incy] += alpha * x[idx * incx]; }
}

template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_stride>
auto make_strided_device_vector_view(ElementType* ptr, IndexType n, IndexType stride)
{
  vector_extent<IndexType> exts{n};
  std::array<IndexType, 1> strides{stride};
  auto layout = typename LayoutPolicy::mapping<vector_extent<IndexType>>{exts, strides};
  return device_vector_view<ElementType, IndexType, LayoutPolicy>{ptr, layout};
}

template <typename InType, typename IndexType = int, typename OutType = InType>
struct AxpyInputs {
  OutType tolerance;
  IndexType len;
  InType alpha;
  IndexType incx;
  IndexType incy;
  unsigned long long int seed;
};

template <typename T>
class AxpyTest : public ::testing::TestWithParam<AxpyInputs<T>> {
 protected:
  raft::handle_t handle;
  AxpyInputs<T> params;
  rmm::device_uvector<T> refy;
  rmm::device_uvector<T> y;

 public:
  AxpyTest()
    : testing::TestWithParam<AxpyInputs<T>>(),
      refy(0, handle.get_stream()),
      y(0, handle.get_stream())
  {
    handle.sync_stream();
  }

 protected:
  void SetUp() override
  {
    params = ::testing::TestWithParam<AxpyInputs<T>>::GetParam();

    cudaStream_t stream = handle.get_stream();

    raft::random::RngState r(params.seed);

    int x_len = params.len * params.incx;
    int y_len = params.len * params.incy;
    rmm::device_uvector<T> x(x_len, stream);
    y.resize(y_len, stream);
    refy.resize(y_len, stream);

    uniform(handle, r, x.data(), x_len, T(-1.0), T(1.0));
    uniform(handle, r, y.data(), y_len, T(-1.0), T(1.0));

    // Take a copy of the random generated values in y for the naive reference implementation
    // this is necessary since axpy uses y for both input and output
    raft::copy(refy.data(), y.data(), y_len, stream);

    int threads = 64;
    int blocks  = raft::ceildiv<int>(params.len, threads);

    naiveAxpy<T><<<blocks, threads, 0, stream>>>(
      params.len, params.alpha, x.data(), refy.data(), params.incx, params.incy);

    if ((params.incx > 1) && (params.incy > 1)) {
      axpy(handle,
           make_host_scalar_view<const T>(&params.alpha),
           make_strided_device_vector_view<const T>(x.data(), params.len, params.incx),
           make_strided_device_vector_view<T>(y.data(), params.len, params.incy));
    } else if (params.incx > 1) {
      axpy(handle,
           make_host_scalar_view<const T>(&params.alpha),
           make_strided_device_vector_view<const T>(x.data(), params.len, params.incx),
           make_device_vector_view<T>(y.data(), params.len));
    } else if (params.incy > 1) {
      axpy(handle,
           make_host_scalar_view<const T>(&params.alpha),
           make_device_vector_view<const T>(x.data(), params.len),
           make_strided_device_vector_view<T>(y.data(), params.len, params.incy));
    } else {
      axpy(handle,
           make_host_scalar_view<const T>(&params.alpha),
           make_device_vector_view<const T>(x.data(), params.len),
           make_device_vector_view<T>(y.data(), params.len));
    }

    handle.sync_stream();
  }

  void TearDown() override {}
};

const std::vector<AxpyInputs<float>> inputsf = {
  {0.000001f, 1024 * 1024, 2.f, 1, 1, 1234ULL},
  {0.000001f, 16 * 1024 * 1024, 128.f, 1, 1, 1234ULL},
  {0.000001f, 98689, 4.f, 1, 1, 1234ULL},
  {0.000001f, 4 * 1024 * 1024, -1, 1, 1, 1234ULL},
  {0.000001f, 1024 * 1024, 6, 4, 1, 1234ULL},
  {0.000001f, 1024 * 1024, 7, 1, 3, 1234ULL},
  {0.000001f, 1024 * 1024, 8, 4, 3, 1234ULL},
};

const std::vector<AxpyInputs<double>> inputsd = {
  {0.000001f, 1024 * 1024, 2.f, 1, 1, 1234ULL},
  {0.000001f, 16 * 1024 * 1024, 128.f, 1, 1, 1234ULL},
  {0.000001f, 98689, 4.f, 1, 1, 1234ULL},
  {0.000001f, 4 * 1024 * 1024, -1, 1, 1, 1234ULL},
  {0.000001f, 1024 * 1024, 6, 4, 1, 1234ULL},
  {0.000001f, 1024 * 1024, 7, 1, 3, 1234ULL},
  {0.000001f, 1024 * 1024, 8, 4, 3, 1234ULL},
};

typedef AxpyTest<float> AxpyTestF;
TEST_P(AxpyTestF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(
    refy.data(), y.data(), params.len * params.incy, raft::CompareApprox<float>(params.tolerance)));
}

typedef AxpyTest<double> AxpyTestD;
TEST_P(AxpyTestD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(refy.data(),
                                y.data(),
                                params.len * params.incy,
                                raft::CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_SUITE_P(AxpyTests, AxpyTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(AxpyTests, AxpyTestD, ::testing::ValuesIn(inputsd));

}  // end namespace linalg
}  // end namespace raft
