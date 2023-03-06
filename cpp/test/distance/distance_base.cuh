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
#include <gtest/gtest.h>
#include <raft/common/nvtx.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/operators.hpp>
#include <raft/distance/distance.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#if defined RAFT_DISTANCE_COMPILED
#include <raft/distance/specializations.cuh>
#endif
#include <raft/random/rng.cuh>

namespace raft {
namespace distance {

template <typename DataType>
__global__ void naiveDistanceKernel(DataType* dist,
                                    const DataType* x,
                                    const DataType* y,
                                    int m,
                                    int n,
                                    int k,
                                    raft::distance::DistanceType type,
                                    bool isRowMajor)
{
  int midx = threadIdx.x + blockIdx.x * blockDim.x;
  int nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) return;
  DataType acc = DataType(0);
  for (int i = 0; i < k; ++i) {
    int xidx  = isRowMajor ? i + midx * k : i * m + midx;
    int yidx  = isRowMajor ? i + nidx * k : i * n + nidx;
    auto diff = x[xidx] - y[yidx];
    acc += diff * diff;
  }
  if (type == raft::distance::DistanceType::L2SqrtExpanded ||
      type == raft::distance::DistanceType::L2SqrtUnexpanded)
    acc = raft::sqrt(acc);
  int outidx   = isRowMajor ? midx * n + nidx : midx + m * nidx;
  dist[outidx] = acc;
}

template <typename DataType>
__global__ void naiveL1_Linf_CanberraDistanceKernel(DataType* dist,
                                                    const DataType* x,
                                                    const DataType* y,
                                                    int m,
                                                    int n,
                                                    int k,
                                                    raft::distance::DistanceType type,
                                                    bool isRowMajor)
{
  int midx = threadIdx.x + blockIdx.x * blockDim.x;
  int nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) { return; }

  DataType acc = DataType(0);
  for (int i = 0; i < k; ++i) {
    int xidx  = isRowMajor ? i + midx * k : i * m + midx;
    int yidx  = isRowMajor ? i + nidx * k : i * n + nidx;
    auto a    = x[xidx];
    auto b    = y[yidx];
    auto diff = (a > b) ? (a - b) : (b - a);
    if (type == raft::distance::DistanceType::Linf) {
      acc = raft::max(acc, diff);
    } else if (type == raft::distance::DistanceType::Canberra) {
      const auto add = raft::abs(a) + raft::abs(b);
      // deal with potential for 0 in denominator by
      // forcing 1/0 instead
      acc += ((add != 0) * diff / (add + (add == 0)));
    } else {
      acc += diff;
    }
  }

  int outidx   = isRowMajor ? midx * n + nidx : midx + m * nidx;
  dist[outidx] = acc;
}

template <typename DataType>
__global__ void naiveCosineDistanceKernel(
  DataType* dist, const DataType* x, const DataType* y, int m, int n, int k, bool isRowMajor)
{
  int midx = threadIdx.x + blockIdx.x * blockDim.x;
  int nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) { return; }

  DataType acc_a  = DataType(0);
  DataType acc_b  = DataType(0);
  DataType acc_ab = DataType(0);

  for (int i = 0; i < k; ++i) {
    int xidx = isRowMajor ? i + midx * k : i * m + midx;
    int yidx = isRowMajor ? i + nidx * k : i * n + nidx;
    auto a   = x[xidx];
    auto b   = y[yidx];
    acc_a += a * a;
    acc_b += b * b;
    acc_ab += a * b;
  }

  int outidx = isRowMajor ? midx * n + nidx : midx + m * nidx;

  // Use 1.0 - (cosine similarity) to calc the distance
  dist[outidx] = (DataType)1.0 - acc_ab / (raft::sqrt(acc_a) * raft::sqrt(acc_b));
}

template <typename DataType>
__global__ void naiveInnerProductKernel(
  DataType* dist, const DataType* x, const DataType* y, int m, int n, int k, bool isRowMajor)
{
  int midx = threadIdx.x + blockIdx.x * blockDim.x;
  int nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) { return; }

  DataType acc_ab = DataType(0);

  for (int i = 0; i < k; ++i) {
    int xidx = isRowMajor ? i + midx * k : i * m + midx;
    int yidx = isRowMajor ? i + nidx * k : i * n + nidx;
    auto a   = x[xidx];
    auto b   = y[yidx];
    acc_ab += a * b;
  }

  int outidx   = isRowMajor ? midx * n + nidx : midx + m * nidx;
  dist[outidx] = acc_ab;
}

template <typename DataType>
__global__ void naiveHellingerDistanceKernel(
  DataType* dist, const DataType* x, const DataType* y, int m, int n, int k, bool isRowMajor)
{
  int midx = threadIdx.x + blockIdx.x * blockDim.x;
  int nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) { return; }

  DataType acc_ab = DataType(0);

  for (int i = 0; i < k; ++i) {
    int xidx = isRowMajor ? i + midx * k : i * m + midx;
    int yidx = isRowMajor ? i + nidx * k : i * n + nidx;
    auto a   = x[xidx];
    auto b   = y[yidx];
    acc_ab += raft::sqrt(a) * raft::sqrt(b);
  }

  int outidx = isRowMajor ? midx * n + nidx : midx + m * nidx;

  // Adjust to replace NaN in sqrt with 0 if input to sqrt is negative
  acc_ab         = 1 - acc_ab;
  auto rectifier = (!signbit(acc_ab));
  dist[outidx]   = raft::sqrt(rectifier * acc_ab);
}

template <typename DataType>
__global__ void naiveLpUnexpDistanceKernel(DataType* dist,
                                           const DataType* x,
                                           const DataType* y,
                                           int m,
                                           int n,
                                           int k,
                                           bool isRowMajor,
                                           DataType p)
{
  int midx = threadIdx.x + blockIdx.x * blockDim.x;
  int nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) return;
  DataType acc = DataType(0);
  for (int i = 0; i < k; ++i) {
    int xidx  = isRowMajor ? i + midx * k : i * m + midx;
    int yidx  = isRowMajor ? i + nidx * k : i * n + nidx;
    auto a    = x[xidx];
    auto b    = y[yidx];
    auto diff = raft::abs(a - b);
    acc += raft::pow(diff, p);
  }
  auto one_over_p = 1 / p;
  acc             = raft::pow(acc, one_over_p);
  int outidx      = isRowMajor ? midx * n + nidx : midx + m * nidx;
  dist[outidx]    = acc;
}

template <typename DataType>
__global__ void naiveHammingDistanceKernel(
  DataType* dist, const DataType* x, const DataType* y, int m, int n, int k, bool isRowMajor)
{
  int midx = threadIdx.x + blockIdx.x * blockDim.x;
  int nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) return;
  DataType acc = DataType(0);
  for (int i = 0; i < k; ++i) {
    int xidx = isRowMajor ? i + midx * k : i * m + midx;
    int yidx = isRowMajor ? i + nidx * k : i * n + nidx;
    auto a   = x[xidx];
    auto b   = y[yidx];
    acc += (a != b);
  }
  acc          = acc / k;
  int outidx   = isRowMajor ? midx * n + nidx : midx + m * nidx;
  dist[outidx] = acc;
}

template <typename DataType>
__global__ void naiveJensenShannonDistanceKernel(
  DataType* dist, const DataType* x, const DataType* y, int m, int n, int k, bool isRowMajor)
{
  int midx = threadIdx.x + blockIdx.x * blockDim.x;
  int nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) return;
  DataType acc = DataType(0);
  for (int i = 0; i < k; ++i) {
    int xidx = isRowMajor ? i + midx * k : i * m + midx;
    int yidx = isRowMajor ? i + nidx * k : i * n + nidx;
    auto a   = x[xidx];
    auto b   = y[yidx];

    DataType m  = 0.5f * (a + b);
    bool a_zero = a == 0;
    bool b_zero = b == 0;

    DataType p = (!a_zero * m) / (a_zero + a);
    DataType q = (!b_zero * m) / (b_zero + b);

    bool p_zero = p == 0;
    bool q_zero = q == 0;

    acc += (-a * (!p_zero * log(p + p_zero))) + (-b * (!q_zero * log(q + q_zero)));
  }
  acc          = raft::sqrt(0.5f * acc);
  int outidx   = isRowMajor ? midx * n + nidx : midx + m * nidx;
  dist[outidx] = acc;
}

template <typename DataType, typename OutType>
__global__ void naiveRussellRaoDistanceKernel(
  OutType* dist, const DataType* x, const DataType* y, int m, int n, int k, bool isRowMajor)
{
  int midx = threadIdx.x + blockIdx.x * blockDim.x;
  int nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) return;
  OutType acc = OutType(0);
  for (int i = 0; i < k; ++i) {
    int xidx = isRowMajor ? i + midx * k : i * m + midx;
    int yidx = isRowMajor ? i + nidx * k : i * n + nidx;
    auto a   = x[xidx];
    auto b   = y[yidx];
    acc += (a * b);
  }
  acc          = (k - acc) / k;
  int outidx   = isRowMajor ? midx * n + nidx : midx + m * nidx;
  dist[outidx] = acc;
}

template <typename DataType, typename OutType>
__global__ void naiveKLDivergenceDistanceKernel(
  OutType* dist, const DataType* x, const DataType* y, int m, int n, int k, bool isRowMajor)
{
  int midx = threadIdx.x + blockIdx.x * blockDim.x;
  int nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) return;
  OutType acc = OutType(0);
  for (int i = 0; i < k; ++i) {
    int xidx    = isRowMajor ? i + midx * k : i * m + midx;
    int yidx    = isRowMajor ? i + nidx * k : i * n + nidx;
    auto a      = x[xidx];
    auto b      = y[yidx];
    bool b_zero = (b == 0);
    bool a_zero = (a == 0);
    acc += a * (log(a + a_zero) - log(b + b_zero));
  }
  acc          = 0.5f * acc;
  int outidx   = isRowMajor ? midx * n + nidx : midx + m * nidx;
  dist[outidx] = acc;
}

template <typename DataType, typename OutType>
__global__ void naiveCorrelationDistanceKernel(
  OutType* dist, const DataType* x, const DataType* y, int m, int n, int k, bool isRowMajor)
{
  int midx = threadIdx.x + blockIdx.x * blockDim.x;
  int nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) return;
  OutType acc    = OutType(0);
  auto a_norm    = DataType(0);
  auto b_norm    = DataType(0);
  auto a_sq_norm = DataType(0);
  auto b_sq_norm = DataType(0);
  for (int i = 0; i < k; ++i) {
    int xidx = isRowMajor ? i + midx * k : i * m + midx;
    int yidx = isRowMajor ? i + nidx * k : i * n + nidx;
    auto a   = x[xidx];
    auto b   = y[yidx];
    a_norm += a;
    b_norm += b;
    a_sq_norm += (a * a);
    b_sq_norm += (b * b);
    acc += (a * b);
  }

  auto numer   = k * acc - (a_norm * b_norm);
  auto Q_denom = k * a_sq_norm - (a_norm * a_norm);
  auto R_denom = k * b_sq_norm - (b_norm * b_norm);

  acc = 1 - (numer / raft::sqrt(Q_denom * R_denom));

  int outidx   = isRowMajor ? midx * n + nidx : midx + m * nidx;
  dist[outidx] = acc;
}

template <typename DataType>
void naiveDistance(DataType* dist,
                   const DataType* x,
                   const DataType* y,
                   int m,
                   int n,
                   int k,
                   raft::distance::DistanceType type,
                   bool isRowMajor,
                   DataType metric_arg = 2.0f,
                   cudaStream_t stream = 0)
{
  static const dim3 TPB(16, 32, 1);
  dim3 nblks(raft::ceildiv(m, (int)TPB.x), raft::ceildiv(n, (int)TPB.y), 1);

  switch (type) {
    case raft::distance::DistanceType::Canberra:
    case raft::distance::DistanceType::Linf:
    case raft::distance::DistanceType::L1:
      naiveL1_Linf_CanberraDistanceKernel<DataType>
        <<<nblks, TPB, 0, stream>>>(dist, x, y, m, n, k, type, isRowMajor);
      break;
    case raft::distance::DistanceType::L2SqrtUnexpanded:
    case raft::distance::DistanceType::L2Unexpanded:
    case raft::distance::DistanceType::L2SqrtExpanded:
    case raft::distance::DistanceType::L2Expanded:
      naiveDistanceKernel<DataType>
        <<<nblks, TPB, 0, stream>>>(dist, x, y, m, n, k, type, isRowMajor);
      break;
    case raft::distance::DistanceType::CosineExpanded:
      naiveCosineDistanceKernel<DataType>
        <<<nblks, TPB, 0, stream>>>(dist, x, y, m, n, k, isRowMajor);
      break;
    case raft::distance::DistanceType::HellingerExpanded:
      naiveHellingerDistanceKernel<DataType>
        <<<nblks, TPB, 0, stream>>>(dist, x, y, m, n, k, isRowMajor);
      break;
    case raft::distance::DistanceType::LpUnexpanded:
      naiveLpUnexpDistanceKernel<DataType>
        <<<nblks, TPB, 0, stream>>>(dist, x, y, m, n, k, isRowMajor, metric_arg);
      break;
    case raft::distance::DistanceType::HammingUnexpanded:
      naiveHammingDistanceKernel<DataType>
        <<<nblks, TPB, 0, stream>>>(dist, x, y, m, n, k, isRowMajor);
      break;
    case raft::distance::DistanceType::InnerProduct:
      naiveInnerProductKernel<DataType><<<nblks, TPB, 0, stream>>>(dist, x, y, m, n, k, isRowMajor);
      break;
    case raft::distance::DistanceType::JensenShannon:
      naiveJensenShannonDistanceKernel<DataType>
        <<<nblks, TPB, 0, stream>>>(dist, x, y, m, n, k, isRowMajor);
      break;
    case raft::distance::DistanceType::RusselRaoExpanded:
      naiveRussellRaoDistanceKernel<DataType>
        <<<nblks, TPB, 0, stream>>>(dist, x, y, m, n, k, isRowMajor);
      break;
    case raft::distance::DistanceType::KLDivergence:
      naiveKLDivergenceDistanceKernel<DataType>
        <<<nblks, TPB, 0, stream>>>(dist, x, y, m, n, k, isRowMajor);
      break;
    case raft::distance::DistanceType::CorrelationExpanded:
      naiveCorrelationDistanceKernel<DataType>
        <<<nblks, TPB, 0, stream>>>(dist, x, y, m, n, k, isRowMajor);
      break;
    default: FAIL() << "should be here\n";
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename DataType>
struct DistanceInputs {
  DataType tolerance;
  int m, n, k;
  bool isRowMajor;
  unsigned long long int seed;
  DataType metric_arg = 2.0f;
};

template <typename DataType>
::std::ostream& operator<<(::std::ostream& os, const DistanceInputs<DataType>& dims)
{
  return os;
}

template <raft::distance::DistanceType distanceType, typename DataType, typename layout>
void distanceLauncher(raft::device_resources const& handle,
                      DataType* x,
                      DataType* y,
                      DataType* dist,
                      DataType* dist2,
                      int m,
                      int n,
                      int k,
                      DistanceInputs<DataType>& params,
                      DataType threshold,
                      DataType metric_arg = 2.0f)
{
  auto x_v    = make_device_matrix_view<DataType, int, layout>(x, m, k);
  auto y_v    = make_device_matrix_view<DataType, int, layout>(y, n, k);
  auto dist_v = make_device_matrix_view<DataType, int, layout>(dist, m, n);

  raft::distance::distance<distanceType, DataType, DataType, DataType, layout>(
    handle, x_v, y_v, dist_v, metric_arg);
}

template <raft::distance::DistanceType distanceType, typename DataType>
class DistanceTest : public ::testing::TestWithParam<DistanceInputs<DataType>> {
 public:
  DistanceTest()
    : params(::testing::TestWithParam<DistanceInputs<DataType>>::GetParam()),
      stream(handle.get_stream()),
      x(params.m * params.k, stream),
      y(params.n * params.k, stream),
      dist_ref(params.m * params.n, stream),
      dist(params.m * params.n, stream),
      dist2(params.m * params.n, stream)
  {
  }

  void SetUp() override
  {
    auto testInfo = testing::UnitTest::GetInstance()->current_test_info();
    common::nvtx::range fun_scope("test::%s/%s", testInfo->test_suite_name(), testInfo->name());

    raft::random::RngState r(params.seed);
    int m               = params.m;
    int n               = params.n;
    int k               = params.k;
    DataType metric_arg = params.metric_arg;
    bool isRowMajor     = params.isRowMajor;
    if (distanceType == raft::distance::DistanceType::HellingerExpanded ||
        distanceType == raft::distance::DistanceType::JensenShannon ||
        distanceType == raft::distance::DistanceType::KLDivergence) {
      // Hellinger works only on positive numbers
      uniform(handle, r, x.data(), m * k, DataType(0.0), DataType(1.0));
      uniform(handle, r, y.data(), n * k, DataType(0.0), DataType(1.0));
    } else if (distanceType == raft::distance::DistanceType::RusselRaoExpanded) {
      uniform(handle, r, x.data(), m * k, DataType(0.0), DataType(1.0));
      uniform(handle, r, y.data(), n * k, DataType(0.0), DataType(1.0));
      // Russel rao works on boolean values.
      bernoulli(handle, r, x.data(), m * k, 0.5f);
      bernoulli(handle, r, y.data(), n * k, 0.5f);
    } else {
      uniform(handle, r, x.data(), m * k, DataType(-1.0), DataType(1.0));
      uniform(handle, r, y.data(), n * k, DataType(-1.0), DataType(1.0));
    }
    naiveDistance(
      dist_ref.data(), x.data(), y.data(), m, n, k, distanceType, isRowMajor, metric_arg, stream);

    DataType threshold = -10000.f;

    if (isRowMajor) {
      distanceLauncher<distanceType, DataType, layout_c_contiguous>(handle,
                                                                    x.data(),
                                                                    y.data(),
                                                                    dist.data(),
                                                                    dist2.data(),
                                                                    m,
                                                                    n,
                                                                    k,
                                                                    params,
                                                                    threshold,
                                                                    metric_arg);

    } else {
      distanceLauncher<distanceType, DataType, layout_f_contiguous>(handle,
                                                                    x.data(),
                                                                    y.data(),
                                                                    dist.data(),
                                                                    dist2.data(),
                                                                    m,
                                                                    n,
                                                                    k,
                                                                    params,
                                                                    threshold,
                                                                    metric_arg);
    }
    handle.sync_stream(stream);
  }

 protected:
  raft::device_resources handle;
  cudaStream_t stream;

  DistanceInputs<DataType> params;
  rmm::device_uvector<DataType> x, y, dist_ref, dist, dist2;
};

template <raft::distance::DistanceType distanceType>
class BigMatrixDistanceTest : public ::testing::Test {
 public:
  BigMatrixDistanceTest()
    : x(m * k, handle.get_stream()), dist(std::size_t(m) * m, handle.get_stream()){};
  void SetUp() override
  {
    auto testInfo = testing::UnitTest::GetInstance()->current_test_info();
    common::nvtx::range fun_scope("test::%s/%s", testInfo->test_suite_name(), testInfo->name());

    raft::distance::distance<distanceType, float, float, float>(
      handle, x.data(), x.data(), dist.data(), m, n, k, true, 0.0f);

    RAFT_CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));
  }

 protected:
  raft::device_resources handle;
  int m = 48000;
  int n = 48000;
  int k = 1;
  rmm::device_uvector<float> x, dist;
};
}  // end namespace distance
}  // end namespace raft
