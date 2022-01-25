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

#include "../test_utils.h"
#include <gtest/gtest.h>
#include <raft/common/nvtx.hpp>
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/distance/distance.hpp>
#if defined RAFT_DISTANCE_COMPILED
#include <raft/distance/specializations.hpp>
#endif
#include <raft/random/rng.hpp>

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
    acc = raft::mySqrt(acc);
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
      acc = raft::myMax(acc, diff);
    } else if (type == raft::distance::DistanceType::Canberra) {
      const auto add = raft::myAbs(a) + raft::myAbs(b);
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
  dist[outidx] = (DataType)1.0 - acc_ab / (raft::mySqrt(acc_a) * raft::mySqrt(acc_b));
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
    acc_ab += raft::mySqrt(a) * raft::mySqrt(b);
  }

  int outidx = isRowMajor ? midx * n + nidx : midx + m * nidx;

  // Adjust to replace NaN in sqrt with 0 if input to sqrt is negative
  acc_ab         = 1 - acc_ab;
  auto rectifier = (!signbit(acc_ab));
  dist[outidx]   = raft::mySqrt(rectifier * acc_ab);
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
    auto diff = raft::L1Op<DataType>()(a - b);
    acc += raft::myPow(diff, p);
  }
  auto one_over_p = 1 / p;
  acc             = raft::myPow(acc, one_over_p);
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
  acc          = raft::mySqrt(0.5f * acc);
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
    int xidx          = isRowMajor ? i + midx * k : i * m + midx;
    int yidx          = isRowMajor ? i + nidx * k : i * n + nidx;
    auto a            = x[xidx];
    auto b            = y[yidx];
    bool b_zero       = (b == 0);
    const auto m      = (!b_zero) * (a / b);
    const bool m_zero = (m == 0);
    acc += (a * (!m_zero) * log(m + m_zero));
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

  acc = 1 - (numer / raft::mySqrt(Q_denom * R_denom));

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

template <raft::distance::DistanceType distanceType, typename DataType>
void distanceLauncher(DataType* x,
                      DataType* y,
                      DataType* dist,
                      DataType* dist2,
                      int m,
                      int n,
                      int k,
                      DistanceInputs<DataType>& params,
                      DataType threshold,
                      char* workspace,
                      size_t worksize,
                      cudaStream_t stream,
                      bool isRowMajor,
                      DataType metric_arg = 2.0f)
{
  raft::distance::distance<distanceType, DataType, DataType, DataType>(
    x, y, dist, m, n, k, workspace, worksize, stream, isRowMajor, metric_arg);
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

    raft::random::Rng r(params.seed);
    int m               = params.m;
    int n               = params.n;
    int k               = params.k;
    DataType metric_arg = params.metric_arg;
    bool isRowMajor     = params.isRowMajor;
    if (distanceType == raft::distance::DistanceType::HellingerExpanded ||
        distanceType == raft::distance::DistanceType::JensenShannon ||
        distanceType == raft::distance::DistanceType::KLDivergence) {
      // Hellinger works only on positive numbers
      r.uniform(x.data(), m * k, DataType(0.0), DataType(1.0), stream);
      r.uniform(y.data(), n * k, DataType(0.0), DataType(1.0), stream);
    } else if (distanceType == raft::distance::DistanceType::RusselRaoExpanded) {
      r.uniform(x.data(), m * k, DataType(0.0), DataType(1.0), stream);
      r.uniform(y.data(), n * k, DataType(0.0), DataType(1.0), stream);
      // Russel rao works on boolean values.
      r.bernoulli(x.data(), m * k, 0.5f, stream);
      r.bernoulli(y.data(), n * k, 0.5f, stream);
    } else {
      r.uniform(x.data(), m * k, DataType(-1.0), DataType(1.0), stream);
      r.uniform(y.data(), n * k, DataType(-1.0), DataType(1.0), stream);
    }
    naiveDistance(
      dist_ref.data(), x.data(), y.data(), m, n, k, distanceType, isRowMajor, metric_arg, stream);
    size_t worksize = raft::distance::getWorkspaceSize<distanceType, DataType, DataType, DataType>(
      x.data(), y.data(), m, n, k);
    rmm::device_uvector<char> workspace(worksize, stream);

    DataType threshold = -10000.f;
    distanceLauncher<distanceType, DataType>(x.data(),
                                             y.data(),
                                             dist.data(),
                                             dist2.data(),
                                             m,
                                             n,
                                             k,
                                             params,
                                             threshold,
                                             workspace.data(),
                                             workspace.size(),
                                             stream,
                                             isRowMajor,
                                             metric_arg);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;

  DistanceInputs<DataType> params;
  rmm::device_uvector<DataType> x, y, dist_ref, dist, dist2;
};

}  // end namespace distance
}  // end namespace raft
