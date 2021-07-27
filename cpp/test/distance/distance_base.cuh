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

#include <gtest/gtest.h>
#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>
#include <raft/distance/distance.cuh>
#include <raft/random/rng.cuh>
#include "../test_utils.h"

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
void naiveDistance(DataType* dist,
                   const DataType* x,
                   const DataType* y,
                   int m,
                   int n,
                   int k,
                   raft::distance::DistanceType type,
                   bool isRowMajor,
                   DataType metric_arg = 2.0f)
{
  static const dim3 TPB(16, 32, 1);
  dim3 nblks(raft::ceildiv(m, (int)TPB.x), raft::ceildiv(n, (int)TPB.y), 1);

  switch (type) {
    case raft::distance::DistanceType::Canberra:
    case raft::distance::DistanceType::Linf:
    case raft::distance::DistanceType::L1:
      naiveL1_Linf_CanberraDistanceKernel<DataType>
        <<<nblks, TPB>>>(dist, x, y, m, n, k, type, isRowMajor);
      break;
    case raft::distance::DistanceType::L2SqrtUnexpanded:
    case raft::distance::DistanceType::L2Unexpanded:
    case raft::distance::DistanceType::L2SqrtExpanded:
    case raft::distance::DistanceType::L2Expanded:
      naiveDistanceKernel<DataType><<<nblks, TPB>>>(dist, x, y, m, n, k, type, isRowMajor);
      break;
    case raft::distance::DistanceType::CosineExpanded:
      naiveCosineDistanceKernel<DataType><<<nblks, TPB>>>(dist, x, y, m, n, k, isRowMajor);
      break;
    case raft::distance::DistanceType::HellingerExpanded:
      naiveHellingerDistanceKernel<DataType><<<nblks, TPB>>>(dist, x, y, m, n, k, isRowMajor);
      break;
    case raft::distance::DistanceType::LpUnexpanded:
      naiveLpUnexpDistanceKernel<DataType>
        <<<nblks, TPB>>>(dist, x, y, m, n, k, isRowMajor, metric_arg);
      break;
    default: FAIL() << "should be here\n";
  }
  CUDA_CHECK(cudaPeekAtLastError());
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
  auto fin_op = [dist2, threshold] __device__(DataType d_val, int g_d_idx) {
    dist2[g_d_idx] = (d_val < threshold) ? 0.f : d_val;
    return d_val;
  };
  raft::distance::distance<distanceType, DataType, DataType, DataType>(
    x, y, dist, m, n, k, workspace, worksize, fin_op, stream, isRowMajor, metric_arg);
}

template <raft::distance::DistanceType distanceType, typename DataType>
class DistanceTest : public ::testing::TestWithParam<DistanceInputs<DataType>> {
 public:
  void SetUp() override
  {
    params = ::testing::TestWithParam<DistanceInputs<DataType>>::GetParam();
    raft::random::Rng r(params.seed);
    int m               = params.m;
    int n               = params.n;
    int k               = params.k;
    DataType metric_arg = params.metric_arg;
    bool isRowMajor     = params.isRowMajor;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    raft::allocate(x, m * k);
    raft::allocate(y, n * k);
    raft::allocate(dist_ref, m * n);
    raft::allocate(dist, m * n);
    raft::allocate(dist2, m * n);
    if (distanceType == raft::distance::DistanceType::HellingerExpanded) {
      // Hellinger works only on positive numbers
      r.uniform(x, m * k, DataType(0.0), DataType(1.0), stream);
      r.uniform(y, n * k, DataType(0.0), DataType(1.0), stream);
    } else {
      r.uniform(x, m * k, DataType(-1.0), DataType(1.0), stream);
      r.uniform(y, n * k, DataType(-1.0), DataType(1.0), stream);
    }

    naiveDistance(dist_ref, x, y, m, n, k, distanceType, isRowMajor, metric_arg);
    char* workspace = nullptr;
    size_t worksize =
      raft::distance::getWorkspaceSize<distanceType, DataType, DataType, DataType>(x, y, m, n, k);
    if (worksize != 0) { raft::allocate(workspace, worksize); }

    DataType threshold = -10000.f;
    distanceLauncher<distanceType, DataType>(x,
                                             y,
                                             dist,
                                             dist2,
                                             m,
                                             n,
                                             k,
                                             params,
                                             threshold,
                                             workspace,
                                             worksize,
                                             stream,
                                             isRowMajor,
                                             metric_arg);
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(workspace));
  }

  void TearDown() override
  {
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
    CUDA_CHECK(cudaFree(dist_ref));
    CUDA_CHECK(cudaFree(dist));
    CUDA_CHECK(cudaFree(dist2));
  }

 protected:
  DistanceInputs<DataType> params;
  DataType *x, *y, *dist_ref, *dist, *dist2;
};

}  // end namespace distance
}  // end namespace raft
