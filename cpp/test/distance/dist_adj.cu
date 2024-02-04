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
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>

#include "dist_adj.cuh"

namespace raft {
namespace distance {

template <typename DataType>
RAFT_KERNEL naiveDistanceAdjKernel(uint8_t* dist,
                                   const DataType* x,
                                   const DataType* y,
                                   int m,
                                   int n,
                                   int k,
                                   DataType eps,
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
  int outidx   = isRowMajor ? midx * n + nidx : midx + m * nidx;
  dist[outidx] = acc <= eps;
}

template <typename DataType>
void naiveDistanceAdj(uint8_t* dist,
                      const DataType* x,
                      const DataType* y,
                      int m,
                      int n,
                      int k,
                      DataType eps,
                      bool isRowMajor,
                      cudaStream_t stream)
{
  static const dim3 TPB(16, 32, 1);
  dim3 nblks(raft::ceildiv(m, (int)TPB.x), raft::ceildiv(n, (int)TPB.y), 1);
  naiveDistanceAdjKernel<DataType><<<nblks, TPB, 0, stream>>>(dist, x, y, m, n, k, eps, isRowMajor);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename DataType>
struct DistanceAdjInputs {
  DataType eps;
  int m, n, k;
  bool isRowMajor;
  unsigned long long int seed;
};

template <typename DataType>
::std::ostream& operator<<(::std::ostream& os, const DistanceAdjInputs<DataType>& dims)
{
  return os;
}

template <typename DataType>
class DistanceAdjTest : public ::testing::TestWithParam<DistanceAdjInputs<DataType>> {
 public:
  DistanceAdjTest()
    : params(::testing::TestWithParam<DistanceAdjInputs<DataType>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      dist(params.m * params.n, stream),
      dist_ref(params.m * params.n, stream)
  {
  }

  void SetUp() override
  {
    raft::random::RngState r(params.seed);
    int m           = params.m;
    int n           = params.n;
    int k           = params.k;
    bool isRowMajor = params.isRowMajor;

    rmm::device_uvector<DataType> x(m * k, stream);
    rmm::device_uvector<DataType> y(n * k, stream);

    uniform(handle, r, x.data(), m * k, DataType(-1.0), DataType(1.0));
    uniform(handle, r, y.data(), n * k, DataType(-1.0), DataType(1.0));

    DataType threshold = params.eps;

    naiveDistanceAdj(dist_ref.data(), x.data(), y.data(), m, n, k, threshold, isRowMajor, stream);
    size_t worksize = raft::distance::
      getWorkspaceSize<raft::distance::DistanceType::L2Expanded, DataType, DataType, uint8_t>(
        x.data(), y.data(), m, n, k);
    rmm::device_uvector<char> workspace(worksize, stream);

    using threshold_final_op_ = threshold_final_op<DataType, DataType, uint8_t, int>;
    threshold_final_op_ threshold_op(threshold);

    raft::distance::distance<raft::distance::DistanceType::L2Expanded,
                             DataType,
                             DataType,
                             uint8_t,
                             threshold_final_op_>(handle,
                                                  x.data(),
                                                  y.data(),
                                                  dist.data(),
                                                  m,
                                                  n,
                                                  k,
                                                  workspace.data(),
                                                  worksize,
                                                  threshold_op,
                                                  isRowMajor);
    resource::sync_stream(handle, stream);
  }

  void TearDown() override {}

 protected:
  DistanceAdjInputs<DataType> params;
  // We use uint8_t even if the output in this test is a bool because
  // cutlass doesn't support bool as output buffer yet. In cuda
  // sizeof(bool) is 1 byte hence it doesn't increase
  // memory consumption if we use uint8_t instead of bool.
  rmm::device_uvector<uint8_t> dist_ref;
  rmm::device_uvector<uint8_t> dist;
  raft::resources handle;
  cudaStream_t stream;
};

const std::vector<DistanceAdjInputs<float>> inputsf = {
  {0.01f, 1024, 1024, 32, true, 1234ULL},
  {0.1f, 1024, 1024, 32, true, 1234ULL},
  {1.0f, 1024, 1024, 32, true, 1234ULL},
  {10.0f, 1024, 1024, 32, true, 1234ULL},
  {0.01f, 1024, 1024, 32, false, 1234ULL},
  {0.1f, 1024, 1024, 32, false, 1234ULL},
  {1.0f, 1024, 1024, 32, false, 1234ULL},
  {10.0f, 1024, 1024, 32, false, 1234ULL},
};
typedef DistanceAdjTest<float> DistanceAdjTestF;
TEST_P(DistanceAdjTestF, Result)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(devArrMatch(dist_ref.data(), dist.data(), m, n, raft::Compare<uint8_t>(), stream));
}
INSTANTIATE_TEST_CASE_P(DistanceAdjTests, DistanceAdjTestF, ::testing::ValuesIn(inputsf));

const std::vector<DistanceAdjInputs<double>> inputsd = {
  {0.01, 1024, 1024, 32, true, 1234ULL},
  {0.1, 1024, 1024, 32, true, 1234ULL},
  {1.0, 1024, 1024, 32, true, 1234ULL},
  {10.0, 1024, 1024, 32, true, 1234ULL},
  {0.01, 1024, 1024, 32, false, 1234ULL},
  {0.1, 1024, 1024, 32, false, 1234ULL},
  {1.0, 1024, 1024, 32, false, 1234ULL},
  {10.0, 1024, 1024, 32, false, 1234ULL},
};
typedef DistanceAdjTest<double> DistanceAdjTestD;
TEST_P(DistanceAdjTestD, Result)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(devArrMatch(dist_ref.data(), dist.data(), m, n, raft::Compare<uint8_t>(), stream));
}
INSTANTIATE_TEST_CASE_P(DistanceAdjTests, DistanceAdjTestD, ::testing::ValuesIn(inputsd));

}  // namespace distance
}  // end namespace raft
