/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
#include <raft/core/kvp.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/distance/detail/fused_l2_nn.cuh>
#include <raft/distance/fused_l2_nn.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

namespace raft {
namespace distance {

template <typename LabelT, typename DataT>
struct RaftKVPMinReduce {
  typedef raft::KeyValuePair<LabelT, DataT> KVP;

  DI KVP operator()(LabelT rit, const KVP& a, const KVP& b) { return b.value < a.value ? b : a; }

  DI KVP operator()(const KVP& a, const KVP& b) { return b.value < a.value ? b : a; }

};  // KVPMinReduce

template <typename DataT, bool Sqrt, typename ReduceOpT, int NWARPS>
RAFT_KERNEL naiveKernel(raft::KeyValuePair<int, DataT>* min,
                        DataT* x,
                        DataT* y,
                        int m,
                        int n,
                        int k,
                        int* workspace,
                        DataT maxVal)
{
  int midx  = threadIdx.y + blockIdx.y * blockDim.y;
  int nidx  = threadIdx.x + blockIdx.x * blockDim.x;
  DataT acc = DataT(0);
  for (int i = 0; i < k; ++i) {
    int xidx  = i + midx * k;
    int yidx  = i + nidx * k;
    auto diff = midx >= m || nidx >= n ? DataT(0) : x[xidx] - y[yidx];
    acc += diff * diff;
  }
  if (Sqrt) { acc = raft::sqrt(acc); }
  ReduceOpT redOp;
  typedef cub::WarpReduce<raft::KeyValuePair<int, DataT>> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp[NWARPS];
  int warpId = threadIdx.x / raft::WarpSize;
  raft::KeyValuePair<int, DataT> tmp;
  tmp.key   = nidx;
  tmp.value = midx >= m || nidx >= n ? maxVal : acc;
  tmp       = WarpReduce(temp[warpId]).Reduce(tmp, RaftKVPMinReduce<int, DataT>());
  if (threadIdx.x % raft::WarpSize == 0 && midx < m) {
    while (atomicCAS(workspace + midx, 0, 1) == 1)
      ;
    __threadfence();
    redOp(midx, min + midx, tmp);
    __threadfence();
    atomicCAS(workspace + midx, 1, 0);
  }
}

template <typename DataT, bool Sqrt>
void naive(raft::KeyValuePair<int, DataT>* min,
           DataT* x,
           DataT* y,
           int m,
           int n,
           int k,
           int* workspace,
           cudaStream_t stream)
{
  static const dim3 TPB(32, 16, 1);
  dim3 nblks(raft::ceildiv(n, (int)TPB.x), raft::ceildiv(m, (int)TPB.y), 1);
  RAFT_CUDA_TRY(cudaMemsetAsync(workspace, 0, sizeof(int) * m, stream));
  auto blks = raft::ceildiv(m, 256);
  MinAndDistanceReduceOp<int, DataT> op;
  detail::initKernel<DataT, raft::KeyValuePair<int, DataT>, int>
    <<<blks, 256, 0, stream>>>(min, m, std::numeric_limits<DataT>::max(), op);
  RAFT_CUDA_TRY(cudaGetLastError());
  naiveKernel<DataT, Sqrt, MinAndDistanceReduceOp<int, DataT>, 16>
    <<<nblks, TPB, 0, stream>>>(min, x, y, m, n, k, workspace, std::numeric_limits<DataT>::max());
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename DataT>
struct Inputs {
  DataT tolerance;
  int m, n, k;
  unsigned long long int seed;

  friend std::ostream& operator<<(std::ostream& os, const Inputs& p)
  {
    return os << "m: " << p.m
              << ", "
                 "n: "
              << p.n
              << ", "
                 "k: "
              << p.k
              << ", "
                 "seed: "
              << p.seed
              << ", "
                 "tol: "
              << p.tolerance;
  }
};

template <typename DataT, bool Sqrt>
class FusedL2NNTest : public ::testing::TestWithParam<Inputs<DataT>> {
 public:
  FusedL2NNTest()
    : params(::testing::TestWithParam<Inputs<DataT>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      x(params.m * params.k, stream),
      y(params.n * params.k, stream),
      xn(params.m, stream),
      yn(params.n, stream),
      min(params.m, stream),
      min_ref(params.m, stream),
      workspace(params.m * sizeof(int), stream)
  {
  }

 protected:
  void SetUp() override
  {
    raft::random::RngState r(params.seed);
    int m = params.m;
    int n = params.n;
    int k = params.k;
    uniform(handle, r, x.data(), m * k, DataT(-1.0), DataT(1.0));
    uniform(handle, r, y.data(), n * k, DataT(-1.0), DataT(1.0));
    generateGoldenResult();
    raft::linalg::rowNorm(xn.data(), x.data(), k, m, raft::linalg::L2Norm, true, stream);
    raft::linalg::rowNorm(yn.data(), y.data(), k, n, raft::linalg::L2Norm, true, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;
  Inputs<DataT> params;
  rmm::device_uvector<DataT> x;
  rmm::device_uvector<DataT> y;
  rmm::device_uvector<DataT> xn;
  rmm::device_uvector<DataT> yn;
  rmm::device_uvector<raft::KeyValuePair<int, DataT>> min;
  rmm::device_uvector<raft::KeyValuePair<int, DataT>> min_ref;
  rmm::device_uvector<char> workspace;

  virtual void generateGoldenResult()
  {
    int m = params.m;
    int n = params.n;
    int k = params.k;
    naive<DataT, Sqrt>(min_ref.data(), x.data(), y.data(), m, n, k, (int*)workspace.data(), stream);
  }

  void runTest(raft::KeyValuePair<int, DataT>* out)
  {
    int m = params.m;
    int n = params.n;
    int k = params.k;

    const bool init_out_buffer = true;
    fusedL2NNMinReduce<DataT, raft::KeyValuePair<int, DataT>, int>(out,
                                                                   x.data(),
                                                                   y.data(),
                                                                   xn.data(),
                                                                   yn.data(),
                                                                   m,
                                                                   n,
                                                                   k,
                                                                   (void*)workspace.data(),
                                                                   Sqrt,
                                                                   init_out_buffer,
                                                                   stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }
};

template <typename T>
struct CompareApproxAbsKVP {
  typedef typename raft::KeyValuePair<int, T> KVP;
  CompareApproxAbsKVP(T eps_) : eps(eps_) {}
  bool operator()(const KVP& a, const KVP& b) const
  {
    T diff  = std::abs(std::abs(a.value) - std::abs(b.value));
    T m     = std::max(std::abs(a.value), std::abs(b.value));
    T ratio = m >= eps ? diff / m : diff;
    return (ratio <= eps);
  }

 private:
  T eps;
};

template <typename T>
struct CompareExactKVP {
  typedef typename raft::KeyValuePair<int, T> KVP;
  bool operator()(const KVP& a, const KVP& b) const
  {
    if (a.value != b.value) return false;
    return true;
  }
};

template <typename K, typename V, typename L>
::testing::AssertionResult devArrMatch(const raft::KeyValuePair<K, V>* expected,
                                       const raft::KeyValuePair<K, V>* actual,
                                       size_t size,
                                       L eq_compare,
                                       cudaStream_t stream = 0)
{
  typedef typename raft::KeyValuePair<K, V> KVP;
  std::shared_ptr<KVP> exp_h(new KVP[size]);
  std::shared_ptr<KVP> act_h(new KVP[size]);
  raft::update_host<KVP>(exp_h.get(), expected, size, stream);
  raft::update_host<KVP>(act_h.get(), actual, size, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  for (size_t i(0); i < size; ++i) {
    auto exp = exp_h.get()[i];
    auto act = act_h.get()[i];
    if (!eq_compare(exp, act)) {
      return ::testing::AssertionFailure()
             << "actual=" << act.key << "," << act.value << " != expected=" << exp.key << ","
             << exp.value << " @" << i;
    }
  }
  return ::testing::AssertionSuccess();
}

const std::vector<Inputs<float>> inputsf = {
  {0.001f, 32, 32, 32, 1234ULL},
  {0.001f, 32, 64, 32, 1234ULL},
  {0.001f, 64, 32, 32, 1234ULL},
  {0.001f, 64, 64, 32, 1234ULL},
  {0.001f, 128, 32, 32, 1234ULL},
  {0.001f, 128, 64, 32, 1234ULL},
  {0.001f, 128, 128, 64, 1234ULL},
  {0.001f, 64, 128, 128, 1234ULL},

  {0.001f, 32, 32, 34, 1234ULL},
  {0.001f, 32, 64, 34, 1234ULL},
  {0.001f, 64, 32, 34, 1234ULL},
  {0.001f, 64, 64, 34, 1234ULL},
  {0.001f, 128, 32, 34, 1234ULL},
  {0.001f, 128, 64, 34, 1234ULL},
  {0.001f, 128, 128, 66, 1234ULL},
  {0.001f, 64, 128, 130, 1234ULL},

  {0.001f, 32, 32, 33, 1234ULL},
  {0.001f, 32, 64, 33, 1234ULL},
  {0.001f, 64, 32, 33, 1234ULL},
  {0.001f, 64, 64, 33, 1234ULL},
  {0.001f, 128, 32, 33, 1234ULL},
  {0.001f, 128, 64, 33, 1234ULL},
  {0.001f, 128, 128, 65, 1234ULL},
  {0.001f, 64, 128, 129, 1234ULL},
  {0.006f, 1805, 134, 2, 1234ULL},
  {0.006f, 8192, 1024, 64, 1234ULL},
  {0.006f, 8192, 1025, 64, 1234ULL},

  // Repeat with smaller values of k
  {0.006f, 32, 32, 1, 1234ULL},
  {0.001f, 32, 64, 2, 1234ULL},
  {0.001f, 64, 32, 3, 1234ULL},
  {0.001f, 64, 64, 4, 1234ULL},
  {0.001f, 128, 32, 5, 1234ULL},
  {0.001f, 128, 64, 6, 1234ULL},
  {0.001f, 128, 128, 7, 1234ULL},
  {0.001f, 64, 128, 8, 1234ULL},

  {0.001f, 32, 32, 9, 1234ULL},
  {0.001f, 32, 64, 10, 1234ULL},
  {0.001f, 64, 32, 11, 1234ULL},
  {0.001f, 64, 64, 12, 1234ULL},
  {0.001f, 128, 32, 13, 1234ULL},
  {0.001f, 128, 64, 14, 1234ULL},
  {0.001f, 128, 128, 15, 1234ULL},
  {0.001f, 64, 128, 16, 1234ULL},

  {0.001f, 32, 32, 17, 1234ULL},
  {0.001f, 32, 64, 18, 1234ULL},
  {0.001f, 64, 32, 19, 1234ULL},
  {0.001f, 64, 64, 20, 1234ULL},
  {0.001f, 128, 32, 21, 1234ULL},
  {0.001f, 128, 64, 22, 1234ULL},
  {0.001f, 128, 128, 23, 1234ULL},
  {0.00001, 64, 128, 24, 1234ULL},
  {0.001f, 1805, 134, 25, 1234ULL},
  {0.006f, 8192, 1024, 25, 1234ULL},
  {0.006f, 8192, 1024, 66, 1234ULL},
};
typedef FusedL2NNTest<float, false> FusedL2NNTestF_Sq;
TEST_P(FusedL2NNTestF_Sq, Result)
{
  runTest(min.data());
  ASSERT_TRUE(devArrMatch(
    min_ref.data(), min.data(), params.m, CompareApproxAbsKVP<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(FusedL2NNTests, FusedL2NNTestF_Sq, ::testing::ValuesIn(inputsf));
typedef FusedL2NNTest<float, true> FusedL2NNTestF_Sqrt;
TEST_P(FusedL2NNTestF_Sqrt, Result)
{
  runTest(min.data());
  ASSERT_TRUE(devArrMatch(
    min_ref.data(), min.data(), params.m, CompareApproxAbsKVP<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(FusedL2NNTests, FusedL2NNTestF_Sqrt, ::testing::ValuesIn(inputsf));

const std::vector<Inputs<double>> inputsd = {
  {0.00001, 32, 32, 32, 1234ULL},   {0.00001, 32, 64, 32, 1234ULL},
  {0.00001, 64, 32, 32, 1234ULL},   {0.00001, 64, 64, 32, 1234ULL},
  {0.00001, 128, 32, 32, 1234ULL},  {0.00001, 128, 64, 32, 1234ULL},
  {0.00001, 128, 128, 64, 1234ULL}, {0.00001, 64, 128, 128, 1234ULL},

  {0.00001, 32, 32, 34, 1234ULL},   {0.00001, 32, 64, 34, 1234ULL},
  {0.00001, 64, 32, 34, 1234ULL},   {0.00001, 64, 64, 34, 1234ULL},
  {0.00001, 128, 32, 34, 1234ULL},  {0.00001, 128, 64, 34, 1234ULL},
  {0.00001, 128, 128, 66, 1234ULL}, {0.00001, 64, 128, 130, 1234ULL},

  {0.00001, 32, 32, 33, 1234ULL},   {0.00001, 32, 64, 33, 1234ULL},
  {0.00001, 64, 32, 33, 1234ULL},   {0.00001, 64, 64, 33, 1234ULL},
  {0.00001, 128, 32, 33, 1234ULL},  {0.00001, 128, 64, 33, 1234ULL},
  {0.00001, 128, 128, 65, 1234ULL}, {0.00001, 64, 128, 129, 1234ULL},

  {0.00001, 1805, 134, 2, 1234ULL}, {0.00001, 8192, 1024, 25, 1234ULL},
};
typedef FusedL2NNTest<double, false> FusedL2NNTestD_Sq;
TEST_P(FusedL2NNTestD_Sq, Result)
{
  runTest(min.data());
  ASSERT_TRUE(devArrMatch(
    min_ref.data(), min.data(), params.m, CompareApproxAbsKVP<double>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(FusedL2NNTests, FusedL2NNTestD_Sq, ::testing::ValuesIn(inputsd));
typedef FusedL2NNTest<double, true> FusedL2NNTestD_Sqrt;
TEST_P(FusedL2NNTestD_Sqrt, Result)
{
  runTest(min.data());
  ASSERT_TRUE(devArrMatch(
    min_ref.data(), min.data(), params.m, CompareApproxAbsKVP<double>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(FusedL2NNTests, FusedL2NNTestD_Sqrt, ::testing::ValuesIn(inputsd));

/// This is to test output determinism of the prim
template <typename DataT, bool Sqrt>
class FusedL2NNDetTest : public FusedL2NNTest<DataT, Sqrt> {
 public:
  FusedL2NNDetTest() : stream(resource::get_cuda_stream(handle)), min1(0, stream) {}

  void SetUp() override
  {
    FusedL2NNTest<DataT, Sqrt>::SetUp();
    int m = this->params.m;
    min1.resize(m, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  void TearDown() override { FusedL2NNTest<DataT, Sqrt>::TearDown(); }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  rmm::device_uvector<raft::KeyValuePair<int, DataT>> min1;

  static const int NumRepeats = 3;

  void generateGoldenResult() override {}
};

typedef FusedL2NNDetTest<float, false> FusedL2NNDetTestF_Sq;
TEST_P(FusedL2NNDetTestF_Sq, Result)
{
  runTest(min.data());  // assumed to be golden
  for (int i = 0; i < NumRepeats; ++i) {
    runTest(min1.data());
    ASSERT_TRUE(devArrMatch(min.data(), min1.data(), params.m, CompareExactKVP<float>(), stream));
  }
}
INSTANTIATE_TEST_CASE_P(FusedL2NNDetTests, FusedL2NNDetTestF_Sq, ::testing::ValuesIn(inputsf));
typedef FusedL2NNDetTest<float, true> FusedL2NNDetTestF_Sqrt;
TEST_P(FusedL2NNDetTestF_Sqrt, Result)
{
  runTest(min.data());  // assumed to be golden
  for (int i = 0; i < NumRepeats; ++i) {
    runTest(min1.data());
    ASSERT_TRUE(devArrMatch(min.data(), min1.data(), params.m, CompareExactKVP<float>(), stream));
  }
}
INSTANTIATE_TEST_CASE_P(FusedL2NNDetTests, FusedL2NNDetTestF_Sqrt, ::testing::ValuesIn(inputsf));

typedef FusedL2NNDetTest<double, false> FusedL2NNDetTestD_Sq;
TEST_P(FusedL2NNDetTestD_Sq, Result)
{
  runTest(min.data());  // assumed to be golden
  for (int i = 0; i < NumRepeats; ++i) {
    runTest(min1.data());
    ASSERT_TRUE(devArrMatch(min.data(), min1.data(), params.m, CompareExactKVP<double>(), stream));
  }
}
INSTANTIATE_TEST_CASE_P(FusedL2NNDetTests, FusedL2NNDetTestD_Sq, ::testing::ValuesIn(inputsd));
typedef FusedL2NNDetTest<double, true> FusedL2NNDetTestD_Sqrt;
TEST_P(FusedL2NNDetTestD_Sqrt, Result)
{
  runTest(min.data());  // assumed to be golden
  for (int i = 0; i < NumRepeats; ++i) {
    runTest(min1.data());
    ASSERT_TRUE(devArrMatch(min.data(), min1.data(), params.m, CompareExactKVP<double>(), stream));
  }
}
INSTANTIATE_TEST_CASE_P(FusedL2NNDetTests, FusedL2NNDetTestD_Sqrt, ::testing::ValuesIn(inputsd));

}  // end namespace distance
}  // end namespace raft
