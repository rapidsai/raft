/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <raft/core/kvp.hpp>
#include <raft/distance/detail/masked_l2_nn.cuh>
#include <raft/distance/masked_l2_nn.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <iostream>
#include <stdio.h>

namespace raft {
namespace distance {
namespace masked_l2_nn {

template <typename LabelT, typename DataT>
struct RaftKVPMinReduce {
  typedef raft::KeyValuePair<LabelT, DataT> KVP;

  DI KVP operator()(LabelT rit, const KVP& a, const KVP& b) { return b.value < a.value ? b : a; }

  DI KVP operator()(const KVP& a, const KVP& b) { return b.value < a.value ? b : a; }

};  // KVPMinReduce

template <typename DataT, bool Sqrt, typename ReduceOpT, int NWARPS>
__global__ __launch_bounds__(32 * NWARPS, 2) void naiveKernel(raft::KeyValuePair<int, DataT>* min,
                                                              DataT* x,
                                                              DataT* y,
                                                              bool* adj,
                                                              int* group_idxs,
                                                              int m,
                                                              int n,
                                                              int k,
                                                              int num_groups,
                                                              int* workspace,
                                                              DataT maxVal)
{
  const int m_stride = blockDim.y * gridDim.y;
  const int m_offset = threadIdx.y + blockIdx.y * blockDim.y;
  const int n_stride = blockDim.x * gridDim.x;
  const int n_offset = threadIdx.x + blockIdx.x * blockDim.x;

  for (int m_grid = 0; m_grid < m; m_grid += m_stride) {
    for (int n_grid = 0; n_grid < n; n_grid += n_stride) {
      int midx = m_grid + m_offset;
      int nidx = n_grid + n_offset;

      // Do a reverse linear search to determine the group index.
      int group_idx = 0;
      for (int i = num_groups; 0 <= i; --i) {
        if (nidx < group_idxs[i]) { group_idx = i; }
      }
      const bool include_dist = adj[group_idx * m + midx] && midx < m && nidx < n;

      // Compute L2 metric.
      DataT acc = DataT(0);
      for (int i = 0; i < k; ++i) {
        int xidx  = i + midx * k;
        int yidx  = i + nidx * k;
        auto diff = x[xidx] - y[yidx];
        acc += diff * diff;
      }
      if (Sqrt) { acc = raft::sqrt(acc); }
      ReduceOpT redOp;
      typedef cub::WarpReduce<raft::KeyValuePair<int, DataT>> WarpReduce;
      __shared__ typename WarpReduce::TempStorage temp[NWARPS];
      int warpId = threadIdx.x / raft::WarpSize;
      raft::KeyValuePair<int, DataT> tmp;
      tmp.key   = include_dist ? nidx : -1;
      tmp.value = include_dist ? acc : maxVal;
      tmp       = WarpReduce(temp[warpId]).Reduce(tmp, RaftKVPMinReduce<int, DataT>());
      if (threadIdx.x % raft::WarpSize == 0 && midx < m) {
        while (atomicCAS(workspace + midx, 0, 1) == 1)
          ;
        __threadfence();
        redOp(midx, min + midx, tmp);
        __threadfence();
        atomicCAS(workspace + midx, 1, 0);
      }
      __syncthreads();
    }
  }
}

template <typename DataT, bool Sqrt>
void naive(raft::KeyValuePair<int, DataT>* min,
           DataT* x,
           DataT* y,
           bool* adj,
           int* group_idxs,
           int m,
           int n,
           int k,
           int num_groups,
           int* workspace,
           cudaStream_t stream)
{
  RAFT_CUDA_TRY(cudaMemsetAsync(workspace, 0, sizeof(int) * m, stream));
  auto blks = raft::ceildiv(m, 256);
  MinAndDistanceReduceOp<int, DataT> op;
  raft::distance::detail::initKernel<DataT, raft::KeyValuePair<int, DataT>, int>
    <<<blks, 256, 0, stream>>>(min, m, std::numeric_limits<DataT>::max(), op);
  RAFT_CUDA_TRY(cudaGetLastError());

  const int nwarps = 16;
  static const dim3 TPB(32, nwarps, 1);
  dim3 nblks(1, 200, 1);
  naiveKernel<DataT, Sqrt, MinAndDistanceReduceOp<int, DataT>, nwarps><<<nblks, TPB, 0, stream>>>(
    min, x, y, adj, group_idxs, m, n, k, num_groups, workspace, std::numeric_limits<DataT>::max());
  RAFT_CUDA_TRY(cudaGetLastError());
}

enum AdjacencyPattern {
  checkerboard    = 0,
  checkerboard_4  = 1,
  checkerboard_64 = 2,
  all_true        = 3,
  all_false       = 4
};

template <typename DataT>
struct Inputs {
  DataT tolerance;
  int m, n, k, num_groups;
  unsigned long long int seed;

  AdjacencyPattern pattern;

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
                 "num_groups: "
              << p.num_groups
              << ", "
                 "seed: "
              << p.seed
              << ", "
                 "tol: "
              << p.tolerance;
  }
};

__global__ void init_adj(
  int m, int n, int num_groups, AdjacencyPattern pattern, bool* adj, int* group_idxs)
{
  for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < num_groups; i += blockDim.y * gridDim.y) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < m; j += blockDim.x * gridDim.x) {
      switch (pattern) {
        case checkerboard: adj[i * m + j] = (i + j) % 2; break;
        case checkerboard_4: adj[i * m + j] = (i + (j / 4)) % 2; break;
        case checkerboard_64: adj[i * m + j] = (i + (j / 64)) % 2; break;
        case all_true: adj[i * m + j] = true; break;
        case all_false: adj[i * m + j] = false; break;
        default: assert(false && "unknown pattern");
      }
    }
  }
  // Each group is of size n / num_groups.
  //
  // - group_idxs[j] indicates the start of group j + 1 (i.e. is the inclusive
  // scan of the group lengths)
  //
  // - The first group always starts at index zero, so we do not store it.
  //
  // - The group_idxs[num_groups - 1] should always equal n.

  if (blockIdx.y == 0 && threadIdx.y == 0) {
    const int j_stride = blockDim.x * gridDim.x;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < num_groups; j += j_stride) {
      group_idxs[j] = (j + 1) * (n / num_groups);
    }
    group_idxs[num_groups - 1] = n;
  }
}

template <typename DataT, bool Sqrt>
class MaskedL2NNTest : public ::testing::TestWithParam<Inputs<DataT>> {
 public:
  MaskedL2NNTest()
    : params(::testing::TestWithParam<Inputs<DataT>>::GetParam()),
      stream(handle.get_stream()),
      x(params.m * params.k, stream),
      y(params.n * params.k, stream),
      adj(params.m * params.num_groups, stream),
      group_idxs(params.num_groups, stream),
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
    int m          = params.m;
    int n          = params.n;
    int k          = params.k;
    int num_groups = params.num_groups;
    uniform(handle, r, x.data(), m * k, DataT(-1.0), DataT(1.0));
    uniform(handle, r, y.data(), n * k, DataT(-1.0), DataT(1.0));

    dim3 block(32, 32);
    dim3 grid(10, 10);
    init_adj<<<grid, block, 0, stream>>>(
      m, n, num_groups, params.pattern, adj.data(), group_idxs.data());
    RAFT_CUDA_TRY(cudaGetLastError());

    generateGoldenResult();
    raft::linalg::rowNorm(xn.data(), x.data(), k, m, raft::linalg::L2Norm, true, stream);
    raft::linalg::rowNorm(yn.data(), y.data(), k, n, raft::linalg::L2Norm, true, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

 protected:
  Inputs<DataT> params;
  rmm::device_uvector<DataT> x;
  rmm::device_uvector<DataT> y;
  rmm::device_uvector<bool> adj;
  rmm::device_uvector<int> group_idxs;
  rmm::device_uvector<DataT> xn;
  rmm::device_uvector<DataT> yn;
  rmm::device_uvector<raft::KeyValuePair<int, DataT>> min;
  rmm::device_uvector<raft::KeyValuePair<int, DataT>> min_ref;
  rmm::device_uvector<char> workspace;
  raft::handle_t handle;
  cudaStream_t stream;

  virtual void generateGoldenResult()
  {
    int m          = params.m;
    int n          = params.n;
    int k          = params.k;
    int num_groups = params.num_groups;

    naive<DataT, Sqrt>(min_ref.data(),
                       x.data(),
                       y.data(),
                       adj.data(),
                       group_idxs.data(),
                       m,
                       n,
                       k,
                       num_groups,
                       (int*)workspace.data(),
                       stream);
  }

  void runTest(raft::KeyValuePair<int, DataT>* out)
  {
    int m          = params.m;
    int n          = params.n;
    int k          = params.k;
    int num_groups = params.num_groups;

    MinAndDistanceReduceOp<int, DataT> redOp;
    maskedL2NN<DataT, raft::KeyValuePair<int, DataT>, int>(
      handle,
      out,
      x.data(),
      y.data(),
      xn.data(),
      yn.data(),
      adj.data(),
      group_idxs.data(),
      num_groups,
      m,
      n,
      k,
      (void*)workspace.data(),
      redOp,
      raft::distance::KVPMinReduce<int, DataT>(),
      Sqrt,
      true);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }
};

template <typename T>
struct CompareApproxAbsKVP {
  typedef typename raft::KeyValuePair<int, T> KVP;
  CompareApproxAbsKVP(T eps_) : eps(eps_) {}
  bool operator()(const KVP& a, const KVP& b) const
  {
    T diff  = raft::abs(raft::abs(a.value) - raft::abs(b.value));
    T m     = std::max(raft::abs(a.value), raft::abs(b.value));
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
  {0.001f, 32, 32, 32, 2, 1234ULL, AdjacencyPattern::all_true},
  {0.001f, 512, 512, 8, 32, 1234ULL, AdjacencyPattern::all_true},
  {0.001f, 512, 512, 8, 32, 1234ULL, AdjacencyPattern::all_false},
  {0.001f, 512, 512, 8, 32, 1234ULL, AdjacencyPattern::checkerboard},
  {0.001f, 512, 512, 8, 32, 1234ULL, AdjacencyPattern::checkerboard_4},
  {0.001f, 512, 512, 8, 32, 1234ULL, AdjacencyPattern::checkerboard_64},
  {0.001f, 1 << 9, 1 << 16, 8, 1 << 9, 1234ULL, AdjacencyPattern::all_true},
  {0.001f, 1 << 9, 1 << 16, 8, 1 << 9, 1234ULL, AdjacencyPattern::all_false},
  {0.001f, 1 << 9, 1 << 16, 8, 1 << 9, 1234ULL, AdjacencyPattern::checkerboard},
  {0.001f, 1 << 9, 1 << 16, 8, 1 << 9, 1234ULL, AdjacencyPattern::checkerboard_4},
  {0.001f, 1 << 9, 1 << 16, 8, 1 << 9, 1234ULL, AdjacencyPattern::checkerboard_64},
  {0.001f, (1 << 15) + 19, (1 << 9) + 17, 8, 32, 1234ULL, AdjacencyPattern::all_true},
  {0.001f, (1 << 15) + 19, (1 << 9) + 17, 8, 32, 1234ULL, AdjacencyPattern::all_false},
  {0.001f, (1 << 15) + 19, (1 << 9) + 17, 8, 32, 1234ULL, AdjacencyPattern::checkerboard},
};

typedef MaskedL2NNTest<float, false> MaskedL2NNTestF_Sq;
TEST_P(MaskedL2NNTestF_Sq, Result)
{
  runTest(min.data());
  ASSERT_TRUE(devArrMatch(
    min_ref.data(), min.data(), params.m, CompareApproxAbsKVP<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(MaskedL2NNTests, MaskedL2NNTestF_Sq, ::testing::ValuesIn(inputsf));
typedef MaskedL2NNTest<float, true> MaskedL2NNTestF_Sqrt;
TEST_P(MaskedL2NNTestF_Sqrt, Result)
{
  runTest(min.data());
  ASSERT_TRUE(devArrMatch(
    min_ref.data(), min.data(), params.m, CompareApproxAbsKVP<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(MaskedL2NNTests, MaskedL2NNTestF_Sqrt, ::testing::ValuesIn(inputsf));

const std::vector<Inputs<double>> inputsd = {
  {0.00001, 32, 32, 32, 2, 1234ULL, AdjacencyPattern::all_true},

  {0.00001, 512, 512, 8, 32, 1234ULL, AdjacencyPattern::all_true},
  {0.00001, 512, 512, 8, 32, 1234ULL, AdjacencyPattern::all_false},
  {0.00001, 512, 512, 8, 32, 1234ULL, AdjacencyPattern::checkerboard},
  {0.00001, 512, 512, 8, 32, 1234ULL, AdjacencyPattern::checkerboard_4},
  {0.00001, 512, 512, 8, 32, 1234ULL, AdjacencyPattern::checkerboard_64},

  {0.00001, 1 << 9, 1 << 16, 8, 1 << 9, 1234ULL, AdjacencyPattern::all_true},
  {0.00001, 1 << 9, 1 << 16, 8, 1 << 9, 1234ULL, AdjacencyPattern::all_false},
  {0.00001, 1 << 9, 1 << 16, 8, 1 << 9, 1234ULL, AdjacencyPattern::checkerboard},
  {0.00001, 1 << 9, 1 << 16, 8, 1 << 9, 1234ULL, AdjacencyPattern::checkerboard_4},
  {0.00001, 1 << 9, 1 << 16, 8, 1 << 9, 1234ULL, AdjacencyPattern::checkerboard_64},
};
typedef MaskedL2NNTest<double, false> MaskedL2NNTestD_Sq;
TEST_P(MaskedL2NNTestD_Sq, Result)
{
  runTest(min.data());
  ASSERT_TRUE(devArrMatch(
    min_ref.data(), min.data(), params.m, CompareApproxAbsKVP<double>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(MaskedL2NNTests, MaskedL2NNTestD_Sq, ::testing::ValuesIn(inputsd));
typedef MaskedL2NNTest<double, true> MaskedL2NNTestD_Sqrt;
TEST_P(MaskedL2NNTestD_Sqrt, Result)
{
  runTest(min.data());
  ASSERT_TRUE(devArrMatch(
    min_ref.data(), min.data(), params.m, CompareApproxAbsKVP<double>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(MaskedL2NNTests, MaskedL2NNTestD_Sqrt, ::testing::ValuesIn(inputsd));

/// This is to test output determinism of the prim
template <typename DataT, bool Sqrt>
class MaskedL2NNDetTest : public MaskedL2NNTest<DataT, Sqrt> {
 public:
  MaskedL2NNDetTest() : stream(handle.get_stream()), min1(0, stream) {}

  void SetUp() override
  {
    MaskedL2NNTest<DataT, Sqrt>::SetUp();
    int m = this->params.m;
    min1.resize(m, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  void TearDown() override { MaskedL2NNTest<DataT, Sqrt>::TearDown(); }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;

  rmm::device_uvector<raft::KeyValuePair<int, DataT>> min1;

  static const int NumRepeats = 100;

  void generateGoldenResult() override {}
};

typedef MaskedL2NNDetTest<float, false> MaskedL2NNDetTestF_Sq;
TEST_P(MaskedL2NNDetTestF_Sq, Result)
{
  runTest(min.data());  // assumed to be golden
  for (int i = 0; i < NumRepeats; ++i) {
    runTest(min1.data());
    ASSERT_TRUE(devArrMatch(min.data(), min1.data(), params.m, CompareExactKVP<float>(), stream));
  }
}
INSTANTIATE_TEST_CASE_P(MaskedL2NNDetTests, MaskedL2NNDetTestF_Sq, ::testing::ValuesIn(inputsf));
typedef MaskedL2NNDetTest<float, true> MaskedL2NNDetTestF_Sqrt;
TEST_P(MaskedL2NNDetTestF_Sqrt, Result)
{
  runTest(min.data());  // assumed to be golden
  for (int i = 0; i < NumRepeats; ++i) {
    runTest(min1.data());
    ASSERT_TRUE(devArrMatch(min.data(), min1.data(), params.m, CompareExactKVP<float>(), stream));
  }
}
INSTANTIATE_TEST_CASE_P(MaskedL2NNDetTests, MaskedL2NNDetTestF_Sqrt, ::testing::ValuesIn(inputsf));

typedef MaskedL2NNDetTest<double, false> MaskedL2NNDetTestD_Sq;
TEST_P(MaskedL2NNDetTestD_Sq, Result)
{
  runTest(min.data());  // assumed to be golden
  for (int i = 0; i < NumRepeats; ++i) {
    runTest(min1.data());
    ASSERT_TRUE(devArrMatch(min.data(), min1.data(), params.m, CompareExactKVP<double>(), stream));
  }
}
INSTANTIATE_TEST_CASE_P(MaskedL2NNDetTests, MaskedL2NNDetTestD_Sq, ::testing::ValuesIn(inputsd));
typedef MaskedL2NNDetTest<double, true> MaskedL2NNDetTestD_Sqrt;
TEST_P(MaskedL2NNDetTestD_Sqrt, Result)
{
  runTest(min.data());  // assumed to be golden
  for (int i = 0; i < NumRepeats; ++i) {
    runTest(min1.data());
    ASSERT_TRUE(devArrMatch(min.data(), min1.data(), params.m, CompareExactKVP<double>(), stream));
  }
}
INSTANTIATE_TEST_CASE_P(MaskedL2NNDetTests, MaskedL2NNDetTestD_Sqrt, ::testing::ValuesIn(inputsd));

}  // end namespace masked_l2_nn
}  // end namespace distance
}  // end namespace raft
