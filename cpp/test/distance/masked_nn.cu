/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <iostream>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/kvp.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/distance/detail/masked_nn.cuh>
#include <raft/distance/masked_nn.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/itertools.hpp>

namespace raft::distance::masked_nn {

// The adjacency pattern determines what distances get computed.
enum AdjacencyPattern {
  checkerboard = 0,  // adjacency matrix looks like a checkerboard (half the distances are computed)
  checkerboard_4  = 1,  // checkerboard with tiles of size 4x4
  checkerboard_64 = 2,  // checkerboard with tiles of size 64x64
  all_true        = 3,  // no distance computations can be skipped
  all_false       = 4   // all distance computations can be skipped
};

// Kernels:
// - init_adj: to initialize the adjacency kernel with a specific adjacency pattern
// - referenceKernel: to produce the ground-truth output

RAFT_KERNEL init_adj(AdjacencyPattern pattern,
                     int n,
                     raft::device_matrix_view<bool, int, raft::layout_c_contiguous> adj,
                     raft::device_vector_view<int, int, raft::layout_c_contiguous> group_idxs)
{
  int m          = adj.extent(0);
  int num_groups = adj.extent(1);

  for (int idx_m = blockIdx.y * blockDim.y + threadIdx.y; idx_m < m;
       idx_m += blockDim.y * gridDim.y) {
    for (int idx_g = blockIdx.x * blockDim.x + threadIdx.x; idx_g < num_groups;
         idx_g += blockDim.x * gridDim.x) {
      switch (pattern) {
        case checkerboard: adj(idx_m, idx_g) = (idx_m + idx_g) % 2; break;
        case checkerboard_4: adj(idx_m, idx_g) = (idx_m / 4 + idx_g) % 2; break;
        case checkerboard_64: adj(idx_m, idx_g) = (idx_m / 64 + idx_g) % 2; break;
        case all_true: adj(idx_m, idx_g) = true; break;
        case all_false: adj(idx_m, idx_g) = false; break;
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
    const int g_stride = blockDim.x * gridDim.x;
    for (int idx_g = blockIdx.x * blockDim.x + threadIdx.x; idx_g < num_groups; idx_g += g_stride) {
      group_idxs(idx_g) = (idx_g + 1) * (n / num_groups);
    }
    group_idxs(num_groups - 1) = n;
  }
}

template <typename DataT, typename ReduceOpT, int NWARPS>
__launch_bounds__(32 * NWARPS, 2) RAFT_KERNEL referenceKernel(raft::KeyValuePair<int, DataT>* min,
                                                              DataT* x,
                                                              DataT* y,
                                                              bool* adj,
                                                              int* group_idxs,
                                                              int m,
                                                              int n,
                                                              int k,
                                                              int num_groups,
                                                              bool sqrt,
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
      const bool include_dist = adj[midx * num_groups + group_idx] && midx < m && nidx < n;

      // Compute L2 metric.
      DataT acc = DataT(0);
      for (int i = 0; i < k; ++i) {
        int xidx  = i + midx * k;
        int yidx  = i + nidx * k;
        auto diff = x[xidx] - y[yidx];
        acc += diff * diff;
      }
      if (sqrt) { acc = raft::sqrt(acc); }
      ReduceOpT redOp;
      typedef cub::WarpReduce<raft::KeyValuePair<int, DataT>> WarpReduce;
      __shared__ typename WarpReduce::TempStorage temp[NWARPS];
      int warpId = threadIdx.x / raft::WarpSize;
      raft::KeyValuePair<int, DataT> tmp;
      tmp.key   = include_dist ? nidx : -1;
      tmp.value = include_dist ? acc : maxVal;
      tmp       = WarpReduce(temp[warpId]).Reduce(tmp, raft::distance::KVPMinReduce<int, DataT>{});
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

// Structs
// - Params: holds parameters for test case
// - Inputs: holds the inputs to the functions under test (x, y, adj, group_idxs). Is generated from
//   the inputs.
struct Params {
  double tolerance;
  int m, n, k, num_groups;
  bool sqrt;
  unsigned long long int seed;
  AdjacencyPattern pattern;
};

inline auto operator<<(std::ostream& os, const Params& p) -> std::ostream&
{
  os << "m: " << p.m << ", n: " << p.n << ", k: " << p.k << ", num_groups: " << p.num_groups
     << ", sqrt: " << p.sqrt << ", seed: " << p.seed << ", tol: " << p.tolerance;
  return os;
}

template <typename DataT>
struct Inputs {
  using IdxT = int;

  raft::device_matrix<DataT, IdxT> x, y;
  raft::device_matrix<bool, IdxT> adj;
  raft::device_vector<IdxT, IdxT> group_idxs;

  Inputs(const raft::handle_t& handle, const Params& p)
    : x{raft::make_device_matrix<DataT, IdxT>(handle, p.m, p.k)},
      y{raft::make_device_matrix<DataT, IdxT>(handle, p.n, p.k)},
      adj{raft::make_device_matrix<bool, IdxT>(handle, p.m, p.num_groups)},
      group_idxs{raft::make_device_vector<IdxT, IdxT>(handle, p.num_groups)}
  {
    // Initialize x, y
    raft::random::RngState r(p.seed);
    uniform(handle, r, x.data_handle(), p.m * p.k, DataT(-1.0), DataT(1.0));
    uniform(handle, r, y.data_handle(), p.n * p.k, DataT(-1.0), DataT(1.0));

    // Initialize adj, group_idxs.
    dim3 block(32, 32);
    dim3 grid(10, 10);
    init_adj<<<grid, block, 0, resource::get_cuda_stream(handle)>>>(
      p.pattern, p.n, adj.view(), group_idxs.view());
    RAFT_CUDA_TRY(cudaGetLastError());
  }
};

template <typename DataT, typename OutT = raft::KeyValuePair<int, DataT>>
auto reference(const raft::handle_t& handle, Inputs<DataT> inp, const Params& p)
  -> raft::device_vector<OutT, int>
{
  int m          = inp.x.extent(0);
  int n          = inp.y.extent(0);
  int k          = inp.x.extent(1);
  int num_groups = inp.group_idxs.extent(0);

  if (m == 0 || n == 0 || k == 0 || num_groups == 0) {
    return raft::make_device_vector<OutT, int>(handle, 0);
  }

  // Initialize workspace
  auto stream = resource::get_cuda_stream(handle);
  rmm::device_uvector<char> workspace(p.m * sizeof(int), stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(workspace.data(), 0, sizeof(int) * m, stream));

  // Initialize output
  auto out  = raft::make_device_vector<OutT, int>(handle, m);
  auto blks = raft::ceildiv(m, 256);
  MinAndDistanceReduceOp<int, DataT> op;
  raft::distance::detail::initKernel<DataT, raft::KeyValuePair<int, DataT>, int>
    <<<blks, 256, 0, stream>>>(out.data_handle(), m, std::numeric_limits<DataT>::max(), op);
  RAFT_CUDA_TRY(cudaGetLastError());

  // Launch reference kernel
  const int nwarps = 16;
  static const dim3 TPB(32, nwarps, 1);
  dim3 nblks(1, 200, 1);
  referenceKernel<DataT, decltype(op), nwarps>
    <<<nblks, TPB, 0, stream>>>(out.data_handle(),
                                inp.x.data_handle(),
                                inp.y.data_handle(),
                                inp.adj.data_handle(),
                                inp.group_idxs.data_handle(),
                                m,
                                n,
                                k,
                                num_groups,
                                p.sqrt,
                                (int*)workspace.data(),
                                std::numeric_limits<DataT>::max());
  RAFT_CUDA_TRY(cudaGetLastError());

  return out;
}

template <typename DataT, typename OutT = raft::KeyValuePair<int, DataT>>
auto run_masked_nn(const raft::handle_t& handle, Inputs<DataT> inp, const Params& p)
  -> raft::device_vector<OutT, int>
{
  // Compute norms:
  auto x_norm = raft::make_device_vector<DataT, int>(handle, p.m);
  auto y_norm = raft::make_device_vector<DataT, int>(handle, p.n);

  raft::linalg::norm(handle,
                     std::as_const(inp.x).view(),
                     x_norm.view(),
                     raft::linalg::L2Norm,
                     raft::linalg::Apply::ALONG_ROWS);
  raft::linalg::norm(handle,
                     std::as_const(inp.y).view(),
                     y_norm.view(),
                     raft::linalg::L2Norm,
                     raft::linalg::Apply::ALONG_ROWS);

  // Create parameters for masked_l2_nn
  using IdxT       = int;
  using RedOpT     = MinAndDistanceReduceOp<int, DataT>;
  using PairRedOpT = raft::distance::KVPMinReduce<int, DataT>;
  using ParamT     = raft::distance::masked_l2_nn_params<RedOpT, PairRedOpT>;

  bool init_out = true;
  ParamT masked_l2_params{RedOpT{}, PairRedOpT{}, p.sqrt, init_out};

  // Create output
  auto out = raft::make_device_vector<OutT, IdxT, raft::layout_c_contiguous>(handle, p.m);

  // Launch kernel
  raft::distance::masked_l2_nn<DataT, OutT, IdxT>(handle,
                                                  masked_l2_params,
                                                  inp.x.view(),
                                                  inp.y.view(),
                                                  x_norm.view(),
                                                  y_norm.view(),
                                                  inp.adj.view(),
                                                  inp.group_idxs.view(),
                                                  out.view());

  resource::sync_stream(handle);

  return out;
}

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

inline auto gen_params() -> std::vector<Params>
{
  // Regular powers of two
  auto regular = raft::util::itertools::product<Params>({0.001f},       // tolerance
                                                        {32, 64, 512},  // m
                                                        {32, 64, 512},  // n
                                                        {8, 32},        // k
                                                        {2, 32},        // num_groups
                                                        {true, false},  // sqrt
                                                        {1234ULL},      // seed
                                                        {AdjacencyPattern::all_true,
                                                         AdjacencyPattern::checkerboard,
                                                         AdjacencyPattern::checkerboard_64,
                                                         AdjacencyPattern::all_false});

  // Irregular sizes to check tiling and bounds checking
  auto irregular = raft::util::itertools::product<Params>({0.001f},         // tolerance
                                                          {511, 512, 513},  // m
                                                          {127, 128, 129},  // n
                                                          {5},              // k
                                                          {3, 9},           // num_groups
                                                          {true, false},    // sqrt
                                                          {1234ULL},        // seed
                                                          {AdjacencyPattern::all_true,
                                                           AdjacencyPattern::checkerboard,
                                                           AdjacencyPattern::checkerboard_64});

  regular.insert(regular.end(), irregular.begin(), irregular.end());

  return regular;
}

class MaskedL2NNTest : public ::testing::TestWithParam<Params> {
  // Empty.
};

//
TEST_P(MaskedL2NNTest, ReferenceCheckFloat)
{
  using DataT = float;

  // Get parameters; create handle and input data.
  Params p = GetParam();
  raft::handle_t handle{};
  Inputs<DataT> inputs{handle, p};

  // Calculate reference and test output
  auto out_reference = reference(handle, inputs, p);
  auto out_fast      = run_masked_nn(handle, inputs, p);

  // Check for differences.
  ASSERT_TRUE(devArrMatch(out_reference.data_handle(),
                          out_fast.data_handle(),
                          p.m,
                          CompareApproxAbsKVP<DataT>(p.tolerance),
                          resource::get_cuda_stream(handle)));
}

// This test checks whether running the masked_l2_nn twice returns the same
// output.
TEST_P(MaskedL2NNTest, DeterminismCheck)
{
  using DataT = float;

  // Get parameters; create handle and input data.
  Params p = GetParam();
  raft::handle_t handle{};
  Inputs<DataT> inputs{handle, p};

  // Calculate reference and test output
  auto out1 = run_masked_nn(handle, inputs, p);
  auto out2 = run_masked_nn(handle, inputs, p);

  // Check for differences.
  ASSERT_TRUE(devArrMatch(out1.data_handle(),
                          out2.data_handle(),
                          p.m,
                          CompareApproxAbsKVP<DataT>(p.tolerance),
                          resource::get_cuda_stream(handle)));
}

TEST_P(MaskedL2NNTest, ReferenceCheckDouble)
{
  using DataT = double;

  // Get parameters; create handle and input data.
  Params p = GetParam();
  raft::handle_t handle{};
  Inputs<DataT> inputs{handle, p};

  // Calculate reference and test output
  auto out_reference = reference(handle, inputs, p);
  auto out_fast      = run_masked_nn(handle, inputs, p);

  // Check for differences.
  ASSERT_TRUE(devArrMatch(out_reference.data_handle(),
                          out_fast.data_handle(),
                          p.m,
                          CompareApproxAbsKVP<DataT>(p.tolerance),
                          resource::get_cuda_stream(handle)));
}

INSTANTIATE_TEST_CASE_P(MaskedL2NNTests, MaskedL2NNTest, ::testing::ValuesIn(gen_params()));

}  // end namespace raft::distance::masked_nn
