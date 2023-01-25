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

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <common/benchmark.hpp>
#include <limits>
#include <raft/distance/sparse_l2_nn.cuh>
#include <raft/handle.hpp>
#include <raft/linalg/norm.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#if defined RAFT_NN_COMPILED
#include <raft/spatial/knn/specializations.hpp>
#endif

namespace raft::bench::spatial::sparse {

// Introduce various sparsity patterns
enum SparsityPattern {
  checkerboard    = 0,
  checkerboard_4  = 1,
  checkerboard_64 = 2,
  all_true        = 3,
  all_false       = 4
};

struct sparse_l2_nn_inputs {
  int m, n, k, num_groups;
  SparsityPattern pattern;
};  // struct sparse_l2_nn_inputs

__global__ void init_adj(
  int m, int n, int num_groups, SparsityPattern pattern, bool* adj, int* group_idxs)
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
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < num_groups;
         j += blockDim.x * gridDim.x) {
      group_idxs[j] = (j + 1) * (n / num_groups);
    }
    group_idxs[num_groups - 1] = n;
  }
}

template <typename T>
struct sparse_l2_nn : public fixture {
  sparse_l2_nn(const sparse_l2_nn_inputs& p)
    : params(p),
      out(p.m, stream),
      x(p.m * p.k, stream),
      y(p.n * p.k, stream),
      xn(p.m, stream),
      yn(p.n, stream),
      adj(p.m * p.num_groups, stream),
      group_idxs(p.num_groups, stream),
      workspace(p.m, stream)
  {
    raft::handle_t handle{stream};
    raft::random::RngState r(123456ULL);

    uniform(handle, r, x.data(), p.m * p.k, T(-1.0), T(1.0));
    uniform(handle, r, y.data(), p.n * p.k, T(-1.0), T(1.0));
    raft::linalg::rowNorm(xn.data(), x.data(), p.k, p.m, raft::linalg::L2Norm, true, stream);
    raft::linalg::rowNorm(yn.data(), y.data(), p.k, p.n, raft::linalg::L2Norm, true, stream);
    raft::distance::initialize<T, cub::KeyValuePair<int, T>, int>(
      handle, out.data(), p.m, std::numeric_limits<T>::max(), op);

    dim3 block(32, 32);
    dim3 grid(10, 10);
    init_adj<<<grid, block, 0, stream>>>(
      p.m, p.n, p.num_groups, p.pattern, adj.data(), group_idxs.data());
    RAFT_CUDA_TRY(cudaGetLastError());
  }

  void run_benchmark(::benchmark::State& state) override
  {
    loop_on_state(state, [this]() {
      // It is sufficient to only benchmark the L2-squared metric
      raft::distance::sparseL2NN<T, cub::KeyValuePair<int, T>, int>(out.data(),
                                                                    x.data(),
                                                                    y.data(),
                                                                    xn.data(),
                                                                    yn.data(),
                                                                    adj.data(),
                                                                    group_idxs.data(),
                                                                    params.num_groups,
                                                                    params.m,
                                                                    params.n,
                                                                    params.k,
                                                                    (void*)workspace.data(),
                                                                    op,
                                                                    pairRedOp,
                                                                    false,
                                                                    false,
                                                                    stream);
    });
  }

 private:
  sparse_l2_nn_inputs params;
  rmm::device_uvector<T> x, y, xn, yn;
  rmm::device_uvector<bool> adj;
  rmm::device_uvector<int> group_idxs;
  rmm::device_uvector<cub::KeyValuePair<int, T>> out;
  rmm::device_uvector<int> workspace;
  raft::distance::KVPMinReduce<int, T> pairRedOp;
  raft::distance::MinAndDistanceReduceOp<int, T> op;
};  // struct SparseL2NN

// TODO: Consider thinning the list of benchmark cases..
const std::vector<sparse_l2_nn_inputs> sparse_l2_nn_input_vecs = {
  // Very fat matrices...
  {32, 16384, 16384, 32, SparsityPattern::checkerboard},
  {64, 16384, 16384, 32, SparsityPattern::checkerboard},
  {128, 16384, 16384, 32, SparsityPattern::checkerboard},
  {256, 16384, 16384, 32, SparsityPattern::checkerboard},
  {512, 16384, 16384, 32, SparsityPattern::checkerboard},
  {1024, 16384, 16384, 32, SparsityPattern::checkerboard},
  {16384, 32, 16384, 32, SparsityPattern::checkerboard},
  {16384, 64, 16384, 32, SparsityPattern::checkerboard},
  {16384, 128, 16384, 32, SparsityPattern::checkerboard},
  {16384, 256, 16384, 32, SparsityPattern::checkerboard},
  {16384, 512, 16384, 32, SparsityPattern::checkerboard},
  {16384, 1024, 16384, 32, SparsityPattern::checkerboard},

  // Representative matrices...
  {16384, 16384, 32, 32, SparsityPattern::checkerboard},
  {16384, 16384, 64, 32, SparsityPattern::checkerboard},
  {16384, 16384, 128, 32, SparsityPattern::checkerboard},
  {16384, 16384, 256, 32, SparsityPattern::checkerboard},
  {16384, 16384, 512, 32, SparsityPattern::checkerboard},
  {16384, 16384, 1024, 32, SparsityPattern::checkerboard},
  {16384, 16384, 16384, 32, SparsityPattern::checkerboard},

  {16384, 16384, 32, 32, SparsityPattern::checkerboard_4},
  {16384, 16384, 64, 32, SparsityPattern::checkerboard_4},
  {16384, 16384, 128, 32, SparsityPattern::checkerboard_4},
  {16384, 16384, 256, 32, SparsityPattern::checkerboard_4},
  {16384, 16384, 512, 32, SparsityPattern::checkerboard_4},
  {16384, 16384, 1024, 32, SparsityPattern::checkerboard_4},
  {16384, 16384, 16384, 32, SparsityPattern::checkerboard_4},

  {16384, 16384, 32, 32, SparsityPattern::checkerboard_64},
  {16384, 16384, 64, 32, SparsityPattern::checkerboard_64},
  {16384, 16384, 128, 32, SparsityPattern::checkerboard_64},
  {16384, 16384, 256, 32, SparsityPattern::checkerboard_64},
  {16384, 16384, 512, 32, SparsityPattern::checkerboard_64},
  {16384, 16384, 1024, 32, SparsityPattern::checkerboard_64},
  {16384, 16384, 16384, 32, SparsityPattern::checkerboard_64},
};

RAFT_BENCH_REGISTER(sparse_l2_nn<float>, "", sparse_l2_nn_input_vecs);
// Do not benchmark double.

}  // namespace raft::bench::spatial::sparse
