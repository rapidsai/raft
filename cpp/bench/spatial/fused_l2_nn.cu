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

#include <common/benchmark.hpp>
#include <limits>
#include <raft/cudart_utils.h>
#include <raft/distance/fused_l2_nn.hpp>
#include <raft/handle.hpp>
#include <raft/linalg/norm.hpp>
#include <raft/random/rng_launch.cuh>

#if defined RAFT_NN_COMPILED
#include <raft/spatial/knn/specializations.hpp>
#endif

namespace raft::bench::spatial {

struct fused_l2_nn_inputs {
  int m, n, k;
};  // struct fused_l2_nn_inputs

template <typename T>
struct fused_l2_nn : public fixture {
  fused_l2_nn(const fused_l2_nn_inputs& p)
    : params(p),
      out(p.m, stream),
      x(p.m * p.k, stream),
      y(p.n * p.k, stream),
      xn(p.m, stream),
      yn(p.n, stream),
      workspace(p.m, stream)
  {
    raft::handle_t handle{stream};
    raft::random::RngState r(123456ULL);

    uniform(r, x.data(), p.m * p.k, T(-1.0), T(1.0), stream);
    uniform(r, y.data(), p.n * p.k, T(-1.0), T(1.0), stream);
    raft::linalg::rowNorm(xn.data(), x.data(), p.k, p.m, raft::linalg::L2Norm, true, stream);
    raft::linalg::rowNorm(yn.data(), y.data(), p.k, p.n, raft::linalg::L2Norm, true, stream);
    raft::distance::initialize<T, cub::KeyValuePair<int, T>, int>(
      handle, out.data(), p.m, std::numeric_limits<T>::max(), op);
  }

  void run_benchmark(::benchmark::State& state) override
  {
    loop_on_state(state, [this]() {
      // it is enough to only benchmark the L2-squared metric
      raft::distance::fusedL2NN<T, cub::KeyValuePair<int, T>, int>(out.data(),
                                                                   x.data(),
                                                                   y.data(),
                                                                   xn.data(),
                                                                   yn.data(),
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
  fused_l2_nn_inputs params;
  rmm::device_uvector<T> x, y, xn, yn;
  rmm::device_uvector<cub::KeyValuePair<int, T>> out;
  rmm::device_uvector<int> workspace;
  raft::distance::KVPMinReduce<int, T> pairRedOp;
  raft::distance::MinAndDistanceReduceOp<int, T> op;
};  // struct FusedL2NN

const std::vector<fused_l2_nn_inputs> fused_l2_nn_input_vecs = {
  {32, 16384, 16384},  {64, 16384, 16384},   {128, 16384, 16384},   {256, 16384, 16384},
  {512, 16384, 16384}, {1024, 16384, 16384}, {16384, 32, 16384},    {16384, 64, 16384},
  {16384, 128, 16384}, {16384, 256, 16384},  {16384, 512, 16384},   {16384, 1024, 16384},
  {16384, 16384, 32},  {16384, 16384, 64},   {16384, 16384, 128},   {16384, 16384, 256},
  {16384, 16384, 512}, {16384, 16384, 1024}, {16384, 16384, 16384},

};

RAFT_BENCH_REGISTER(fused_l2_nn<float>, "", fused_l2_nn_input_vecs);
RAFT_BENCH_REGISTER(fused_l2_nn<double>, "", fused_l2_nn_input_vecs);

}  // namespace raft::bench::spatial
