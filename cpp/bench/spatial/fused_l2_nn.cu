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
#include <raft/distance/fused_l2_nn.cuh>
#include <raft/handle.hpp>
#include <raft/linalg/norm.cuh>
#include <raft/random/rng.cuh>

// TODO: Once fusedL2NN is specialized in the raft_distance shared library, add
// back
//
// #if defined RAFT_NN_COMPILED
// #include <raft/spatial/knn/specializations.hpp>
// #endif

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

    uniform(handle, r, x.data(), p.m * p.k, T(-1.0), T(1.0));
    uniform(handle, r, y.data(), p.n * p.k, T(-1.0), T(1.0));
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

    // Num distance calculations
    int64_t num_dist_calcs = (int64_t)params.n * (int64_t)params.m;

    int64_t num_flops = 3 * num_dist_calcs * params.k;

    int64_t read_elts  = (int64_t)params.n * params.k + (int64_t)params.m * params.k;
    int64_t write_elts = (int64_t)params.n;

    state.counters["D/s"] = benchmark::Counter(num_dist_calcs,
                                               benchmark::Counter::kIsIterationInvariantRate,
                                               benchmark::Counter::OneK::kIs1000);

    state.counters["FLOP/s"] = benchmark::Counter(
      num_flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);

    state.counters["BW Wr"] = benchmark::Counter(write_elts * sizeof(cub::KeyValuePair<int, T>),
                                                 benchmark::Counter::kIsIterationInvariantRate,
                                                 benchmark::Counter::OneK::kIs1000);
    state.counters["BW Rd"] = benchmark::Counter(read_elts * sizeof(float),
                                                 benchmark::Counter::kIsIterationInvariantRate,
                                                 benchmark::Counter::OneK::kIs1000);

    state.counters["K"] = benchmark::Counter(params.k);
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
  {16384, 16384, 2},   {16384, 16384, 4},    {16384, 16384, 8},     {16384, 16384, 16},
  {16384, 16384, 32},  {16384, 16384, 64},   {16384, 16384, 128},   {16384, 16384, 256},
  {16384, 16384, 512}, {16384, 16384, 1024}, {16384, 16384, 16384},
};

RAFT_BENCH_REGISTER(fused_l2_nn<float>, "", fused_l2_nn_input_vecs);
RAFT_BENCH_REGISTER(fused_l2_nn<double>, "", fused_l2_nn_input_vecs);

}  // namespace raft::bench::spatial
