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

// Tuning benchmarks.
//
// Goals:
//
// 1. Fast compile times to maintain iteration speed.
// 2. Create benchmarks that can inform the design of the kernels.
//
// Non-goals:
//
// 1. Measure every distance operation. Instead measures just one distance
//    operation at the same time.
// 2. Be useful for finding performance regressions. This is handled by the
//    normal benchmarks.
//
// So far, both goals are partly achieved.
//
// RE (1), COMPILE TIMES: kernel.cu is fast to compile. This file is not.
// When the internals of a pairwise distance kernel is changed, this file is not
// recompiled.
//
// RE 2, benchmarks with intent: this file contains a benchmark to check the
// maximal throughput of a kernel. Measuring other things, like performance on
// skinny or wide matrices is not yet implemented.

#include "kernel.cuh"                                       // launch_kernel
#include <algorithm>                                        // std::min
#include <common/benchmark.hpp>                             // RAFT_BENCH_REGISTER
#include <raft/distance/detail/pairwise_matrix/params.cuh>  // pairwise_matrix_params
#include <rmm/device_uvector.hpp>                           // rmm::device_uvector
#include <vector>                                           // std::vector

namespace raft::bench::distance::tune {

// Max throughput benchmark.
//
// Goal: Measure the maximum distances/sec that can be computed.
//
// To achieve this, we make sure that:
//
// - Input data size is a multiple of the block tile size.
//
// - Perfect distribution of work between SMs, i.e. the number of block tiles is
//   a large multiple (num_waves) of the number of blocks (#SMs * occupancy).
//
// - Multiple iterations over Kblk are executed (num_k_iters).
struct throughput_param {
  int num_waves;
  int occupancy;
  int num_k_iters;
};

const std::vector<throughput_param> throughput_params{
  // 32 waves, requested occupancy of 4, and 32 k iterations typically achieves
  // maximum throughput. No need to pick higher values.
  {32, 4, 32},
};

struct throughput_bench : public fixture {
  const throughput_param p;

  throughput_bench(const throughput_param& p_) : p(p_) {}

  void run_benchmark(::benchmark::State& state) override
  {
    // Get block size:
    int block_m, block_n, block_k;
    get_block_size(block_m, block_n, block_k);

    // Determine number of blocks that will be launched. This informs the size
    // of the inputs as well as the grid size.
    const int num_sms       = raft::getMultiProcessorCount();
    const int max_occupancy = get_max_occupancy();
    const int occupancy     = std::min(p.occupancy, max_occupancy);
    const int num_blocks    = occupancy * num_sms;
    dim3 grid(num_blocks);

    // Create input sizes that are a multiple of the block tile size.
    size_t m = block_m;
    size_t n = block_n * p.num_waves * num_blocks;
    size_t k = block_k * p.num_k_iters;

    // DataT, OutT, IdxT, etc, are defined in tuned_kernel.cuh
    rmm::device_uvector<DataT> x_vec(m * k, stream);
    rmm::device_uvector<DataT> y_vec(n * k, stream);
    rmm::device_uvector<DataT> x_norm_vec(m, stream);
    rmm::device_uvector<DataT> y_norm_vec(n, stream);
    rmm::device_uvector<OutT> out_vec(m * n, stream);

    auto x      = x_vec.data();
    auto y      = y_vec.data();
    auto x_norm = x_norm_vec.data();
    auto y_norm = y_norm_vec.data();
    auto out    = out_vec.data();
    FinOpT fin_op{};

    // Create kernel parameter struct. Flip x and y if column major.
    IdxT ldx    = row_major ? k : m;
    IdxT ldy    = row_major ? k : n;
    IdxT ld_out = row_major ? n : m;

    // Template parameters of pairwise_matrix_params are defined in kernel.cuh
    pairwise_matrix_params kparams{
      IdxT(m), IdxT(n), IdxT(k), ldx, ldy, ld_out, x, y, x_norm, y_norm, out, fin_op, row_major};

    // Run benchmark
    loop_on_state(state, [&]() { launch_kernel(kparams, grid, stream); });

    // Report metrics. We don't report flop/s because we do not know for each
    // distance operation how many flops it costs. For L2_unexp and l1, we can
    // double this number to get the flop/s. For l2 expanded, core_ops/s should
    // equal flop/s (modulo the sqrt and subtracting from the norm).
    size_t num_core_ops = m * n * k;
    size_t read_elts    = n * k + m * k;
    size_t write_elts   = m * n;

    state.counters["m"]         = benchmark::Counter(m);
    state.counters["n"]         = benchmark::Counter(n);
    state.counters["k"]         = benchmark::Counter(k);
    state.counters["occupancy"] = benchmark::Counter(occupancy);
    state.counters["# waves"]   = benchmark::Counter(p.num_waves);
    state.counters["# k iters"] = benchmark::Counter(p.num_k_iters);

    state.counters["core_ops/s"] = benchmark::Counter(num_core_ops,
                                                      benchmark::Counter::kIsIterationInvariantRate,
                                                      benchmark::Counter::OneK::kIs1000);

    state.counters["BW"] = benchmark::Counter(write_elts * sizeof(OutT) + read_elts * sizeof(DataT),
                                              benchmark::Counter::kIsIterationInvariantRate,
                                              benchmark::Counter::OneK::kIs1000);
  }
};

RAFT_BENCH_REGISTER(throughput_bench, "", throughput_params);

}  // namespace raft::bench::distance::tune
