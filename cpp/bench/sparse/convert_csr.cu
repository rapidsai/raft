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

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <stdio.h>

#include <common/benchmark.hpp>
#include <raft/sparse/convert/csr.cuh>
#include <rmm/device_uvector.hpp>

namespace raft::bench::sparse {

struct bench_param {
  size_t num_cols;
  size_t num_rows;
  size_t divisor;
};

template <typename index_t>
__global__ void init_adj(bool* adj, index_t num_rows, index_t num_cols, int divisor)
{
  index_t r = blockDim.y * blockIdx.y + threadIdx.y;
  index_t c = blockDim.x * blockIdx.x + threadIdx.x;

  if (r < num_rows && c < num_cols) { adj[r * num_cols + c] = c % divisor == 0; }
}

template <typename index_t>
struct bench_base : public fixture {
  bench_base(const bench_param& p)
    : params(p),
      handle(stream),
      adj(p.num_rows * p.num_cols, stream),
      row_ind(p.num_rows, stream),
      row_ind_host(p.num_rows),
      row_counters(p.num_rows, stream),
      col_ind(p.num_rows * p.num_cols,
              stream)  // This is over-dimensioned because nnz is unknown at this point
  {
    dim3 block(32, 32);
    dim3 grid(p.num_cols / 32 + 1, p.num_rows / 32 + 1);
    init_adj<index_t><<<grid, block, 0, stream>>>(adj.data(), p.num_rows, p.num_cols, p.divisor);

    std::vector<index_t> row_ind_host(p.num_rows);
    for (size_t i = 0; i < row_ind_host.size(); ++i) {
      size_t nnz_per_row = raft::ceildiv(p.num_cols, p.divisor);
      row_ind_host[i]    = nnz_per_row * i;
    }
    raft::update_device(row_ind.data(), row_ind_host.data(), row_ind.size(), stream);
  }

  void run_benchmark(::benchmark::State& state) override
  {
    loop_on_state(state, [this]() {
      raft::sparse::convert::dense_bool_to_unsorted_csr<index_t>(handle,
                                                                 adj.data(),
                                                                 row_ind.data(),
                                                                 params.num_rows,
                                                                 params.num_cols,
                                                                 row_counters.data(),
                                                                 col_ind.data());
    });

    // Estimate bandwidth:
    index_t num_entries = params.num_rows * params.num_cols;
    index_t bytes_read  = num_entries * sizeof(bool);
    index_t bytes_write = num_entries / params.divisor * sizeof(index_t);

    state.counters["BW"]      = benchmark::Counter(bytes_read + bytes_write,
                                              benchmark::Counter::kIsIterationInvariantRate,
                                              benchmark::Counter::OneK::kIs1024);
    state.counters["BW read"] = benchmark::Counter(
      bytes_read, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
    state.counters["BW write"] = benchmark::Counter(bytes_write,
                                                    benchmark::Counter::kIsIterationInvariantRate,
                                                    benchmark::Counter::OneK::kIs1024);

    state.counters["Fraction nz"] = benchmark::Counter(100.0 / ((double)params.divisor));
    state.counters["Columns"]     = benchmark::Counter(params.num_cols);
    state.counters["Rows"]        = benchmark::Counter(params.num_rows);
  }

 protected:
  raft::handle_t handle;
  bench_param params;
  rmm::device_uvector<bool> adj;
  rmm::device_uvector<index_t> row_ind;
  std::vector<index_t> row_ind_host;
  rmm::device_uvector<index_t> row_counters;
  rmm::device_uvector<index_t> col_ind;
};  // struct bench_base

const int64_t num_cols = 1 << 30;

const std::vector<bench_param> bench_params = {
  {num_cols, 1, 8},
  {num_cols >> 3, 1 << 3, 8},
  {num_cols >> 6, 1 << 6, 8},

  {num_cols, 1, 64},
  {num_cols >> 3, 1 << 3, 64},
  {num_cols >> 6, 1 << 6, 64},

  {num_cols, 1, 2048},
  {num_cols >> 3, 1 << 3, 2048},
  {num_cols >> 6, 1 << 6, 2048},
};

RAFT_BENCH_REGISTER(bench_base<int64_t>, "", bench_params);
// RAFT_BENCH_REGISTER(bench_base<int>, "", bench_params);

}  // namespace raft::bench::sparse
