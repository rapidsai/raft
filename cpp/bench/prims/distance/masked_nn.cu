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

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <common/benchmark.hpp>
#include <limits>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/distance/masked_nn.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

namespace raft::bench::distance::masked_nn {

// Introduce various sparsity patterns
enum AdjacencyPattern {
  checkerboard    = 0,
  checkerboard_4  = 1,
  checkerboard_64 = 2,
  all_true        = 3,
  all_false       = 4
};

struct Params {
  int m, n, k, num_groups;
  AdjacencyPattern pattern;
};  // struct Params

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

template <typename T>
struct masked_l2_nn : public fixture {
  using DataT      = T;
  using IdxT       = int;
  using OutT       = raft::KeyValuePair<IdxT, DataT>;
  using RedOpT     = raft::distance::MinAndDistanceReduceOp<int, DataT>;
  using PairRedOpT = raft::distance::KVPMinReduce<int, DataT>;
  using ParamT     = raft::distance::masked_l2_nn_params<RedOpT, PairRedOpT>;

  // Parameters
  Params params;
  // Data
  raft::device_vector<OutT, IdxT> out;
  raft::device_matrix<T, IdxT> x, y;
  raft::device_vector<DataT, IdxT> xn, yn;
  raft::device_matrix<bool, IdxT> adj;
  raft::device_vector<IdxT, IdxT> group_idxs;

  masked_l2_nn(const Params& p)
    : params(p),
      out{raft::make_device_vector<OutT, IdxT>(handle, p.m)},
      x{raft::make_device_matrix<DataT, IdxT>(handle, p.m, p.k)},
      y{raft::make_device_matrix<DataT, IdxT>(handle, p.n, p.k)},
      xn{raft::make_device_vector<DataT, IdxT>(handle, p.m)},
      yn{raft::make_device_vector<DataT, IdxT>(handle, p.n)},
      adj{raft::make_device_matrix<bool, IdxT>(handle, p.m, p.num_groups)},
      group_idxs{raft::make_device_vector<IdxT, IdxT>(handle, p.num_groups)}
  {
    raft::random::RngState r(123456ULL);

    uniform(handle, r, x.data_handle(), p.m * p.k, T(-1.0), T(1.0));
    uniform(handle, r, y.data_handle(), p.n * p.k, T(-1.0), T(1.0));
    raft::linalg::rowNorm(
      xn.data_handle(), x.data_handle(), p.k, p.m, raft::linalg::L2Norm, true, stream);
    raft::linalg::rowNorm(
      yn.data_handle(), y.data_handle(), p.k, p.n, raft::linalg::L2Norm, true, stream);
    raft::distance::initialize<T, raft::KeyValuePair<int, T>, int>(
      handle, out.data_handle(), p.m, std::numeric_limits<T>::max(), RedOpT{});

    dim3 block(32, 32);
    dim3 grid(10, 10);
    init_adj<<<grid, block, 0, stream>>>(p.pattern, p.n, adj.view(), group_idxs.view());
    RAFT_CUDA_TRY(cudaGetLastError());
  }

  void run_benchmark(::benchmark::State& state) override
  {
    bool init_out = true;
    bool sqrt     = false;
    ParamT masked_l2_params{RedOpT{}, PairRedOpT{}, sqrt, init_out};

    loop_on_state(state, [this, masked_l2_params]() {
      // It is sufficient to only benchmark the L2-squared metric
      raft::distance::masked_l2_nn<DataT, OutT, IdxT>(handle,
                                                      masked_l2_params,
                                                      x.view(),
                                                      y.view(),
                                                      xn.view(),
                                                      yn.view(),
                                                      adj.view(),
                                                      group_idxs.view(),
                                                      out.view());
    });

    // Virtual flop count if no skipping had occurred.
    size_t virtual_flops = size_t(2) * size_t(params.m) * size_t(params.n) * size_t(params.k);

    int64_t read_elts  = params.n * params.k + params.m * params.k;
    int64_t write_elts = params.m;

    // Virtual min flops is the number of flops that would have been executed if
    // the algorithm had actually skipped each computation that it could have
    // skipped.
    size_t virtual_min_flops = 0;
    switch (params.pattern) {
      case checkerboard:
      case checkerboard_4:
      case checkerboard_64: virtual_min_flops = virtual_flops / 2; break;
      case all_true: virtual_min_flops = virtual_flops; break;
      case all_false: virtual_min_flops = 0; break;
      default: assert(false && "unknown pattern");
    }

    // VFLOP/s is the "virtual" flop count that would have executed if there was
    // no adjacency pattern. This is useful for comparing to fusedL2NN
    state.counters["VFLOP/s"] = benchmark::Counter(virtual_flops,
                                                   benchmark::Counter::kIsIterationInvariantRate,
                                                   benchmark::Counter::OneK::kIs1000);
    // Virtual min flops is the number of flops that would have been executed if
    // the algorithm had actually skipped each computation that it could have
    // skipped.
    state.counters["VminFLOP/s"] = benchmark::Counter(virtual_min_flops,
                                                      benchmark::Counter::kIsIterationInvariantRate,
                                                      benchmark::Counter::OneK::kIs1000);

    state.counters["BW Wr"] = benchmark::Counter(write_elts * sizeof(OutT),
                                                 benchmark::Counter::kIsIterationInvariantRate,
                                                 benchmark::Counter::OneK::kIs1000);
    state.counters["BW Rd"] = benchmark::Counter(read_elts * sizeof(DataT),
                                                 benchmark::Counter::kIsIterationInvariantRate,
                                                 benchmark::Counter::OneK::kIs1000);

    state.counters["m"]          = benchmark::Counter(params.m);
    state.counters["n"]          = benchmark::Counter(params.n);
    state.counters["k"]          = benchmark::Counter(params.k);
    state.counters["num_groups"] = benchmark::Counter(params.num_groups);
    state.counters["group size"] = benchmark::Counter(params.n / params.num_groups);
    state.counters["Pat"]        = benchmark::Counter(static_cast<int>(params.pattern));

    state.counters["SM count"] = raft::getMultiProcessorCount();
  }
};

const std::vector<Params> masked_l2_nn_input_vecs = {
  // Very fat matrices...
  {32, 16384, 16384, 32, AdjacencyPattern::checkerboard},
  {64, 16384, 16384, 32, AdjacencyPattern::checkerboard},
  {128, 16384, 16384, 32, AdjacencyPattern::checkerboard},
  {256, 16384, 16384, 32, AdjacencyPattern::checkerboard},
  {512, 16384, 16384, 32, AdjacencyPattern::checkerboard},
  {1024, 16384, 16384, 32, AdjacencyPattern::checkerboard},
  {16384, 32, 16384, 32, AdjacencyPattern::checkerboard},
  {16384, 64, 16384, 32, AdjacencyPattern::checkerboard},
  {16384, 128, 16384, 32, AdjacencyPattern::checkerboard},
  {16384, 256, 16384, 32, AdjacencyPattern::checkerboard},
  {16384, 512, 16384, 32, AdjacencyPattern::checkerboard},
  {16384, 1024, 16384, 32, AdjacencyPattern::checkerboard},

  // Representative matrices...
  {16384, 16384, 32, 32, AdjacencyPattern::checkerboard},
  {16384, 16384, 64, 32, AdjacencyPattern::checkerboard},
  {16384, 16384, 128, 32, AdjacencyPattern::checkerboard},
  {16384, 16384, 256, 32, AdjacencyPattern::checkerboard},
  {16384, 16384, 512, 32, AdjacencyPattern::checkerboard},
  {16384, 16384, 1024, 32, AdjacencyPattern::checkerboard},
  {16384, 16384, 16384, 32, AdjacencyPattern::checkerboard},

  {16384, 16384, 32, 32, AdjacencyPattern::checkerboard_4},
  {16384, 16384, 64, 32, AdjacencyPattern::checkerboard_4},
  {16384, 16384, 128, 32, AdjacencyPattern::checkerboard_4},
  {16384, 16384, 256, 32, AdjacencyPattern::checkerboard_4},
  {16384, 16384, 512, 32, AdjacencyPattern::checkerboard_4},
  {16384, 16384, 1024, 32, AdjacencyPattern::checkerboard_4},
  {16384, 16384, 16384, 32, AdjacencyPattern::checkerboard_4},

  {16384, 16384, 32, 32, AdjacencyPattern::checkerboard_64},
  {16384, 16384, 64, 32, AdjacencyPattern::checkerboard_64},
  {16384, 16384, 128, 32, AdjacencyPattern::checkerboard_64},
  {16384, 16384, 256, 32, AdjacencyPattern::checkerboard_64},
  {16384, 16384, 512, 32, AdjacencyPattern::checkerboard_64},
  {16384, 16384, 1024, 32, AdjacencyPattern::checkerboard_64},
  {16384, 16384, 16384, 32, AdjacencyPattern::checkerboard_64},

  {16384, 16384, 32, 32, AdjacencyPattern::all_true},
  {16384, 16384, 64, 32, AdjacencyPattern::all_true},
  {16384, 16384, 128, 32, AdjacencyPattern::all_true},
  {16384, 16384, 256, 32, AdjacencyPattern::all_true},
  {16384, 16384, 512, 32, AdjacencyPattern::all_true},
  {16384, 16384, 1024, 32, AdjacencyPattern::all_true},
  {16384, 16384, 16384, 32, AdjacencyPattern::all_true},

  {16384, 16384, 32, 32, AdjacencyPattern::all_false},
  {16384, 16384, 64, 32, AdjacencyPattern::all_false},
  {16384, 16384, 128, 32, AdjacencyPattern::all_false},
  {16384, 16384, 256, 32, AdjacencyPattern::all_false},
  {16384, 16384, 512, 32, AdjacencyPattern::all_false},
  {16384, 16384, 1024, 32, AdjacencyPattern::all_false},
  {16384, 16384, 16384, 32, AdjacencyPattern::all_false},
};

RAFT_BENCH_REGISTER(masked_l2_nn<float>, "", masked_l2_nn_input_vecs);
// We don't benchmark double to keep compile times in check when not using the
// distance library.

}  // namespace raft::bench::distance::masked_nn
