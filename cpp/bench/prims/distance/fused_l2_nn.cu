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

#include <common/benchmark.hpp>
#include <raft/distance/fused_l2_nn.cuh>
#include <raft/util/cudart_utils.hpp>
#if defined RAFT_COMPILED
#include <raft/distance/specializations.cuh>
#endif
#include <rmm/device_uvector.hpp>

namespace raft::bench::distance {

struct fusedl2nn_inputs {
  int64_t m, n, k;
};  // struct fusedl2nn_inputs

inline auto operator<<(std::ostream& os, const fusedl2nn_inputs& p) -> std::ostream&
{
  os << p.m << "#" << p.n << "#" << p.k;
  return os;
}

template <typename DataT, typename IdxT, typename OutT>
struct fusedl2nn : public fixture {
  fusedl2nn(const fusedl2nn_inputs& p)
    : params(p),
      workspace(this->handle),
      x(this->handle),
      y(this->handle),
      x_norm(this->handle),
      y_norm(this->handle),
      out(this->handle)
  {
  }

  void allocate_data(const ::benchmark::State& state) override
  {
    x      = raft::make_device_matrix<DataT, IdxT>(handle, params.m, params.k);
    y      = raft::make_device_matrix<DataT, IdxT>(handle, params.n, params.k);
    x_norm = raft::make_device_vector<DataT, IdxT>(handle, params.m);
    y_norm = raft::make_device_vector<DataT, IdxT>(handle, params.n);
    out    = raft::make_device_vector<OutT, IdxT>(handle, params.m);

    raft::random::RngState rng{1234};
    raft::random::uniform(
      handle, rng, x.data_handle(), params.m * params.k, (DataT)-1.0, (DataT)1.0);
    raft::random::uniform(
      handle, rng, y.data_handle(), params.n * params.k, (DataT)-1.0, (DataT)1.0);

    // Pre-compute norms
    raft::linalg::rowNorm(x_norm.data_handle(),
                          x.data_handle(),
                          params.k,
                          params.m,
                          raft::linalg::L2Norm,
                          true,
                          stream);
    raft::linalg::rowNorm(y_norm.data_handle(),
                          y.data_handle(),
                          params.k,
                          params.n,
                          raft::linalg::L2Norm,
                          true,
                          stream);
    handle.sync_stream(stream);
  }

  void allocate_temp_buffers(const ::benchmark::State& state) override
  {
    workspace = raft::make_device_vector<char, IdxT>(handle, params.m * sizeof(IdxT));
  }

  void run_benchmark(::benchmark::State& state) override
  {
    std::ostringstream label_stream;
    label_stream << params;
    state.SetLabel(label_stream.str());

    loop_on_state(state, [this]() {
      raft::distance::fusedL2NNMinReduce<DataT, OutT, IdxT>(out.data_handle(),
                                                            x.data_handle(),
                                                            y.data_handle(),
                                                            x_norm.data_handle(),
                                                            y_norm.data_handle(),
                                                            static_cast<IdxT>(params.m),
                                                            static_cast<IdxT>(params.n),
                                                            static_cast<IdxT>(params.k),
                                                            (void*)workspace.data_handle(),
                                                            false,
                                                            true,
                                                            stream);
    });

    int64_t num_flops = 2 * params.m * params.n * params.k;

    int64_t read_elts  = params.n * params.k + params.m * params.k;
    int64_t write_elts = params.m;

    state.counters["FLOP/s"] = benchmark::Counter(
      num_flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1000);

    state.counters["BW Wr"] = benchmark::Counter(write_elts * sizeof(OutT),
                                                 benchmark::Counter::kIsIterationInvariantRate,
                                                 benchmark::Counter::OneK::kIs1000);
    state.counters["BW Rd"] = benchmark::Counter(read_elts * sizeof(DataT),
                                                 benchmark::Counter::kIsIterationInvariantRate,
                                                 benchmark::Counter::OneK::kIs1000);
  }

 private:
  fusedl2nn_inputs params;
  raft::device_matrix<DataT, IdxT> x, y;
  raft::device_vector<DataT, IdxT> x_norm, y_norm;
  raft::device_vector<OutT, IdxT> out;
  raft::device_vector<char, IdxT> workspace;
};  // struct fusedl2nn

template <typename IdxT>
std::vector<fusedl2nn_inputs> getFusedL2NNInputs()
{
  std::vector<fusedl2nn_inputs> inputs;
  std::vector<int64_t> m_list = {100000, 1000000};
  if constexpr (sizeof(IdxT) == 8) { m_list.push_back(10000000); }
  std::vector<int64_t> n_list = {100, 1000, 10000};
  std::vector<int64_t> k_list = {64, 128, 256};
  for (auto m : m_list) {
    for (auto n : n_list) {
      for (auto k : k_list) {
        inputs.push_back({m, n, k});
      }
    }
  }
  return inputs;
}

#define FUSEDL2NN_BENCH(DataT, IdxT, OutT) \
  RAFT_BENCH_REGISTER((fusedl2nn<DataT, IdxT, RAFT_DEPAREN(OutT)>), "", getFusedL2NNInputs<IdxT>())

FUSEDL2NN_BENCH(float, int, float);
FUSEDL2NN_BENCH(double, int, double);
FUSEDL2NN_BENCH(float, int, (raft::KeyValuePair<int, float>));
FUSEDL2NN_BENCH(double, int, (raft::KeyValuePair<int, double>));
FUSEDL2NN_BENCH(float, int64_t, float);
FUSEDL2NN_BENCH(double, int64_t, double);
FUSEDL2NN_BENCH(float, int64_t, (raft::KeyValuePair<int64_t, float>));
FUSEDL2NN_BENCH(double, int64_t, (raft::KeyValuePair<int64_t, double>));

}  // namespace raft::bench::distance
