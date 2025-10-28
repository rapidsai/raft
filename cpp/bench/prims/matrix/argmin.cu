/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/argmin.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/itertools.hpp>

#include <rmm/device_uvector.hpp>

namespace raft::bench::matrix {

template <typename IdxT>
struct ArgminParams {
  IdxT rows, cols;
};

template <typename T, typename OutT, typename IdxT>
struct Argmin : public fixture {
  Argmin(const ArgminParams<IdxT>& p) : params(p), matrix(this->handle), indices(this->handle) {}

  void allocate_data(const ::benchmark::State& state) override
  {
    matrix  = raft::make_device_matrix<T, IdxT>(handle, params.rows, params.cols);
    indices = raft::make_device_vector<OutT, IdxT>(handle, params.rows);

    raft::random::RngState rng{1234};
    raft::random::uniform(
      handle, rng, matrix.data_handle(), params.rows * params.cols, T(-1), T(1));
    resource::sync_stream(handle, stream);
  }

  void run_benchmark(::benchmark::State& state) override
  {
    state.counters.insert({{"rows", params.rows}});
    state.counters.insert({{"cols", params.cols}});
    size_t bytes_processed = 0;
    loop_on_state(state, [this, &bytes_processed]() {
      raft::matrix::argmin(handle, raft::make_const_mdspan(matrix.view()), indices.view());
      bytes_processed += size_t(params.rows) * size_t(params.cols) * sizeof(T);
    });
    state.SetBytesProcessed(bytes_processed);
  }

 private:
  ArgminParams<IdxT> params;
  raft::device_matrix<T, IdxT> matrix;
  raft::device_vector<OutT, IdxT> indices;
};  // struct Argmin

const std::vector<ArgminParams<int64_t>> argmin_inputs_i64 =
  raft::util::itertools::product<ArgminParams<int64_t>>({1000, 10000, 100000, 1000000, 10000000},
                                                        {64, 128, 256, 512, 1024});

RAFT_BENCH_REGISTER((Argmin<float, uint32_t, int64_t>), "", argmin_inputs_i64);
RAFT_BENCH_REGISTER((Argmin<double, uint32_t, int64_t>), "", argmin_inputs_i64);

}  // namespace raft::bench::matrix
