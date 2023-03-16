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
      rng, matrix.data_handle(), params.rows * params.cols, T(-1), T(1), stream);
    handle.sync_stream(stream);
  }

  void run_benchmark(::benchmark::State& state) override
  {
    loop_on_state(state, [this]() {
      raft::matrix::argmin(handle, raft::make_const_mdspan(matrix.view()), indices.view());
    });
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
