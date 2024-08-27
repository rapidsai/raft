/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/itertools.hpp>

#include <rmm/device_uvector.hpp>

namespace raft::bench::linalg {

template <typename IdxT>
struct transpose_input {
  IdxT rows, cols;
};

template <typename IdxT>
inline auto operator<<(std::ostream& os, const transpose_input<IdxT>& p) -> std::ostream&
{
  os << p.rows << "#" << p.cols;
  return os;
}

template <typename T, typename IdxT, typename Layout>
struct TransposeBench : public fixture {
  TransposeBench(const transpose_input<IdxT>& p)
    : params(p), in(p.rows * p.cols, stream), out(p.rows * p.cols, stream)
  {
    raft::random::RngState rng{1234};
    raft::random::uniform(handle, rng, in.data(), p.rows * p.cols, (T)-10.0, (T)10.0);
  }

  void run_benchmark(::benchmark::State& state) override
  {
    std::ostringstream label_stream;
    label_stream << params;
    state.SetLabel(label_stream.str());

    loop_on_state(state, [this]() {
      auto input_view =
        raft::make_device_matrix_view<T, IdxT, Layout>(in.data(), params.rows, params.cols);
      auto output_view = raft::make_device_vector_view<T, IdxT, Layout>(out.data(), params.rows);
      raft::linalg::transpose(handle,
                              input_view.data_handle(),
                              output_view.data_handle(),
                              params.rows,
                              params.cols,
                              handle.get_stream());
    });
  }

 private:
  transpose_input<IdxT> params;
  rmm::device_uvector<T> in, out;
};  // struct TransposeBench

const std::vector<transpose_input<int>> transpose_inputs_i32 =
  raft::util::itertools::product<transpose_input<int>>({10, 128, 256, 512, 1024},
                                                       {10000, 100000, 1000000});

RAFT_BENCH_REGISTER((TransposeBench<float, int, raft::row_major>), "", transpose_inputs_i32);
RAFT_BENCH_REGISTER((TransposeBench<half, int, raft::row_major>), "", transpose_inputs_i32);

RAFT_BENCH_REGISTER((TransposeBench<float, int, raft::col_major>), "", transpose_inputs_i32);
RAFT_BENCH_REGISTER((TransposeBench<half, int, raft::col_major>), "", transpose_inputs_i32);

}  // namespace raft::bench::linalg
