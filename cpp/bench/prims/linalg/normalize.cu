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
#include <raft/linalg/normalize.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/itertools.hpp>

#include <rmm/device_uvector.hpp>

namespace raft::bench::linalg {

template <typename IdxT>
struct normalize_input {
  IdxT rows, cols;
};

template <typename IdxT>
inline auto operator<<(std::ostream& os, const normalize_input<IdxT>& p) -> std::ostream&
{
  os << p.rows << "#" << p.cols;
  return os;
}

template <typename T, typename IdxT>
struct rowNormalize : public fixture {
  rowNormalize(const normalize_input<IdxT>& p)
    : params(p), in(p.rows * p.cols, stream), out(p.rows * p.cols, stream)
  {
    raft::random::RngState rng{1234};
    raft::random::uniform(rng, in.data(), p.rows * p.cols, (T)-10.0, (T)10.0, stream);
  }

  void run_benchmark(::benchmark::State& state) override
  {
    std::ostringstream label_stream;
    label_stream << params;
    state.SetLabel(label_stream.str());

    loop_on_state(state, [this]() {
      auto input_view = raft::make_device_matrix_view<const T, IdxT, raft::row_major>(
        in.data(), params.rows, params.cols);
      auto output_view = raft::make_device_matrix_view<T, IdxT, raft::row_major>(
        out.data(), params.rows, params.cols);
      raft::linalg::row_normalize(handle, input_view, output_view, raft::linalg::L2Norm);
    });
  }

 private:
  normalize_input<IdxT> params;
  rmm::device_uvector<T> in, out;
};  // struct rowNormalize

const std::vector<normalize_input<int>> normalize_inputs_i32 =
  raft::util::itertools::product<normalize_input<int>>(
    {10, 100, 1000, 10000, 100000}, {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384});
const std::vector<normalize_input<int64_t>> normalize_inputs_i64 =
  raft::util::itertools::product<normalize_input<int64_t>>(
    {10, 100, 1000, 10000, 100000}, {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384});

RAFT_BENCH_REGISTER((rowNormalize<float, int>), "", normalize_inputs_i32);
RAFT_BENCH_REGISTER((rowNormalize<double, int>), "", normalize_inputs_i32);
RAFT_BENCH_REGISTER((rowNormalize<float, int64_t>), "", normalize_inputs_i64);
RAFT_BENCH_REGISTER((rowNormalize<double, int64_t>), "", normalize_inputs_i64);

}  // namespace raft::bench::linalg
