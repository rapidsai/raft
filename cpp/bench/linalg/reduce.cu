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
#include <raft/linalg/reduce.hpp>

#include <rmm/device_uvector.hpp>

namespace raft::bench::linalg {

struct params {
  int rows, cols;
  bool alongRows;
};

template <typename T>
struct reduce : public fixture {
  reduce(const params& p)
    : params(p), in(params.rows * params.cols, stream), out(params.rows, stream)
  {
  }

  void run_benchmark(::benchmark::State& state) override
  {
    loop_on_state(state, [this]() {
      raft::linalg::reduce(
        out.data(), in.data(), params.cols, params.rows, T(0.f), true, params.alongRows, stream);
    });
  }

 private:
  params params;
  rmm::device_uvector<T> in, out;
};  // struct reduce

const std::vector<params> inputs{
  {8 * 1024, 1024, false},
  {1024, 8 * 1024, false},
  {8 * 1024, 8 * 1024, false},
  {32 * 1024, 1024, false},
  {1024, 32 * 1024, false},
  {32 * 1024, 32 * 1024, false},

  {8 * 1024, 1024, true},
  {1024, 8 * 1024, true},
  {8 * 1024, 8 * 1024, true},
  {32 * 1024, 1024, true},
  {1024, 32 * 1024, true},
  {32 * 1024, 32 * 1024, true},
};

RAFT_BENCH_REGISTER(reduce<float>, "", inputs);
RAFT_BENCH_REGISTER(reduce<double>, "", inputs);

}  // namespace raft::bench::linalg
