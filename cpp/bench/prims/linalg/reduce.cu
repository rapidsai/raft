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
#include <raft/linalg/reduce.cuh>

#include <rmm/device_uvector.hpp>

namespace raft::bench::linalg {

struct input_size {
  int rows, cols;
  bool along_rows;
};

template <typename T>
struct reduce : public fixture {
  reduce(bool along_rows, const input_size& p)
    : input_size(p), along_rows(along_rows), in(p.rows * p.cols, stream), out(p.rows, stream)
  {
  }

  void run_benchmark(::benchmark::State& state) override
  {
    loop_on_state(state, [this]() {
      raft::linalg::reduce(
        out.data(), in.data(), input_size.cols, input_size.rows, T(0.f), true, along_rows, stream);
    });
  }

 private:
  bool along_rows;
  input_size input_size;
  rmm::device_uvector<T> in, out;
};  // struct reduce

const std::vector<input_size> kInputSizes{{8 * 1024, 1024},
                                          {1024, 8 * 1024},
                                          {8 * 1024, 8 * 1024},
                                          {32 * 1024, 1024},
                                          {1024, 32 * 1024},
                                          {32 * 1024, 32 * 1024}};

const std::vector<bool> kAlongRows{false, true};

RAFT_BENCH_REGISTER(reduce<float>, "", kAlongRows, kInputSizes);
RAFT_BENCH_REGISTER(reduce<double>, "", kAlongRows, kInputSizes);

}  // namespace raft::bench::linalg
