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
#include <raft/linalg/map_then_reduce.cuh>
#include <rmm/device_uvector.hpp>

namespace raft::bench::linalg {

struct map_then_reduce_inputs {
  int len;
};

template <typename Type>
struct Identity {
  HDI Type operator()(Type a) { return a; }
};

template <typename T>
struct map_then_reduce : public fixture {
  map_then_reduce(const map_then_reduce_inputs& p) : params(p), in(p.len, stream), out(1, stream) {}

  void run_benchmark(::benchmark::State& state) override
  {
    loop_on_state(state, [this]() {
      raft::linalg::mapThenSumReduce(out.data(), params.len, Identity<T>(), stream, in.data());
    });
  }

 private:
  map_then_reduce_inputs params;
  rmm::device_uvector<T> out, in;
};  // struct MapThenReduce

const std::vector<map_then_reduce_inputs> map_then_reduce_input_vecs{
  {1024 * 1024},
  {32 * 1024 * 1024},
  {1024 * 1024 * 1024},
  {1024 * 1024 + 2},
  {32 * 1024 * 1024 + 2},
  {1024 * 1024 * 1024 + 2},
  {1024 * 1024 + 1},
  {32 * 1024 * 1024 + 1},
  {1024 * 1024 * 1024 + 1},

};

RAFT_BENCH_REGISTER(map_then_reduce<float>, "", map_then_reduce_input_vecs);
RAFT_BENCH_REGISTER(map_then_reduce<double>, "", map_then_reduce_input_vecs);

}  // namespace raft::bench::linalg
