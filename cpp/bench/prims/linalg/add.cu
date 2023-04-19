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
#include <raft/linalg/add.cuh>
#include <rmm/device_uvector.hpp>

namespace raft::bench::linalg {

struct add_inputs {
  int len;
};  // struct add_inputs

template <typename T>
struct add : public fixture {
  add(const add_inputs& p) : params(p), ptr0(p.len, stream), ptr1(p.len, stream) {}

  void run_benchmark(::benchmark::State& state) override
  {
    loop_on_state(state, [this]() {
      raft::linalg::add(ptr0.data(), ptr0.data(), ptr1.data(), params.len, stream);
    });
  }

 private:
  add_inputs params;
  rmm::device_uvector<T> ptr0, ptr1;
};  // struct add

const std::vector<add_inputs> add_input_vecs{
  {256 * 1024 * 1024}, {256 * 1024 * 1024 + 2}, {256 * 1024 * 1024 + 1}

};

RAFT_BENCH_REGISTER(add<float>, "", add_input_vecs);
RAFT_BENCH_REGISTER(add<double>, "", add_input_vecs);

}  // namespace raft::bench::linalg
