/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <raft/core/bitset.cuh>
#include <raft/core/device_mdspan.hpp>
#include <rmm/device_uvector.hpp>

namespace raft::bench::core {

struct bitset_inputs {
  uint32_t bitset_len;
  uint32_t mask_len;
  uint32_t query_len;
};  // struct bitset_inputs

template <typename bitset_t, typename index_t>
struct bitset_bench : public fixture {
  bitset_bench(const bitset_inputs& p)
    : params(p),
      mask{raft::make_device_vector<index_t, index_t>(res, p.mask_len)},
      queries{raft::make_device_vector<index_t, index_t>(res, p.query_len)},
      outputs{raft::make_device_vector<bool, index_t>(res, p.query_len)}
  {
    raft::random::RngState state{42};
    raft::random::uniformInt(res, state, mask.view(), index_t{0}, index_t{p.bitset_len});
  }

  void run_benchmark(::benchmark::State& state) override
  {
    loop_on_state(state, [this]() {
      auto my_bitset = raft::core::bitset<bitset_t, index_t>(
        this->res, raft::make_const_mdspan(mask.view()), params.bitset_len);
      my_bitset.test(this->res, raft::make_const_mdspan(queries.view()), outputs.view());
    });
  }

 private:
  raft::resources res;
  bitset_inputs params;
  raft::device_vector<index_t, index_t> mask, queries;
  raft::device_vector<bool, index_t> outputs;
};  // struct bitset

const std::vector<bitset_inputs> bitset_input_vecs{
  {256 * 1024 * 1024, 64 * 1024 * 1024, 256 * 1024 * 1024},    // Standard Bench
  {256 * 1024 * 1024, 64 * 1024 * 1024, 1024 * 1024 * 1024},   // Extra queries
  {128 * 1024 * 1024, 1024 * 1024 * 1024, 256 * 1024 * 1024},  // Extra mask to test atomics impact
};

using Uint8_32  = bitset_bench<uint8_t, uint32_t>;
using Uint16_64 = bitset_bench<uint16_t, uint32_t>;
using Uint32_32 = bitset_bench<uint32_t, uint32_t>;
using Uint32_64 = bitset_bench<uint32_t, uint64_t>;

RAFT_BENCH_REGISTER(Uint8_32, "", bitset_input_vecs);
RAFT_BENCH_REGISTER(Uint16_64, "", bitset_input_vecs);
RAFT_BENCH_REGISTER(Uint32_32, "", bitset_input_vecs);
RAFT_BENCH_REGISTER(Uint32_64, "", bitset_input_vecs);

}  // namespace raft::bench::core
