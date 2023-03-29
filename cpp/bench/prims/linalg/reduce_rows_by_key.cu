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
#include <raft/linalg/reduce_rows_by_key.cuh>
#include <raft/random/rng.cuh>

#include <rmm/device_uvector.hpp>

namespace raft::bench::linalg {

struct rrbk_params {
  int64_t rows, cols;
  int64_t keys;
};

template <typename T, typename KeyT>
struct reduce_rows_by_key : public fixture {
  reduce_rows_by_key(const rrbk_params& p)
    : params(p),
      in(p.rows * p.cols, stream),
      out(p.keys * p.cols, stream),
      keys(p.rows, stream),
      workspace(p.rows, stream)
  {
    raft::random::RngState rng{42};
    raft::random::uniformInt(rng, keys.data(), p.rows, (KeyT)0, (KeyT)p.keys, stream);
  }

  void run_benchmark(::benchmark::State& state) override
  {
    loop_on_state(state, [this]() {
      raft::linalg::reduce_rows_by_key(in.data(),
                                       params.cols,
                                       keys.data(),
                                       workspace.data(),
                                       params.rows,
                                       params.cols,
                                       params.keys,
                                       out.data(),
                                       stream,
                                       false);
    });
  }

 protected:
  rrbk_params params;
  rmm::device_uvector<T> in, out;
  rmm::device_uvector<KeyT> keys;
  rmm::device_uvector<char> workspace;
};  // struct reduce_rows_by_key

const std::vector<rrbk_params> kInputSizes{
  {10000, 128, 64},
  {100000, 128, 64},
  {1000000, 128, 64},
  {10000000, 128, 64},
  {10000, 128, 256},
  {100000, 128, 256},
  {1000000, 128, 256},
  {10000000, 128, 256},
  {10000, 128, 1024},
  {100000, 128, 1024},
  {1000000, 128, 1024},
  {10000000, 128, 1024},
  {10000, 128, 4096},
  {100000, 128, 4096},
  {1000000, 128, 4096},
  {10000000, 128, 4096},
};

RAFT_BENCH_REGISTER((reduce_rows_by_key<float, uint32_t>), "", kInputSizes);
RAFT_BENCH_REGISTER((reduce_rows_by_key<double, uint32_t>), "", kInputSizes);

}  // namespace raft::bench::linalg
