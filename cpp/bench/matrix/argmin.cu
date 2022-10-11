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
#include <raft/matrix/math.cuh>

#include <rmm/device_uvector.hpp>

namespace raft::bench::linalg {

struct ArgminParams {
  int64_t rows, cols;
};

template <typename T, typename OutT>
struct Argmin : public fixture {
  Argmin(const ArgminParams& p)
    : params(p), matrix(p.rows * p.cols, stream), indices(p.cols, stream)
  {
  }

  void run_benchmark(::benchmark::State& state) override
  {
    loop_on_state(state, [this]() {
      raft::matrix::argmin(matrix.data(), params.rows, params.cols, indices.data(), stream);
    });
  }

 private:
  ArgminParams params;
  rmm::device_uvector<T> matrix;
  rmm::device_uvector<OutT> indices;
};  // struct Argmin

const std::vector<ArgminParams> kInputSizes{
  {64, 1000},     {128, 1000},     {256, 1000},     {512, 1000},     {1024, 1000},
  {64, 10000},    {128, 10000},    {256, 10000},    {512, 10000},    {1024, 10000},
  {64, 100000},   {128, 100000},   {256, 100000},   {512, 100000},   {1024, 100000},
  {64, 1000000},  {128, 1000000},  {256, 1000000},  {512, 1000000},  {1024, 1000000},
  {64, 10000000}, {128, 10000000}, {256, 10000000}, {512, 10000000}, {1024, 10000000},
};

RAFT_BENCH_REGISTER((Argmin<float, uint32_t>), "", kInputSizes);
RAFT_BENCH_REGISTER((Argmin<double, uint32_t>), "", kInputSizes);

}  // namespace raft::bench::linalg
