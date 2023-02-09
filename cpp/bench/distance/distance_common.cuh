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
#include <raft/distance/distance.cuh>
#include <raft/util/cudart_utils.hpp>
#if defined RAFT_DISTANCE_COMPILED
#include <raft/distance/specializations.cuh>
#endif
#include <rmm/device_uvector.hpp>

namespace raft::bench::distance {

struct distance_params {
  int m, n, k;
  bool isRowMajor;
};  // struct distance_params

template <typename T, raft::distance::DistanceType DType>
struct distance : public fixture {
  distance(const distance_params& p)
    : params(p),
      x(p.m * p.k, stream),
      y(p.n * p.k, stream),
      out(p.m * p.n, stream),
      workspace(0, stream)
  {
    RAFT_CUDA_TRY(cudaMemsetAsync(x.data(), 0, x.size() * sizeof(T), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(y.data(), 0, y.size() * sizeof(T), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(out.data(), 0, out.size() * sizeof(T), stream));
    worksize = raft::distance::getWorkspaceSize<DType, T, T, T>(
      x.data(), y.data(), params.m, params.n, params.k);
    workspace.resize(worksize, stream);
  }

  void run_benchmark(::benchmark::State& state) override
  {
    loop_on_state(state, [this]() {
      raft::distance::distance<DType, T, T, T>(x.data(),
                                               y.data(),
                                               out.data(),
                                               params.m,
                                               params.n,
                                               params.k,
                                               (void*)workspace.data(),
                                               worksize,
                                               stream,
                                               params.isRowMajor);
    });
  }

 private:
  distance_params params;
  rmm::device_uvector<T> x, y, out;
  rmm::device_uvector<char> workspace;
  size_t worksize;
};  // struct Distance

const std::vector<distance_params> dist_input_vecs{
  {32, 16384, 16384, true},    {64, 16384, 16384, true},    {128, 16384, 16384, true},
  {256, 16384, 16384, true},   {512, 16384, 16384, true},   {1024, 16384, 16384, true},
  {16384, 32, 16384, true},    {16384, 64, 16384, true},    {16384, 128, 16384, true},
  {16384, 256, 16384, true},   {16384, 512, 16384, true},   {16384, 1024, 16384, true},
  {16384, 16384, 32, true},    {16384, 16384, 64, true},    {16384, 16384, 128, true},
  {16384, 16384, 256, true},   {16384, 16384, 512, true},   {16384, 16384, 1024, true},
  {16384, 16384, 16384, true}, {32, 16384, 16384, false},   {64, 16384, 16384, false},
  {128, 16384, 16384, false},  {256, 16384, 16384, false},  {512, 16384, 16384, false},
  {1024, 16384, 16384, false}, {16384, 32, 16384, false},   {16384, 64, 16384, false},
  {16384, 128, 16384, false},  {16384, 256, 16384, false},  {16384, 512, 16384, false},
  {16384, 1024, 16384, false}, {16384, 16384, 32, false},   {16384, 16384, 64, false},
  {16384, 16384, 128, false},  {16384, 16384, 256, false},  {16384, 16384, 512, false},
  {16384, 16384, 1024, false}, {16384, 16384, 16384, false}

};

#define DIST_BENCH_REGISTER(Name, Metric)            \
  using Name##F = distance<float, Metric>;           \
  RAFT_BENCH_REGISTER(Name##F, "", dist_input_vecs); \
  using Name##D = distance<double, Metric>;          \
  RAFT_BENCH_REGISTER(Name##D, "", dist_input_vecs);

}  // namespace raft::bench::distance
