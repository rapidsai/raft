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
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

namespace raft::bench::random {

enum RandomType {
  RNG_Normal,
  RNG_LogNormal,
  RNG_Uniform,
  RNG_Gumbel,
  RNG_Logistic,
  RNG_Exp,
  RNG_Rayleigh,
  RNG_Laplace,
  RNG_Fill
};  // enum RandomType

template <typename T>
struct rng_inputs {
  int len;
  RandomType type;
  raft::random::GeneratorType gtype;
  T start, end;
};  // struct rng_inputs

template <typename T>
struct rng : public fixture {
  rng(const rng_inputs<T>& p) : params(p), ptr(p.len, stream) {}

  void run_benchmark(::benchmark::State& state) override
  {
    raft::random::RngState r(123456ULL, params.gtype);
    loop_on_state(state, [this, &r]() {
      switch (params.type) {
        case RNG_Normal: normal(handle, r, ptr.data(), params.len, params.start, params.end); break;
        case RNG_LogNormal:
          lognormal(handle, r, ptr.data(), params.len, params.start, params.end);
          break;
        case RNG_Uniform:
          uniform(handle, r, ptr.data(), params.len, params.start, params.end);
          break;
        case RNG_Gumbel: gumbel(handle, r, ptr.data(), params.len, params.start, params.end); break;
        case RNG_Logistic:
          logistic(handle, r, ptr.data(), params.len, params.start, params.end);
          break;
        case RNG_Exp: exponential(handle, r, ptr.data(), params.len, params.start); break;
        case RNG_Rayleigh: rayleigh(handle, r, ptr.data(), params.len, params.start); break;
        case RNG_Laplace:
          laplace(handle, r, ptr.data(), params.len, params.start, params.end);
          break;
        case RNG_Fill: fill(handle, r, ptr.data(), params.len, params.start); break;
      };
    });
  }

 private:
  rng_inputs<T> params;
  rmm::device_uvector<T> ptr;
};  // struct RngBench

template <typename T>
static std::vector<rng_inputs<T>> get_rng_input_vecs()
{
  using namespace raft::random;
  return {
    {1024 * 1024, RNG_Uniform, GenPhilox, T(-1.0), T(1.0)},
    {32 * 1024 * 1024, RNG_Uniform, GenPhilox, T(-1.0), T(1.0)},
    {1024 * 1024 * 1024, RNG_Uniform, GenPhilox, T(-1.0), T(1.0)},
    {1024 * 1024 + 2, RNG_Uniform, GenPhilox, T(-1.0), T(1.0)},
    {32 * 1024 * 1024 + 2, RNG_Uniform, GenPhilox, T(-1.0), T(1.0)},
    {1024 * 1024 * 1024 + 2, RNG_Uniform, GenPhilox, T(-1.0), T(1.0)},
    {1024 * 1024 + 1, RNG_Uniform, GenPhilox, T(-1.0), T(1.0)},
    {32 * 1024 * 1024 + 1, RNG_Uniform, GenPhilox, T(-1.0), T(1.0)},
    {1024 * 1024 * 1024 + 1, RNG_Uniform, GenPhilox, T(-1.0), T(1.0)},

    {1024 * 1024, RNG_Uniform, GenPC, T(-1.0), T(1.0)},
    {32 * 1024 * 1024, RNG_Uniform, GenPC, T(-1.0), T(1.0)},
    {1024 * 1024 * 1024, RNG_Uniform, GenPC, T(-1.0), T(1.0)},
    {1024 * 1024 + 2, RNG_Uniform, GenPC, T(-1.0), T(1.0)},
    {32 * 1024 * 1024 + 2, RNG_Uniform, GenPC, T(-1.0), T(1.0)},
    {1024 * 1024 * 1024 + 2, RNG_Uniform, GenPC, T(-1.0), T(1.0)},
    {1024 * 1024 + 1, RNG_Uniform, GenPC, T(-1.0), T(1.0)},
    {32 * 1024 * 1024 + 1, RNG_Uniform, GenPC, T(-1.0), T(1.0)},
    {1024 * 1024 * 1024 + 1, RNG_Uniform, GenPC, T(-1.0), T(1.0)},

    {1024 * 1024, RNG_Fill, GenPhilox, T(-1.0), T(1.0)},
    {32 * 1024 * 1024, RNG_Fill, GenPhilox, T(-1.0), T(1.0)},
    {1024 * 1024 * 1024, RNG_Fill, GenPhilox, T(-1.0), T(1.0)},
    {1024 * 1024 + 2, RNG_Fill, GenPhilox, T(-1.0), T(1.0)},
    {32 * 1024 * 1024 + 2, RNG_Fill, GenPhilox, T(-1.0), T(1.0)},
    {1024 * 1024 * 1024 + 2, RNG_Fill, GenPhilox, T(-1.0), T(1.0)},
    {1024 * 1024 + 1, RNG_Fill, GenPhilox, T(-1.0), T(1.0)},
    {32 * 1024 * 1024 + 1, RNG_Fill, GenPhilox, T(-1.0), T(1.0)},
    {1024 * 1024 * 1024 + 1, RNG_Fill, GenPhilox, T(-1.0), T(1.0)},
  };
}

RAFT_BENCH_REGISTER(rng<float>, "", get_rng_input_vecs<float>());
RAFT_BENCH_REGISTER(rng<double>, "", get_rng_input_vecs<double>());

}  // namespace raft::bench::random
