/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/operators.hpp>
#include <raft/random/permute.cuh>
#include <raft/random/rng.cuh>
#include <raft/random/sample_without_replacement.cuh>
#include <raft/spatial/knn/detail/ann_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>

#include <cub/cub.cuh>

namespace raft::bench::random {

struct sample_inputs {
  int n_samples;
  int n_train;
  int method;
};  // struct sample_inputs

// Sample with replacement. We use this as a baseline.
template <typename IdxT>
auto bernoulli_subsample(raft::resources const& res, IdxT n_samples, IdxT n_subsamples, int seed)
  -> raft::device_vector<IdxT, IdxT>
{
  RAFT_EXPECTS(n_subsamples <= n_samples, "Cannot have more training samples than dataset vectors");

  auto indices = raft::make_device_vector<IdxT, IdxT>(res, n_subsamples);
  raft::random::RngState state(123456ULL);
  raft::random::uniformInt(
    res, state, indices.data_handle(), n_subsamples, IdxT(0), IdxT(n_samples));
  return indices;
}

template <typename T>
struct sample : public fixture {
  sample(const sample_inputs& p)
    : params(p),
      in(make_device_vector<T, int64_t>(res, p.n_samples)),
      out(make_device_vector<T, int64_t>(res, p.n_train))
  {
    raft::random::RngState r(123456ULL);
  }

  void run_benchmark(::benchmark::State& state) override
  {
    raft::random::RngState r(123456ULL);
    loop_on_state(state, [this, &r]() {
      if (params.method == 1) {
        this->out =
          bernoulli_subsample<T>(this->res, this->params.n_samples, this->params.n_train, 137);
      } else if (params.method == 2) {
        this->out = raft::random::excess_subsample<T, int64_t>(
          this->res, r, this->params.n_samples, this->params.n_train);
      }
    });
  }

 private:
  raft::device_resources res;
  sample_inputs params;
  raft::device_vector<T, int64_t> out, in;
};  // struct sample

const std::vector<sample_inputs> input_vecs = {{100000000, 10000000, 1},
                                               {100000000, 50000000, 1},
                                               {100000000, 100000000, 1},
                                               {100000000, 10000000, 2},
                                               {100000000, 50000000, 2},
                                               {100000000, 100000000, 2}};

RAFT_BENCH_REGISTER(sample<int64_t>, "", input_vecs);

}  // namespace raft::bench::random
