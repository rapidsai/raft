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
#include <cub/cub.cuh>
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

namespace raft::bench::random {

struct sample_inputs {
  int n_samples;
  int n_train;
  int method;
};  // struct sample_inputs

template <typename IdxT>
auto excess_subsample(raft::resources const& res, IdxT n_samples, IdxT n_subsamples, int seed)
  -> raft::device_vector<IdxT, IdxT>
{
  RAFT_EXPECTS(n_subsamples <= n_samples, "Cannot have more training samples than dataset vectors");
  auto stream = resource::get_cuda_stream(res);

  auto rnd_idx =
    raft::make_device_vector<IdxT, IdxT>(res, std::min<IdxT>(1.5 * n_subsamples, n_samples));
  auto linear_idx = raft::make_device_vector<IdxT, IdxT>(res, rnd_idx.size());
  raft::linalg::map_offset(res, linear_idx.view(), identity_op());

  raft::random::RngState state(137ULL);
  raft::random::uniformInt(
    res, state, rnd_idx.data_handle(), rnd_idx.size(), IdxT(0), IdxT(n_samples));

  // Sort indices according to rnd keys
  size_t workspace_size = 0;
  cub::DeviceMergeSort::SortPairs(nullptr,
                                  workspace_size,
                                  rnd_idx.data_handle(),
                                  linear_idx.data_handle(),
                                  rnd_idx.size(),
                                  raft::less_op{});
  float GiB = 1073741824.0f;
  RAFT_LOG_INFO("worksize sort %6.1f GiB", workspace_size / GiB);
  auto workspace = raft::make_device_vector<char, IdxT>(res, workspace_size);
  cub::DeviceMergeSort::SortPairs(nullptr,
                                  workspace_size,
                                  rnd_idx.data_handle(),
                                  linear_idx.data_handle(),
                                  rnd_idx.size(),
                                  raft::less_op{});

  if (rnd_idx.size() == static_cast<size_t>(n_samples)) {
    // We shuffled the linear_idx array by sorting it according to rnd_idx.
    // We return the first n_subsamples elements.
    if (n_subsamples == n_samples) { return linear_idx; }
    rnd_idx = raft::make_device_vector<IdxT, IdxT>(res, n_subsamples);
    raft::copy(rnd_idx.data_handle(), linear_idx.data_handle(), n_subsamples, stream);
    return rnd_idx;
  }
  // Else we do a rejection sampling (or excess sampling): we generated more random indices than
  // needed and reject the duplicates.
  auto keys_out   = raft::make_device_vector<IdxT, IdxT>(res, rnd_idx.size());
  auto values_out = raft::make_device_vector<IdxT, IdxT>(res, rnd_idx.size());
  rmm::device_scalar<IdxT> num_selected(stream);
  size_t worksize2 = 0;
  cub::DeviceSelect::UniqueByKey(nullptr,
                                 worksize2,
                                 rnd_idx.data_handle(),
                                 linear_idx.data_handle(),
                                 keys_out.data_handle(),
                                 values_out.data_handle(),
                                 num_selected.data(),
                                 rnd_idx.size(),
                                 stream);

  RAFT_LOG_INFO("worksize unique %6.1f GiB", worksize2 / GiB);

  if (worksize2 > workspace.size()) {
    workspace = raft::make_device_vector<char, IdxT>(res, worksize2);
  }

  cub::DeviceSelect::UniqueByKey(workspace.data_handle(),
                                 worksize2,
                                 rnd_idx.data_handle(),
                                 linear_idx.data_handle(),
                                 keys_out.data_handle(),
                                 values_out.data_handle(),
                                 num_selected.data(),
                                 rnd_idx.size(),
                                 stream);

  IdxT selected = num_selected.value(stream);

  if (selected < n_subsamples) {
    RAFT_LOG_WARN("Subsampling returned with less unique indices (%zu) than requested (%zu)",
                  (size_t)selected,
                  (size_t)n_subsamples);

  } else {
    RAFT_LOG_INFO(
      "Subsampling unique indices (%zu) requested (%zu)", (size_t)selected, (size_t)n_subsamples);
  }

  // need to shuffle again
  cub::DeviceMergeSort::SortPairs(workspace.data_handle(),
                                  worksize2,
                                  linear_idx.data_handle(),
                                  rnd_idx.data_handle(),
                                  n_samples,
                                  raft::less_op{});

  if (n_subsamples == n_samples) { return linear_idx; }
  values_out = raft::make_device_vector<IdxT, IdxT>(res, n_subsamples);
  raft::copy(values_out.data_handle(), rnd_idx.data_handle(), n_subsamples, stream);
  return values_out;
}

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
      if (params.method == 0) {
        this->out = raft::spatial::knn::detail::utils::get_subsample_indices<T>(
          this->res, this->params.n_samples, this->params.n_train, 137);
      } else if (params.method == 1) {
        this->out =
          bernoulli_subsample<T>(this->res, this->params.n_samples, this->params.n_train, 137);
      } else if (params.method == 2) {
        this->out =
          excess_subsample<T>(this->res, this->params.n_samples, this->params.n_train, 137);
      }
      //   raft::random::permute(
      //     perms.data(), out.data(), in.data(), params.cols, params.rows, params.rowMajor,
      //     stream);
    });
  }

 private:
  raft::device_resources res;
  sample_inputs params;
  raft::device_vector<T, int64_t> out, in;
};  // struct sample

const std::vector<sample_inputs> input_vecs = {{10000000, 1000000, 0},
                                               {10000000, 10000000, 0},
                                               {100000000, 10000000, 1},
                                               {100000000, 100000000, 1},
                                               {100000000, 10000000, 2},
                                               {100000000, 50000000, 2},
                                               {100000000, 100000000, 2}};

RAFT_BENCH_REGISTER(sample<int64_t>, "", input_vecs);

}  // namespace raft::bench::random
