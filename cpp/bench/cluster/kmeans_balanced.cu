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
#include <raft/cluster/kmeans_balanced.cuh>
#include <raft/random/rng.cuh>

#if defined RAFT_COMPILED
#include <raft/cluster/specializations.cuh>
#endif

namespace raft::bench::cluster {

struct KMeansBalancedBenchParams {
  DatasetParams data;
  uint32_t n_lists;
  raft::cluster::kmeans_balanced_params kb_params;
};

template <typename T, typename IndexT = int>
struct KMeansBalanced : public fixture {
  KMeansBalanced(const KMeansBalancedBenchParams& p) : params(p), X(handle), centroids(handle) {}

  void run_benchmark(::benchmark::State& state) override
  {
    this->loop_on_state(state, [this]() {
      raft::device_matrix_view<const T, IndexT> X_view   = this->X.view();
      raft::device_matrix_view<T, IndexT> centroids_view = this->centroids.view();
      raft::cluster::kmeans_balanced::fit(
        this->handle, this->params.kb_params, X_view, centroids_view);
    });
  }

  void allocate_data(const ::benchmark::State& state) override
  {
    X = raft::make_device_matrix<T, IndexT>(handle, params.data.rows, params.data.cols);

    raft::random::RngState rng{1234};
    constexpr T kRangeMax = std::is_integral_v<T> ? std::numeric_limits<T>::max() : T(1);
    constexpr T kRangeMin = std::is_integral_v<T> ? std::numeric_limits<T>::min() : T(-1);
    if constexpr (std::is_integral_v<T>) {
      raft::random::uniformInt(
        rng, X.data_handle(), params.data.rows * params.data.cols, kRangeMin, kRangeMax, stream);
    } else {
      raft::random::uniform(
        rng, X.data_handle(), params.data.rows * params.data.cols, kRangeMin, kRangeMax, stream);
    }
    handle.sync_stream(stream);
  }

  void allocate_temp_buffers(const ::benchmark::State& state) override
  {
    centroids =
      raft::make_device_matrix<float, IndexT>(this->handle, params.n_lists, params.data.cols);
  }

 private:
  KMeansBalancedBenchParams params;
  raft::device_matrix<T, IndexT> X;
  raft::device_matrix<float, IndexT> centroids;
};  // struct KMeansBalanced

std::vector<KMeansBalancedBenchParams> getKMeansBalancedInputs()
{
  std::vector<KMeansBalancedBenchParams> out;
  KMeansBalancedBenchParams p;
  p.data.row_major                          = true;
  p.kb_params.n_iters                       = 20;
  p.kb_params.metric                        = raft::distance::DistanceType::L2Expanded;
  std::vector<std::pair<int, int>> row_cols = {
    {100000, 128}, {1000000, 128}, {10000000, 128},
    // The following dataset sizes are too large for most GPUs.
    // {100000000, 128},
  };
  for (auto& rc : row_cols) {
    p.data.rows = rc.first;
    p.data.cols = rc.second;
    for (auto n_lists : std::vector<int>({1000, 10000, 100000})) {
      p.n_lists = n_lists;
      out.push_back(p);
    }
  }
  return out;
}

// Note: the datasets sizes are too large for 32-bit index types.
RAFT_BENCH_REGISTER((KMeansBalanced<float, int64_t>), "", getKMeansBalancedInputs());

}  // namespace raft::bench::cluster
