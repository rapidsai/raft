/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#pragma once

#include <common/benchmark.hpp>
#include <raft/neighbors/cagra.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/itertools.hpp>

#include <optional>

namespace raft::bench::neighbors {

struct params {
  /** Size of the dataset. */
  size_t n_samples;
  /** Number of dimensions in the dataset. */
  int n_dims;
  /** The batch size -- number of KNN searches. */
  int n_queries;
  /** Number of nearest neighbours to find for every probe. */
  int k;
  /** kNN graph degree*/
  int degree;
  int itopk_size;
  int block_size;
  int num_parents;
  int max_iterations;
};

template <typename T, typename IdxT>
struct CagraBench : public fixture {
  explicit CagraBench(const params& ps)
    : fixture(true),
      params_(ps),
      queries_(make_device_matrix<T, IdxT>(handle, ps.n_queries, ps.n_dims)),
      dataset_(make_device_matrix<T, IdxT>(handle, ps.n_samples, ps.n_dims)),
      knn_graph_(make_device_matrix<IdxT, IdxT>(handle, ps.n_samples, ps.degree))
  {
    // Generate random dataset and queriees
    raft::random::RngState state{42};
    constexpr T kRangeMax = std::is_integral_v<T> ? std::numeric_limits<T>::max() : T(1);
    constexpr T kRangeMin = std::is_integral_v<T> ? std::numeric_limits<T>::min() : T(-1);
    if constexpr (std::is_integral_v<T>) {
      raft::random::uniformInt(
        state, dataset_.data_handle(), dataset_.size(), kRangeMin, kRangeMax, stream);
      raft::random::uniformInt(
        state, queries_.data_handle(), queries_.size(), kRangeMin, kRangeMax, stream);
    } else {
      raft::random::uniform(
        state, dataset_.data_handle(), dataset_.size(), kRangeMin, kRangeMax, stream);
      raft::random::uniform(
        state, queries_.data_handle(), queries_.size(), kRangeMin, kRangeMax, stream);
    }

    // Generate random knn graph

    raft::random::uniformInt<IdxT>(
      state, knn_graph_.data_handle(), knn_graph_.size(), 0, ps.n_samples - 1, stream);

    auto metric = raft::distance::DistanceType::L2Expanded;

    index_.emplace(raft::neighbors::experimental::cagra::index<T, IdxT>(
      handle, metric, make_const_mdspan(dataset_.view()), make_const_mdspan(knn_graph_.view())));
  }

  void run_benchmark(::benchmark::State& state) override
  {
    raft::neighbors::experimental::cagra::search_params search_params;
    search_params.max_queries       = 1024;
    search_params.itopk_size        = params_.itopk_size;
    search_params.team_size         = 0;
    search_params.thread_block_size = params_.block_size;
    search_params.num_parents       = params_.num_parents;

    auto indices   = make_device_matrix<IdxT, IdxT>(handle, params_.n_queries, params_.k);
    auto distances = make_device_matrix<float, IdxT>(handle, params_.n_queries, params_.k);
    auto ind_v     = make_device_matrix_view<IdxT, IdxT, row_major>(
      indices.data_handle(), params_.n_queries, params_.k);
    auto dist_v = make_device_matrix_view<float, IdxT, row_major>(
      distances.data_handle(), params_.n_queries, params_.k);

    auto queries_v = make_const_mdspan(queries_.view());
    loop_on_state(state, [&]() {
      raft::neighbors::experimental::cagra::search(
        this->handle, search_params, *this->index_, queries_v, ind_v, dist_v);
    });

    double data_size  = params_.n_samples * params_.n_dims * sizeof(T);
    double graph_size = params_.n_samples * params_.degree * sizeof(IdxT);

    int iterations = params_.max_iterations;
    if (iterations == 0) {
      // see search_plan_impl::adjust_search_params()
      double r   = params_.itopk_size / static_cast<float>(params_.num_parents);
      iterations = 1 + std::min(r * 1.1, r + 10);
    }
    state.counters["dataset (GiB)"] = data_size / (1 << 30);
    state.counters["graph (GiB)"]   = graph_size / (1 << 30);
    state.counters["n_rows"]        = params_.n_samples;
    state.counters["n_cols"]        = params_.n_dims;
    state.counters["degree"]        = params_.degree;
    state.counters["n_queries"]     = params_.n_queries;
    state.counters["k"]             = params_.k;
    state.counters["itopk_size"]    = params_.itopk_size;
    state.counters["block_size"]    = params_.block_size;
    state.counters["num_parents"]   = params_.num_parents;
    state.counters["iterations"]    = iterations;
  }

 private:
  const params params_;
  std::optional<const raft::neighbors::experimental::cagra::index<T, IdxT>> index_;
  raft::device_matrix<T, IdxT, row_major> queries_;
  raft::device_matrix<T, IdxT, row_major> dataset_;
  raft::device_matrix<IdxT, IdxT, row_major> knn_graph_;
};

inline const std::vector<params> generate_inputs()
{
  std::vector<params> inputs =
    raft::util::itertools::product<params>({2000000ull},           // n_samples
                                           {128, 256, 512, 1024},  // dataset dim
                                           {1000},                 // n_queries
                                           {32},                   // k
                                           {64},                   // knn graph degree
                                           {64},                   // itopk_size
                                           {0},                    // block_size
                                           {1},                    // num_parents
                                           {0}                     // max_iterations
    );
  auto inputs2 = raft::util::itertools::product<params>({2000000ull, 10000000ull},  // n_samples
                                                        {128},                      // dataset dim
                                                        {1000},                     // n_queries
                                                        {32},                       // k
                                                        {64},  // knn graph degree
                                                        {64},  // itopk_size
                                                        {64, 128, 256, 512, 1024},  // block_size
                                                        {1},                        // num_parents
                                                        {0}  // max_iterations
  );
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());
  return inputs;
}

const std::vector<params> kCagraInputs = generate_inputs();

#define CAGRA_REGISTER(ValT, IdxT, inputs)                \
  namespace BENCHMARK_PRIVATE_NAME(knn) {                 \
  using AnnCagra = CagraBench<ValT, IdxT>;                \
  RAFT_BENCH_REGISTER(AnnCagra, #ValT "/" #IdxT, inputs); \
  }

}  // namespace raft::bench::neighbors
