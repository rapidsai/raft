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
#include <raft/neighbors/sample_filter.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/itertools.hpp>
#include <thrust/sequence.h>

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
  int search_width;
  int max_iterations;
  /** Ratio of removed indices. */
  double removed_ratio;
};

template <typename T, typename IdxT>
struct CagraBench : public fixture {
  explicit CagraBench(const params& ps)
    : fixture(true),
      params_(ps),
      queries_(make_device_matrix<T, int64_t>(handle, ps.n_queries, ps.n_dims)),
      dataset_(make_device_matrix<T, int64_t>(handle, ps.n_samples, ps.n_dims)),
      knn_graph_(make_device_matrix<IdxT, int64_t>(handle, ps.n_samples, ps.degree)),
      removed_indices_bitset_(handle, ps.n_samples)
  {
    // Generate random dataset and queriees
    raft::random::RngState state{42};
    constexpr T kRangeMax = std::is_integral_v<T> ? std::numeric_limits<T>::max() : T(1);
    constexpr T kRangeMin = std::is_integral_v<T> ? std::numeric_limits<T>::min() : T(-1);
    if constexpr (std::is_integral_v<T>) {
      raft::random::uniformInt(
        handle, state, dataset_.data_handle(), dataset_.size(), kRangeMin, kRangeMax);
      raft::random::uniformInt(
        handle, state, queries_.data_handle(), queries_.size(), kRangeMin, kRangeMax);
    } else {
      raft::random::uniform(
        handle, state, dataset_.data_handle(), dataset_.size(), kRangeMin, kRangeMax);
      raft::random::uniform(
        handle, state, queries_.data_handle(), queries_.size(), kRangeMin, kRangeMax);
    }

    // Generate random knn graph

    raft::random::uniformInt<IdxT>(
      handle, state, knn_graph_.data_handle(), knn_graph_.size(), 0, ps.n_samples - 1);

    auto metric = raft::distance::DistanceType::L2Expanded;

    auto removed_indices =
      raft::make_device_vector<IdxT, int64_t>(handle, ps.removed_ratio * ps.n_samples);
    thrust::sequence(
      resource::get_thrust_policy(handle),
      thrust::device_pointer_cast(removed_indices.data_handle()),
      thrust::device_pointer_cast(removed_indices.data_handle() + removed_indices.extent(0)));
    removed_indices_bitset_.set(removed_indices.view());
    index_.emplace(raft::neighbors::cagra::index<T, IdxT>(
      handle, metric, make_const_mdspan(dataset_.view()), make_const_mdspan(knn_graph_.view())));
  }

  void run_benchmark(::benchmark::State& state) override
  {
    raft::neighbors::cagra::search_params search_params;
    search_params.max_queries       = 1024;
    search_params.itopk_size        = params_.itopk_size;
    search_params.team_size         = 0;
    search_params.thread_block_size = params_.block_size;
    search_params.search_width      = params_.search_width;

    auto indices   = make_device_matrix<IdxT, int64_t>(handle, params_.n_queries, params_.k);
    auto distances = make_device_matrix<float, int64_t>(handle, params_.n_queries, params_.k);
    auto ind_v     = make_device_matrix_view<IdxT, int64_t, row_major>(
      indices.data_handle(), params_.n_queries, params_.k);
    auto dist_v = make_device_matrix_view<float, int64_t, row_major>(
      distances.data_handle(), params_.n_queries, params_.k);

    auto queries_v = make_const_mdspan(queries_.view());
    if (params_.removed_ratio > 0) {
      auto filter = raft::neighbors::filtering::bitset_filter(removed_indices_bitset_.view());
      loop_on_state(state, [&]() {
        raft::neighbors::cagra::search_with_filtering(
          this->handle, search_params, *this->index_, queries_v, ind_v, dist_v, filter);
      });
    } else {
      loop_on_state(state, [&]() {
        raft::neighbors::cagra::search(
          this->handle, search_params, *this->index_, queries_v, ind_v, dist_v);
      });
    }

    double data_size  = params_.n_samples * params_.n_dims * sizeof(T);
    double graph_size = params_.n_samples * params_.degree * sizeof(IdxT);

    int iterations = params_.max_iterations;
    if (iterations == 0) {
      // see search_plan_impl::adjust_search_params()
      double r   = params_.itopk_size / static_cast<float>(params_.search_width);
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
    state.counters["search_width"]  = params_.search_width;
    state.counters["iterations"]    = iterations;
    state.counters["removed_ratio"] = params_.removed_ratio;
  }

 private:
  const params params_;
  std::optional<const raft::neighbors::cagra::index<T, IdxT>> index_;
  raft::device_matrix<T, int64_t, row_major> queries_;
  raft::device_matrix<T, int64_t, row_major> dataset_;
  raft::device_matrix<IdxT, int64_t, row_major> knn_graph_;
  raft::core::bitset<std::uint32_t, IdxT> removed_indices_bitset_;
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
                                           {1},                    // search_width
                                           {0},                    // max_iterations
                                           {0.0}                   // removed_ratio
    );
  auto inputs2 = raft::util::itertools::product<params>({2000000ull, 10000000ull},  // n_samples
                                                        {128},                      // dataset dim
                                                        {1000},                     // n_queries
                                                        {32},                       // k
                                                        {64},  // knn graph degree
                                                        {64},  // itopk_size
                                                        {64, 128, 256, 512, 1024},  // block_size
                                                        {1},                        // search_width
                                                        {0},   // max_iterations
                                                        {0.0}  // removed_ratio
  );
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  inputs2 = raft::util::itertools::product<params>(
    {2000000ull, 10000000ull},                 // n_samples
    {128},                                     // dataset dim
    {1, 10, 10000},                            // n_queries
    {255},                                     // k
    {64},                                      // knn graph degree
    {300},                                     // itopk_size
    {256},                                     // block_size
    {2},                                       // search_width
    {0},                                       // max_iterations
    {0.0, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64}  // removed_ratio
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
