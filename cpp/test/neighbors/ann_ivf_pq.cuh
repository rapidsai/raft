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
#pragma once

#include "../test_utils.cuh"
#include "ann_utils.cuh"

#include <raft_internal/neighbors/naive_knn.cuh>

#include <raft/core/logger.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/ivf_pq.cuh>
#include <raft/random/rng.cuh>
#ifdef RAFT_COMPILED
#include <raft/neighbors/specializations.cuh>
#else
#pragma message("NN specializations are not enabled; expect very long building times.")
#endif

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>

#include <gtest/gtest.h>

#include <cub/cub.cuh>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <optional>
#include <vector>

namespace raft::neighbors::ivf_pq {

struct ivf_pq_inputs {
  uint32_t num_db_vecs             = 4096;
  uint32_t num_queries             = 1024;
  uint32_t dim                     = 64;
  uint32_t k                       = 32;
  std::optional<double> min_recall = std::nullopt;

  ivf_pq::index_params index_params;
  ivf_pq::search_params search_params;

  // Set some default parameters for tests
  ivf_pq_inputs()
  {
    index_params.n_lists                  = max(32u, min(1024u, num_db_vecs / 128u));
    index_params.kmeans_trainset_fraction = 1.0;
  }
};

inline auto operator<<(std::ostream& os, const ivf_pq::codebook_gen& p) -> std::ostream&
{
  switch (p) {
    case ivf_pq::codebook_gen::PER_CLUSTER: os << "codebook_gen::PER_CLUSTER"; break;
    case ivf_pq::codebook_gen::PER_SUBSPACE: os << "codebook_gen::PER_SUBSPACE"; break;
    default: RAFT_FAIL("unreachable code");
  }
  return os;
}

inline auto operator<<(std::ostream& os, const ivf_pq_inputs& p) -> std::ostream&
{
  ivf_pq_inputs dflt;
  bool need_comma = false;
#define PRINT_DIFF_V(spec, val)       \
  do {                                \
    if (dflt spec != p spec) {        \
      if (need_comma) { os << ", "; } \
      os << #spec << " = " << val;    \
      need_comma = true;              \
    }                                 \
  } while (0)
#define PRINT_DIFF(spec) PRINT_DIFF_V(spec, p spec)

  os << "ivf_pq_inputs {";
  PRINT_DIFF(.num_db_vecs);
  PRINT_DIFF(.num_queries);
  PRINT_DIFF(.dim);
  PRINT_DIFF(.k);
  PRINT_DIFF_V(.min_recall, p.min_recall.value_or(0));
  PRINT_DIFF_V(.index_params.metric, print_metric{p.index_params.metric});
  PRINT_DIFF(.index_params.metric_arg);
  PRINT_DIFF(.index_params.add_data_on_build);
  PRINT_DIFF(.index_params.n_lists);
  PRINT_DIFF(.index_params.kmeans_n_iters);
  PRINT_DIFF(.index_params.kmeans_trainset_fraction);
  PRINT_DIFF(.index_params.pq_bits);
  PRINT_DIFF(.index_params.pq_dim);
  PRINT_DIFF(.index_params.codebook_kind);
  PRINT_DIFF(.index_params.force_random_rotation);
  PRINT_DIFF(.search_params.n_probes);
  PRINT_DIFF_V(.search_params.lut_dtype, print_dtype{p.search_params.lut_dtype});
  PRINT_DIFF_V(.search_params.internal_distance_dtype,
               print_dtype{p.search_params.internal_distance_dtype});
  os << "}";
  return os;
}

template <typename IdxT>
auto min_output_size(const raft::device_resources& handle,
                     const ivf_pq::index<IdxT>& index,
                     uint32_t n_probes) -> IdxT
{
  auto acc_sizes        = index.accum_sorted_sizes();
  uint32_t last_nonzero = index.n_lists();
  while (last_nonzero > 0 && acc_sizes(last_nonzero - 1) == acc_sizes(last_nonzero)) {
    last_nonzero--;
  }
  return acc_sizes(last_nonzero) - acc_sizes(last_nonzero - std::min(last_nonzero, n_probes));
}

template <typename EvalT, typename DataT, typename IdxT>
class ivf_pq_test : public ::testing::TestWithParam<ivf_pq_inputs> {
 public:
  ivf_pq_test()
    : stream_(handle_.get_stream()),
      ps(::testing::TestWithParam<ivf_pq_inputs>::GetParam()),
      database(0, stream_),
      search_queries(0, stream_)
  {
  }

 protected:
  void gen_data()
  {
    database.resize(size_t{ps.num_db_vecs} * size_t{ps.dim}, stream_);
    search_queries.resize(size_t{ps.num_queries} * size_t{ps.dim}, stream_);

    raft::random::Rng r(1234ULL);
    if constexpr (std::is_same<DataT, float>{}) {
      r.uniform(database.data(), ps.num_db_vecs * ps.dim, DataT(0.1), DataT(2.0), stream_);
      r.uniform(search_queries.data(), ps.num_queries * ps.dim, DataT(0.1), DataT(2.0), stream_);
    } else {
      r.uniformInt(database.data(), ps.num_db_vecs * ps.dim, DataT(1), DataT(20), stream_);
      r.uniformInt(search_queries.data(), ps.num_queries * ps.dim, DataT(1), DataT(20), stream_);
    }
    handle_.sync_stream(stream_);
  }

  void calc_ref()
  {
    size_t queries_size = size_t{ps.num_queries} * size_t{ps.k};
    rmm::device_uvector<EvalT> distances_naive_dev(queries_size, stream_);
    rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);
    naive_knn<EvalT, DataT, IdxT>(distances_naive_dev.data(),
                                  indices_naive_dev.data(),
                                  search_queries.data(),
                                  database.data(),
                                  ps.num_queries,
                                  ps.num_db_vecs,
                                  ps.dim,
                                  ps.k,
                                  ps.index_params.metric,
                                  stream_);
    distances_ref.resize(queries_size);
    update_host(distances_ref.data(), distances_naive_dev.data(), queries_size, stream_);
    indices_ref.resize(queries_size);
    update_host(indices_ref.data(), indices_naive_dev.data(), queries_size, stream_);
    handle_.sync_stream(stream_);
  }

  index<IdxT> build_only()
  {
    auto ipams              = ps.index_params;
    ipams.add_data_on_build = true;

    auto index_view =
      raft::make_device_matrix_view<DataT, IdxT>(database.data(), ps.num_db_vecs, ps.dim);
    return ivf_pq::build<DataT, IdxT>(handle_, ipams, index_view);
  }

  index<IdxT> build_2_extends()
  {
    rmm::device_uvector<IdxT> db_indices(ps.num_db_vecs, stream_);
    thrust::sequence(handle_.get_thrust_policy(),
                     thrust::device_pointer_cast(db_indices.data()),
                     thrust::device_pointer_cast(db_indices.data() + ps.num_db_vecs));
    handle_.sync_stream(stream_);
    auto size_1 = IdxT(ps.num_db_vecs) / 2;
    auto size_2 = IdxT(ps.num_db_vecs) - size_1;
    auto vecs_1 = database.data();
    auto vecs_2 = database.data() + size_t(size_1) * size_t(ps.dim);
    auto inds_1 = db_indices.data();
    auto inds_2 = db_indices.data() + size_t(size_1);

    auto ipams              = ps.index_params;
    ipams.add_data_on_build = false;

    auto database_view =
      raft::make_device_matrix_view<DataT, IdxT>(database.data(), ps.num_db_vecs, ps.dim);
    auto idx = ivf_pq::build<DataT, IdxT>(handle_, ipams, database_view);

    auto vecs_2_view = raft::make_device_matrix_view<DataT, IdxT>(vecs_2, size_2, ps.dim);
    auto inds_2_view = raft::make_device_matrix_view<IdxT, IdxT>(inds_2, size_2, 1);
    ivf_pq::extend<DataT, IdxT>(handle_, vecs_2_view, inds_2_view, &idx);

    auto vecs_1_view =
      raft::make_device_matrix_view<DataT, IdxT, row_major>(vecs_1, size_1, ps.dim);
    auto inds_1_view =
      raft::make_device_matrix_view<const IdxT, IdxT, row_major>(inds_1, size_1, 1);
    ivf_pq::extend<DataT, IdxT>(handle_, vecs_1_view, inds_1_view, &idx);
    return idx;
  }

  index<IdxT> build_serialize()
  {
    ivf_pq::serialize<IdxT>(handle_, "ivf_pq_index", build_only());
    return ivf_pq::deserialize<IdxT>(handle_, "ivf_pq_index");
  }

  template <typename BuildIndex>
  void run(BuildIndex build_index)
  {
    index<IdxT> index = build_index();

    size_t queries_size = ps.num_queries * ps.k;
    std::vector<IdxT> indices_ivf_pq(queries_size);
    std::vector<EvalT> distances_ivf_pq(queries_size);

    rmm::device_uvector<EvalT> distances_ivf_pq_dev(queries_size, stream_);
    rmm::device_uvector<IdxT> indices_ivf_pq_dev(queries_size, stream_);

    auto query_view =
      raft::make_device_matrix_view<DataT, IdxT>(search_queries.data(), ps.num_queries, ps.dim);
    auto inds_view =
      raft::make_device_matrix_view<IdxT, IdxT>(indices_ivf_pq_dev.data(), ps.num_queries, ps.k);
    auto dists_view =
      raft::make_device_matrix_view<EvalT, IdxT>(distances_ivf_pq_dev.data(), ps.num_queries, ps.k);

    ivf_pq::search<DataT, IdxT>(
      handle_, ps.search_params, index, query_view, inds_view, dists_view);

    update_host(distances_ivf_pq.data(), distances_ivf_pq_dev.data(), queries_size, stream_);
    update_host(indices_ivf_pq.data(), indices_ivf_pq_dev.data(), queries_size, stream_);
    handle_.sync_stream(stream_);

    // A very conservative lower bound on recall
    double min_recall =
      static_cast<double>(ps.search_params.n_probes) / static_cast<double>(ps.index_params.n_lists);
    double low_precision_factor =
      static_cast<double>(ps.dim * 8) / static_cast<double>(index.pq_dim() * index.pq_bits());
    // Using a heuristic to lower the required recall due to code-packing errors
    min_recall =
      std::min(std::erfc(0.05 * low_precision_factor / std::max(min_recall, 0.5)), min_recall);
    // Use explicit per-test min recall value if provided.
    min_recall = ps.min_recall.value_or(min_recall);

    ASSERT_TRUE(eval_neighbours(indices_ref,
                                indices_ivf_pq,
                                distances_ref,
                                distances_ivf_pq,
                                ps.num_queries,
                                ps.k,
                                0.0001 * low_precision_factor,
                                min_recall))
      << ps;

    // Test a few extra invariants
    IdxT min_results = min_output_size(handle_, index, ps.search_params.n_probes);
    IdxT max_oob     = ps.k <= min_results ? 0 : ps.k - min_results;
    IdxT found_oob   = 0;
    for (uint32_t query_ix = 0; query_ix < ps.num_queries; query_ix++) {
      for (uint32_t k = 0; k < ps.k; k++) {
        auto flat_i   = query_ix * ps.k + k;
        auto found_ix = indices_ivf_pq[flat_i];
        if (found_ix == ivf_pq::kOutOfBoundsRecord<IdxT>) {
          found_oob++;
          continue;
        }
        ASSERT_NE(found_ix, ivf::kInvalidRecord<IdxT>)
          << "got an invalid record at query_ix = " << query_ix << ", k = " << k
          << " (distance = " << distances_ivf_pq[flat_i] << ")";
        ASSERT_LT(found_ix, ps.num_db_vecs)
          << "got an impossible index = " << found_ix << " at query_ix = " << query_ix
          << ", k = " << k << " (distance = " << distances_ivf_pq[flat_i] << ")";
      }
    }
    ASSERT_LE(found_oob, max_oob)
      << "got too many records out-of-bounds (see ivf_pq::kOutOfBoundsRecord<IdxT>).";
    if (found_oob > 0) {
      RAFT_LOG_WARN(
        "Got %zu results out-of-bounds because of large top-k (%zu) and small n_probes (%u) and "
        "small DB size/n_lists ratio (%zu / %u)",
        size_t(found_oob),
        size_t(ps.k),
        ps.search_params.n_probes,
        size_t(ps.num_db_vecs),
        ps.index_params.n_lists);
    }
  }

  void SetUp() override  // NOLINT
  {
    gen_data();
    calc_ref();
  }

  void TearDown() override  // NOLINT
  {
    cudaGetLastError();
    handle_.sync_stream(stream_);
    database.resize(0, stream_);
    search_queries.resize(0, stream_);
  }

 private:
  raft::device_resources handle_;
  rmm::cuda_stream_view stream_;
  ivf_pq_inputs ps;                           // NOLINT
  rmm::device_uvector<DataT> database;        // NOLINT
  rmm::device_uvector<DataT> search_queries;  // NOLINT
  std::vector<IdxT> indices_ref;              // NOLINT
  std::vector<EvalT> distances_ref;           // NOLINT
};

/* Test cases */
using test_cases_t = std::vector<ivf_pq_inputs>;

// concatenate parameter sets for different type
template <typename T>
auto operator+(const std::vector<T>& a, const std::vector<T>& b) -> std::vector<T>
{
  std::vector<T> res = a;
  res.insert(res.end(), b.begin(), b.end());
  return res;
}

inline auto defaults() -> test_cases_t { return {ivf_pq_inputs{}}; }

template <typename B, typename A, typename F>
auto map(const std::vector<A>& xs, F f) -> std::vector<B>
{
  std::vector<B> ys(xs.size());
  std::transform(xs.begin(), xs.end(), ys.begin(), f);
  return ys;
}

inline auto with_dims(const std::vector<uint32_t>& dims) -> test_cases_t
{
  return map<ivf_pq_inputs>(dims, [](uint32_t d) {
    ivf_pq_inputs x;
    x.dim = d;
    return x;
  });
}

/** These will surely trigger the fastest kernel available. */
inline auto small_dims() -> test_cases_t { return with_dims({1, 2, 3, 4, 5, 8, 15, 16, 17}); }

inline auto small_dims_per_cluster() -> test_cases_t
{
  return map<ivf_pq_inputs>(small_dims(), [](const ivf_pq_inputs& x) {
    ivf_pq_inputs y(x);
    y.index_params.codebook_kind = ivf_pq::codebook_gen::PER_CLUSTER;
    return y;
  });
}

inline auto big_dims() -> test_cases_t
{
  // with_dims({512, 513, 1023, 1024, 1025, 2048, 2049, 2050, 2053, 6144, 8192, 12288, 16384});
  auto xs = with_dims({512, 513, 1023, 1024, 1025, 2048, 2049, 2050, 2053, 6144});
  return map<ivf_pq_inputs>(xs, [](const ivf_pq_inputs& x) {
    ivf_pq_inputs y(x);
    uint32_t pq_len       = 2;
    y.index_params.pq_dim = div_rounding_up_safe(x.dim, pq_len);
    // This comes from pure experimentation, also the recall depens a lot on pq_len.
    y.min_recall = 0.48 + 0.028 * std::log2(x.dim);
    return y;
  });
}

/** These will surely trigger no-smem-lut kernel.  */
inline auto big_dims_moderate_lut() -> test_cases_t
{
  return map<ivf_pq_inputs>(big_dims(), [](const ivf_pq_inputs& x) {
    ivf_pq_inputs y(x);
    uint32_t pq_len           = 2;
    y.index_params.pq_dim     = round_up_safe(div_rounding_up_safe(x.dim, pq_len), 4u);
    y.index_params.pq_bits    = 6;
    y.search_params.lut_dtype = CUDA_R_16F;
    y.min_recall              = 0.69;
    return y;
  });
}

/** Some of these should trigger no-basediff kernel.  */
inline auto big_dims_small_lut() -> test_cases_t
{
  return map<ivf_pq_inputs>(big_dims(), [](const ivf_pq_inputs& x) {
    ivf_pq_inputs y(x);
    uint32_t pq_len           = 8;
    y.index_params.pq_dim     = round_up_safe(div_rounding_up_safe(x.dim, pq_len), 4u);
    y.index_params.pq_bits    = 6;
    y.search_params.lut_dtype = CUDA_R_8U;
    y.min_recall              = 0.21;
    return y;
  });
}

/**
 * A minimal set of tests to check various enum-like parameters.
 */
inline auto enum_variety() -> test_cases_t
{
  test_cases_t xs;
#define ADD_CASE(f)                               \
  do {                                            \
    xs.push_back({});                             \
    ([](ivf_pq_inputs & x) f)(xs[xs.size() - 1]); \
  } while (0);

  ADD_CASE({
    x.index_params.codebook_kind = ivf_pq::codebook_gen::PER_CLUSTER;
    x.min_recall                 = 0.86;
  });
  ADD_CASE({
    x.index_params.codebook_kind = ivf_pq::codebook_gen::PER_SUBSPACE;
    x.min_recall                 = 0.86;
  });
  ADD_CASE({
    x.index_params.codebook_kind = ivf_pq::codebook_gen::PER_CLUSTER;
    x.index_params.pq_bits       = 4;
    x.min_recall                 = 0.79;
  });
  ADD_CASE({
    x.index_params.codebook_kind = ivf_pq::codebook_gen::PER_CLUSTER;
    x.index_params.pq_bits       = 5;
    x.min_recall                 = 0.83;
  });

  ADD_CASE({
    x.index_params.pq_bits = 6;
    x.min_recall           = 0.84;
  });
  ADD_CASE({
    x.index_params.pq_bits = 7;
    x.min_recall           = 0.85;
  });
  ADD_CASE({
    x.index_params.pq_bits = 8;
    x.min_recall           = 0.86;
  });

  ADD_CASE({
    x.index_params.force_random_rotation = true;
    x.min_recall                         = 0.86;
  });
  ADD_CASE({
    x.index_params.force_random_rotation = false;
    x.min_recall                         = 0.86;
  });

  ADD_CASE({
    x.search_params.lut_dtype = CUDA_R_32F;
    x.min_recall              = 0.86;
  });
  ADD_CASE({
    x.search_params.lut_dtype = CUDA_R_16F;
    x.min_recall              = 0.86;
  });
  ADD_CASE({
    x.search_params.lut_dtype = CUDA_R_8U;
    x.min_recall              = 0.84;
  });

  ADD_CASE({
    x.search_params.internal_distance_dtype = CUDA_R_32F;
    x.min_recall                            = 0.86;
  });
  ADD_CASE({
    x.search_params.internal_distance_dtype = CUDA_R_16F;
    x.search_params.lut_dtype               = CUDA_R_16F;
    x.min_recall                            = 0.86;
  });

  return xs;
}

inline auto enum_variety_l2() -> test_cases_t
{
  return map<ivf_pq_inputs>(enum_variety(), [](const ivf_pq_inputs& x) {
    ivf_pq_inputs y(x);
    y.index_params.metric = distance::DistanceType::L2Expanded;
    return y;
  });
}

inline auto enum_variety_ip() -> test_cases_t
{
  return map<ivf_pq_inputs>(enum_variety(), [](const ivf_pq_inputs& x) {
    ivf_pq_inputs y(x);
    if (y.min_recall.has_value()) {
      if (y.search_params.lut_dtype == CUDA_R_8U) {
        // InnerProduct score is signed,
        // thus we're forced to used signed 8-bit representation,
        // thus we have one bit less precision
        y.min_recall = y.min_recall.value() * 0.90;
      } else {
        // In other cases it seems to perform a little bit better, still worse than L2
        y.min_recall = y.min_recall.value() * 0.94;
      }
    }
    y.index_params.metric = distance::DistanceType::InnerProduct;
    return y;
  });
}

inline auto enum_variety_l2sqrt() -> test_cases_t
{
  return map<ivf_pq_inputs>(enum_variety(), [](const ivf_pq_inputs& x) {
    ivf_pq_inputs y(x);
    y.index_params.metric = distance::DistanceType::L2SqrtExpanded;
    return y;
  });
}

/**
 * Try different number of n_probes, some of which may trigger the non-fused version of the search
 * kernel.
 */
inline auto var_n_probes() -> test_cases_t
{
  ivf_pq_inputs dflt;
  std::vector<uint32_t> xs;
  for (auto x = dflt.index_params.n_lists; x >= 1; x /= 2) {
    xs.push_back(x);
  }
  return map<ivf_pq_inputs>(xs, [](uint32_t n_probes) {
    ivf_pq_inputs x;
    x.search_params.n_probes = n_probes;
    return x;
  });
}

/**
 * Try different number of nearest neighbours.
 * Values smaller than 32 test if the code behaves well when Capacity (== 32) does not change,
 * but `k <= Capacity` changes.
 *
 * Values between `32 and ivf_pq::detail::kMaxCapacity` test various instantiations of the
 * main kernel (Capacity-templated)
 *
 * Values above ivf_pq::detail::kMaxCapacity should trigger the non-fused version of the kernel
 * (manage_local_topk = false).
 *
 * Also we test here various values that are close-but-not-power-of-two to catch any problems
 * related to rounding/alignment.
 *
 * Note, we cannot control explicitly which instance of the search kernel to choose, hence it's
 * important to try a variety of different values of `k` to make sure all paths are triggered.
 *
 * Set the log level to DEBUG (5) or above to inspect the selected kernel instances.
 */
inline auto var_k() -> test_cases_t
{
  return map<ivf_pq_inputs, uint32_t>(
    {1, 2, 3, 5, 8, 15, 16, 32, 63, 65, 127, 128, 256, 257, 1023, 2048, 2049}, [](uint32_t k) {
      ivf_pq_inputs x;
      x.k = k;
      // when there's not enough data, try more cluster probes
      x.search_params.n_probes = max(x.search_params.n_probes, min(x.index_params.n_lists, k));
      return x;
    });
}

/**
 * Cases brought up from downstream projects.
 */
inline auto special_cases() -> test_cases_t
{
  test_cases_t xs;

#define ADD_CASE(f)                               \
  do {                                            \
    xs.push_back({});                             \
    ([](ivf_pq_inputs & x) f)(xs[xs.size() - 1]); \
  } while (0);

  ADD_CASE({
    x.num_db_vecs                = 1183514;
    x.dim                        = 100;
    x.num_queries                = 10000;
    x.k                          = 10;
    x.index_params.codebook_kind = ivf_pq::codebook_gen::PER_SUBSPACE;
    x.index_params.pq_dim        = 10;
    x.index_params.pq_bits       = 8;
    x.index_params.n_lists       = 1024;
    x.search_params.n_probes     = 50;
  });

  ADD_CASE({
    x.num_db_vecs                = 10000;
    x.dim                        = 16;
    x.num_queries                = 500;
    x.k                          = 128;
    x.index_params.metric        = distance::DistanceType::L2Expanded;
    x.index_params.codebook_kind = ivf_pq::codebook_gen::PER_SUBSPACE;
    x.index_params.pq_bits       = 8;
    x.index_params.n_lists       = 100;
    x.search_params.n_probes     = 100;
  });

  ADD_CASE({
    x.num_db_vecs                = 10000;
    x.dim                        = 16;
    x.num_queries                = 500;
    x.k                          = 129;
    x.index_params.metric        = distance::DistanceType::L2Expanded;
    x.index_params.codebook_kind = ivf_pq::codebook_gen::PER_SUBSPACE;
    x.index_params.pq_bits       = 8;
    x.index_params.n_lists       = 100;
    x.search_params.n_probes     = 100;
  });

  return xs;
}

/* Test instantiations */

#define TEST_BUILD_SEARCH(type)                         \
  TEST_P(type, build_search) /* NOLINT */               \
  {                                                     \
    this->run([this]() { return this->build_only(); }); \
  }

#define TEST_BUILD_EXTEND_SEARCH(type)                       \
  TEST_P(type, build_extend_search) /* NOLINT */             \
  {                                                          \
    this->run([this]() { return this->build_2_extends(); }); \
  }

#define TEST_BUILD_SERIALIZE_SEARCH(type)                    \
  TEST_P(type, build_serialize_search) /* NOLINT */          \
  {                                                          \
    this->run([this]() { return this->build_serialize(); }); \
  }

#define INSTANTIATE(type, vals) \
  INSTANTIATE_TEST_SUITE_P(IvfPq, type, ::testing::ValuesIn(vals)); /* NOLINT */

}  // namespace raft::neighbors::ivf_pq
