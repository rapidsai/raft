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

#include "../test_utils.h"
#include "ann_utils.cuh"

#include <raft/core/logger.hpp>
#include <raft/distance/distance_type.hpp>
#include <raft/random/rng.cuh>
#include <raft/spatial/knn/ivf_pq.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>

#include <gtest/gtest.h>

#include <thrust/sequence.h>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <vector>

namespace raft::spatial::knn {

struct ivf_pq_inputs {
  uint32_t num_db_vecs = 4096;
  uint32_t num_queries = 1024;
  uint32_t dim         = 64;
  uint32_t k           = 32;
  raft::spatial::knn::ivf_pq::index_params index_params;
  raft::spatial::knn::ivf_pq::search_params search_params;

  // Set some default parameters for tests
  ivf_pq_inputs()
  {
    index_params.n_lists                  = max(32u, min(1024u, num_db_vecs / 128u));
    index_params.kmeans_trainset_fraction = 1.0;
  }
};

auto operator<<(std::ostream& os, const ivf_pq::codebook_gen& p) -> std::ostream&
{
  switch (p) {
    case ivf_pq::codebook_gen::PER_CLUSTER: os << "codebook_gen::PER_CLUSTER"; break;
    case ivf_pq::codebook_gen::PER_SUBSPACE: os << "codebook_gen::PER_SUBSPACE"; break;
    default: RAFT_FAIL("unreachable code");
  }
  return os;
}

auto operator<<(std::ostream& os, const ivf_pq_inputs& p) -> std::ostream&
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
  PRINT_DIFF_V(.index_params.metric, print_metric{p.index_params.metric});
  PRINT_DIFF(.index_params.metric_arg);
  PRINT_DIFF(.index_params.add_data_on_build);
  PRINT_DIFF(.index_params.n_lists);
  PRINT_DIFF(.index_params.kmeans_n_iters);
  PRINT_DIFF(.index_params.kmeans_trainset_fraction);
  PRINT_DIFF(.index_params.pq_bits);
  PRINT_DIFF(.index_params.pq_dim);
  PRINT_DIFF(.index_params.random_rotation);
  PRINT_DIFF(.index_params.codebook_kind);
  PRINT_DIFF(.search_params.n_probes);
  PRINT_DIFF_V(.search_params.lut_dtype, print_dtype{p.search_params.lut_dtype});
  PRINT_DIFF_V(.search_params.internal_distance_dtype,
               print_dtype{p.search_params.internal_distance_dtype});
  PRINT_DIFF(.search_params.preferred_thread_block_size);
  os << "}";
  return os;
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
  void run()
  {
    size_t queries_size = ps.num_queries * ps.k;
    std::vector<IdxT> indices_ivf_pq(queries_size);
    std::vector<IdxT> indices_naive(queries_size);
    std::vector<EvalT> distances_ivf_pq(queries_size);
    std::vector<EvalT> distances_naive(queries_size);

    {
      rmm::device_uvector<EvalT> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);
      naiveBfKnn<EvalT, DataT, IdxT>(distances_naive_dev.data(),
                                     indices_naive_dev.data(),
                                     search_queries.data(),
                                     database.data(),
                                     ps.num_queries,
                                     ps.num_db_vecs,
                                     ps.dim,
                                     ps.k,
                                     ps.index_params.metric,
                                     stream_);
      update_host(distances_naive.data(), distances_naive_dev.data(), queries_size, stream_);
      update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
      handle_.sync_stream(stream_);
    }

    {
      // unless something is really wrong with clustering, this could serve as a lower bound on
      // recall
      double min_recall = static_cast<double>(ps.search_params.n_probes) /
                          static_cast<double>(ps.index_params.n_lists);

      rmm::device_uvector<EvalT> distances_ivf_pq_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_ivf_pq_dev(queries_size, stream_);

      {
        auto size_1 = IdxT(ps.num_db_vecs) / 2;
        auto size_2 = IdxT(ps.num_db_vecs) - size_1;
        auto vecs_1 = database.data();
        auto vecs_2 = database.data() + size_t(size_1) * size_t(ps.dim);
        rmm::device_uvector<IdxT> db_indices(ps.num_db_vecs, stream_);
        thrust::sequence(handle_.get_thrust_policy(),
                         thrust::device_pointer_cast(db_indices.data()),
                         thrust::device_pointer_cast(db_indices.data() + ps.num_db_vecs));
        handle_.sync_stream(stream_);

        auto index = ivf_pq::build<DataT, IdxT>(handle_, ps.index_params, vecs_1, size_1, ps.dim);
        handle_.sync_stream(stream_);

        auto index_2 =
          ivf_pq::extend<DataT, IdxT>(handle_, index, vecs_2, db_indices.data() + size_1, size_2);
        handle_.sync_stream(stream_);

        // finally, search!
        ivf_pq::search<DataT, IdxT>(handle_,
                                    ps.search_params,
                                    index_2,
                                    search_queries.data(),
                                    ps.num_queries,
                                    ps.k,
                                    indices_ivf_pq_dev.data(),
                                    distances_ivf_pq_dev.data());
        handle_.sync_stream(stream_);

        update_host(distances_ivf_pq.data(), distances_ivf_pq_dev.data(), queries_size, stream_);
        update_host(indices_ivf_pq.data(), indices_ivf_pq_dev.data(), queries_size, stream_);
        handle_.sync_stream(stream_);
      }
      handle_.sync_stream(stream_);
      ASSERT_TRUE(eval_neighbours(indices_naive,
                                  indices_ivf_pq,
                                  distances_naive,
                                  distances_ivf_pq,
                                  ps.num_queries,
                                  ps.k,
                                  0.001,
                                  min_recall));
    }
  }

  void SetUp() override  // NOLINT
  {
    database.resize(ps.num_db_vecs * ps.dim, stream_);
    search_queries.resize(ps.num_queries * ps.dim, stream_);

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

  void TearDown() override  // NOLINT
  {
    cudaGetLastError();
    handle_.sync_stream(stream_);
    database.resize(0, stream_);
    search_queries.resize(0, stream_);
  }

 private:
  raft::handle_t handle_;
  rmm::cuda_stream_view stream_;
  ivf_pq_inputs ps;                           // NOLINT
  rmm::device_uvector<DataT> database;        // NOLINT
  rmm::device_uvector<DataT> search_queries;  // NOLINT
};

/* Type instantiations to test */
using f32_f32_u64 = ivf_pq_test<float, float, uint64_t>;
using f32_f32_i64 = ivf_pq_test<float, float, int64_t>;
using f32_f32_u32 = ivf_pq_test<float, float, uint32_t>;
using f32_u08_u64 = ivf_pq_test<float, uint8_t, uint64_t>;
using f32_i08_u64 = ivf_pq_test<float, int8_t, uint64_t>;

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

auto defaults() -> test_cases_t { return {ivf_pq_inputs{}}; }

template <typename B, typename A, typename F>
auto map(const std::vector<A>& xs, F f) -> std::vector<B>
{
  std::vector<B> ys(xs.size());
  std::transform(xs.begin(), xs.end(), ys.begin(), f);
  return ys;
}

auto with_dims(const std::vector<uint32_t>& dims) -> test_cases_t
{
  return map<ivf_pq_inputs>(dims, [](uint32_t d) {
    ivf_pq_inputs x;
    x.dim = d;
    return x;
  });
}

auto small_dims() -> test_cases_t { return with_dims({1, 2, 3, 4, 5, 8, 15, 16, 17}); }

auto big_dims() -> test_cases_t
{
  return map<ivf_pq_inputs>(with_dims({511, 512, 513, 1023, 1024, 1025, 2048, 2049, 2050, 2053}),
                            [](const ivf_pq_inputs& x) {
                              ivf_pq_inputs y(x);
                              // disable random rotation for now here, to reduce the exec time
                              // (TODO: move the matrix generation to GPU)
                              y.index_params.random_rotation = false;
                              return y;
                            });
}

auto enum_variety() -> test_cases_t
{
  test_cases_t xs;
#define ADD_CASE(f)                               \
  do {                                            \
    xs.push_back({});                             \
    ([](ivf_pq_inputs & x) f)(xs[xs.size() - 1]); \
  } while (0);

  ADD_CASE({ x.index_params.codebook_kind = ivf_pq::codebook_gen::PER_CLUSTER; });
  ADD_CASE({ x.index_params.codebook_kind = ivf_pq::codebook_gen::PER_SUBSPACE; });
  ADD_CASE({
    x.index_params.codebook_kind = ivf_pq::codebook_gen::PER_CLUSTER;
    x.index_params.pq_bits       = 4;
  });
  ADD_CASE({
    x.index_params.codebook_kind = ivf_pq::codebook_gen::PER_CLUSTER;
    x.index_params.pq_bits       = 5;
  });

  ADD_CASE({ x.index_params.pq_bits = 6; });
  ADD_CASE({ x.index_params.pq_bits = 7; });
  ADD_CASE({ x.index_params.pq_bits = 8; });
  ADD_CASE({ x.index_params.random_rotation = false; });
  ADD_CASE({ x.index_params.random_rotation = true; });

  ADD_CASE({ x.search_params.lut_dtype = CUDA_R_32F; });
  ADD_CASE({ x.search_params.lut_dtype = CUDA_R_16F; });
  ADD_CASE({ x.search_params.lut_dtype = CUDA_R_8U; });

  ADD_CASE({ x.search_params.internal_distance_dtype = CUDA_R_32F; });
  ADD_CASE({ x.search_params.internal_distance_dtype = CUDA_R_16F; });

  ADD_CASE({ x.search_params.preferred_thread_block_size = 256; });
  ADD_CASE({ x.search_params.preferred_thread_block_size = 512; });
  ADD_CASE({ x.search_params.preferred_thread_block_size = 1024; });

  return xs;
}

auto enum_variety_l2() -> test_cases_t
{
  return map<ivf_pq_inputs>(enum_variety(), [](const ivf_pq_inputs& x) {
    ivf_pq_inputs y(x);
    y.index_params.metric = distance::DistanceType::L2Expanded;
    return y;
  });
}

auto enum_variety_ip() -> test_cases_t
{
  return map<ivf_pq_inputs>(enum_variety(), [](const ivf_pq_inputs& x) {
    ivf_pq_inputs y(x);
    y.index_params.metric = distance::DistanceType::InnerProduct;
    return y;
  });
}

auto var_n_probes() -> test_cases_t
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

auto var_k() -> test_cases_t
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

/* Test instantiations */
#define INSTANTIATE_IVF_PQ(type, vals)                                           \
  TEST_P(type, run) { this->run(); }                                /* NOLINT */ \
  INSTANTIATE_TEST_SUITE_P(IvfPq, type, ::testing::ValuesIn(vals)); /* NOLINT */

INSTANTIATE_IVF_PQ(f32_f32_u64, defaults() + small_dims() + big_dims());
INSTANTIATE_IVF_PQ(f32_f32_i64, enum_variety_l2() + enum_variety_ip());
INSTANTIATE_IVF_PQ(f32_f32_u32, defaults() + var_n_probes() + var_k());
INSTANTIATE_IVF_PQ(f32_u08_u64, small_dims() + enum_variety());
INSTANTIATE_IVF_PQ(f32_i08_u64, defaults() + big_dims() + var_k());

}  // namespace raft::spatial::knn
