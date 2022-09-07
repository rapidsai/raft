/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include "spatial_data.h"
#include <raft/cudart_utils.h>
#include <raft/distance/distance_type.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/spatial/knn/ball_cover.cuh>
#include <raft/spatial/knn/detail/knn_brute_force_faiss.cuh>
#if defined RAFT_NN_COMPILED
#include <raft/spatial/knn/specializations.cuh>
#endif

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/transform.h>

#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

namespace raft {
namespace spatial {
namespace knn {

using namespace std;

template <typename value_idx, typename value_t>
__global__ void count_discrepancies_kernel(value_idx* actual_idx,
                                           value_idx* expected_idx,
                                           value_t* actual,
                                           value_t* expected,
                                           uint32_t m,
                                           uint32_t n,
                                           uint32_t* out,
                                           float thres = 1e-3)
{
  uint32_t row = blockDim.x * blockIdx.x + threadIdx.x;

  int n_diffs = 0;
  if (row < m) {
    for (uint32_t i = 0; i < n; i++) {
      value_t d    = actual[row * n + i] - expected[row * n + i];
      bool matches = (fabsf(d) <= thres) || (actual_idx[row * n + i] == expected_idx[row * n + i] &&
                                             actual_idx[row * n + i] == row);

      if (!matches) {
        printf(
          "row=%ud, n=%ud, actual_dist=%f, actual_ind=%ld, expected_dist=%f, expected_ind=%ld\n",
          row,
          i,
          actual[row * n + i],
          actual_idx[row * n + i],
          expected[row * n + i],
          expected_idx[row * n + i]);
      }
      n_diffs += !matches;
      out[row] = n_diffs;
    }
  }
}

struct is_nonzero {
  __host__ __device__ bool operator()(uint32_t& i) { return i > 0; }
};

template <typename value_idx, typename value_t>
uint32_t count_discrepancies(value_idx* actual_idx,
                             value_idx* expected_idx,
                             value_t* actual,
                             value_t* expected,
                             uint32_t m,
                             uint32_t n,
                             uint32_t* out,
                             cudaStream_t stream)
{
  uint32_t tpb = 256;
  count_discrepancies_kernel<<<raft::ceildiv(m, tpb), tpb, 0, stream>>>(
    actual_idx, expected_idx, actual, expected, m, n, out);

  auto exec_policy = rmm::exec_policy(stream);

  uint32_t result = thrust::count_if(exec_policy, out, out + m, is_nonzero());
  return result;
}

template <typename value_t>
void compute_bfknn(const raft::handle_t& handle,
                   const value_t* X1,
                   const value_t* X2,
                   uint32_t n_rows,
                   uint32_t n_query_rows,
                   uint32_t d,
                   uint32_t k,
                   const raft::distance::DistanceType metric,
                   value_t* dists,
                   int64_t* inds)
{
  std::vector<value_t*> input_vec = {const_cast<value_t*>(X1)};
  std::vector<uint32_t> sizes_vec = {n_rows};

  std::vector<int64_t>* translations = nullptr;

  raft::spatial::knn::detail::brute_force_knn_impl<uint32_t, int64_t>(handle,
                                                                      input_vec,
                                                                      sizes_vec,
                                                                      d,
                                                                      const_cast<value_t*>(X2),
                                                                      n_query_rows,
                                                                      inds,
                                                                      dists,
                                                                      k,
                                                                      true,
                                                                      true,
                                                                      translations,
                                                                      metric);
}

struct ToRadians {
  __device__ __host__ float operator()(float a) { return a * (CUDART_PI_F / 180.0); }
};

struct BallCoverInputs {
  uint32_t k;
  uint32_t n_rows;
  uint32_t n_cols;
  float weight;
  uint32_t n_query;
  raft::distance::DistanceType metric;
};

template <typename value_idx, typename value_t>
class BallCoverKNNQueryTest : public ::testing::TestWithParam<BallCoverInputs> {
 protected:
  void basicTest()
  {
    params = ::testing::TestWithParam<BallCoverInputs>::GetParam();
    raft::handle_t handle;

    uint32_t k         = params.k;
    uint32_t n_centers = 25;
    float weight       = params.weight;
    auto metric        = params.metric;

    rmm::device_uvector<value_t> X(params.n_rows * params.n_cols, handle.get_stream());
    rmm::device_uvector<uint32_t> Y(params.n_rows, handle.get_stream());

    // Make sure the train and query sets are completely disjoint
    rmm::device_uvector<value_t> X2(params.n_query * params.n_cols, handle.get_stream());
    rmm::device_uvector<uint32_t> Y2(params.n_query, handle.get_stream());

    raft::random::make_blobs(
      X.data(), Y.data(), params.n_rows, params.n_cols, n_centers, handle.get_stream());

    raft::random::make_blobs(
      X2.data(), Y2.data(), params.n_query, params.n_cols, n_centers, handle.get_stream());

    rmm::device_uvector<value_idx> d_ref_I(params.n_query * k, handle.get_stream());
    rmm::device_uvector<value_t> d_ref_D(params.n_query * k, handle.get_stream());

    if (metric == raft::distance::DistanceType::Haversine) {
      thrust::transform(
        handle.get_thrust_policy(), X.data(), X.data() + X.size(), X.data(), ToRadians());
      thrust::transform(
        handle.get_thrust_policy(), X2.data(), X2.data() + X2.size(), X2.data(), ToRadians());
    }

    compute_bfknn(handle,
                  X.data(),
                  X2.data(),
                  params.n_rows,
                  params.n_query,
                  params.n_cols,
                  k,
                  metric,
                  d_ref_D.data(),
                  d_ref_I.data());

    handle.sync_stream();

    // Allocate predicted arrays
    rmm::device_uvector<value_idx> d_pred_I(params.n_query * k, handle.get_stream());
    rmm::device_uvector<value_t> d_pred_D(params.n_query * k, handle.get_stream());

    BallCoverIndex<value_idx, value_t> index(
      handle, X.data(), params.n_rows, params.n_cols, metric);

    raft::spatial::knn::rbc_build_index(handle, index);
    raft::spatial::knn::rbc_knn_query(
      handle, index, k, X2.data(), params.n_query, d_pred_I.data(), d_pred_D.data(), true, weight);

    handle.sync_stream();
    // What we really want are for the distances to match exactly. The
    // indices may or may not match exactly, depending upon the ordering which
    // can be nondeterministic.

    rmm::device_uvector<uint32_t> discrepancies(params.n_query, handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 discrepancies.data(),
                 discrepancies.data() + discrepancies.size(),
                 0);
    //
    int res = count_discrepancies(d_ref_I.data(),
                                  d_pred_I.data(),
                                  d_ref_D.data(),
                                  d_pred_D.data(),
                                  params.n_query,
                                  k,
                                  discrepancies.data(),
                                  handle.get_stream());

    ASSERT_TRUE(res == 0);
  }

  void SetUp() override {}

  void TearDown() override {}

 protected:
  uint32_t d = 2;
  BallCoverInputs params;
};

template <typename value_idx, typename value_t>
class BallCoverAllKNNTest : public ::testing::TestWithParam<BallCoverInputs> {
 protected:
  void basicTest()
  {
    params = ::testing::TestWithParam<BallCoverInputs>::GetParam();
    raft::handle_t handle;

    uint32_t k         = params.k;
    uint32_t n_centers = 25;
    float weight       = params.weight;
    auto metric        = params.metric;

    rmm::device_uvector<value_t> X(params.n_rows * params.n_cols, handle.get_stream());
    rmm::device_uvector<uint32_t> Y(params.n_rows, handle.get_stream());

    raft::random::make_blobs(
      X.data(), Y.data(), params.n_rows, params.n_cols, n_centers, handle.get_stream());

    rmm::device_uvector<value_idx> d_ref_I(params.n_rows * k, handle.get_stream());
    rmm::device_uvector<value_t> d_ref_D(params.n_rows * k, handle.get_stream());

    if (metric == raft::distance::DistanceType::Haversine) {
      thrust::transform(
        handle.get_thrust_policy(), X.data(), X.data() + X.size(), X.data(), ToRadians());
    }

    compute_bfknn(handle,
                  X.data(),
                  X.data(),
                  params.n_rows,
                  params.n_rows,
                  params.n_cols,
                  k,
                  metric,
                  d_ref_D.data(),
                  d_ref_I.data());

    handle.sync_stream();

    // Allocate predicted arrays
    rmm::device_uvector<value_idx> d_pred_I(params.n_rows * k, handle.get_stream());
    rmm::device_uvector<value_t> d_pred_D(params.n_rows * k, handle.get_stream());

    BallCoverIndex<value_idx, value_t> index(
      handle, X.data(), params.n_rows, params.n_cols, metric);

    raft::spatial::knn::rbc_all_knn_query(
      handle, index, k, d_pred_I.data(), d_pred_D.data(), true, weight);

    handle.sync_stream();
    // What we really want are for the distances to match exactly. The
    // indices may or may not match exactly, depending upon the ordering which
    // can be nondeterministic.

    rmm::device_uvector<uint32_t> discrepancies(params.n_rows, handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 discrepancies.data(),
                 discrepancies.data() + discrepancies.size(),
                 0);
    //
    uint32_t res = count_discrepancies(d_ref_I.data(),
                                       d_pred_I.data(),
                                       d_ref_D.data(),
                                       d_pred_D.data(),
                                       params.n_rows,
                                       k,
                                       discrepancies.data(),
                                       handle.get_stream());

    // TODO: There seem to be discrepancies here only when
    // the entire test suite is executed.
    // Ref: https://github.com/rapidsai/raft/issues/
    // 1-5 mismatches in 8000 samples is 0.0125% - 0.0625%
    ASSERT_TRUE(res <= 5);
  }

  void SetUp() override {}

  void TearDown() override {}

 protected:
  BallCoverInputs params;
};

typedef BallCoverAllKNNTest<int64_t, float> BallCoverAllKNNTestF;
typedef BallCoverKNNQueryTest<int64_t, float> BallCoverKNNQueryTestF;

const std::vector<BallCoverInputs> ballcover_inputs = {
  {11, 5000, 2, 1.0, 10000, raft::distance::DistanceType::Haversine},
  {25, 10000, 2, 1.0, 5000, raft::distance::DistanceType::Haversine},
  {2, 10000, 2, 1.0, 5000, raft::distance::DistanceType::L2SqrtUnexpanded},
  {2, 5000, 2, 1.0, 10000, raft::distance::DistanceType::Haversine},
  {11, 10000, 2, 1.0, 5000, raft::distance::DistanceType::L2SqrtUnexpanded},
  {25, 5000, 2, 1.0, 10000, raft::distance::DistanceType::L2SqrtUnexpanded},
  {5, 8000, 3, 1.0, 10000, raft::distance::DistanceType::L2SqrtUnexpanded},
  {11, 6000, 3, 1.0, 10000, raft::distance::DistanceType::L2SqrtUnexpanded},
  {25, 10000, 3, 1.0, 5000, raft::distance::DistanceType::L2SqrtUnexpanded}};

INSTANTIATE_TEST_CASE_P(BallCoverAllKNNTest,
                        BallCoverAllKNNTestF,
                        ::testing::ValuesIn(ballcover_inputs));
INSTANTIATE_TEST_CASE_P(BallCoverKNNQueryTest,
                        BallCoverKNNQueryTestF,
                        ::testing::ValuesIn(ballcover_inputs));

TEST_P(BallCoverAllKNNTestF, Fit) { basicTest(); }
TEST_P(BallCoverKNNQueryTestF, Fit) { basicTest(); }

}  // namespace knn
}  // namespace spatial
}  // namespace raft
