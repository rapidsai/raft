/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

<<<<<<< HEAD
#include <gtest/gtest.h>
#include <raft/cudart_utils.h>
#include <raft/linalg/distance_type.h>
#include <iostream>
=======
#include <raft/cudart_utils.h>
#include <raft/linalg/distance_type.h>
>>>>>>> branch-21.10
#include <raft/spatial/knn/ball_cover.hpp>
#include <raft/spatial/knn/detail/haversine_distance.cuh>
#include <raft/spatial/knn/detail/knn_brute_force_faiss.cuh>
#include <rmm/device_uvector.hpp>
<<<<<<< HEAD
#include <vector>
=======
>>>>>>> branch-21.10
#include "../test_utils.h"
#include "spatial_data.h"

#include <thrust/transform.h>
#include <rmm/exec_policy.hpp>

<<<<<<< HEAD
=======
#include <gtest/gtest.h>
#include <cstdint>
#include <iostream>
#include <vector>

>>>>>>> branch-21.10
namespace raft {
namespace spatial {
namespace knn {

using namespace std;

template <typename value_idx, typename value_t>
__global__ void count_discrepancies_kernel(value_idx *actual_idx,
                                           value_idx *expected_idx,
                                           value_t *actual, value_t *expected,
<<<<<<< HEAD
                                           int m, int n, int *out,
                                           float thres = 1e-1) {
  int row = blockDim.x * blockIdx.x + threadIdx.x;

  int n_diffs = 0;
  if (row < m) {
    for (int i = 0; i < n; i++) {
      value_t d = actual[row * n + i] - expected[row * n + i];
      bool matches = fabsf(d) <= thres;
=======
                                           uint32_t m, uint32_t n,
                                           uint32_t *out, float thres = 1e-3) {
  uint32_t row = blockDim.x * blockIdx.x + threadIdx.x;

  int n_diffs = 0;
  if (row < m) {
    for (uint32_t i = 0; i < n; i++) {
      value_t d = actual[row * n + i] - expected[row * n + i];
      bool matches = fabsf(d) <= thres;
      if (!matches) {
        //          printf("row=%d, actual_idx=%ld, actual=%f, expected_id=%ld, expected=%f\n",
        //                 row, actual_idx[row*n+i], actual[row*n+i], expected_idx[row*n+i], expected[row*n+i]);
      }

>>>>>>> branch-21.10
      n_diffs += !matches;
      out[row] = n_diffs;
    }
  }
}

struct is_nonzero {
<<<<<<< HEAD
  __host__ __device__ bool operator()(int &i) { return i > 0; }
};

template <typename value_idx, typename value_t>
int count_discrepancies(value_idx *actual_idx, value_idx *expected_idx,
                        value_t *actual, value_t *expected, int m, int n,
                        int *out, cudaStream_t stream) {
  count_discrepancies_kernel<<<raft::ceildiv(m, 256), 256, 0, stream>>>(
=======
  __host__ __device__ bool operator()(uint32_t &i) { return i > 0; }
};

template <typename value_idx, typename value_t>
uint32_t count_discrepancies(value_idx *actual_idx, value_idx *expected_idx,
                             value_t *actual, value_t *expected, uint32_t m,
                             uint32_t n, uint32_t *out, cudaStream_t stream) {
  uint32_t tpb = 256;
  count_discrepancies_kernel<<<raft::ceildiv(m, tpb), tpb, 0, stream>>>(
>>>>>>> branch-21.10
    actual_idx, expected_idx, actual, expected, m, n, out);

  auto exec_policy = rmm::exec_policy(stream);

<<<<<<< HEAD
  int result = thrust::count_if(exec_policy, out, out + m, is_nonzero());
=======
  uint32_t result = thrust::count_if(exec_policy, out, out + m, is_nonzero());
>>>>>>> branch-21.10
  return result;
}

struct ToRadians {
  __device__ __host__ float operator()(float a) {
    return a * (CUDART_PI_F / 180.0);
  }
};

struct BallCoverInputs {
<<<<<<< HEAD
  int k = 2;
  float weight = 1.0;
  raft::distance::DistanceType metric = raft::distance::DistanceType::Haversine;
=======
  uint32_t k;
  float weight;
  raft::distance::DistanceType metric;
>>>>>>> branch-21.10
};

template <typename value_idx, typename value_t>
class BallCoverKNNQueryTest : public ::testing::TestWithParam<BallCoverInputs> {
 protected:
  void basicTest() {
    params = ::testing::TestWithParam<BallCoverInputs>::GetParam();
    raft::handle_t handle;

<<<<<<< HEAD
    int k = params.k;
    int weight = params.weight;
=======
    uint32_t k = params.k;
    float weight = params.weight;
>>>>>>> branch-21.10
    auto metric = params.metric;

    std::vector<value_t> h_train_inputs = spatial_data;

<<<<<<< HEAD
    int n = h_train_inputs.size() / d;
=======
    uint32_t n = h_train_inputs.size() / d;
>>>>>>> branch-21.10

    rmm::device_uvector<value_idx> d_ref_I(n * k, handle.get_stream());
    rmm::device_uvector<value_t> d_ref_D(n * k, handle.get_stream());

    // Allocate input
    rmm::device_uvector<value_t> d_train_inputs(n * d, handle.get_stream());
    raft::update_device(d_train_inputs.data(), h_train_inputs.data(), n * d,
                        handle.get_stream());

    if (metric == raft::distance::DistanceType::Haversine) {
      thrust::transform(handle.get_thrust_policy(), d_train_inputs.data(),
                        d_train_inputs.data() + d_train_inputs.size(),
                        d_train_inputs.data(), ToRadians());
    }

    cudaStream_t *int_streams = nullptr;
    std::vector<int64_t> *translations = nullptr;

    std::vector<float *> input_vec = {d_train_inputs.data()};
<<<<<<< HEAD
    std::vector<int> sizes_vec = {n};

    raft::spatial::knn::detail::brute_force_knn_impl<int, int64_t>(
=======
    std::vector<uint32_t> sizes_vec = {n};

    raft::spatial::knn::detail::brute_force_knn_impl<uint32_t, int64_t>(
>>>>>>> branch-21.10
      input_vec, sizes_vec, d, d_train_inputs.data(), n, d_ref_I.data(),
      d_ref_D.data(), k, handle.get_stream(), int_streams, 0, true, true,
      translations, metric);

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    // Allocate predicted arrays
    rmm::device_uvector<value_idx> d_pred_I(n * k, handle.get_stream());
    rmm::device_uvector<value_t> d_pred_D(n * k, handle.get_stream());

    BallCoverIndex<value_idx, value_t> index(handle, d_train_inputs.data(), n,
                                             d, metric);

<<<<<<< HEAD
    raft::spatial::knn::rbc_build_index(handle, index, k);
=======
    raft::spatial::knn::rbc_build_index(handle, index);
>>>>>>> branch-21.10
    raft::spatial::knn::rbc_knn_query(handle, index, k, d_train_inputs.data(),
                                      n, d_pred_I.data(), d_pred_D.data(), true,
                                      weight);

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
    // What we really want are for the distances to match exactly. The
    // indices may or may not match exactly, depending upon the ordering which
    // can be nondeterministic.

<<<<<<< HEAD
    rmm::device_uvector<int> discrepancies(n, handle.get_stream());
=======
    rmm::device_uvector<uint32_t> discrepancies(n, handle.get_stream());
>>>>>>> branch-21.10
    thrust::fill(handle.get_thrust_policy(), discrepancies.data(),
                 discrepancies.data() + discrepancies.size(), 0);
    //
    int res = count_discrepancies(d_ref_I.data(), d_pred_I.data(),
                                  d_ref_D.data(), d_pred_D.data(), n, k,
                                  discrepancies.data(), handle.get_stream());

<<<<<<< HEAD
    printf("res=%d\n", res);

=======
>>>>>>> branch-21.10
    ASSERT_TRUE(res == 0);
  }

  void SetUp() override {}

  void TearDown() override {}

 protected:
<<<<<<< HEAD
  int d = 2;
=======
  uint32_t d = 2;
>>>>>>> branch-21.10
  BallCoverInputs params;
};

template <typename value_idx, typename value_t>
class BallCoverAllKNNTest : public ::testing::TestWithParam<BallCoverInputs> {
 protected:
  void basicTest() {
    params = ::testing::TestWithParam<BallCoverInputs>::GetParam();
    raft::handle_t handle;

<<<<<<< HEAD
    int k = params.k;
    int weight = params.weight;
=======
    uint32_t k = params.k;
    float weight = params.weight;
>>>>>>> branch-21.10
    auto metric = params.metric;

    std::vector<value_t> h_train_inputs = spatial_data;

<<<<<<< HEAD
    int n = h_train_inputs.size() / d;
=======
    uint32_t n = h_train_inputs.size() / d;
>>>>>>> branch-21.10

    rmm::device_uvector<value_idx> d_ref_I(n * k, handle.get_stream());
    rmm::device_uvector<value_t> d_ref_D(n * k, handle.get_stream());

    // Allocate input
    rmm::device_uvector<value_t> d_train_inputs(n * d, handle.get_stream());
    raft::update_device(d_train_inputs.data(), h_train_inputs.data(), n * d,
                        handle.get_stream());

    if (metric == raft::distance::DistanceType::Haversine) {
      thrust::transform(handle.get_thrust_policy(), d_train_inputs.data(),
                        d_train_inputs.data() + d_train_inputs.size(),
                        d_train_inputs.data(), ToRadians());
    }

    cudaStream_t *int_streams = nullptr;
    std::vector<int64_t> *translations = nullptr;

    std::vector<float *> input_vec = {d_train_inputs.data()};
<<<<<<< HEAD
    std::vector<int> sizes_vec = {n};

    raft::spatial::knn::detail::brute_force_knn_impl<int, int64_t>(
=======
    std::vector<uint32_t> sizes_vec = {n};

    raft::spatial::knn::detail::brute_force_knn_impl<uint32_t, int64_t>(
>>>>>>> branch-21.10
      input_vec, sizes_vec, d, d_train_inputs.data(), n, d_ref_I.data(),
      d_ref_D.data(), k, handle.get_stream(), int_streams, 0, true, true,
      translations, metric);

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    // Allocate predicted arrays
    rmm::device_uvector<value_idx> d_pred_I(n * k, handle.get_stream());
    rmm::device_uvector<value_t> d_pred_D(n * k, handle.get_stream());

    BallCoverIndex<value_idx, value_t> index(handle, d_train_inputs.data(), n,
                                             d, metric);

    raft::spatial::knn::rbc_all_knn_query(handle, index, k, d_pred_I.data(),
                                          d_pred_D.data(), true, weight);

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
    // What we really want are for the distances to match exactly. The
    // indices may or may not match exactly, depending upon the ordering which
    // can be nondeterministic.

<<<<<<< HEAD
    rmm::device_uvector<int> discrepancies(n, handle.get_stream());
    thrust::fill(handle.get_thrust_policy(), discrepancies.data(),
                 discrepancies.data() + discrepancies.size(), 0);
    //
    int res = count_discrepancies(d_ref_I.data(), d_pred_I.data(),
                                  d_ref_D.data(), d_pred_D.data(), n, k,
                                  discrepancies.data(), handle.get_stream());
=======
    rmm::device_uvector<uint32_t> discrepancies(n, handle.get_stream());
    thrust::fill(handle.get_thrust_policy(), discrepancies.data(),
                 discrepancies.data() + discrepancies.size(), 0);
    //
    uint32_t res = count_discrepancies(
      d_ref_I.data(), d_pred_I.data(), d_ref_D.data(), d_pred_D.data(), n, k,
      discrepancies.data(), handle.get_stream());
>>>>>>> branch-21.10
    ASSERT_TRUE(res == 0);
  }

  void SetUp() override {}

  void TearDown() override {}

 protected:
<<<<<<< HEAD
  int d = 2;
=======
  uint32_t d = 2;
>>>>>>> branch-21.10
  BallCoverInputs params;
};

typedef BallCoverAllKNNTest<int64_t, float> BallCoverAllKNNTestF;
typedef BallCoverKNNQueryTest<int64_t, float> BallCoverKNNQueryTestF;

const std::vector<BallCoverInputs> ballcover_inputs = {
  {2, 1.0, raft::distance::DistanceType::Haversine},
<<<<<<< HEAD
  {7, 1.0, raft::distance::DistanceType::Haversine},
  {64, 1.0, raft::distance::DistanceType::Haversine},
  {2, 1.0, raft::distance::DistanceType::L2Unexpanded},
  {7, 1.0, raft::distance::DistanceType::L2Unexpanded},
  {64, 1.0, raft::distance::DistanceType::L2Unexpanded},
  {2, 1.0, raft::distance::DistanceType::L2SqrtUnexpanded},
  {7, 1.0, raft::distance::DistanceType::L2SqrtUnexpanded},
  {64, 1.0, raft::distance::DistanceType::L2SqrtUnexpanded},
=======
  {4, 1.0, raft::distance::DistanceType::Haversine},
  {7, 1.0, raft::distance::DistanceType::Haversine},
  {2, 1.0, raft::distance::DistanceType::L2SqrtUnexpanded},
  {4, 1.0, raft::distance::DistanceType::L2SqrtUnexpanded},
  {7, 1.0, raft::distance::DistanceType::L2SqrtUnexpanded},
>>>>>>> branch-21.10
};

INSTANTIATE_TEST_CASE_P(BallCoverAllKNNTest, BallCoverAllKNNTestF,
                        ::testing::ValuesIn(ballcover_inputs));
INSTANTIATE_TEST_CASE_P(BallCoverKNNQueryTest, BallCoverKNNQueryTestF,
                        ::testing::ValuesIn(ballcover_inputs));

TEST_P(BallCoverAllKNNTestF, Fit) { basicTest(); }
TEST_P(BallCoverKNNQueryTestF, Fit) { basicTest(); }

}  // namespace knn
}  // namespace spatial
}  // namespace raft
