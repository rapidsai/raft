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

#include <gtest/gtest.h>
#include <raft/cudart_utils.h>
#include <raft/linalg/distance_type.h>
#include <iostream>
#include <raft/spatial/knn/ball_cover.hpp>
#include <raft/spatial/knn/detail/haversine_distance.cuh>
#include <raft/spatial/knn/detail/knn_brute_force_faiss.cuh>
#include <rmm/device_uvector.hpp>
#include <vector>
#include "../test_utils.h"
#include "spatial_data.h"

#include <thrust/transform.h>
#include <rmm/exec_policy.hpp>

namespace raft {
namespace spatial {
namespace knn {

using namespace std;

template <typename value_idx, typename value_t>
__global__ void count_discrepancies_kernel(value_idx *actual_idx,
                                           value_idx *expected_idx,
                                           value_t *actual, value_t *expected,
                                           int m, int n, int *out,
                                           float thres = 1e-1) {
  int row = blockDim.x * blockIdx.x + threadIdx.x;

  int n_diffs = 0;
  if (row < m) {
    for (int i = 0; i < n; i++) {
      value_t d = actual[row * n + i] - expected[row * n + i];
      bool matches = fabsf(d) <= thres;
      n_diffs += !matches;
      //      if(!matches)
      //        printf("Diff in idx=%d, expected=%ld, actual=%ld, dist1=%f, dist2=%f\n",
      //               row, expected_idx[row*n+i], actual_idx[row*n+i], expected[row*n+i], actual[row*n+i]);
      out[row] = n_diffs;
    }
  }
}

struct is_nonzero {
  __host__ __device__ bool operator()(int &i) { return i > 0; }
};

template <typename value_idx, typename value_t>
int count_discrepancies(value_idx *actual_idx, value_idx *expected_idx,
                        value_t *actual, value_t *expected, int m, int n,
                        int *out, cudaStream_t stream) {
  count_discrepancies_kernel<<<raft::ceildiv(m, 256), 256, 0, stream>>>(
    actual_idx, expected_idx, actual, expected, m, n, out);

  auto exec_policy = rmm::exec_policy(stream);

  int result = thrust::count_if(exec_policy, out, out + m, is_nonzero());
  return result;
}

struct ToRadians {
  __device__ __host__ float operator()(float a) {
    return a * (CUDART_PI_F / 180.0);
  }
};

template <typename value_idx, typename value_t>
class BallCoverKNNTest : public ::testing::Test {
 protected:
  void basicTest() {
    raft::handle_t handle;

    auto exec_policy = rmm::exec_policy(handle.get_stream());

    cout << "Reading CSV" << endl;

    int dim_mult = 1;
    d = d * dim_mult;

    std::vector<value_t> h_train_inputs = spatial_data;

    int n = h_train_inputs.size() / d;

    rmm::device_uvector<value_idx> d_ref_I(n * k, handle.get_stream());
    rmm::device_uvector<value_t> d_ref_D(n * k, handle.get_stream());

    cout << "Done" << endl;

    cout << "n_inputs " << n << endl;

    // Allocate input
    rmm::device_uvector<value_t> d_train_inputs(n * d, handle.get_stream());
    raft::update_device(d_train_inputs.data(), h_train_inputs.data(), n * d,
                        handle.get_stream());

    /**
     * FOr haversine, convert degrees to radians
     */
    thrust::transform(exec_policy, d_train_inputs.data(),
                      d_train_inputs.data() + d_train_inputs.size(),
                      d_train_inputs.data(), ToRadians());

    cout << "Calling brute force knn " << endl;

    /**
     * Execute brute-force knn to get reference data
     */

    cudaStream_t *int_streams = nullptr;
    std::vector<int64_t> *translations = nullptr;
    auto bfknn_start = curTimeMillis();

    std::vector<float *> input_vec = {d_train_inputs.data()};
    std::vector<int> sizes_vec = {n};

    raft::spatial::knn::detail::brute_force_knn_impl<int, int64_t>(
      input_vec, sizes_vec, d, d_train_inputs.data(), n, d_ref_I.data(),
      d_ref_D.data(), k, handle.get_stream(), int_streams, 0, true, true,
      translations, raft::distance::DistanceType::Haversine);

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
    cout << "Done in: " << curTimeMillis() - bfknn_start << "ms." << endl;

    // Allocate predicted arrays
    rmm::device_uvector<value_idx> d_pred_I(n * k, handle.get_stream());
    rmm::device_uvector<value_t> d_pred_D(n * k, handle.get_stream());

    cout << "Calling ball cover" << endl;
    auto rbc_start = curTimeMillis();

    BallCoverIndex<value_idx, value_t> index(
      handle, d_train_inputs.data(), n, d,
      raft::distance::DistanceType::Haversine);

    printf("n_landmarks=%d\n", index.n_landmarks);

    float weight = 1.0;
    raft::spatial::knn::rbc_all_knn_query(handle, index, k, d_pred_I.data(),
                                          d_pred_D.data(), true, weight);

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    cout << "Done in: " << curTimeMillis() - rbc_start << "ms." << endl;

    printf("Done.\n");

    // What we really want are for the distances to match exactly. The
    // indices may or may not match exactly, depending upon the ordering which
    // can be nondeterministic.

    /**
     * Evaluate discrepancies for debugging
     */

    rmm::device_uvector<int> discrepancies(n, handle.get_stream());
    thrust::fill(exec_policy, discrepancies.data(),
                 discrepancies.data() + discrepancies.size(), 0);
    //
    int res = count_discrepancies(d_ref_I.data(), d_pred_I.data(),
                                  d_ref_D.data(), d_pred_D.data(), n, k,
                                  discrepancies.data(), handle.get_stream());

    printf("res=%d\n", res);

    ASSERT_TRUE(res == 0);
  }

  void SetUp() override {}

  void TearDown() override {}

 protected:
  int d = 2;
  int k = 64;
};

typedef BallCoverKNNTest<int64_t, float> BallCoverKNNTestF;

TEST_F(BallCoverKNNTestF, Fit) { basicTest(); }

}  // namespace knn
}  // namespace spatial
}  // namespace raft
