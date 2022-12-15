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
#include <gtest/gtest.h>
#include <optional>
#include <vector>

#include <raft/cluster/kmeans_balanced.cuh>
#include <raft/core/cudart_utils.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/stats/adjusted_rand_index.cuh>
#include <raft/util/cuda_utils.cuh>
#include <rmm/device_uvector.hpp>
#include <thrust/fill.h>

#if defined RAFT_DISTANCE_COMPILED
#include <raft/distance/specializations.cuh>
#endif

/* This test takes advantage of the fact that make_blobs generates balanced clusters.
 * It doesn't currently test whether the algorithm can make balanced clusters with an imbalanced
 * dataset.
 */

namespace raft {

template <typename T>
struct KmeansBalancedInputs {
  int n_row;
  int n_col;
  int n_clusters;
  int max_iter;
  T tol;
};

template <typename T>
class KmeansBalancedTest : public ::testing::TestWithParam<KmeansBalancedInputs<T>> {
 protected:
  KmeansBalancedTest()
    : stream(handle.get_stream()),
      d_labels(0, stream),
      d_labels_ref(0, stream),
      d_centroids(0, stream)
  {
  }

  void basicTest()
  {
    testparams = ::testing::TestWithParam<KmeansBalancedInputs<T>>::GetParam();

    int n_samples              = testparams.n_row;
    int n_features             = testparams.n_col;
    params.n_clusters          = testparams.n_clusters;
    params.tol                 = testparams.tol;
    params.n_init              = 5;
    params.rng_state.seed      = 1;
    params.oversampling_factor = 0;

    auto X      = raft::make_device_matrix<T, int>(handle, n_samples, n_features);
    auto labels = raft::make_device_vector<int, int>(handle, n_samples);

    raft::random::make_blobs<T, int>(X.data_handle(),
                                     labels.data_handle(),
                                     n_samples,
                                     n_features,
                                     params.n_clusters,
                                     stream,
                                     true,
                                     nullptr,
                                     nullptr,
                                     T(1.0),
                                     true,
                                     (T)-10.0f,
                                     (T)10.0f,
                                     (uint64_t)1234);

    d_labels.resize(n_samples, stream);
    d_labels_ref.resize(n_samples, stream);
    d_centroids.resize(params.n_clusters * n_features, stream);

    raft::copy(d_labels_ref.data(), labels.data_handle(), n_samples, stream);

    auto X_view =
      raft::make_device_matrix_view<const T, int>(X.data_handle(), X.extent(0), X.extent(1));
    auto d_centroids_view =
      raft::make_device_matrix_view<T, int>(d_centroids.data(), params.n_clusters, n_features);
    auto d_labels_view = raft::make_device_vector_view<int, int>(d_labels.data(), n_samples);

    // todo: pass metric, mapping_op
    raft::cluster::kmeans_balanced::fit_predict(
      handle, X_view, d_centroids_view, d_labels_view, params.max_iter);

    handle.sync_stream(stream);

    score = raft::stats::adjusted_rand_index(
      d_labels_ref.data(), d_labels.data(), n_samples, handle.get_stream());

    if (score < 1.0) {
      std::stringstream ss;
      ss << "Expected: " << raft::arr2Str(d_labels_ref.data(), 25, "d_labels_ref", stream);
      std::cout << (ss.str().c_str()) << '\n';
      ss.str(std::string());
      ss << "Actual: " << raft::arr2Str(d_labels.data(), 25, "d_labels", stream);
      std::cout << (ss.str().c_str()) << '\n';
      std::cout << "Score = " << score << '\n';
    }
  }

  void SetUp() override { basicTest(); }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;
  KmeansBalancedInputs<T> testparams;
  rmm::device_uvector<int> d_labels;
  rmm::device_uvector<int> d_labels_ref;
  rmm::device_uvector<T> d_centroids;
  double score;
  raft::cluster::KMeansParams params;
};

const std::vector<KmeansBalancedInputs<float>> inputsf2 = {{1000, 32, 5, 20, 0.0001f},
                                                           {1000, 100, 20, 20, 0.0001f},
                                                           {10000, 32, 10, 20, 0.0001f},
                                                           {10000, 100, 50, 20, 0.0001f},
                                                           {10000, 500, 100, 20, 0.0001f}};

const std::vector<KmeansBalancedInputs<double>> inputsd2 = {{1000, 32, 5, 20, 0.0001},
                                                            {1000, 100, 20, 20, 0.0001},
                                                            {10000, 32, 10, 20, 0.0001},
                                                            {10000, 100, 50, 20, 0.0001},
                                                            {10000, 500, 100, 20, 0.0001}};

typedef KmeansBalancedTest<float> KmeansBalancedTestF;
TEST_P(KmeansBalancedTestF, Result) { ASSERT_TRUE(score == 1.0); }

typedef KmeansBalancedTest<double> KmeansBalancedTestD;
TEST_P(KmeansBalancedTestD, Result) { ASSERT_TRUE(score == 1.0); }

INSTANTIATE_TEST_CASE_P(KmeansBalancedTests, KmeansBalancedTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(KmeansBalancedTests, KmeansBalancedTestD, ::testing::ValuesIn(inputsd2));

}  // namespace raft
