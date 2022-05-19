/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <test_utils.h>
#include <vector>

#include <raft/cluster/kmeans.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/stats/adjusted_rand_index.cuh>
#include <thrust/fill.h>

namespace raft {

template <typename T>
struct KmeansInputs {
  int n_row;
  int n_col;
  int n_clusters;
  T tol;
  bool weighted;
};

template <typename T>
class KmeansTest : public ::testing::TestWithParam<KmeansInputs<T>> {
 protected:
  KmeansTest()
    : d_labels(0, stream),
      d_labels_ref(0, stream),
      d_centroids(0, stream),
      d_sample_weight(0, stream)
  {
  }

  void basicTest()
  {
    raft::handle_t handle;
    testparams = ::testing::TestWithParam<KmeansInputs<T>>::GetParam();

    int n_samples              = testparams.n_row;
    int n_features             = testparams.n_col;
    params.n_clusters          = testparams.n_clusters;
    params.tol                 = testparams.tol;
    params.n_init              = 5;
    params.seed                = 1;
    params.oversampling_factor = 0;

    auto stream = handle.get_stream();
    auto X      = raft::make_device_matrix<T>(n_samples, n_features, stream);
    auto labels = raft::make_device_vector<int>(n_samples, stream);

    raft::random::make_blobs<T, int>(X.data(),
                                     labels.data(),
                                     n_samples,
                                     n_features,
                                     params.n_clusters,
                                     stream,
                                     true,
                                     nullptr,
                                     nullptr,
                                     T(1.0),
                                     false,
                                     (T)-10.0f,
                                     (T)10.0f,
                                     (uint64_t)1234);

    d_labels.resize(n_samples, stream);
    d_labels_ref.resize(n_samples, stream);
    d_centroids.resize(params.n_clusters * n_features, stream);

    std::optional<raft::device_vector_view<const T>> d_sw = std::nullopt;
    auto d_centroids_view =
      raft::make_device_matrix_view<T>(d_centroids.data(), params.n_clusters, n_features);
    if (testparams.weighted) {
      d_sample_weight.resize(n_samples, stream);
      d_sw = std::make_optional(
        raft::make_device_vector_view<const T>(d_sample_weight.data(), n_samples));
      thrust::fill(thrust::cuda::par.on(stream),
                   d_sample_weight.data(),
                   d_sample_weight.data() + n_samples,
                   1);
    }

    raft::copy(d_labels_ref.data(), labels.data(), n_samples, stream);
    handle.sync_stream(stream);

    T inertia   = 0;
    int n_iter  = 0;
    auto X_view = (raft::device_matrix_view<const T>)X.view();

    raft::cluster::kmeans_fit_predict<T, int>(
      handle,
      params,
      X_view,
      d_sw,
      d_centroids_view,
      raft::make_device_vector_view<int>(d_labels.data(), n_samples),
      raft::make_host_scalar_view(&inertia),
      raft::make_host_scalar_view(&n_iter));

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
  cudaStream_t stream = 0;
  KmeansInputs<T> testparams;
  rmm::device_uvector<int> d_labels;
  rmm::device_uvector<int> d_labels_ref;
  rmm::device_uvector<T> d_centroids;
  rmm::device_uvector<T> d_sample_weight;
  double score;
  raft::cluster::KMeansParams params;
};

const std::vector<KmeansInputs<float>> inputsf2 = {{1000, 32, 5, 0.0001, true},
                                                   {1000, 32, 5, 0.0001, false},
                                                   {1000, 100, 20, 0.0001, true},
                                                   {1000, 100, 20, 0.0001, false},
                                                   {10000, 32, 10, 0.0001, true},
                                                   {10000, 32, 10, 0.0001, false},
                                                   {10000, 100, 50, 0.0001, true},
                                                   {10000, 100, 50, 0.0001, false},
                                                   {10000, 1000, 200, 0.0001, true},
                                                   {10000, 1000, 200, 0.0001, false}};

const std::vector<KmeansInputs<double>> inputsd2 = {{1000, 32, 5, 0.0001, true},
                                                    {1000, 32, 5, 0.0001, false},
                                                    {1000, 100, 20, 0.0001, true},
                                                    {1000, 100, 20, 0.0001, false},
                                                    {10000, 32, 10, 0.0001, true},
                                                    {10000, 32, 10, 0.0001, false},
                                                    {10000, 100, 50, 0.0001, true},
                                                    {10000, 100, 50, 0.0001, false},
                                                    {10000, 1000, 200, 0.0001, true},
                                                    {10000, 1000, 200, 0.0001, false}};

typedef KmeansTest<float> KmeansTestF;
TEST_P(KmeansTestF, Result) { ASSERT_TRUE(score == 1.0); }

typedef KmeansTest<double> KmeansTestD;
TEST_P(KmeansTestD, Result) { ASSERT_TRUE(score == 1.0); }

INSTANTIATE_TEST_CASE_P(KmeansTests, KmeansTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(KmeansTests, KmeansTestD, ::testing::ValuesIn(inputsd2));

}  // namespace raft
