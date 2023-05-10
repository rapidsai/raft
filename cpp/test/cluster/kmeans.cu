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

#include "../test_utils.cuh"
#include <gtest/gtest.h>
#include <optional>
#include <raft/core/resource/cuda_stream.hpp>
#include <vector>

#include <raft/cluster/kmeans.cuh>
#include <raft/core/cudart_utils.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/stats/adjusted_rand_index.cuh>
#include <raft/util/cuda_utils.cuh>
#include <rmm/device_uvector.hpp>
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

template <typename DataT, typename IndexT>
void run_cluster_cost(const raft::resources& handle,
                      raft::device_vector_view<DataT, IndexT> minClusterDistance,
                      rmm::device_uvector<char>& workspace,
                      raft::device_scalar_view<DataT> clusterCost)
{
  raft::cluster::kmeans::cluster_cost(
    handle, minClusterDistance, workspace, clusterCost, raft::add_op{});
}

template <typename T>
class KmeansTest : public ::testing::TestWithParam<KmeansInputs<T>> {
 protected:
  KmeansTest()
    : d_labels(0, resource::get_cuda_stream(handle)),
      d_labels_ref(0, resource::get_cuda_stream(handle)),
      d_centroids(0, resource::get_cuda_stream(handle)),
      d_sample_weight(0, resource::get_cuda_stream(handle))
  {
  }

  void apiTest()
  {
    testparams = ::testing::TestWithParam<KmeansInputs<T>>::GetParam();

    auto stream                = resource::get_cuda_stream(handle);
    int n_samples              = testparams.n_row;
    int n_features             = testparams.n_col;
    params.n_clusters          = testparams.n_clusters;
    params.tol                 = testparams.tol;
    params.n_init              = 1;
    params.rng_state.seed      = 1;
    params.oversampling_factor = 0;

    raft::random::RngState rng(params.rng_state.seed, params.rng_state.type);

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
                                     false,
                                     (T)-10.0f,
                                     (T)10.0f,
                                     (uint64_t)1234);
    d_labels.resize(n_samples, stream);
    d_labels_ref.resize(n_samples, stream);
    d_centroids.resize(params.n_clusters * n_features, stream);
    raft::copy(d_labels_ref.data(), labels.data_handle(), n_samples, stream);
    rmm::device_uvector<T> d_sample_weight(n_samples, stream);
    thrust::fill(
      thrust::cuda::par.on(stream), d_sample_weight.data(), d_sample_weight.data() + n_samples, 1);
    auto weight_view =
      raft::make_device_vector_view<const T, int>(d_sample_weight.data(), n_samples);

    T inertia  = 0;
    int n_iter = 0;
    rmm::device_uvector<char> workspace(0, stream);
    rmm::device_uvector<T> L2NormBuf_OR_DistBuf(0, stream);
    rmm::device_uvector<T> inRankCp(0, stream);
    auto X_view = raft::make_const_mdspan(X.view());
    auto centroids_view =
      raft::make_device_matrix_view<T, int>(d_centroids.data(), params.n_clusters, n_features);
    auto miniX = raft::make_device_matrix<T, int>(handle, n_samples / 4, n_features);

    // Initialize kmeans on a portion of X
    raft::cluster::kmeans::shuffle_and_gather(
      handle,
      X_view,
      raft::make_device_matrix_view<T, int>(miniX.data_handle(), miniX.extent(0), miniX.extent(1)),
      miniX.extent(0),
      params.rng_state.seed);

    raft::cluster::kmeans::init_plus_plus(
      handle, params, raft::make_const_mdspan(miniX.view()), centroids_view, workspace);

    auto minClusterDistance = raft::make_device_vector<T, int>(handle, n_samples);
    auto minClusterAndDistance =
      raft::make_device_vector<raft::KeyValuePair<int, T>, int>(handle, n_samples);
    auto L2NormX           = raft::make_device_vector<T, int>(handle, n_samples);
    auto clusterCostBefore = raft::make_device_scalar<T>(handle, 0);
    auto clusterCostAfter  = raft::make_device_scalar<T>(handle, 0);

    raft::linalg::rowNorm(L2NormX.data_handle(),
                          X.data_handle(),
                          X.extent(1),
                          X.extent(0),
                          raft::linalg::L2Norm,
                          true,
                          stream);

    raft::cluster::kmeans::min_cluster_distance(handle,
                                                X_view,
                                                centroids_view,
                                                minClusterDistance.view(),
                                                L2NormX.view(),
                                                L2NormBuf_OR_DistBuf,
                                                params.metric,
                                                params.batch_samples,
                                                params.batch_centroids,
                                                workspace);

    run_cluster_cost(handle, minClusterDistance.view(), workspace, clusterCostBefore.view());

    // Run a fit of kmeans
    raft::cluster::kmeans::fit_main(handle,
                                    params,
                                    X_view,
                                    weight_view,
                                    centroids_view,
                                    raft::make_host_scalar_view(&inertia),
                                    raft::make_host_scalar_view(&n_iter),
                                    workspace);

    // Check that the cluster cost decreased
    raft::cluster::kmeans::min_cluster_distance(handle,
                                                X_view,
                                                centroids_view,
                                                minClusterDistance.view(),
                                                L2NormX.view(),
                                                L2NormBuf_OR_DistBuf,
                                                params.metric,
                                                params.batch_samples,
                                                params.batch_centroids,
                                                workspace);

    run_cluster_cost(handle, minClusterDistance.view(), workspace, clusterCostAfter.view());
    T h_clusterCostBefore = T(0);
    T h_clusterCostAfter  = T(0);
    raft::update_host(&h_clusterCostBefore, clusterCostBefore.data_handle(), 1, stream);
    raft::update_host(&h_clusterCostAfter, clusterCostAfter.data_handle(), 1, stream);
    ASSERT_TRUE(h_clusterCostAfter < h_clusterCostBefore);

    // Count samples in clusters using 2 methods and compare them
    // Fill minClusterAndDistance
    raft::cluster::kmeans::min_cluster_and_distance(
      handle,
      X_view,
      raft::make_device_matrix_view<const T, int>(
        d_centroids.data(), params.n_clusters, n_features),
      minClusterAndDistance.view(),
      L2NormX.view(),
      L2NormBuf_OR_DistBuf,
      params.metric,
      params.batch_samples,
      params.batch_centroids,
      workspace);
    raft::cluster::kmeans::KeyValueIndexOp<int, T> conversion_op;
    cub::TransformInputIterator<int,
                                raft::cluster::kmeans::KeyValueIndexOp<int, T>,
                                raft::KeyValuePair<int, T>*>
      itr(minClusterAndDistance.data_handle(), conversion_op);

    auto sampleCountInCluster = raft::make_device_vector<T, int>(handle, params.n_clusters);
    auto weigthInCluster      = raft::make_device_vector<T, int>(handle, params.n_clusters);
    auto newCentroids = raft::make_device_matrix<T, int>(handle, params.n_clusters, n_features);
    raft::cluster::kmeans::update_centroids(handle,
                                            X_view,
                                            weight_view,
                                            raft::make_device_matrix_view<const T, int>(
                                              d_centroids.data(), params.n_clusters, n_features),
                                            itr,
                                            weigthInCluster.view(),
                                            newCentroids.view());
    raft::cluster::kmeans::count_samples_in_cluster(handle,
                                                    params,
                                                    X_view,
                                                    L2NormX.view(),
                                                    newCentroids.view(),
                                                    workspace,
                                                    sampleCountInCluster.view());

    ASSERT_TRUE(devArrMatch(sampleCountInCluster.data_handle(),
                            weigthInCluster.data_handle(),
                            params.n_clusters,
                            CompareApprox<T>(params.tol)));
  }

  void basicTest()
  {
    testparams = ::testing::TestWithParam<KmeansInputs<T>>::GetParam();

    int n_samples              = testparams.n_row;
    int n_features             = testparams.n_col;
    params.n_clusters          = testparams.n_clusters;
    params.tol                 = testparams.tol;
    params.n_init              = 5;
    params.rng_state.seed      = 1;
    params.oversampling_factor = 0;

    auto X      = raft::make_device_matrix<T, int>(handle, n_samples, n_features);
    auto labels = raft::make_device_vector<int, int>(handle, n_samples);
    auto stream = resource::get_cuda_stream(handle);

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
                                     false,
                                     (T)-10.0f,
                                     (T)10.0f,
                                     (uint64_t)1234);

    d_labels.resize(n_samples, stream);
    d_labels_ref.resize(n_samples, stream);
    d_centroids.resize(params.n_clusters * n_features, stream);

    std::optional<raft::device_vector_view<const T>> d_sw = std::nullopt;
    auto d_centroids_view =
      raft::make_device_matrix_view<T, int>(d_centroids.data(), params.n_clusters, n_features);
    if (testparams.weighted) {
      d_sample_weight.resize(n_samples, stream);
      d_sw = std::make_optional(
        raft::make_device_vector_view<const T, int>(d_sample_weight.data(), n_samples));
      thrust::fill(thrust::cuda::par.on(stream),
                   d_sample_weight.data(),
                   d_sample_weight.data() + n_samples,
                   1);
    }

    raft::copy(d_labels_ref.data(), labels.data_handle(), n_samples, stream);

    T inertia   = 0;
    int n_iter  = 0;
    auto X_view = raft::make_const_mdspan(X.view());

    raft::cluster::kmeans_fit_predict<T, int>(
      handle,
      params,
      X_view,
      d_sw,
      d_centroids_view,
      raft::make_device_vector_view<int, int>(d_labels.data(), n_samples),
      raft::make_host_scalar_view<T>(&inertia),
      raft::make_host_scalar_view<int>(&n_iter));

    resource::sync_stream(handle, stream);

    score = raft::stats::adjusted_rand_index(
      d_labels_ref.data(), d_labels.data(), n_samples, resource::get_cuda_stream(handle));

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

  void SetUp() override
  {
    basicTest();
    apiTest();
  }

 protected:
  raft::resources handle;
  KmeansInputs<T> testparams;
  rmm::device_uvector<int> d_labels;
  rmm::device_uvector<int> d_labels_ref;
  rmm::device_uvector<T> d_centroids;
  rmm::device_uvector<T> d_sample_weight;
  double score;
  raft::cluster::KMeansParams params;
};

const std::vector<KmeansInputs<float>> inputsf2 = {{1000, 32, 5, 0.0001f, true},
                                                   {1000, 32, 5, 0.0001f, false},
                                                   {1000, 100, 20, 0.0001f, true},
                                                   {1000, 100, 20, 0.0001f, false},
                                                   {10000, 32, 10, 0.0001f, true},
                                                   {10000, 32, 10, 0.0001f, false},
                                                   {10000, 100, 50, 0.0001f, true},
                                                   {10000, 100, 50, 0.0001f, false},
                                                   {10000, 500, 100, 0.0001f, true},
                                                   {10000, 500, 100, 0.0001f, false}};

const std::vector<KmeansInputs<double>> inputsd2 = {{1000, 32, 5, 0.0001, true},
                                                    {1000, 32, 5, 0.0001, false},
                                                    {1000, 100, 20, 0.0001, true},
                                                    {1000, 100, 20, 0.0001, false},
                                                    {10000, 32, 10, 0.0001, true},
                                                    {10000, 32, 10, 0.0001, false},
                                                    {10000, 100, 50, 0.0001, true},
                                                    {10000, 100, 50, 0.0001, false},
                                                    {10000, 500, 100, 0.0001, true},
                                                    {10000, 500, 100, 0.0001, false}};

typedef KmeansTest<float> KmeansTestF;
TEST_P(KmeansTestF, Result) { ASSERT_TRUE(score == 1.0); }

typedef KmeansTest<double> KmeansTestD;
TEST_P(KmeansTestD, Result) { ASSERT_TRUE(score == 1.0); }

INSTANTIATE_TEST_CASE_P(KmeansTests, KmeansTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(KmeansTests, KmeansTestD, ::testing::ValuesIn(inputsd2));

}  // namespace raft
