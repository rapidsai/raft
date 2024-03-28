/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <raft/cluster/detail/kmeans_common.cuh>
#include <raft/cluster/kmeans_types.hpp>
#include <raft/common/nvtx.hpp>
#include <raft/core/cudart_utils.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/kvp.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/reduce_cols_by_key.cuh>
#include <raft/linalg/reduce_rows_by_key.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda.h>
#include <thrust/fill.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <optional>
#include <random>

namespace raft {
namespace cluster {
namespace detail {

// =========================================================
// Init functions
// =========================================================

// Selects 'n_clusters' samples randomly from X
template <typename DataT, typename IndexT>
void initRandom(raft::resources const& handle,
                const KMeansParams& params,
                raft::device_matrix_view<const DataT, IndexT> X,
                raft::device_matrix_view<DataT, IndexT> centroids)
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope("initRandom");
  cudaStream_t stream = resource::get_cuda_stream(handle);
  auto n_clusters     = params.n_clusters;
  detail::shuffleAndGather<DataT, IndexT>(handle, X, centroids, n_clusters, params.rng_state.seed);
}

/*
 * @brief Selects 'n_clusters' samples from the input X using kmeans++ algorithm.

 * @note  This is the algorithm described in
 *        "k-means++: the advantages of careful seeding". 2007, Arthur, D. and Vassilvitskii, S.
 *        ACM-SIAM symposium on Discrete algorithms.
 *
 * Scalable kmeans++ pseudocode
 * 1: C = sample a point uniformly at random from X
 * 2: while |C| < k
 * 3:   Sample x in X with probability p_x = d^2(x, C) / phi_X (C)
 * 4:   C = C U {x}
 * 5: end for
 */
template <typename DataT, typename IndexT>
void kmeansPlusPlus(raft::resources const& handle,
                    const KMeansParams& params,
                    raft::device_matrix_view<const DataT, IndexT> X,
                    raft::device_matrix_view<DataT, IndexT> centroidsRawData,
                    rmm::device_uvector<char>& workspace)
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope("kmeansPlusPlus");
  cudaStream_t stream = resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = params.n_clusters;
  auto metric         = params.metric;

  // number of seeding trials for each center (except the first)
  auto n_trials = 2 + static_cast<int>(std::ceil(log(n_clusters)));

  RAFT_LOG_DEBUG(
    "Run sequential k-means++ to select %d centroids from %d input samples "
    "(%d seeding trials per iterations)",
    n_clusters,
    n_samples,
    n_trials);

  auto dataBatchSize = getDataBatchSize(params.batch_samples, n_samples);

  // temporary buffers
  auto indices            = raft::make_device_vector<IndexT, IndexT>(handle, n_trials);
  auto centroidCandidates = raft::make_device_matrix<DataT, IndexT>(handle, n_trials, n_features);
  auto costPerCandidate   = raft::make_device_vector<DataT, IndexT>(handle, n_trials);
  auto minClusterDistance = raft::make_device_vector<DataT, IndexT>(handle, n_samples);
  auto distBuffer         = raft::make_device_matrix<DataT, IndexT>(handle, n_trials, n_samples);

  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);
  rmm::device_scalar<DataT> clusterCost(stream);
  rmm::device_scalar<cub::KeyValuePair<int, DataT>> minClusterIndexAndDistance(stream);

  // Device and matrix views
  raft::device_vector_view<IndexT, IndexT> indices_view(indices.data_handle(), n_trials);
  auto const_weights_view =
    raft::make_device_vector_view<const DataT, IndexT>(minClusterDistance.data_handle(), n_samples);
  auto const_indices_view =
    raft::make_device_vector_view<const IndexT, IndexT>(indices.data_handle(), n_trials);
  auto const_X_view =
    raft::make_device_matrix_view<const DataT, IndexT>(X.data_handle(), n_samples, n_features);
  raft::device_matrix_view<DataT, IndexT> candidates_view(
    centroidCandidates.data_handle(), n_trials, n_features);

  // L2 norm of X: ||c||^2
  auto L2NormX = raft::make_device_vector<DataT, IndexT>(handle, n_samples);

  if (metric == raft::distance::DistanceType::L2Expanded ||
      metric == raft::distance::DistanceType::L2SqrtExpanded) {
    raft::linalg::rowNorm(L2NormX.data_handle(),
                          X.data_handle(),
                          X.extent(1),
                          X.extent(0),
                          raft::linalg::L2Norm,
                          true,
                          stream);
  }

  raft::random::RngState rng(params.rng_state.seed, params.rng_state.type);
  std::mt19937 gen(params.rng_state.seed);
  std::uniform_int_distribution<> dis(0, n_samples - 1);

  // <<< Step-1 >>>: C <-- sample a point uniformly at random from X
  auto initialCentroid = raft::make_device_matrix_view<const DataT, IndexT>(
    X.data_handle() + dis(gen) * n_features, 1, n_features);
  int n_clusters_picked = 1;

  // store the chosen centroid in the buffer
  raft::copy(
    centroidsRawData.data_handle(), initialCentroid.data_handle(), initialCentroid.size(), stream);

  //  C = initial set of centroids
  auto centroids = raft::make_device_matrix_view<DataT, IndexT>(
    centroidsRawData.data_handle(), initialCentroid.extent(0), initialCentroid.extent(1));
  // <<< End of Step-1 >>>

  // Calculate cluster distance, d^2(x, C), for all the points x in X to the nearest centroid
  detail::minClusterDistanceCompute<DataT, IndexT>(handle,
                                                   X,
                                                   centroids,
                                                   minClusterDistance.view(),
                                                   L2NormX.view(),
                                                   L2NormBuf_OR_DistBuf,
                                                   params.metric,
                                                   params.batch_samples,
                                                   params.batch_centroids,
                                                   workspace);

  RAFT_LOG_DEBUG(" k-means++ - Sampled %d/%d centroids", n_clusters_picked, n_clusters);

  // <<<< Step-2 >>> : while |C| < k
  while (n_clusters_picked < n_clusters) {
    // <<< Step-3 >>> : Sample x in X with probability p_x = d^2(x, C) / phi_X (C)
    // Choose 'n_trials' centroid candidates from X with probability proportional to the squared
    // distance to the nearest existing cluster

    raft::random::discrete(handle, rng, indices_view, const_weights_view);
    raft::matrix::gather(handle, const_X_view, const_indices_view, candidates_view);

    // Calculate pairwise distance between X and the centroid candidates
    // Output - pwd [n_trials x n_samples]
    auto pwd = distBuffer.view();
    detail::pairwise_distance_kmeans<DataT, IndexT>(
      handle, centroidCandidates.view(), X, pwd, workspace, metric);

    // Update nearest cluster distance for each centroid candidate
    // Note pwd and minDistBuf points to same buffer which currently holds pairwise distance values.
    // Outputs minDistanceBuf[n_trials x n_samples] where minDistance[i, :] contains updated
    // minClusterDistance that includes candidate-i
    auto minDistBuf = distBuffer.view();
    raft::linalg::matrixVectorOp(minDistBuf.data_handle(),
                                 pwd.data_handle(),
                                 minClusterDistance.data_handle(),
                                 pwd.extent(1),
                                 pwd.extent(0),
                                 true,
                                 true,
                                 raft::min_op{},
                                 stream);

    // Calculate costPerCandidate[n_trials] where costPerCandidate[i] is the cluster cost when using
    // centroid candidate-i
    raft::linalg::reduce(costPerCandidate.data_handle(),
                         minDistBuf.data_handle(),
                         minDistBuf.extent(1),
                         minDistBuf.extent(0),
                         static_cast<DataT>(0),
                         true,
                         true,
                         stream);

    // Greedy Choice - Choose the candidate that has minimum cluster cost
    // ArgMin operation below identifies the index of minimum cost in costPerCandidate
    {
      // Determine temporary device storage requirements
      size_t temp_storage_bytes = 0;
      cub::DeviceReduce::ArgMin(nullptr,
                                temp_storage_bytes,
                                costPerCandidate.data_handle(),
                                minClusterIndexAndDistance.data(),
                                costPerCandidate.extent(0),
                                stream);

      // Allocate temporary storage
      workspace.resize(temp_storage_bytes, stream);

      // Run argmin-reduction
      cub::DeviceReduce::ArgMin(workspace.data(),
                                temp_storage_bytes,
                                costPerCandidate.data_handle(),
                                minClusterIndexAndDistance.data(),
                                costPerCandidate.extent(0),
                                stream);

      int bestCandidateIdx = -1;
      raft::copy(&bestCandidateIdx, &minClusterIndexAndDistance.data()->key, 1, stream);
      resource::sync_stream(handle);
      /// <<< End of Step-3 >>>

      /// <<< Step-4 >>>: C = C U {x}
      // Update minimum cluster distance corresponding to the chosen centroid candidate
      raft::copy(minClusterDistance.data_handle(),
                 minDistBuf.data_handle() + bestCandidateIdx * n_samples,
                 n_samples,
                 stream);

      raft::copy(centroidsRawData.data_handle() + n_clusters_picked * n_features,
                 centroidCandidates.data_handle() + bestCandidateIdx * n_features,
                 n_features,
                 stream);

      ++n_clusters_picked;
      /// <<< End of Step-4 >>>
    }

    RAFT_LOG_DEBUG(" k-means++ - Sampled %d/%d centroids", n_clusters_picked, n_clusters);
  }  /// <<<< Step-5 >>>
}

/**
 *
 * @tparam DataT
 * @tparam IndexT
 * @param handle
 * @param[in] X input matrix (size n_samples, n_features)
 * @param[in] weight number of samples currently assigned to each centroid
 * @param[in] cur_centroids matrix of current centroids (size n_clusters, n_features)
 * @param[in] l2norm_x
 * @param[out] min_cluster_and_dist
 * @param[out] new_centroids
 * @param[out] new_weight
 * @param[inout] workspace
 */
template <typename DataT, typename IndexT, typename LabelsIterator>
void update_centroids(raft::resources const& handle,
                      raft::device_matrix_view<const DataT, IndexT, row_major> X,
                      raft::device_vector_view<const DataT, IndexT> sample_weights,
                      raft::device_matrix_view<const DataT, IndexT, row_major> centroids,

                      // TODO: Figure out how to best wrap iterator types in mdspan
                      LabelsIterator cluster_labels,
                      raft::device_vector_view<DataT, IndexT> weight_per_cluster,
                      raft::device_matrix_view<DataT, IndexT, row_major> new_centroids,
                      rmm::device_uvector<char>& workspace)
{
  auto n_clusters = centroids.extent(0);
  auto n_samples  = X.extent(0);

  workspace.resize(n_samples, resource::get_cuda_stream(handle));

  // Calculates weighted sum of all the samples assigned to cluster-i and stores the
  // result in new_centroids[i]
  raft::linalg::reduce_rows_by_key((DataT*)X.data_handle(),
                                   X.extent(1),
                                   cluster_labels,
                                   sample_weights.data_handle(),
                                   workspace.data(),
                                   X.extent(0),
                                   X.extent(1),
                                   n_clusters,
                                   new_centroids.data_handle(),
                                   resource::get_cuda_stream(handle));

  // Reduce weights by key to compute weight in each cluster
  raft::linalg::reduce_cols_by_key(sample_weights.data_handle(),
                                   cluster_labels,
                                   weight_per_cluster.data_handle(),
                                   (IndexT)1,
                                   (IndexT)sample_weights.extent(0),
                                   (IndexT)n_clusters,
                                   resource::get_cuda_stream(handle));

  // Computes new_centroids[i] = new_centroids[i]/weight_per_cluster[i] where
  //   new_centroids[n_clusters x n_features] - 2D array, new_centroids[i] has sum of all the
  //   samples assigned to cluster-i
  //   weight_per_cluster[n_clusters] - 1D array, weight_per_cluster[i] contains sum of weights in
  //   cluster-i.
  // Note - when weight_per_cluster[i] is 0, new_centroids[i] is reset to 0
  raft::linalg::matrixVectorOp(new_centroids.data_handle(),
                               new_centroids.data_handle(),
                               weight_per_cluster.data_handle(),
                               new_centroids.extent(1),
                               new_centroids.extent(0),
                               true,
                               false,
                               raft::div_checkzero_op{},
                               resource::get_cuda_stream(handle));

  // copy centroids[i] to new_centroids[i] when weight_per_cluster[i] is 0
  cub::ArgIndexInputIterator<DataT*> itr_wt(weight_per_cluster.data_handle());
  raft::matrix::gather_if(
    const_cast<DataT*>(centroids.data_handle()),
    static_cast<int>(centroids.extent(1)),
    static_cast<int>(centroids.extent(0)),
    itr_wt,
    itr_wt,
    static_cast<int>(weight_per_cluster.size()),
    new_centroids.data_handle(),
    [=] __device__(raft::KeyValuePair<ptrdiff_t, DataT> map) {  // predicate
      // copy when the sum of weights in the cluster is 0
      return map.value == 0;
    },
    raft::key_op{},
    resource::get_cuda_stream(handle));
}

// TODO: Resizing is needed to use mdarray instead of rmm::device_uvector
template <typename DataT, typename IndexT>
void kmeans_fit_main(raft::resources const& handle,
                     const KMeansParams& params,
                     raft::device_matrix_view<const DataT, IndexT> X,
                     raft::device_vector_view<const DataT, IndexT> weight,
                     raft::device_matrix_view<DataT, IndexT> centroidsRawData,
                     raft::host_scalar_view<DataT> inertia,
                     raft::host_scalar_view<IndexT> n_iter,
                     rmm::device_uvector<char>& workspace)
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope("kmeans_fit_main");
  logger::get(RAFT_NAME).set_level(params.verbosity);
  cudaStream_t stream = resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = params.n_clusters;
  auto metric         = params.metric;

  // stores (key, value) pair corresponding to each sample where
  //   - key is the index of nearest cluster
  //   - value is the distance to the nearest cluster
  auto minClusterAndDistance =
    raft::make_device_vector<raft::KeyValuePair<IndexT, DataT>, IndexT>(handle, n_samples);

  // temporary buffer to store L2 norm of centroids or distance matrix,
  // destructor releases the resource
  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);

  // temporary buffer to store intermediate centroids, destructor releases the
  // resource
  auto newCentroids = raft::make_device_matrix<DataT, IndexT>(handle, n_clusters, n_features);

  // temporary buffer to store weights per cluster, destructor releases the
  // resource
  auto wtInCluster = raft::make_device_vector<DataT, IndexT>(handle, n_clusters);

  rmm::device_scalar<DataT> clusterCostD(stream);

  // L2 norm of X: ||x||^2
  auto L2NormX = raft::make_device_vector<DataT, IndexT>(handle, n_samples);
  auto l2normx_view =
    raft::make_device_vector_view<const DataT, IndexT>(L2NormX.data_handle(), n_samples);

  if (metric == raft::distance::DistanceType::L2Expanded ||
      metric == raft::distance::DistanceType::L2SqrtExpanded) {
    raft::linalg::rowNorm(L2NormX.data_handle(),
                          X.data_handle(),
                          X.extent(1),
                          X.extent(0),
                          raft::linalg::L2Norm,
                          true,
                          stream);
  }

  RAFT_LOG_DEBUG(
    "Calling KMeans.fit with %d samples of input data and the initialized "
    "cluster centers",
    n_samples);

  DataT priorClusteringCost = 0;
  for (n_iter[0] = 1; n_iter[0] <= params.max_iter; ++n_iter[0]) {
    RAFT_LOG_DEBUG(
      "KMeans.fit: Iteration-%d: fitting the model using the initialized "
      "cluster centers",
      n_iter[0]);

    auto centroids = raft::make_device_matrix_view<DataT, IndexT>(
      centroidsRawData.data_handle(), n_clusters, n_features);

    // computes minClusterAndDistance[0:n_samples) where
    // minClusterAndDistance[i] is a <key, value> pair where
    //   'key' is index to a sample in 'centroids' (index of the nearest
    //   centroid) and 'value' is the distance between the sample 'X[i]' and the
    //   'centroid[key]'
    detail::minClusterAndDistanceCompute<DataT, IndexT>(handle,
                                                        X,
                                                        centroids,
                                                        minClusterAndDistance.view(),
                                                        l2normx_view,
                                                        L2NormBuf_OR_DistBuf,
                                                        params.metric,
                                                        params.batch_samples,
                                                        params.batch_centroids,
                                                        workspace);

    // Using TransformInputIteratorT to dereference an array of
    // raft::KeyValuePair and converting them to just return the Key to be used
    // in reduce_rows_by_key prims
    detail::KeyValueIndexOp<IndexT, DataT> conversion_op;
    cub::TransformInputIterator<IndexT,
                                detail::KeyValueIndexOp<IndexT, DataT>,
                                raft::KeyValuePair<IndexT, DataT>*>
      itr(minClusterAndDistance.data_handle(), conversion_op);

    update_centroids(handle,
                     X,
                     weight,
                     raft::make_device_matrix_view<const DataT, IndexT>(
                       centroidsRawData.data_handle(), n_clusters, n_features),
                     itr,
                     wtInCluster.view(),
                     newCentroids.view(),
                     workspace);

    // compute the squared norm between the newCentroids and the original
    // centroids, destructor releases the resource
    auto sqrdNorm = raft::make_device_scalar(handle, DataT(0));
    raft::linalg::mapThenSumReduce(sqrdNorm.data_handle(),
                                   newCentroids.size(),
                                   raft::sqdiff_op{},
                                   stream,
                                   centroids.data_handle(),
                                   newCentroids.data_handle());

    DataT sqrdNormError = 0;
    raft::copy(&sqrdNormError, sqrdNorm.data_handle(), sqrdNorm.size(), stream);

    raft::copy(
      centroidsRawData.data_handle(), newCentroids.data_handle(), newCentroids.size(), stream);

    bool done = false;
    if (params.inertia_check) {
      // calculate cluster cost phi_x(C)
      detail::computeClusterCost(handle,
                                 minClusterAndDistance.view(),
                                 workspace,
                                 raft::make_device_scalar_view(clusterCostD.data()),
                                 raft::value_op{},
                                 raft::add_op{});

      DataT curClusteringCost = clusterCostD.value(stream);

      ASSERT(curClusteringCost != (DataT)0.0,
             "Too few points and centroids being found is getting 0 cost from "
             "centers");

      if (n_iter[0] > 1) {
        DataT delta = curClusteringCost / priorClusteringCost;
        if (delta > 1 - params.tol) done = true;
      }
      priorClusteringCost = curClusteringCost;
    }

    resource::sync_stream(handle, stream);
    if (sqrdNormError < params.tol) done = true;

    if (done) {
      RAFT_LOG_DEBUG("Threshold triggered after %d iterations. Terminating early.", n_iter[0]);
      break;
    }
  }

  auto centroids = raft::make_device_matrix_view<DataT, IndexT>(
    centroidsRawData.data_handle(), n_clusters, n_features);

  detail::minClusterAndDistanceCompute<DataT, IndexT>(handle,
                                                      X,
                                                      centroids,
                                                      minClusterAndDistance.view(),
                                                      l2normx_view,
                                                      L2NormBuf_OR_DistBuf,
                                                      params.metric,
                                                      params.batch_samples,
                                                      params.batch_centroids,
                                                      workspace);

  // TODO: add different templates for InType of binaryOp to avoid thrust transform
  thrust::transform(resource::get_thrust_policy(handle),
                    minClusterAndDistance.data_handle(),
                    minClusterAndDistance.data_handle() + minClusterAndDistance.size(),
                    weight.data_handle(),
                    minClusterAndDistance.data_handle(),
                    [=] __device__(const raft::KeyValuePair<IndexT, DataT> kvp, DataT wt) {
                      raft::KeyValuePair<IndexT, DataT> res;
                      res.value = kvp.value * wt;
                      res.key   = kvp.key;
                      return res;
                    });

  // calculate cluster cost phi_x(C)
  detail::computeClusterCost(handle,
                             minClusterAndDistance.view(),
                             workspace,
                             raft::make_device_scalar_view(clusterCostD.data()),
                             raft::value_op{},
                             raft::add_op{});

  inertia[0] = clusterCostD.value(stream);

  RAFT_LOG_DEBUG("KMeans.fit: completed after %d iterations with %f inertia[0] ",
                 n_iter[0] > params.max_iter ? n_iter[0] - 1 : n_iter[0],
                 inertia[0]);
}

/*
 * @brief Selects 'n_clusters' samples from X using scalable kmeans++ algorithm.

 * @note  This is the algorithm described in
 *        "Scalable K-Means++", 2012, Bahman Bahmani, Benjamin Moseley,
 *         Andrea Vattani, Ravi Kumar, Sergei Vassilvitskii,
 *         https://arxiv.org/abs/1203.6402

 * Scalable kmeans++ pseudocode
 * 1: C = sample a point uniformly at random from X
 * 2: psi = phi_X (C)
 * 3: for O( log(psi) ) times do
 * 4:   C' = sample each point x in X independently with probability
 *           p_x = l * (d^2(x, C) / phi_X (C) )
 * 5:   C = C U C'
 * 6: end for
 * 7: For x in C, set w_x to be the number of points in X closer to x than any
 * other point in C
 * 8: Recluster the weighted points in C into k clusters

 * TODO: Resizing is needed to use mdarray instead of rmm::device_uvector

 */
template <typename DataT, typename IndexT>
void initScalableKMeansPlusPlus(raft::resources const& handle,
                                const KMeansParams& params,
                                raft::device_matrix_view<const DataT, IndexT> X,
                                raft::device_matrix_view<DataT, IndexT> centroidsRawData,
                                rmm::device_uvector<char>& workspace)
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope("initScalableKMeansPlusPlus");
  cudaStream_t stream = resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = params.n_clusters;
  auto metric         = params.metric;

  raft::random::RngState rng(params.rng_state.seed, params.rng_state.type);

  // <<<< Step-1 >>> : C <- sample a point uniformly at random from X
  std::mt19937 gen(params.rng_state.seed);
  std::uniform_int_distribution<> dis(0, n_samples - 1);

  auto cIdx            = dis(gen);
  auto initialCentroid = raft::make_device_matrix_view<const DataT, IndexT>(
    X.data_handle() + cIdx * n_features, 1, n_features);

  // flag the sample that is chosen as initial centroid
  std::vector<uint8_t> h_isSampleCentroid(n_samples);
  std::fill(h_isSampleCentroid.begin(), h_isSampleCentroid.end(), 0);
  h_isSampleCentroid[cIdx] = 1;

  // device buffer to flag the sample that is chosen as initial centroid
  auto isSampleCentroid = raft::make_device_vector<uint8_t, IndexT>(handle, n_samples);

  raft::copy(
    isSampleCentroid.data_handle(), h_isSampleCentroid.data(), isSampleCentroid.size(), stream);

  rmm::device_uvector<DataT> centroidsBuf(initialCentroid.size(), stream);

  // reset buffer to store the chosen centroid
  raft::copy(centroidsBuf.data(), initialCentroid.data_handle(), initialCentroid.size(), stream);

  auto potentialCentroids = raft::make_device_matrix_view<DataT, IndexT>(
    centroidsBuf.data(), initialCentroid.extent(0), initialCentroid.extent(1));
  // <<< End of Step-1 >>>

  // temporary buffer to store L2 norm of centroids or distance matrix,
  // destructor releases the resource
  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);

  // L2 norm of X: ||x||^2
  auto L2NormX = raft::make_device_vector<DataT, IndexT>(handle, n_samples);
  if (metric == raft::distance::DistanceType::L2Expanded ||
      metric == raft::distance::DistanceType::L2SqrtExpanded) {
    raft::linalg::rowNorm(L2NormX.data_handle(),
                          X.data_handle(),
                          X.extent(1),
                          X.extent(0),
                          raft::linalg::L2Norm,
                          true,
                          stream);
  }

  auto minClusterDistanceVec = raft::make_device_vector<DataT, IndexT>(handle, n_samples);
  auto uniformRands          = raft::make_device_vector<DataT, IndexT>(handle, n_samples);
  rmm::device_scalar<DataT> clusterCost(stream);

  // <<< Step-2 >>>: psi <- phi_X (C)
  detail::minClusterDistanceCompute<DataT, IndexT>(handle,
                                                   X,
                                                   potentialCentroids,
                                                   minClusterDistanceVec.view(),
                                                   L2NormX.view(),
                                                   L2NormBuf_OR_DistBuf,
                                                   params.metric,
                                                   params.batch_samples,
                                                   params.batch_centroids,
                                                   workspace);

  // compute partial cluster cost from the samples in rank
  detail::computeClusterCost(handle,
                             minClusterDistanceVec.view(),
                             workspace,
                             raft::make_device_scalar_view(clusterCost.data()),
                             raft::identity_op{},
                             raft::add_op{});

  auto psi = clusterCost.value(stream);

  // <<< End of Step-2 >>>

  // Scalable kmeans++ paper claims 8 rounds is sufficient
  resource::sync_stream(handle, stream);
  int niter = std::min(8, (int)ceil(log(psi)));
  RAFT_LOG_DEBUG("KMeans||: psi = %g, log(psi) = %g, niter = %d ", psi, log(psi), niter);

  // <<<< Step-3 >>> : for O( log(psi) ) times do
  for (int iter = 0; iter < niter; ++iter) {
    RAFT_LOG_DEBUG("KMeans|| - Iteration %d: # potential centroids sampled - %d",
                   iter,
                   potentialCentroids.extent(0));

    detail::minClusterDistanceCompute<DataT, IndexT>(handle,
                                                     X,
                                                     potentialCentroids,
                                                     minClusterDistanceVec.view(),
                                                     L2NormX.view(),
                                                     L2NormBuf_OR_DistBuf,
                                                     params.metric,
                                                     params.batch_samples,
                                                     params.batch_centroids,
                                                     workspace);

    detail::computeClusterCost(handle,
                               minClusterDistanceVec.view(),
                               workspace,
                               raft::make_device_scalar_view<DataT>(clusterCost.data()),
                               raft::identity_op{},
                               raft::add_op{});

    psi = clusterCost.value(stream);

    // <<<< Step-4 >>> : Sample each point x in X independently and identify new
    // potentialCentroids
    raft::random::uniform(
      handle, rng, uniformRands.data_handle(), uniformRands.extent(0), (DataT)0, (DataT)1);

    detail::SamplingOp<DataT, IndexT> select_op(psi,
                                                params.oversampling_factor,
                                                n_clusters,
                                                uniformRands.data_handle(),
                                                isSampleCentroid.data_handle());

    rmm::device_uvector<DataT> CpRaw(0, stream);
    detail::sampleCentroids<DataT, IndexT>(handle,
                                           X,
                                           minClusterDistanceVec.view(),
                                           isSampleCentroid.view(),
                                           select_op,
                                           CpRaw,
                                           workspace);
    auto Cp = raft::make_device_matrix_view<DataT, IndexT>(
      CpRaw.data(), CpRaw.size() / n_features, n_features);
    /// <<<< End of Step-4 >>>>

    /// <<<< Step-5 >>> : C = C U C'
    // append the data in Cp to the buffer holding the potentialCentroids
    centroidsBuf.resize(centroidsBuf.size() + Cp.size(), stream);
    raft::copy(
      centroidsBuf.data() + centroidsBuf.size() - Cp.size(), Cp.data_handle(), Cp.size(), stream);

    IndexT tot_centroids = potentialCentroids.extent(0) + Cp.extent(0);
    potentialCentroids =
      raft::make_device_matrix_view<DataT, IndexT>(centroidsBuf.data(), tot_centroids, n_features);
    /// <<<< End of Step-5 >>>
  }  /// <<<< Step-6 >>>

  RAFT_LOG_DEBUG("KMeans||: total # potential centroids sampled - %d",
                 potentialCentroids.extent(0));

  if ((int)potentialCentroids.extent(0) > n_clusters) {
    // <<< Step-7 >>>: For x in C, set w_x to be the number of pts closest to X
    // temporary buffer to store the sample count per cluster, destructor
    // releases the resource
    auto weight = raft::make_device_vector<DataT, IndexT>(handle, potentialCentroids.extent(0));

    detail::countSamplesInCluster<DataT, IndexT>(
      handle, params, X, L2NormX.view(), potentialCentroids, workspace, weight.view());

    // <<< end of Step-7 >>>

    // Step-8: Recluster the weighted points in C into k clusters
    detail::kmeansPlusPlus<DataT, IndexT>(
      handle, params, potentialCentroids, centroidsRawData, workspace);

    auto inertia = make_host_scalar<DataT>(0);
    auto n_iter  = make_host_scalar<IndexT>(0);
    KMeansParams default_params;
    default_params.n_clusters = params.n_clusters;

    detail::kmeans_fit_main<DataT, IndexT>(handle,
                                           default_params,
                                           potentialCentroids,
                                           weight.view(),
                                           centroidsRawData,
                                           inertia.view(),
                                           n_iter.view(),
                                           workspace);

  } else if ((int)potentialCentroids.extent(0) < n_clusters) {
    // supplement with random
    auto n_random_clusters = n_clusters - potentialCentroids.extent(0);

    RAFT_LOG_DEBUG(
      "[Warning!] KMeans||: found fewer than %d centroids during "
      "initialization (found %d centroids, remaining %d centroids will be "
      "chosen randomly from input samples)",
      n_clusters,
      potentialCentroids.extent(0),
      n_random_clusters);

    // generate `n_random_clusters` centroids
    KMeansParams rand_params;
    rand_params.init       = KMeansParams::InitMethod::Random;
    rand_params.n_clusters = n_random_clusters;
    initRandom<DataT, IndexT>(handle, rand_params, X, centroidsRawData);

    // copy centroids generated during kmeans|| iteration to the buffer
    raft::copy(centroidsRawData.data_handle() + n_random_clusters * n_features,
               potentialCentroids.data_handle(),
               potentialCentroids.size(),
               stream);
  } else {
    // found the required n_clusters
    raft::copy(centroidsRawData.data_handle(),
               potentialCentroids.data_handle(),
               potentialCentroids.size(),
               stream);
  }
}

/**
 * @brief Find clusters with k-means algorithm.
 *   Initial centroids are chosen with k-means++ algorithm. Empty
 *   clusters are reinitialized by choosing new centroids with
 *   k-means++ algorithm.
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. It must be noted
 * that the data must be in row-major format and stored in device accessible
 * location.
 * @param[in]     n_samples     Number of samples in the input X.
 * @param[in]     n_features    Number of features or the dimensions of each
 * sample.
 * @param[in]     sample_weight Optional weights for each observation in X.
 * @param[inout]  centroids     [in] When init is InitMethod::Array, use
 * centroids as the initial cluster centers
 *                              [out] Otherwise, generated centroids from the
 * kmeans algorithm is stored at the address pointed by 'centroids'.
 * @param[out]    inertia       Sum of squared distances of samples to their
 * closest cluster center.
 * @param[out]    n_iter        Number of iterations run.
 */
template <typename DataT, typename IndexT>
void kmeans_fit(raft::resources const& handle,
                const KMeansParams& params,
                raft::device_matrix_view<const DataT, IndexT> X,
                std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight,
                raft::device_matrix_view<DataT, IndexT> centroids,
                raft::host_scalar_view<DataT> inertia,
                raft::host_scalar_view<IndexT> n_iter)
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope("kmeans_fit");
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = params.n_clusters;
  cudaStream_t stream = resource::get_cuda_stream(handle);
  // Check that parameters are valid
  if (sample_weight.has_value())
    RAFT_EXPECTS(sample_weight.value().extent(0) == n_samples,
                 "invalid parameter (sample_weight!=n_samples)");
  RAFT_EXPECTS(n_clusters > 0, "invalid parameter (n_clusters<=0)");
  RAFT_EXPECTS(params.tol > 0, "invalid parameter (tol<=0)");
  RAFT_EXPECTS(params.oversampling_factor >= 0, "invalid parameter (oversampling_factor<0)");
  RAFT_EXPECTS((int)centroids.extent(0) == params.n_clusters,
               "invalid parameter (centroids.extent(0) != n_clusters)");
  RAFT_EXPECTS(centroids.extent(1) == n_features,
               "invalid parameter (centroids.extent(1) != n_features)");

  // Display a message if the batch size is smaller than n_samples but will be ignored
  if (params.batch_samples < (int)n_samples &&
      (params.metric == raft::distance::DistanceType::L2Expanded ||
       params.metric == raft::distance::DistanceType::L2SqrtExpanded)) {
    RAFT_LOG_DEBUG(
      "batch_samples=%d was passed, but batch_samples=%d will be used (reason: "
      "batch_samples has no impact on the memory footprint when FusedL2NN can be used)",
      params.batch_samples,
      (int)n_samples);
  }
  // Display a message if batch_centroids is set and a fusedL2NN-compatible metric is used
  if (params.batch_centroids != 0 && params.batch_centroids != params.n_clusters &&
      (params.metric == raft::distance::DistanceType::L2Expanded ||
       params.metric == raft::distance::DistanceType::L2SqrtExpanded)) {
    RAFT_LOG_DEBUG(
      "batch_centroids=%d was passed, but batch_centroids=%d will be used (reason: "
      "batch_centroids has no impact on the memory footprint when FusedL2NN can be used)",
      params.batch_centroids,
      params.n_clusters);
  }

  logger::get(RAFT_NAME).set_level(params.verbosity);

  // Allocate memory
  rmm::device_uvector<char> workspace(0, stream);
  auto weight = raft::make_device_vector<DataT>(handle, n_samples);
  if (sample_weight.has_value())
    raft::copy(weight.data_handle(), sample_weight.value().data_handle(), n_samples, stream);
  else
    thrust::fill(resource::get_thrust_policy(handle),
                 weight.data_handle(),
                 weight.data_handle() + weight.size(),
                 1);

  // check if weights sum up to n_samples
  checkWeight<DataT>(handle, weight.view(), workspace);

  auto centroidsRawData = raft::make_device_matrix<DataT, IndexT>(handle, n_clusters, n_features);

  auto n_init = params.n_init;
  if (params.init == KMeansParams::InitMethod::Array && n_init != 1) {
    RAFT_LOG_DEBUG(
      "Explicit initial center position passed: performing only one init in "
      "k-means instead of n_init=%d",
      n_init);
    n_init = 1;
  }

  std::mt19937 gen(params.rng_state.seed);
  inertia[0] = std::numeric_limits<DataT>::max();

  for (auto seed_iter = 0; seed_iter < n_init; ++seed_iter) {
    KMeansParams iter_params   = params;
    iter_params.rng_state.seed = gen();

    DataT iter_inertia    = std::numeric_limits<DataT>::max();
    IndexT n_current_iter = 0;
    if (iter_params.init == KMeansParams::InitMethod::Random) {
      // initializing with random samples from input dataset
      RAFT_LOG_DEBUG(
        "KMeans.fit (Iteration-%d/%d): initialize cluster centers by "
        "randomly choosing from the "
        "input data.",
        seed_iter + 1,
        n_init);
      initRandom<DataT, IndexT>(handle, iter_params, X, centroidsRawData.view());
    } else if (iter_params.init == KMeansParams::InitMethod::KMeansPlusPlus) {
      // default method to initialize is kmeans++
      RAFT_LOG_DEBUG(
        "KMeans.fit (Iteration-%d/%d): initialize cluster centers using "
        "k-means++ algorithm.",
        seed_iter + 1,
        n_init);
      if (iter_params.oversampling_factor == 0)
        detail::kmeansPlusPlus<DataT, IndexT>(
          handle, iter_params, X, centroidsRawData.view(), workspace);
      else
        detail::initScalableKMeansPlusPlus<DataT, IndexT>(
          handle, iter_params, X, centroidsRawData.view(), workspace);
    } else if (iter_params.init == KMeansParams::InitMethod::Array) {
      RAFT_LOG_DEBUG(
        "KMeans.fit (Iteration-%d/%d): initialize cluster centers from "
        "the ndarray array input "
        "passed to init argument.",
        seed_iter + 1,
        n_init);
      raft::copy(
        centroidsRawData.data_handle(), centroids.data_handle(), n_clusters * n_features, stream);
    } else {
      THROW("unknown initialization method to select initial centers");
    }

    detail::kmeans_fit_main<DataT, IndexT>(handle,
                                           iter_params,
                                           X,
                                           weight.view(),
                                           centroidsRawData.view(),
                                           raft::make_host_scalar_view<DataT>(&iter_inertia),
                                           raft::make_host_scalar_view<IndexT>(&n_current_iter),
                                           workspace);
    if (iter_inertia < inertia[0]) {
      inertia[0] = iter_inertia;
      n_iter[0]  = n_current_iter;
      raft::copy(
        centroids.data_handle(), centroidsRawData.data_handle(), n_clusters * n_features, stream);
    }
    RAFT_LOG_DEBUG("KMeans.fit after iteration-%d/%d: inertia - %f, n_iter[0] - %d",
                   seed_iter + 1,
                   n_init,
                   inertia[0],
                   n_iter[0]);
  }
  RAFT_LOG_DEBUG("KMeans.fit: async call returned (fit could still be running on the device)");
}

template <typename DataT, typename IndexT = int>
void kmeans_fit(raft::resources const& handle,
                const KMeansParams& params,
                const DataT* X,
                const DataT* sample_weight,
                DataT* centroids,
                IndexT n_samples,
                IndexT n_features,
                DataT& inertia,
                IndexT& n_iter)
{
  auto XView = raft::make_device_matrix_view<const DataT, IndexT>(X, n_samples, n_features);
  auto centroidsView =
    raft::make_device_matrix_view<DataT, IndexT>(centroids, params.n_clusters, n_features);
  std::optional<raft::device_vector_view<const DataT>> sample_weightView = std::nullopt;
  if (sample_weight)
    sample_weightView =
      raft::make_device_vector_view<const DataT, IndexT>(sample_weight, n_samples);
  auto inertiaView = raft::make_host_scalar_view(&inertia);
  auto n_iterView  = raft::make_host_scalar_view(&n_iter);

  detail::kmeans_fit<DataT, IndexT>(
    handle, params, XView, sample_weightView, centroidsView, inertiaView, n_iterView);
}

template <typename DataT, typename IndexT>
void kmeans_predict(raft::resources const& handle,
                    const KMeansParams& params,
                    raft::device_matrix_view<const DataT, IndexT> X,
                    std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight,
                    raft::device_matrix_view<const DataT, IndexT> centroids,
                    raft::device_vector_view<IndexT, IndexT> labels,
                    bool normalize_weight,
                    raft::host_scalar_view<DataT> inertia)
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope("kmeans_predict");
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  cudaStream_t stream = resource::get_cuda_stream(handle);
  // Check that parameters are valid
  if (sample_weight.has_value())
    RAFT_EXPECTS(sample_weight.value().extent(0) == n_samples,
                 "invalid parameter (sample_weight!=n_samples)");
  RAFT_EXPECTS(params.n_clusters > 0, "invalid parameter (n_clusters<=0)");
  RAFT_EXPECTS(params.tol > 0, "invalid parameter (tol<=0)");
  RAFT_EXPECTS(params.oversampling_factor >= 0, "invalid parameter (oversampling_factor<0)");
  RAFT_EXPECTS((int)centroids.extent(0) == params.n_clusters,
               "invalid parameter (centroids.extent(0) != n_clusters)");
  RAFT_EXPECTS(centroids.extent(1) == n_features,
               "invalid parameter (centroids.extent(1) != n_features)");

  logger::get(RAFT_NAME).set_level(params.verbosity);
  auto metric = params.metric;

  // Allocate memory
  // Device-accessible allocation of expandable storage used as temporary buffers
  rmm::device_uvector<char> workspace(0, stream);
  auto weight = raft::make_device_vector<DataT, IndexT>(handle, n_samples);
  if (sample_weight.has_value())
    raft::copy(weight.data_handle(), sample_weight.value().data_handle(), n_samples, stream);
  else
    thrust::fill(resource::get_thrust_policy(handle),
                 weight.data_handle(),
                 weight.data_handle() + weight.size(),
                 1);

  // check if weights sum up to n_samples
  if (normalize_weight) checkWeight(handle, weight.view(), workspace);

  auto minClusterAndDistance =
    raft::make_device_vector<raft::KeyValuePair<IndexT, DataT>, IndexT>(handle, n_samples);
  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);

  // L2 norm of X: ||x||^2
  auto L2NormX = raft::make_device_vector<DataT, IndexT>(handle, n_samples);
  if (metric == raft::distance::DistanceType::L2Expanded ||
      metric == raft::distance::DistanceType::L2SqrtExpanded) {
    raft::linalg::rowNorm(L2NormX.data_handle(),
                          X.data_handle(),
                          X.extent(1),
                          X.extent(0),
                          raft::linalg::L2Norm,
                          true,
                          stream);
  }

  // computes minClusterAndDistance[0:n_samples) where  minClusterAndDistance[i]
  // is a <key, value> pair where
  //   'key' is index to a sample in 'centroids' (index of the nearest
  //   centroid) and 'value' is the distance between the sample 'X[i]' and the
  //   'centroid[key]'
  auto l2normx_view =
    raft::make_device_vector_view<const DataT, IndexT>(L2NormX.data_handle(), n_samples);
  detail::minClusterAndDistanceCompute<DataT, IndexT>(handle,
                                                      X,
                                                      centroids,
                                                      minClusterAndDistance.view(),
                                                      l2normx_view,
                                                      L2NormBuf_OR_DistBuf,
                                                      params.metric,
                                                      params.batch_samples,
                                                      params.batch_centroids,
                                                      workspace);

  // calculate cluster cost phi_x(C)
  rmm::device_scalar<DataT> clusterCostD(stream);
  // TODO: add different templates for InType of binaryOp to avoid thrust transform
  thrust::transform(resource::get_thrust_policy(handle),
                    minClusterAndDistance.data_handle(),
                    minClusterAndDistance.data_handle() + minClusterAndDistance.size(),
                    weight.data_handle(),
                    minClusterAndDistance.data_handle(),
                    [=] __device__(const raft::KeyValuePair<IndexT, DataT> kvp, DataT wt) {
                      raft::KeyValuePair<IndexT, DataT> res;
                      res.value = kvp.value * wt;
                      res.key   = kvp.key;
                      return res;
                    });

  detail::computeClusterCost(handle,
                             minClusterAndDistance.view(),
                             workspace,
                             raft::make_device_scalar_view(clusterCostD.data()),
                             raft::value_op{},
                             raft::add_op{});

  thrust::transform(resource::get_thrust_policy(handle),
                    minClusterAndDistance.data_handle(),
                    minClusterAndDistance.data_handle() + minClusterAndDistance.size(),
                    labels.data_handle(),
                    raft::key_op{});

  inertia[0] = clusterCostD.value(stream);
}

template <typename DataT, typename IndexT = int>
void kmeans_predict(raft::resources const& handle,
                    const KMeansParams& params,
                    const DataT* X,
                    const DataT* sample_weight,
                    const DataT* centroids,
                    IndexT n_samples,
                    IndexT n_features,
                    IndexT* labels,
                    bool normalize_weight,
                    DataT& inertia)
{
  auto XView = raft::make_device_matrix_view<const DataT, IndexT>(X, n_samples, n_features);
  auto centroidsView =
    raft::make_device_matrix_view<const DataT, IndexT>(centroids, params.n_clusters, n_features);
  std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weightView{std::nullopt};
  if (sample_weight)
    sample_weightView.emplace(
      raft::make_device_vector_view<const DataT, IndexT>(sample_weight, n_samples));
  auto labelsView  = raft::make_device_vector_view<IndexT, IndexT>(labels, n_samples);
  auto inertiaView = raft::make_host_scalar_view(&inertia);

  detail::kmeans_predict<DataT, IndexT>(handle,
                                        params,
                                        XView,
                                        sample_weightView,
                                        centroidsView,
                                        labelsView,
                                        normalize_weight,
                                        inertiaView);
}

template <typename DataT, typename IndexT = int>
void kmeans_fit_predict(raft::resources const& handle,
                        const KMeansParams& params,
                        raft::device_matrix_view<const DataT, IndexT> X,
                        std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight,
                        std::optional<raft::device_matrix_view<DataT, IndexT>> centroids,
                        raft::device_vector_view<IndexT, IndexT> labels,
                        raft::host_scalar_view<DataT> inertia,
                        raft::host_scalar_view<IndexT> n_iter)
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope("kmeans_fit_predict");
  if (!centroids.has_value()) {
    auto n_features = X.extent(1);
    auto centroids_matrix =
      raft::make_device_matrix<DataT, IndexT>(handle, params.n_clusters, n_features);
    detail::kmeans_fit<DataT, IndexT>(
      handle, params, X, sample_weight, centroids_matrix.view(), inertia, n_iter);
    detail::kmeans_predict<DataT, IndexT>(
      handle, params, X, sample_weight, centroids_matrix.view(), labels, true, inertia);
  } else {
    detail::kmeans_fit<DataT, IndexT>(
      handle, params, X, sample_weight, centroids.value(), inertia, n_iter);
    detail::kmeans_predict<DataT, IndexT>(
      handle, params, X, sample_weight, centroids.value(), labels, true, inertia);
  }
}

template <typename DataT, typename IndexT = int>
void kmeans_fit_predict(raft::resources const& handle,
                        const KMeansParams& params,
                        const DataT* X,
                        const DataT* sample_weight,
                        DataT* centroids,
                        IndexT n_samples,
                        IndexT n_features,
                        IndexT* labels,
                        DataT& inertia,
                        IndexT& n_iter)
{
  auto XView = raft::make_device_matrix_view<const DataT, IndexT>(X, n_samples, n_features);
  std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weightView{std::nullopt};
  if (sample_weight)
    sample_weightView.emplace(
      raft::make_device_vector_view<const DataT, IndexT>(sample_weight, n_samples));
  std::optional<raft::device_matrix_view<DataT, IndexT>> centroidsView{std::nullopt};
  if (centroids)
    centroidsView.emplace(
      raft::make_device_matrix_view<DataT, IndexT>(centroids, params.n_clusters, n_features));
  auto labelsView  = raft::make_device_vector_view<IndexT, IndexT>(labels, n_samples);
  auto inertiaView = raft::make_host_scalar_view(&inertia);
  auto n_iterView  = raft::make_host_scalar_view(&n_iter);

  detail::kmeans_fit_predict<DataT, IndexT>(
    handle, params, XView, sample_weightView, centroidsView, labelsView, inertiaView, n_iterView);
}

/**
 * @brief Transform X to a cluster-distance space.
 *
 * @param[in]     handle        The handle to the cuML library context that
 * manages the CUDA resources.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must
 * be in row-major format
 * @param[in]     centroids     Cluster centroids. The data must be in row-major format.
 * @param[out]    X_new         X transformed in the new space..
 */
template <typename DataT, typename IndexT = int>
void kmeans_transform(raft::resources const& handle,
                      const KMeansParams& params,
                      raft::device_matrix_view<const DataT> X,
                      raft::device_matrix_view<const DataT> centroids,
                      raft::device_matrix_view<DataT> X_new)
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope("kmeans_transform");
  logger::get(RAFT_NAME).set_level(params.verbosity);
  cudaStream_t stream = resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = params.n_clusters;
  auto metric         = params.metric;

  // Device-accessible allocation of expandable storage used as temporary buffers
  rmm::device_uvector<char> workspace(0, stream);
  auto dataBatchSize = getDataBatchSize(params.batch_samples, n_samples);

  // tile over the input data and calculate distance matrix [n_samples x
  // n_clusters]
  for (IndexT dIdx = 0; dIdx < (IndexT)n_samples; dIdx += dataBatchSize) {
    // # of samples for the current batch
    auto ns = std::min(static_cast<IndexT>(dataBatchSize), static_cast<IndexT>(n_samples - dIdx));

    // datasetView [ns x n_features] - view representing the current batch of
    // input dataset
    auto datasetView = raft::make_device_matrix_view<const DataT, IndexT>(
      X.data_handle() + n_features * dIdx, ns, n_features);

    // pairwiseDistanceView [ns x n_clusters]
    auto pairwiseDistanceView = raft::make_device_matrix_view<DataT, IndexT>(
      X_new.data_handle() + n_clusters * dIdx, ns, n_clusters);

    // calculate pairwise distance between cluster centroids and current batch
    // of input dataset
    pairwise_distance_kmeans<DataT, IndexT>(
      handle, datasetView, centroids, pairwiseDistanceView, workspace, metric);
  }
}

template <typename DataT, typename IndexT = int>
void kmeans_transform(raft::resources const& handle,
                      const KMeansParams& params,
                      const DataT* X,
                      const DataT* centroids,
                      IndexT n_samples,
                      IndexT n_features,
                      DataT* X_new)
{
  auto XView = raft::make_device_matrix_view<const DataT, IndexT>(X, n_samples, n_features);
  auto centroidsView =
    raft::make_device_matrix_view<const DataT, IndexT>(centroids, params.n_clusters, n_features);
  auto X_newView = raft::make_device_matrix_view<DataT, IndexT>(X_new, n_samples, n_features);

  detail::kmeans_transform<DataT, IndexT>(handle, params, XView, centroidsView, X_newView);
}
}  // namespace detail
}  // namespace cluster
}  // namespace raft
