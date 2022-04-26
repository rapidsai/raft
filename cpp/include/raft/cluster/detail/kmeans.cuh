/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <optional>
#include <random>

#include <cuda.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/scan.h>

#include <raft/cluster/detail/kmeans_helper.cuh>
#include <raft/cluster/kmeans_params.hpp>
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/distance/distance_type.hpp>
#include <raft/handle.hpp>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/reduce_cols_by_key.cuh>
#include <raft/linalg/reduce_rows_by_key.cuh>
#include <raft/mdarray.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace raft {
namespace cluster {
namespace detail {

// =========================================================
// Init functions
// =========================================================

// Selects 'n_clusters' samples randomly from X
template <typename DataT, typename IndexT>
void initRandom(const raft::handle_t& handle,
                const KMeansParams& params,
                raft::device_matrix_view<const DataT> X,
                rmm::device_uvector<DataT>& centroidsRawData)
{
  cudaStream_t stream = handle.get_stream();
  auto n_features     = X.extent(1);
  auto n_clusters     = params.n_clusters;
  // allocate centroids buffer
  centroidsRawData.resize(n_clusters * n_features, stream);
  auto centroids =
    raft::make_device_matrix_view<DataT>(centroidsRawData.data(), n_clusters, n_features);

  shuffleAndGather<DataT, IndexT>(handle, X, centroids, n_clusters, params.seed, stream);
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
void kmeansPlusPlus(const raft::handle_t& handle,
                    const KMeansParams& params,
                    raft::device_matrix_view<const DataT> X,
                    raft::distance::DistanceType metric,
                    rmm::device_uvector<char>& workspace,
                    rmm::device_uvector<DataT>& centroidsRawData,
                    cudaStream_t stream)
{
  auto n_samples  = X.extent(0);
  auto n_features = X.extent(1);
  auto n_clusters = params.n_clusters;

  // number of seeding trials for each center (except the first)
  auto n_trials = 2 + static_cast<int>(std::ceil(log(n_clusters)));

  /*RAFT_LOG_INFO(
    "Run sequential k-means++ to select %d centroids from %d input samples "
    "(%d seeding trials per iterations)",
    n_clusters,
    n_samples,
    n_trials);*/

  auto dataBatchSize = getDataBatchSize(params, n_samples);

  // temporary buffers
  std::vector<DataT> h_wt(n_samples);
  auto centroidCandidates = raft::make_device_matrix<DataT>(n_trials, n_features, stream);
  auto costPerCandidate   = raft::make_device_vector<DataT>(n_trials, stream);
  auto minClusterDistance = raft::make_device_vector<DataT>(n_samples, stream);

  rmm::device_uvector<DataT> distBuffer(n_trials * n_samples, stream);
  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);
  rmm::device_scalar<DataT> clusterCost(stream);
  rmm::device_scalar<cub::KeyValuePair<int, DataT>> minClusterIndexAndDistance(stream);

  // L2 norm of X: ||c||^2
  auto L2NormX = raft::make_device_vector<DataT>(n_samples, stream);

  if (metric == raft::distance::DistanceType::L2Expanded ||
      metric == raft::distance::DistanceType::L2SqrtExpanded) {
    raft::linalg::rowNorm(
      L2NormX.data(), X.data(), X.extent(1), X.extent(0), raft::linalg::L2Norm, true, stream);
  }

  std::mt19937 gen(params.seed);
  std::uniform_int_distribution<> dis(0, n_samples - 1);

  // <<< Step-1 >>>: C <-- sample a point uniformly at random from X
  auto initialCentroid =
    raft::make_device_matrix_view<const DataT>(X.data() + dis(gen) * n_features, 1, n_features);
  int n_clusters_picked = 1;

  // reset buffer to store the chosen centroid
  centroidsRawData.resize(initialCentroid.size(), stream);
  raft::copy(centroidsRawData.begin(), initialCentroid.data(), initialCentroid.size(), stream);

  //  C = initial set of centroids
  auto centroids = raft::make_device_matrix_view<DataT>(
    centroidsRawData.data(), initialCentroid.extent(0), initialCentroid.extent(1));
  // <<< End of Step-1 >>>

  // Calculate cluster distance, d^2(x, C), for all the points x in X to the nearest centroid
  minClusterDistanceCompute<DataT, IndexT>(handle,
                                           params,
                                           X,
                                           centroids,
                                           minClusterDistance.view(),
                                           L2NormX.view(),
                                           L2NormBuf_OR_DistBuf,
                                           workspace,
                                           metric,
                                           stream);

  // RAFT_LOG_INFO(" k-means++ - Sampled %d/%d centroids", n_clusters_picked, n_clusters);

  // <<<< Step-2 >>> : while |C| < k
  while (n_clusters_picked < n_clusters) {
    // <<< Step-3 >>> : Sample x in X with probability p_x = d^2(x, C) / phi_X (C)
    // Choose 'n_trials' centroid candidates from X with probability proportional to the squared
    // distance to the nearest existing cluster
    raft::copy(h_wt.data(), minClusterDistance.data(), minClusterDistance.size(), stream);
    handle.sync_stream(stream);

    // Note - n_trials is relative small here, we don't need raft::gather call
    std::discrete_distribution<> d(h_wt.begin(), h_wt.end());
    for (int cIdx = 0; cIdx < n_trials; ++cIdx) {
      auto rand_idx = d(gen);
      auto randCentroid =
        raft::make_device_matrix_view<const DataT>(X.data() + n_features * rand_idx, 1, n_features);
      raft::copy(centroidCandidates.data() + cIdx * n_features,
                 randCentroid.data(),
                 randCentroid.size(),
                 stream);
    }

    // Calculate pairwise distance between X and the centroid candidates
    // Output - pwd [n_trails x n_samples]
    auto pwd = raft::make_device_matrix_view<DataT>(distBuffer.data(), n_trials, n_samples);
    pairwise_distance_kmeans<DataT, IndexT>(
      handle, centroidCandidates.view(), X, pwd, workspace, metric, stream);

    // Update nearest cluster distance for each centroid candidate
    // Note pwd and minDistBuf points to same buffer which currently holds pairwise distance values.
    // Outputs minDistanceBuf[m_trails x n_samples] where minDistance[i, :] contains updated
    // minClusterDistance that includes candidate-i
    auto minDistBuf = raft::make_device_matrix_view<DataT>(distBuffer.data(), n_trials, n_samples);
    raft::linalg::matrixVectorOp(
      minDistBuf.data(),
      pwd.data(),
      minClusterDistance.data(),
      pwd.extent(1),
      pwd.extent(0),
      true,
      true,
      [=] __device__(DataT mat, DataT vec) { return vec <= mat ? vec : mat; },
      stream);

    // Calculate costPerCandidate[n_trials] where costPerCandidate[i] is the cluster cost when using
    // centroid candidate-i
    raft::linalg::reduce(costPerCandidate.data(),
                         minDistBuf.data(),
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
                                costPerCandidate.data(),
                                minClusterIndexAndDistance.data(),
                                costPerCandidate.extent(0));

      // Allocate temporary storage
      workspace.resize(temp_storage_bytes, stream);

      // Run argmin-reduction
      cub::DeviceReduce::ArgMin(workspace.data(),
                                temp_storage_bytes,
                                costPerCandidate.data(),
                                minClusterIndexAndDistance.data(),
                                costPerCandidate.extent(0));

      int bestCandidateIdx = -1;
      raft::copy(&bestCandidateIdx, &minClusterIndexAndDistance.data()->key, 1, stream);
      /// <<< End of Step-3 >>>

      /// <<< Step-4 >>>: C = C U {x}
      // Update minimum cluster distance corresponding to the chosen centroid candidate
      raft::copy(minClusterDistance.data(),
                 minDistBuf.data() + bestCandidateIdx * n_samples,
                 n_samples,
                 stream);

      raft::copy(centroidsRawData.data() + n_clusters_picked * n_features,
                 centroidCandidates.data() + bestCandidateIdx * n_features,
                 n_features,
                 stream);

      ++n_clusters_picked;
      /// <<< End of Step-4 >>>
    }

    // RAFT_LOG_INFO(" k-means++ - Sampled %d/%d centroids", n_clusters_picked, n_clusters);
  }  /// <<<< Step-5 >>>
}

template <typename DataT, typename IndexT>
void initKMeansPlusPlus(const raft::handle_t& handle,
                        const KMeansParams& params,
                        raft::device_matrix_view<const DataT> X,
                        rmm::device_uvector<DataT>& centroidsRawData,
                        rmm::device_uvector<char>& workspace)
{
  cudaStream_t stream = handle.get_stream();
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = params.n_clusters;
  auto metric         = static_cast<raft::distance::DistanceType>(params.metric);
  centroidsRawData.resize(n_clusters * n_features, stream);
  kmeansPlusPlus<DataT, IndexT>(handle, params, X, metric, workspace, centroidsRawData, stream);
}

template <typename DataT, typename IndexT>
void kmeans_fit_main(const raft::handle_t& handle,
                     const KMeansParams& params,
                     raft::device_matrix_view<const DataT> X,
                     raft::device_vector_view<const DataT> weight,
                     rmm::device_uvector<DataT>& centroidsRawData,
                     DataT& inertia,
                     IndexT& n_iter,
                     rmm::device_uvector<char>& workspace)
{
  // logger::get(RAFT_NAME).set_level(params.verbosity);
  cudaStream_t stream = handle.get_stream();
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = params.n_clusters;
  auto metric         = static_cast<raft::distance::DistanceType>(params.metric);

  // stores (key, value) pair corresponding to each sample where
  //   - key is the index of nearest cluster
  //   - value is the distance to the nearest cluster
  auto minClusterAndDistance =
    raft::make_device_vector<cub::KeyValuePair<IndexT, DataT>>(n_samples, stream);

  // temporary buffer to store L2 norm of centroids or distance matrix,
  // destructor releases the resource
  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);

  // temporary buffer to store intermediate centroids, destructor releases the
  // resource
  auto newCentroids = raft::make_device_matrix<DataT>(n_clusters, n_features, stream);

  // temporary buffer to store weights per cluster, destructor releases the
  // resource
  auto wtInCluster = raft::make_device_vector<DataT>(n_clusters, stream);

  rmm::device_scalar<cub::KeyValuePair<IndexT, DataT>> clusterCostD(stream);

  // L2 norm of X: ||x||^2
  auto L2NormX = raft::make_device_vector<DataT>(n_samples, stream);
  if (metric == raft::distance::DistanceType::L2Expanded ||
      metric == raft::distance::DistanceType::L2SqrtExpanded) {
    raft::linalg::rowNorm(
      L2NormX.data(), X.data(), X.extent(1), X.extent(0), raft::linalg::L2Norm, true, stream);
  }

  /*RAFT_LOG_INFO(
    "Calling KMeans.fit with %d samples of input data and the initialized "
    "cluster centers",
    n_samples);*/

  DataT priorClusteringCost = 0;
  for (n_iter = 1; n_iter <= params.max_iter; ++n_iter) {
    /*RAFT_LOG_INFO(
      "KMeans.fit: Iteration-%d: fitting the model using the initialized "
      "cluster centers",
      n_iter);*/

    auto centroids = raft::make_device_matrix_view(centroidsRawData.data(), n_clusters, n_features);

    // computes minClusterAndDistance[0:n_samples) where
    // minClusterAndDistance[i] is a <key, value> pair where
    //   'key' is index to an sample in 'centroids' (index of the nearest
    //   centroid) and 'value' is the distance between the sample 'X[i]' and the
    //   'centroid[key]'
    minClusterAndDistanceCompute<DataT, IndexT>(handle,
                                                params,
                                                X,
                                                centroids,
                                                minClusterAndDistance.view(),
                                                L2NormX.view(),
                                                L2NormBuf_OR_DistBuf,
                                                workspace,
                                                metric,
                                                stream);

    // Using TransformInputIteratorT to dereference an array of
    // cub::KeyValuePair and converting them to just return the Key to be used
    // in reduce_rows_by_key prims
    KeyValueIndexOp<IndexT, DataT> conversion_op;
    cub::TransformInputIterator<IndexT,
                                KeyValueIndexOp<IndexT, DataT>,
                                cub::KeyValuePair<IndexT, DataT>*>
      itr(minClusterAndDistance.data(), conversion_op);

    workspace.resize(n_samples, stream);

    // Calculates weighted sum of all the samples assigned to cluster-i and store the
    // result in newCentroids[i]
    raft::linalg::reduce_rows_by_key((DataT*)X.data(),
                                     X.extent(1),
                                     itr,
                                     weight.data(),
                                     workspace.data(),
                                     X.extent(0),
                                     X.extent(1),
                                     n_clusters,
                                     newCentroids.data(),
                                     stream);

    // Reduce weights by key to compute weight in each cluster
    raft::linalg::reduce_cols_by_key(weight.data(),
                                     itr,
                                     wtInCluster.data(),
                                     (IndexT)1,
                                     (IndexT)weight.extent(0),
                                     (IndexT)n_clusters,
                                     stream);

    // Computes newCentroids[i] = newCentroids[i]/wtInCluster[i] where
    //   newCentroids[n_clusters x n_features] - 2D array, newCentroids[i] has sum of all the
    //   samples assigned to cluster-i wtInCluster[n_clusters] - 1D array, wtInCluster[i] contains #
    //   of samples in cluster-i.
    // Note - when wtInCluster[i] is 0, newCentroid[i] is reset to 0
    raft::linalg::matrixVectorOp(
      newCentroids.data(),
      newCentroids.data(),
      wtInCluster.data(),
      newCentroids.extent(1),
      newCentroids.extent(0),
      true,
      false,
      [=] __device__(DataT mat, DataT vec) {
        if (vec == 0)
          return DataT(0);
        else
          return mat / vec;
      },
      stream);

    // copy centroids[i] to newCentroids[i] when wtInCluster[i] is 0
    cub::ArgIndexInputIterator<DataT*> itr_wt(wtInCluster.data());
    raft::matrix::gather_if(
      centroids.data(),
      centroids.extent(1),
      centroids.extent(0),
      itr_wt,
      itr_wt,
      wtInCluster.size(),
      newCentroids.data(),
      [=] __device__(cub::KeyValuePair<ptrdiff_t, DataT> map) {  // predicate
        // copy when the # of samples in the cluster is 0
        if (map.value == 0)
          return true;
        else
          return false;
      },
      [=] __device__(cub::KeyValuePair<ptrdiff_t, DataT> map) {  // map
        return map.key;
      },
      stream);

    // compute the squared norm between the newCentroids and the original
    // centroids, destructor releases the resource
    auto sqrdNorm = raft::make_device_scalar(DataT(0), stream);
    raft::linalg::mapThenSumReduce(
      sqrdNorm.data(),
      newCentroids.size(),
      [=] __device__(const DataT a, const DataT b) {
        DataT diff = a - b;
        return diff * diff;
      },
      stream,
      centroids.data(),
      newCentroids.data());

    DataT sqrdNormError = 0;
    raft::copy(&sqrdNormError, sqrdNorm.data(), sqrdNorm.size(), stream);

    raft::copy(centroidsRawData.data(), newCentroids.data(), newCentroids.size(), stream);

    bool done = false;
    if (params.inertia_check) {
      // calculate cluster cost phi_x(C)
      computeClusterCost(
        handle,
        minClusterAndDistance.view(),
        workspace,
        clusterCostD.data(),
        [] __device__(const cub::KeyValuePair<IndexT, DataT>& a,
                      const cub::KeyValuePair<IndexT, DataT>& b) {
          cub::KeyValuePair<IndexT, DataT> res;
          res.key   = 0;
          res.value = a.value + b.value;
          return res;
        },
        stream);

      DataT curClusteringCost = 0;
      raft::copy(&curClusteringCost, &(clusterCostD.data()->value), 1, stream);

      handle.sync_stream(stream);
      ASSERT(curClusteringCost != (DataT)0.0,
             "Too few points and centriods being found is getting 0 cost from "
             "centers");

      if (n_iter > 1) {
        DataT delta = curClusteringCost / priorClusteringCost;
        if (delta > 1 - params.tol) done = true;
      }
      priorClusteringCost = curClusteringCost;
    }

    handle.sync_stream(stream);
    if (sqrdNormError < params.tol) done = true;

    if (done) {
      // RAFT_LOG_INFO("Threshold triggered after %d iterations. Terminating early.", n_iter);
      break;
    }
  }

  auto centroids = raft::make_device_matrix_view(centroidsRawData.data(), n_clusters, n_features);

  minClusterAndDistanceCompute<DataT, IndexT>(handle,
                                              params,
                                              X,
                                              centroids,
                                              minClusterAndDistance.view(),
                                              L2NormX.view(),
                                              L2NormBuf_OR_DistBuf,
                                              workspace,
                                              metric,
                                              stream);

  thrust::transform(handle.get_thrust_policy(),
                    minClusterAndDistance.data(),
                    minClusterAndDistance.data() + minClusterAndDistance.size(),
                    weight.data(),
                    minClusterAndDistance.data(),
                    [=] __device__(const cub::KeyValuePair<IndexT, DataT> kvp, DataT wt) {
                      cub::KeyValuePair<IndexT, DataT> res;
                      res.value = kvp.value * wt;
                      res.key   = kvp.key;
                      return res;
                    });

  // calculate cluster cost phi_x(C)
  computeClusterCost(
    handle,
    minClusterAndDistance.view(),
    workspace,
    clusterCostD.data(),
    [] __device__(const cub::KeyValuePair<IndexT, DataT>& a,
                  const cub::KeyValuePair<IndexT, DataT>& b) {
      cub::KeyValuePair<IndexT, DataT> res;
      res.key   = 0;
      res.value = a.value + b.value;
      return res;
    },
    stream);

  raft::copy(&inertia, &(clusterCostD.data()->value), 1, stream);

  /*RAFT_LOG_INFO("KMeans.fit: completed after %d iterations with %f inertia ",
                n_iter > params.max_iter ? n_iter - 1 : n_iter,
                inertia);*/
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

 */
template <typename DataT, typename IndexT>
void initScalableKMeansPlusPlus(const raft::handle_t& handle,
                                const KMeansParams& params,
                                raft::device_matrix_view<const DataT> X,
                                rmm::device_uvector<DataT>& centroidsRawData,
                                rmm::device_uvector<char>& workspace)
{
  cudaStream_t stream = handle.get_stream();
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = params.n_clusters;
  auto metric         = static_cast<raft::distance::DistanceType>(params.metric);

  raft::random::Rng rng(params.seed, raft::random::GeneratorType::GenPhilox);

  // <<<< Step-1 >>> : C <- sample a point uniformly at random from X
  std::mt19937 gen(params.seed);
  std::uniform_int_distribution<> dis(0, n_samples - 1);

  auto cIdx            = dis(gen);
  auto initialCentroid = raft::make_device_matrix_view(X.data() + cIdx * n_features, 1, n_features);

  // flag the sample that is chosen as initial centroid
  std::vector<IndexT> h_isSampleCentroid(n_samples);
  std::fill(h_isSampleCentroid.begin(), h_isSampleCentroid.end(), 0);
  h_isSampleCentroid[cIdx] = 1;

  // device buffer to flag the sample that is chosen as initial centroid
  auto isSampleCentroid = raft::make_device_vector<IndexT>(n_samples, stream);

  raft::copy(isSampleCentroid.data(), h_isSampleCentroid.data(), isSampleCentroid.size(), stream);

  rmm::device_uvector<DataT> centroidsBuf(initialCentroid.size(), stream);

  // reset buffer to store the chosen centroid
  raft::copy(centroidsBuf.data(), initialCentroid.data(), initialCentroid.size(), stream);

  auto potentialCentroids = raft::make_device_matrix_view<DataT>(
    centroidsBuf.data(), initialCentroid.extent(0), initialCentroid.extent(1));
  // <<< End of Step-1 >>>

  // temporary buffer to store L2 norm of centroids or distance matrix,
  // destructor releases the resource
  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);

  // L2 norm of X: ||x||^2
  auto L2NormX = raft::make_device_vector<DataT>(n_samples, stream);
  if (metric == raft::distance::DistanceType::L2Expanded ||
      metric == raft::distance::DistanceType::L2SqrtExpanded) {
    raft::linalg::rowNorm(
      L2NormX.data(), X.data(), X.extent(1), X.extent(0), raft::linalg::L2Norm, true, stream);
  }

  auto minClusterDistanceVec = raft::make_device_vector<DataT>(n_samples, stream);
  auto uniformRands          = raft::make_device_vector<DataT>(n_samples, stream);
  rmm::device_scalar<DataT> clusterCost(stream);

  // <<< Step-2 >>>: psi <- phi_X (C)
  minClusterDistanceCompute<DataT, IndexT>(handle,
                                           params,
                                           X,
                                           potentialCentroids,
                                           minClusterDistanceVec.view(),
                                           L2NormX.view(),
                                           L2NormBuf_OR_DistBuf,
                                           workspace,
                                           metric,
                                           stream);

  // compute partial cluster cost from the samples in rank
  computeClusterCost(
    handle,
    minClusterDistanceVec.view(),
    workspace,
    clusterCost.data(),
    [] __device__(const DataT& a, const DataT& b) { return a + b; },
    stream);

  auto psi = clusterCost.value(stream);

  // <<< End of Step-2 >>>

  // Scalable kmeans++ paper claims 8 rounds is sufficient
  handle.sync_stream(stream);
  int niter = std::min(8, (int)ceil(log(psi)));
  // RAFT_LOG_INFO("KMeans||: psi = %g, log(psi) = %g, niter = %d ", psi, log(psi), niter);

  // <<<< Step-3 >>> : for O( log(psi) ) times do
  for (int iter = 0; iter < niter; ++iter) {
    /*RAFT_LOG_INFO(
        "KMeans|| - Iteration %d: # potential centroids sampled - %d",
        iter,
        potentialCentroids.extent(0));*/

    minClusterDistanceCompute<DataT, IndexT>(handle,
                                             params,
                                             X,
                                             potentialCentroids,
                                             minClusterDistanceVec.view(),
                                             L2NormX.view(),
                                             L2NormBuf_OR_DistBuf,
                                             workspace,
                                             metric,
                                             stream);

    computeClusterCost(
      handle,
      minClusterDistanceVec.view(),
      workspace,
      clusterCost.data(),
      [] __device__(const DataT& a, const DataT& b) { return a + b; },
      stream);

    psi = clusterCost.value(stream);

    // <<<< Step-4 >>> : Sample each point x in X independently and identify new
    // potentialCentroids
    rng.uniform(uniformRands.data(), uniformRands.extent(0), (DataT)0, (DataT)1, stream);

    SamplingOp<DataT, IndexT> select_op(
      psi, params.oversampling_factor, n_clusters, uniformRands.data(), isSampleCentroid.data());

    auto Cp = sampleCentroids<DataT, IndexT>(handle,
                                             X,
                                             minClusterDistanceVec.view(),
                                             isSampleCentroid.view(),
                                             select_op,
                                             workspace,
                                             stream);
    /// <<<< End of Step-4 >>>>

    /// <<<< Step-5 >>> : C = C U C'
    // append the data in Cp to the buffer holding the potentialCentroids
    centroidsBuf.resize(centroidsBuf.size() + Cp.size(), stream);
    raft::copy(centroidsBuf.data() + centroidsBuf.size() - Cp.size(), Cp.data(), Cp.size(), stream);

    IndexT tot_centroids = potentialCentroids.extent(0) + Cp.extent(0);
    potentialCentroids =
      raft::make_device_matrix_view<DataT>(centroidsBuf.data(), tot_centroids, n_features);
    /// <<<< End of Step-5 >>>
  }  /// <<<< Step-6 >>>

  // RAFT_LOG_INFO("KMeans||: total # potential centroids sampled - %d",
  // potentialCentroids.extent(0));

  if ((int)potentialCentroids.extent(0) > n_clusters) {
    // <<< Step-7 >>>: For x in C, set w_x to be the number of pts closest to X
    // temporary buffer to store the sample count per cluster, destructor
    // releases the resource
    auto weight = raft::make_device_vector<DataT>(potentialCentroids.extent(0), stream);

    countSamplesInCluster<DataT, IndexT>(handle,
                                         params,
                                         X,
                                         L2NormX.view(),
                                         potentialCentroids,
                                         workspace,
                                         metric,
                                         weight.view(),
                                         stream);

    // <<< end of Step-7 >>>

    // Step-8: Recluster the weighted points in C into k clusters
    centroidsRawData.resize(n_clusters * n_features, stream);
    kmeansPlusPlus<DataT, IndexT>(
      handle, params, potentialCentroids, metric, workspace, centroidsRawData, stream);

    DataT inertia = 0;
    int n_iter    = 0;
    KMeansParams default_params;
    default_params.n_clusters = params.n_clusters;

    kmeans_fit_main<DataT, IndexT>(handle,
                                   default_params,
                                   potentialCentroids,
                                   weight.view(),
                                   centroidsRawData,
                                   inertia,
                                   n_iter,
                                   workspace);

  } else if ((int)potentialCentroids.extent(0) < n_clusters) {
    // supplement with random
    auto n_random_clusters = n_clusters - potentialCentroids.extent(0);

    /*RAFT_LOG_INFO(
      "[Warning!] KMeans||: found fewer than %d centroids during "
      "initialization (found %d centroids, remaining %d centroids will be "
      "chosen randomly from input samples)",
      n_clusters,
      potentialCentroids.extent(0),
      n_random_clusters);*/

    // reset buffer to store the chosen centroid
    centroidsRawData.resize(n_clusters * n_features, stream);

    // generate `n_random_clusters` centroids
    KMeansParams rand_params;
    rand_params.init       = KMeansParams::InitMethod::Random;
    rand_params.n_clusters = n_random_clusters;
    initRandom<DataT, IndexT>(handle, rand_params, X, centroidsRawData);

    // copy centroids generated during kmeans|| iteration to the buffer
    raft::copy(centroidsRawData.data() + n_random_clusters * n_features,
               potentialCentroids.data(),
               potentialCentroids.size(),
               stream);
  } else {
    // found the required n_clusters
    centroidsRawData.resize(n_clusters * n_features, stream);
    raft::copy(
      centroidsRawData.data(), potentialCentroids.data(), potentialCentroids.size(), stream);
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
void kmeans_fit(handle_t const& handle,
                const KMeansParams& params,
                raft::device_matrix_view<const DataT> X,
                std::optional<raft::device_vector_view<const DataT>> sample_weight,
                raft::device_matrix_view<DataT> centroids,
                DataT& inertia,
                IndexT& n_iter)
{
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  cudaStream_t stream = handle.get_stream();
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

  // logger::get(RAFT_NAME).set_level(params.verbosity);

  // Allocate memory
  rmm::device_uvector<char> workspace(0, stream);
  auto weight = raft::make_device_vector<DataT>(handle, n_samples);
  if (sample_weight.has_value())
    raft::copy(weight.data(), sample_weight.value().data(), n_samples, stream);
  else
    thrust::fill(handle.get_thrust_policy(), weight.data(), weight.data() + weight.size(), 1);

  // check if weights sum up to n_samples
  checkWeight<DataT>(handle, weight.view(), stream);

  rmm::device_uvector<DataT> centroidsRawData(0, stream);

  auto n_init = params.n_init;
  if (params.init == KMeansParams::InitMethod::Array && n_init != 1) {
    /*RAFT_LOG_INFO(
      "Explicit initial center position passed: performing only one init in "
      "k-means instead of n_init=%d",
      n_init);*/
    n_init = 1;
  }

  std::mt19937 gen(params.seed);
  inertia = std::numeric_limits<DataT>::max();

  for (auto seed_iter = 0; seed_iter < n_init; ++seed_iter) {
    KMeansParams iter_params = params;
    iter_params.seed         = gen();

    DataT iter_inertia    = std::numeric_limits<DataT>::max();
    IndexT n_current_iter = 0;
    if (iter_params.init == KMeansParams::InitMethod::Random) {
      // initializing with random samples from input dataset
      /*RAFT_LOG_INFO(
        "KMeans.fit (Iteration-%d/%d): initialize cluster centers by "
        "randomly choosing from the "
        "input data.",
        seed_iter + 1,
        n_init);*/
      initRandom<DataT, IndexT>(handle, iter_params, X, centroidsRawData);
    } else if (iter_params.init == KMeansParams::InitMethod::KMeansPlusPlus) {
      // default method to initialize is kmeans++
      /*RAFT_LOG_INFO(
        "KMeans.fit (Iteration-%d/%d): initialize cluster centers using "
        "k-means++ algorithm.",
        seed_iter + 1,
        n_init);*/
      if (iter_params.oversampling_factor == 0)
        initKMeansPlusPlus<DataT, IndexT>(handle, iter_params, X, centroidsRawData, workspace);
      else
        initScalableKMeansPlusPlus<DataT, IndexT>(
          handle, iter_params, X, centroidsRawData, workspace);
    } else if (iter_params.init == KMeansParams::InitMethod::Array) {
      /*RAFT_LOG_INFO(
        "KMeans.fit (Iteration-%d/%d): initialize cluster centers from "
        "the ndarray array input "
        "passed to init arguement.",
        seed_iter + 1,
        n_init);*/
      centroidsRawData.resize(params.n_clusters * n_features, stream);
      raft::copy(
        centroidsRawData.begin(), centroids.data(), iter_params.n_clusters * n_features, stream);
    } else {
      THROW("unknown initialization method to select initial centers");
    }

    kmeans_fit_main<DataT, IndexT>(handle,
                                   iter_params,
                                   X,
                                   weight.view(),
                                   centroidsRawData,
                                   iter_inertia,
                                   n_current_iter,
                                   workspace);
    if (iter_inertia < inertia) {
      inertia = iter_inertia;
      n_iter  = n_current_iter;
      raft::copy(
        centroids.data(), centroidsRawData.data(), iter_params.n_clusters * n_features, stream);
    }
    /*RAFT_LOG_INFO("KMeans.fit after iteration-%d/%d: inertia - %f, n_iter - %d",
                  seed_iter + 1,
                  n_init,
                  inertia,
                  n_iter);*/
  }
  // RAFT_LOG_INFO("KMeans.fit: async call returned (fit could still be running on the device)");
}

template <typename DataT, typename IndexT>
void kmeans_predict(handle_t const& handle,
                    const KMeansParams& params,
                    raft::device_matrix_view<const DataT> X,
                    std::optional<raft::device_vector_view<const DataT>> sample_weight,
                    raft::device_matrix_view<const DataT> centroids,
                    raft::device_vector_view<IndexT> labels,
                    bool normalize_weight,
                    DataT& inertia)
{
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  cudaStream_t stream = handle.get_stream();
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

  // logger::get(RAFT_NAME).set_level(params.verbosity);
  auto metric = static_cast<raft::distance::DistanceType>(params.metric);

  // Allocate memory
  // Device-accessible allocation of expandable storage used as temorary buffers
  rmm::device_uvector<char> workspace(0, stream);
  auto weight = raft::make_device_vector<DataT>(handle, n_samples);
  if (sample_weight.has_value())
    raft::copy(weight.data(), sample_weight.value().data(), n_samples, stream);
  else
    thrust::fill(handle.get_thrust_policy(), weight.data(), weight.data() + weight.size(), 1);

  // underlying expandable storage that holds labels
  rmm::device_uvector<IndexT> labelsRawData(0, stream);

  // check if weights sum up to n_samples
  if (normalize_weight) checkWeight(handle, weight.view(), stream);

  auto minClusterAndDistance =
    raft::make_device_vector<cub::KeyValuePair<IndexT, DataT>>(n_samples, stream);
  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);

  // L2 norm of X: ||x||^2
  auto L2NormX = raft::make_device_vector<DataT>(n_samples, stream);
  if (metric == raft::distance::DistanceType::L2Expanded ||
      metric == raft::distance::DistanceType::L2SqrtExpanded) {
    raft::linalg::rowNorm(
      L2NormX.data(), X.data(), X.extent(1), X.extent(0), raft::linalg::L2Norm, true, stream);
  }

  // computes minClusterAndDistance[0:n_samples) where  minClusterAndDistance[i]
  // is a <key, value> pair where
  //   'key' is index to an sample in 'centroids' (index of the nearest
  //   centroid) and 'value' is the distance between the sample 'X[i]' and the
  //   'centroid[key]'
  minClusterAndDistanceCompute<DataT, IndexT>(handle,
                                              params,
                                              X,
                                              centroids,
                                              minClusterAndDistance.view(),
                                              L2NormX.view(),
                                              L2NormBuf_OR_DistBuf,
                                              workspace,
                                              metric,
                                              stream);

  // calculate cluster cost phi_x(C)
  rmm::device_scalar<cub::KeyValuePair<IndexT, DataT>> clusterCostD(stream);
  thrust::transform(handle.get_thrust_policy(),
                    minClusterAndDistance.data(),
                    minClusterAndDistance.data() + minClusterAndDistance.size(),
                    weight.data(),
                    minClusterAndDistance.data(),
                    [=] __device__(const cub::KeyValuePair<IndexT, DataT> kvp, DataT wt) {
                      cub::KeyValuePair<IndexT, DataT> res;
                      res.value = kvp.value * wt;
                      res.key   = kvp.key;
                      return res;
                    });

  computeClusterCost(
    handle,
    minClusterAndDistance.view(),
    workspace,
    clusterCostD.data(),
    [] __device__(const cub::KeyValuePair<IndexT, DataT>& a,
                  const cub::KeyValuePair<IndexT, DataT>& b) {
      cub::KeyValuePair<IndexT, DataT> res;
      res.key   = 0;
      res.value = a.value + b.value;
      return res;
    },
    stream);

  raft::copy(&inertia, &(clusterCostD.data()->value), 1, stream);

  labelsRawData.resize(n_samples, stream);

  thrust::transform(handle.get_thrust_policy(),
                    minClusterAndDistance.data(),
                    minClusterAndDistance.data() + minClusterAndDistance.size(),
                    labelsRawData.data(),
                    [=] __device__(cub::KeyValuePair<IndexT, DataT> pair) { return pair.key; });

  raft::copy(labels.data(), labelsRawData.data(), n_samples, stream);
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
void kmeans_transform(const raft::handle_t& handle,
                      const KMeansParams& params,
                      raft::device_matrix_view<const DataT> X,
                      raft::device_matrix_view<const DataT> centroids,
                      raft::device_matrix_view<DataT> X_new)
{
  // logger::get(RAFT_NAME).set_level(params.verbosity);
  cudaStream_t stream = handle.get_stream();
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = params.n_clusters;
  auto metric         = static_cast<raft::distance::DistanceType>(params.metric);

  // Device-accessible allocation of expandable storage used as temorary buffers
  rmm::device_uvector<char> workspace(0, stream);
  auto dataBatchSize = getDataBatchSize(params, n_samples);

  // tile over the input data and calculate distance matrix [n_samples x
  // n_clusters]
  for (IndexT dIdx = 0; dIdx < n_samples; dIdx += dataBatchSize) {
    // # of samples for the current batch
    auto ns = std::min(dataBatchSize, n_samples - dIdx);

    // datasetView [ns x n_features] - view representing the current batch of
    // input dataset
    auto datasetView = raft::make_device_matrix_view(X.data() + n_features * dIdx, ns, n_features);

    // pairwiseDistanceView [ns x n_clusters]
    auto pairwiseDistanceView =
      raft::make_device_matrix_view(X_new.data() + n_clusters * dIdx, ns, n_clusters);

    // calculate pairwise distance between cluster centroids and current batch
    // of input dataset
    pairwise_distance_kmeans<DataT, IndexT>(
      handle, datasetView, centroids, pairwiseDistanceView, workspace, metric, stream);
  }
}
}  // namespace detail
}  // namespace cluster
}  // namespace raft
