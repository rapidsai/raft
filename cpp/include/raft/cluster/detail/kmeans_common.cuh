/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <raft/cluster/kmeans_types.hpp>
#include <raft/core/cudart_utils.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/kvp.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance.cuh>
#include <raft/distance/distance_types.hpp>
#include <raft/distance/fused_l2_nn.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/reduce_rows_by_key.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/random/permute.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>
#include <cuda.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <optional>
#include <random>

namespace raft {
namespace cluster {
namespace detail {

template <typename DataT, typename IndexT>
struct SamplingOp {
  DataT* rnd;
  uint8_t* flag;
  DataT cluster_cost;
  double oversampling_factor;
  IndexT n_clusters;

  CUB_RUNTIME_FUNCTION __forceinline__
  SamplingOp(DataT c, double l, IndexT k, DataT* rand, uint8_t* ptr)
    : cluster_cost(c), oversampling_factor(l), n_clusters(k), rnd(rand), flag(ptr)
  {
  }

  __host__ __device__ __forceinline__ bool operator()(
    const raft::KeyValuePair<ptrdiff_t, DataT>& a) const
  {
    DataT prob_threshold = (DataT)rnd[a.key];

    DataT prob_x = ((oversampling_factor * n_clusters * a.value) / cluster_cost);

    return !flag[a.key] && (prob_x > prob_threshold);
  }
};

template <typename IndexT, typename DataT>
struct KeyValueIndexOp {
  __host__ __device__ __forceinline__ IndexT
  operator()(const raft::KeyValuePair<IndexT, DataT>& a) const
  {
    return a.key;
  }
};

// Computes the intensity histogram from a sequence of labels
template <typename SampleIteratorT, typename CounterT, typename IndexT>
void countLabels(raft::resources const& handle,
                 SampleIteratorT labels,
                 CounterT* count,
                 IndexT n_samples,
                 IndexT n_clusters,
                 rmm::device_uvector<char>& workspace)
{
  cudaStream_t stream = resource::get_cuda_stream(handle);

  // CUB::DeviceHistogram requires a signed index type
  typedef typename std::make_signed_t<IndexT> CubIndexT;

  CubIndexT num_levels  = n_clusters + 1;
  CubIndexT lower_level = 0;
  CubIndexT upper_level = n_clusters;

  size_t temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(nullptr,
                                                    temp_storage_bytes,
                                                    labels,
                                                    count,
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    static_cast<CubIndexT>(n_samples),
                                                    stream));

  workspace.resize(temp_storage_bytes, stream);

  RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(workspace.data(),
                                                    temp_storage_bytes,
                                                    labels,
                                                    count,
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    static_cast<CubIndexT>(n_samples),
                                                    stream));
}

template <typename DataT, typename IndexT>
void checkWeight(raft::resources const& handle,
                 raft::device_vector_view<DataT, IndexT> weight,
                 rmm::device_uvector<char>& workspace)
{
  cudaStream_t stream = resource::get_cuda_stream(handle);
  auto wt_aggr        = raft::make_device_scalar<DataT>(handle, 0);
  auto n_samples      = weight.extent(0);

  size_t temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceReduce::Sum(
    nullptr, temp_storage_bytes, weight.data_handle(), wt_aggr.data_handle(), n_samples, stream));

  workspace.resize(temp_storage_bytes, stream);

  RAFT_CUDA_TRY(cub::DeviceReduce::Sum(workspace.data(),
                                       temp_storage_bytes,
                                       weight.data_handle(),
                                       wt_aggr.data_handle(),
                                       n_samples,
                                       stream));
  DataT wt_sum = 0;
  raft::copy(&wt_sum, wt_aggr.data_handle(), 1, stream);
  resource::sync_stream(handle, stream);

  if (wt_sum != n_samples) {
    RAFT_LOG_DEBUG(
      "[Warning!] KMeans: normalizing the user provided sample weight to "
      "sum up to %d samples",
      n_samples);

    auto scale = static_cast<DataT>(n_samples) / wt_sum;
    raft::linalg::unaryOp(weight.data_handle(),
                          weight.data_handle(),
                          n_samples,
                          raft::mul_const_op<DataT>{scale},
                          stream);
  }
}

template <typename IndexT>
IndexT getDataBatchSize(int batch_samples, IndexT n_samples)
{
  auto minVal = std::min(static_cast<IndexT>(batch_samples), n_samples);
  return (minVal == 0) ? n_samples : minVal;
}

template <typename IndexT>
IndexT getCentroidsBatchSize(int batch_centroids, IndexT n_local_clusters)
{
  auto minVal = std::min(static_cast<IndexT>(batch_centroids), n_local_clusters);
  return (minVal == 0) ? n_local_clusters : minVal;
}

template <typename InputT,
          typename OutputT,
          typename MainOpT,
          typename ReductionOpT,
          typename IndexT = int>
void computeClusterCost(raft::resources const& handle,
                        raft::device_vector_view<InputT, IndexT> minClusterDistance,
                        rmm::device_uvector<char>& workspace,
                        raft::device_scalar_view<OutputT> clusterCost,
                        MainOpT main_op,
                        ReductionOpT reduction_op)
{
  cudaStream_t stream = resource::get_cuda_stream(handle);

  cub::TransformInputIterator<OutputT, MainOpT, InputT*> itr(minClusterDistance.data_handle(),
                                                             main_op);

  size_t temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceReduce::Reduce(nullptr,
                                          temp_storage_bytes,
                                          itr,
                                          clusterCost.data_handle(),
                                          minClusterDistance.size(),
                                          reduction_op,
                                          OutputT(),
                                          stream));

  workspace.resize(temp_storage_bytes, stream);

  RAFT_CUDA_TRY(cub::DeviceReduce::Reduce(workspace.data(),
                                          temp_storage_bytes,
                                          itr,
                                          clusterCost.data_handle(),
                                          minClusterDistance.size(),
                                          reduction_op,
                                          OutputT(),
                                          stream));
}

template <typename DataT, typename IndexT>
void sampleCentroids(raft::resources const& handle,
                     raft::device_matrix_view<const DataT, IndexT> X,
                     raft::device_vector_view<DataT, IndexT> minClusterDistance,
                     raft::device_vector_view<uint8_t, IndexT> isSampleCentroid,
                     SamplingOp<DataT, IndexT>& select_op,
                     rmm::device_uvector<DataT>& inRankCp,
                     rmm::device_uvector<char>& workspace)
{
  cudaStream_t stream  = resource::get_cuda_stream(handle);
  auto n_local_samples = X.extent(0);
  auto n_features      = X.extent(1);

  auto nSelected = raft::make_device_scalar<IndexT>(handle, 0);
  cub::ArgIndexInputIterator<DataT*> ip_itr(minClusterDistance.data_handle());
  auto sampledMinClusterDistance =
    raft::make_device_vector<raft::KeyValuePair<ptrdiff_t, DataT>, IndexT>(handle, n_local_samples);
  size_t temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceSelect::If(nullptr,
                                      temp_storage_bytes,
                                      ip_itr,
                                      sampledMinClusterDistance.data_handle(),
                                      nSelected.data_handle(),
                                      n_local_samples,
                                      select_op,
                                      stream));

  workspace.resize(temp_storage_bytes, stream);

  RAFT_CUDA_TRY(cub::DeviceSelect::If(workspace.data(),
                                      temp_storage_bytes,
                                      ip_itr,
                                      sampledMinClusterDistance.data_handle(),
                                      nSelected.data_handle(),
                                      n_local_samples,
                                      select_op,
                                      stream));

  IndexT nPtsSampledInRank = 0;
  raft::copy(&nPtsSampledInRank, nSelected.data_handle(), 1, stream);
  resource::sync_stream(handle, stream);

  uint8_t* rawPtr_isSampleCentroid = isSampleCentroid.data_handle();
  thrust::for_each_n(resource::get_thrust_policy(handle),
                     sampledMinClusterDistance.data_handle(),
                     nPtsSampledInRank,
                     [=] __device__(raft::KeyValuePair<ptrdiff_t, DataT> val) {
                       rawPtr_isSampleCentroid[val.key] = 1;
                     });

  inRankCp.resize(nPtsSampledInRank * n_features, stream);

  raft::matrix::gather((DataT*)X.data_handle(),
                       X.extent(1),
                       X.extent(0),
                       sampledMinClusterDistance.data_handle(),
                       nPtsSampledInRank,
                       inRankCp.data(),
                       raft::key_op{},
                       stream);
}

// calculate pairwise distance between 'dataset[n x d]' and 'centroids[k x d]',
// result will be stored in 'pairwiseDistance[n x k]'
template <typename DataT, typename IndexT>
void pairwise_distance_kmeans(raft::resources const& handle,
                              raft::device_matrix_view<const DataT, IndexT> X,
                              raft::device_matrix_view<const DataT, IndexT> centroids,
                              raft::device_matrix_view<DataT, IndexT> pairwiseDistance,
                              rmm::device_uvector<char>& workspace,
                              raft::distance::DistanceType metric)
{
  auto n_samples  = X.extent(0);
  auto n_features = X.extent(1);
  auto n_clusters = centroids.extent(0);

  ASSERT(X.extent(1) == centroids.extent(1),
         "# features in dataset and centroids are different (must be same)");

  raft::distance::pairwise_distance(handle,
                                    X.data_handle(),
                                    centroids.data_handle(),
                                    pairwiseDistance.data_handle(),
                                    n_samples,
                                    n_clusters,
                                    n_features,
                                    workspace,
                                    metric);
}

// shuffle and randomly select 'n_samples_to_gather' from input 'in' and stores
// in 'out' does not modify the input
template <typename DataT, typename IndexT>
void shuffleAndGather(raft::resources const& handle,
                      raft::device_matrix_view<const DataT, IndexT> in,
                      raft::device_matrix_view<DataT, IndexT> out,
                      uint32_t n_samples_to_gather,
                      uint64_t seed)
{
  cudaStream_t stream = resource::get_cuda_stream(handle);
  auto n_samples      = in.extent(0);
  auto n_features     = in.extent(1);

  auto indices = raft::make_device_vector<IndexT, IndexT>(handle, n_samples);

  // shuffle indices on device
  raft::random::permute<DataT, IndexT, IndexT>(indices.data_handle(),
                                               nullptr,
                                               nullptr,
                                               (IndexT)in.extent(1),
                                               (IndexT)in.extent(0),
                                               true,
                                               stream);

  raft::matrix::gather((DataT*)in.data_handle(),
                       in.extent(1),
                       in.extent(0),
                       indices.data_handle(),
                       static_cast<IndexT>(n_samples_to_gather),
                       out.data_handle(),
                       stream);
}

// Calculates a <key, value> pair for every sample in input 'X' where key is an
// index to an sample in 'centroids' (index of the nearest centroid) and 'value'
// is the distance between the sample and the 'centroid[key]'
template <typename DataT, typename IndexT>
void minClusterAndDistanceCompute(
  raft::resources const& handle,
  raft::device_matrix_view<const DataT, IndexT> X,
  raft::device_matrix_view<const DataT, IndexT> centroids,
  raft::device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT> minClusterAndDistance,
  raft::device_vector_view<const DataT, IndexT> L2NormX,
  rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,
  raft::distance::DistanceType metric,
  int batch_samples,
  int batch_centroids,
  rmm::device_uvector<char>& workspace)
{
  cudaStream_t stream = resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = centroids.extent(0);
  // todo(lsugy): change batch size computation when using fusedL2NN!
  bool is_fused = metric == raft::distance::DistanceType::L2Expanded ||
                  metric == raft::distance::DistanceType::L2SqrtExpanded;
  auto dataBatchSize = is_fused ? (IndexT)n_samples : getDataBatchSize(batch_samples, n_samples);
  auto centroidsBatchSize = getCentroidsBatchSize(batch_centroids, n_clusters);

  if (is_fused) {
    L2NormBuf_OR_DistBuf.resize(n_clusters, stream);
    raft::linalg::rowNorm(L2NormBuf_OR_DistBuf.data(),
                          centroids.data_handle(),
                          centroids.extent(1),
                          centroids.extent(0),
                          raft::linalg::L2Norm,
                          true,
                          stream);
  } else {
    // TODO: Unless pool allocator is used, passing in a workspace for this
    // isn't really increasing performance because this needs to do a re-allocation
    // anyways. ref https://github.com/rapidsai/raft/issues/930
    L2NormBuf_OR_DistBuf.resize(dataBatchSize * centroidsBatchSize, stream);
  }

  // Note - pairwiseDistance and centroidsNorm share the same buffer
  // centroidsNorm [n_clusters] - tensor wrapper around centroids L2 Norm
  auto centroidsNorm =
    raft::make_device_vector_view<DataT, IndexT>(L2NormBuf_OR_DistBuf.data(), n_clusters);
  // pairwiseDistance[ns x nc] - tensor wrapper around the distance buffer
  auto pairwiseDistance = raft::make_device_matrix_view<DataT, IndexT>(
    L2NormBuf_OR_DistBuf.data(), dataBatchSize, centroidsBatchSize);

  raft::KeyValuePair<IndexT, DataT> initial_value(0, std::numeric_limits<DataT>::max());

  thrust::fill(resource::get_thrust_policy(handle),
               minClusterAndDistance.data_handle(),
               minClusterAndDistance.data_handle() + minClusterAndDistance.size(),
               initial_value);

  // tile over the input dataset
  for (IndexT dIdx = 0; dIdx < n_samples; dIdx += dataBatchSize) {
    // # of samples for the current batch
    auto ns = std::min((IndexT)dataBatchSize, n_samples - dIdx);

    // datasetView [ns x n_features] - view representing the current batch of
    // input dataset
    auto datasetView = raft::make_device_matrix_view<const DataT, IndexT>(
      X.data_handle() + (dIdx * n_features), ns, n_features);

    // minClusterAndDistanceView [ns x n_clusters]
    auto minClusterAndDistanceView =
      raft::make_device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT>(
        minClusterAndDistance.data_handle() + dIdx, ns);

    auto L2NormXView =
      raft::make_device_vector_view<const DataT, IndexT>(L2NormX.data_handle() + dIdx, ns);

    if (is_fused) {
      workspace.resize((sizeof(int)) * ns, stream);

      // todo(lsugy): remove cIdx
      raft::distance::fusedL2NNMinReduce<DataT, raft::KeyValuePair<IndexT, DataT>, IndexT>(
        minClusterAndDistanceView.data_handle(),
        datasetView.data_handle(),
        centroids.data_handle(),
        L2NormXView.data_handle(),
        centroidsNorm.data_handle(),
        ns,
        n_clusters,
        n_features,
        (void*)workspace.data(),
        metric != raft::distance::DistanceType::L2Expanded,
        false,
        stream);
    } else {
      // tile over the centroids
      for (IndexT cIdx = 0; cIdx < n_clusters; cIdx += centroidsBatchSize) {
        // # of centroids for the current batch
        auto nc = std::min((IndexT)centroidsBatchSize, n_clusters - cIdx);

        // centroidsView [nc x n_features] - view representing the current batch
        // of centroids
        auto centroidsView = raft::make_device_matrix_view<const DataT, IndexT>(
          centroids.data_handle() + (cIdx * n_features), nc, n_features);

        // pairwiseDistanceView [ns x nc] - view representing the pairwise
        // distance for current batch
        auto pairwiseDistanceView =
          raft::make_device_matrix_view<DataT, IndexT>(pairwiseDistance.data_handle(), ns, nc);

        // calculate pairwise distance between current tile of cluster centroids
        // and input dataset
        pairwise_distance_kmeans<DataT, IndexT>(
          handle, datasetView, centroidsView, pairwiseDistanceView, workspace, metric);

        // argmin reduction returning <index, value> pair
        // calculates the closest centroid and the distance to the closest
        // centroid
        raft::linalg::coalescedReduction(
          minClusterAndDistanceView.data_handle(),
          pairwiseDistanceView.data_handle(),
          pairwiseDistanceView.extent(1),
          pairwiseDistanceView.extent(0),
          initial_value,
          stream,
          true,
          [=] __device__(const DataT val, const IndexT i) {
            raft::KeyValuePair<IndexT, DataT> pair;
            pair.key   = cIdx + i;
            pair.value = val;
            return pair;
          },
          raft::argmin_op{},
          raft::identity_op{});
      }
    }
  }
}

template <typename DataT, typename IndexT>
void minClusterDistanceCompute(raft::resources const& handle,
                               raft::device_matrix_view<const DataT, IndexT> X,
                               raft::device_matrix_view<DataT, IndexT> centroids,
                               raft::device_vector_view<DataT, IndexT> minClusterDistance,
                               raft::device_vector_view<DataT, IndexT> L2NormX,
                               rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,
                               raft::distance::DistanceType metric,
                               int batch_samples,
                               int batch_centroids,
                               rmm::device_uvector<char>& workspace)
{
  cudaStream_t stream = resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = centroids.extent(0);

  bool is_fused = metric == raft::distance::DistanceType::L2Expanded ||
                  metric == raft::distance::DistanceType::L2SqrtExpanded;
  auto dataBatchSize = is_fused ? (IndexT)n_samples : getDataBatchSize(batch_samples, n_samples);
  auto centroidsBatchSize = getCentroidsBatchSize(batch_centroids, n_clusters);

  if (is_fused) {
    L2NormBuf_OR_DistBuf.resize(n_clusters, stream);
    raft::linalg::rowNorm(L2NormBuf_OR_DistBuf.data(),
                          centroids.data_handle(),
                          centroids.extent(1),
                          centroids.extent(0),
                          raft::linalg::L2Norm,
                          true,
                          stream);
  } else {
    L2NormBuf_OR_DistBuf.resize(dataBatchSize * centroidsBatchSize, stream);
  }

  // Note - pairwiseDistance and centroidsNorm share the same buffer
  // centroidsNorm [n_clusters] - tensor wrapper around centroids L2 Norm
  auto centroidsNorm =
    raft::make_device_vector_view<DataT, IndexT>(L2NormBuf_OR_DistBuf.data(), n_clusters);
  // pairwiseDistance[ns x nc] - tensor wrapper around the distance buffer
  auto pairwiseDistance = raft::make_device_matrix_view<DataT, IndexT>(
    L2NormBuf_OR_DistBuf.data(), dataBatchSize, centroidsBatchSize);

  thrust::fill(resource::get_thrust_policy(handle),
               minClusterDistance.data_handle(),
               minClusterDistance.data_handle() + minClusterDistance.size(),
               std::numeric_limits<DataT>::max());

  // tile over the input data and calculate distance matrix [n_samples x
  // n_clusters]
  for (IndexT dIdx = 0; dIdx < n_samples; dIdx += dataBatchSize) {
    // # of samples for the current batch
    auto ns = std::min((IndexT)dataBatchSize, n_samples - dIdx);

    // datasetView [ns x n_features] - view representing the current batch of
    // input dataset
    auto datasetView = raft::make_device_matrix_view<const DataT, IndexT>(
      X.data_handle() + dIdx * n_features, ns, n_features);

    // minClusterDistanceView [ns x n_clusters]
    auto minClusterDistanceView =
      raft::make_device_vector_view<DataT, IndexT>(minClusterDistance.data_handle() + dIdx, ns);

    auto L2NormXView =
      raft::make_device_vector_view<DataT, IndexT>(L2NormX.data_handle() + dIdx, ns);

    if (is_fused) {
      workspace.resize((sizeof(IndexT)) * ns, stream);

      raft::distance::fusedL2NNMinReduce<DataT, DataT, IndexT>(
        minClusterDistanceView.data_handle(),
        datasetView.data_handle(),
        centroids.data_handle(),
        L2NormXView.data_handle(),
        centroidsNorm.data_handle(),
        ns,
        n_clusters,
        n_features,
        (void*)workspace.data(),
        metric != raft::distance::DistanceType::L2Expanded,
        false,
        stream);
    } else {
      // tile over the centroids
      for (IndexT cIdx = 0; cIdx < n_clusters; cIdx += centroidsBatchSize) {
        // # of centroids for the current batch
        auto nc = std::min((IndexT)centroidsBatchSize, n_clusters - cIdx);

        // centroidsView [nc x n_features] - view representing the current batch
        // of centroids
        auto centroidsView = raft::make_device_matrix_view<DataT, IndexT>(
          centroids.data_handle() + cIdx * n_features, nc, n_features);

        // pairwiseDistanceView [ns x nc] - view representing the pairwise
        // distance for current batch
        auto pairwiseDistanceView =
          raft::make_device_matrix_view<DataT, IndexT>(pairwiseDistance.data_handle(), ns, nc);

        // calculate pairwise distance between current tile of cluster centroids
        // and input dataset
        pairwise_distance_kmeans<DataT, IndexT>(
          handle, datasetView, centroidsView, pairwiseDistanceView, workspace, metric);

        raft::linalg::coalescedReduction(minClusterDistanceView.data_handle(),
                                         pairwiseDistanceView.data_handle(),
                                         pairwiseDistanceView.extent(1),
                                         pairwiseDistanceView.extent(0),
                                         std::numeric_limits<DataT>::max(),
                                         stream,
                                         true,
                                         raft::identity_op{},
                                         raft::min_op{},
                                         raft::identity_op{});
      }
    }
  }
}

template <typename DataT, typename IndexT>
void countSamplesInCluster(raft::resources const& handle,
                           const KMeansParams& params,
                           raft::device_matrix_view<const DataT, IndexT> X,
                           raft::device_vector_view<const DataT, IndexT> L2NormX,
                           raft::device_matrix_view<DataT, IndexT> centroids,
                           rmm::device_uvector<char>& workspace,
                           raft::device_vector_view<DataT, IndexT> sampleCountInCluster)
{
  cudaStream_t stream = resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = centroids.extent(0);

  // stores (key, value) pair corresponding to each sample where
  //   - key is the index of nearest cluster
  //   - value is the distance to the nearest cluster
  auto minClusterAndDistance =
    raft::make_device_vector<raft::KeyValuePair<IndexT, DataT>, IndexT>(handle, n_samples);

  // temporary buffer to store distance matrix, destructor releases the resource
  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);

  // computes minClusterAndDistance[0:n_samples) where  minClusterAndDistance[i]
  // is a <key, value> pair where
  //   'key' is index to an sample in 'centroids' (index of the nearest
  //   centroid) and 'value' is the distance between the sample 'X[i]' and the
  //   'centroid[key]'
  detail::minClusterAndDistanceCompute(handle,
                                       X,
                                       (raft::device_matrix_view<const DataT, IndexT>)centroids,
                                       minClusterAndDistance.view(),
                                       L2NormX,
                                       L2NormBuf_OR_DistBuf,
                                       params.metric,
                                       params.batch_samples,
                                       params.batch_centroids,
                                       workspace);

  // Using TransformInputIteratorT to dereference an array of raft::KeyValuePair
  // and converting them to just return the Key to be used in reduce_rows_by_key
  // prims
  detail::KeyValueIndexOp<IndexT, DataT> conversion_op;
  cub::TransformInputIterator<IndexT,
                              detail::KeyValueIndexOp<IndexT, DataT>,
                              raft::KeyValuePair<IndexT, DataT>*>
    itr(minClusterAndDistance.data_handle(), conversion_op);

  // count # of samples in each cluster
  countLabels(handle,
              itr,
              sampleCountInCluster.data_handle(),
              (IndexT)n_samples,
              (IndexT)n_clusters,
              workspace);
}
}  // namespace detail
}  // namespace cluster
}  // namespace raft
