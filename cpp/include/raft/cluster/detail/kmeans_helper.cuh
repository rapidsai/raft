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

#include <raft/cluster/kmeans_params.hpp>
//#include <raft/common/logger.hpp>
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/distance/distance.cuh>
#include <raft/distance/distance_type.hpp>
#include <raft/distance/fused_l2_nn.cuh>
#include <raft/handle.hpp>
#include <raft/linalg/reduce_cols_by_key.cuh>
#include <raft/linalg/reduce_rows_by_key.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/mdarray.hpp>
#include <raft/random/permute.cuh>
#include <raft/random/rng.cuh>
#include <raft/stats/sum.cuh>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace raft {
namespace cluster {
namespace detail {

template <typename IndexT, typename DataT>
struct FusedL2NNReduceOp {
  IndexT offset;

  FusedL2NNReduceOp(IndexT _offset) : offset(_offset){};

  typedef typename cub::KeyValuePair<IndexT, DataT> KVP;
  DI void operator()(IndexT rit, KVP* out, const KVP& other)
  {
    if (other.value < out->value) {
      out->key   = offset + other.key;
      out->value = other.value;
    }
  }

  DI void operator()(IndexT rit, DataT* out, const KVP& other)
  {
    if (other.value < *out) { *out = other.value; }
  }

  DI void init(DataT* out, DataT maxVal) { *out = maxVal; }
  DI void init(KVP* out, DataT maxVal)
  {
    out->key   = -1;
    out->value = maxVal;
  }
};

template <typename DataT, typename IndexT>
struct SamplingOp {
  DataT* rnd;
  int* flag;
  DataT cluster_cost;
  double oversampling_factor;
  IndexT n_clusters;

  CUB_RUNTIME_FUNCTION __forceinline__
  SamplingOp(DataT c, double l, IndexT k, DataT* rand, int* ptr)
    : cluster_cost(c), oversampling_factor(l), n_clusters(k), rnd(rand), flag(ptr)
  {
  }

  __host__ __device__ __forceinline__ bool operator()(
    const cub::KeyValuePair<ptrdiff_t, DataT>& a) const
  {
    DataT prob_threshold = (DataT)rnd[a.key];

    DataT prob_x = ((oversampling_factor * n_clusters * a.value) / cluster_cost);

    return !flag[a.key] && (prob_x > prob_threshold);
  }
};

template <typename IndexT, typename DataT>
struct KeyValueIndexOp {
  __host__ __device__ __forceinline__ IndexT
  operator()(const cub::KeyValuePair<IndexT, DataT>& a) const
  {
    return a.key;
  }
};

// Computes the intensity histogram from a sequence of labels
template <typename SampleIteratorT, typename CounterT, typename IndexT>
void countLabels(const raft::handle_t& handle,
                 SampleIteratorT labels,
                 CounterT* count,
                 IndexT n_samples,
                 IndexT n_clusters,
                 rmm::device_uvector<char>& workspace,
                 cudaStream_t stream)
{
  IndexT num_levels  = n_clusters + 1;
  IndexT lower_level = 0;
  IndexT upper_level = n_clusters;

  size_t temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(nullptr,
                                                    temp_storage_bytes,
                                                    labels,
                                                    count,
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    n_samples,
                                                    stream));

  workspace.resize(temp_storage_bytes, stream);

  RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(workspace.data(),
                                                    temp_storage_bytes,
                                                    labels,
                                                    count,
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    n_samples,
                                                    stream));
}

template <typename DataT>
void checkWeight(const raft::handle_t& handle,
                 raft::device_vector_view<DataT> weight,
                 cudaStream_t stream)
{
  auto wt_aggr   = raft::make_device_scalar<DataT>(0, handle.get_stream());
  auto n_samples = weight.extent(0);

  raft::stats::sum(wt_aggr.data(), weight.data(), (std::size_t)1, n_samples, true, stream);

  DataT wt_sum = 0;
  raft::copy(&wt_sum, wt_aggr.data(), 1, stream);
  handle.sync_stream(stream);

  if (wt_sum != n_samples) {
    /*RAFT_LOG_INFO(
      "[Warning!] KMeans: normalizing the user provided sample weight to "
      "sum up to %d samples",
      n_samples);*/

    auto scale = static_cast<DataT>(n_samples) / wt_sum;
    raft::linalg::unaryOp(
      weight.data(),
      weight.data(),
      n_samples,
      [=] __device__(const DataT& wt) { return wt * scale; },
      stream);
  }
}

template <typename IndexT>
IndexT getDataBatchSize(const KMeansParams& params, IndexT n_samples)
{
  auto minVal = std::min(static_cast<IndexT>(params.batch_samples), n_samples);
  return (minVal == 0) ? n_samples : minVal;
}

template <typename IndexT>
IndexT getCentroidsBatchSize(const KMeansParams& params, IndexT n_local_clusters)
{
  auto minVal = std::min(static_cast<IndexT>(params.batch_centroids), n_local_clusters);
  return (minVal == 0) ? n_local_clusters : minVal;
}

template <typename DataT, typename ReductionOpT>
void computeClusterCost(const raft::handle_t& handle,
                        raft::device_vector_view<DataT> minClusterDistance,
                        rmm::device_uvector<char>& workspace,
                        DataT* clusterCost,
                        ReductionOpT reduction_op,
                        cudaStream_t stream)
{
  size_t temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceReduce::Reduce(nullptr,
                                          temp_storage_bytes,
                                          minClusterDistance.data(),
                                          clusterCost,
                                          minClusterDistance.size(),
                                          reduction_op,
                                          DataT(),
                                          stream));

  workspace.resize(temp_storage_bytes, stream);

  RAFT_CUDA_TRY(cub::DeviceReduce::Reduce(workspace.data(),
                                          temp_storage_bytes,
                                          minClusterDistance.data(),
                                          clusterCost,
                                          minClusterDistance.size(),
                                          reduction_op,
                                          DataT(),
                                          stream));
}

template <typename DataT, typename IndexT>
raft::device_matrix<DataT> sampleCentroids(const raft::handle_t& handle,
                                           raft::device_matrix_view<const DataT> X,
                                           raft::device_vector_view<DataT> minClusterDistance,
                                           raft::device_vector_view<IndexT> isSampleCentroid,
                                           SamplingOp<DataT, IndexT>& select_op,
                                           rmm::device_uvector<char>& workspace,
                                           cudaStream_t stream)
{
  auto n_local_samples = X.extent(0);
  auto n_features      = X.extent(1);

  auto nSelected = raft::make_device_scalar<IndexT>(0, stream);
  cub::ArgIndexInputIterator<DataT*> ip_itr(minClusterDistance.data());
  auto sampledMinClusterDistance =
    raft::make_device_vector<cub::KeyValuePair<ptrdiff_t, DataT>>(n_local_samples, stream);
  size_t temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceSelect::If(nullptr,
                                      temp_storage_bytes,
                                      ip_itr,
                                      sampledMinClusterDistance.data(),
                                      nSelected.data(),
                                      n_local_samples,
                                      select_op,
                                      stream));

  workspace.resize(temp_storage_bytes, stream);

  RAFT_CUDA_TRY(cub::DeviceSelect::If(workspace.data(),
                                      temp_storage_bytes,
                                      ip_itr,
                                      sampledMinClusterDistance.data(),
                                      nSelected.data(),
                                      n_local_samples,
                                      select_op,
                                      stream));

  IndexT nPtsSampledInRank = 0;
  raft::copy(&nPtsSampledInRank, nSelected.data(), 1, stream);
  handle.sync_stream(stream);

  IndexT* rawPtr_isSampleCentroid = isSampleCentroid.data();
  thrust::for_each_n(handle.get_thrust_policy(),
                     sampledMinClusterDistance.data(),
                     nPtsSampledInRank,
                     [=] __device__(cub::KeyValuePair<ptrdiff_t, DataT> val) {
                       rawPtr_isSampleCentroid[val.key] = 1;
                     });

  auto inRankCp = raft::make_device_matrix<DataT>(nPtsSampledInRank, n_features, stream);

  raft::matrix::gather((DataT*)X.data(),
                       X.extent(1),
                       X.extent(0),
                       sampledMinClusterDistance.data(),
                       nPtsSampledInRank,
                       inRankCp.data(),
                       [=] __device__(cub::KeyValuePair<ptrdiff_t, DataT> val) {  // MapTransformOp
                         return val.key;
                       },
                       stream);

  return inRankCp;
}

// calculate pairwise distance between 'dataset[n x d]' and 'centroids[k x d]',
// result will be stored in 'pairwiseDistance[n x k]'
template <typename DataT, typename IndexT>
void pairwise_distance_kmeans(const raft::handle_t& handle,
                              raft::device_matrix_view<const DataT> X,
                              raft::device_matrix_view<const DataT> centroids,
                              raft::device_matrix_view<DataT> pairwiseDistance,
                              rmm::device_uvector<char>& workspace,
                              raft::distance::DistanceType metric,
                              cudaStream_t stream)
{
  auto n_samples  = X.extent(0);
  auto n_features = X.extent(1);
  auto n_clusters = centroids.extent(0);

  ASSERT(X.extent(1) == centroids.extent(1),
         "# features in dataset and centroids are different (must be same)");

  raft::distance::pairwise_distance<DataT, IndexT>(handle,
                                                   X.data(),
                                                   centroids.data(),
                                                   pairwiseDistance.data(),
                                                   n_samples,
                                                   n_clusters,
                                                   n_features,
                                                   workspace,
                                                   metric);
}

// shuffle and randomly select 'n_samples_to_gather' from input 'in' and stores
// in 'out' does not modify the input
template <typename DataT, typename IndexT>
void shuffleAndGather(const raft::handle_t& handle,
                      raft::device_matrix_view<const DataT> in,
                      raft::device_matrix_view<DataT> out,
                      uint32_t n_samples_to_gather,
                      uint64_t seed,
                      cudaStream_t stream,
                      rmm::device_uvector<char>* workspace = nullptr)
{
  auto n_samples  = in.extent(0);
  auto n_features = in.extent(1);

  auto indices = raft::make_device_vector<IndexT>(n_samples, stream);

  if (workspace) {
    // shuffle indices on device using ml-prims
    raft::random::permute<DataT>(
      indices.data(), nullptr, nullptr, (IndexT)in.extent(1), (IndexT)in.extent(0), true, stream);
  } else {
    // shuffle indices on host and copy to device...
    std::vector<IndexT> ht_indices(n_samples);

    std::iota(ht_indices.begin(), ht_indices.end(), 0);

    std::mt19937 gen(seed);
    std::shuffle(ht_indices.begin(), ht_indices.end(), gen);

    raft::copy(indices.data(), ht_indices.data(), indices.size(), stream);
  }

  raft::matrix::gather((DataT*)in.data(),
                       in.extent(1),
                       in.extent(0),
                       indices.data(),
                       n_samples_to_gather,
                       out.data(),
                       stream);
}

// Calculates a <key, value> pair for every sample in input 'X' where key is an
// index to an sample in 'centroids' (index of the nearest centroid) and 'value'
// is the distance between the sample and the 'centroid[key]'
template <typename DataT, typename IndexT>
void minClusterAndDistanceCompute(
  const raft::handle_t& handle,
  const KMeansParams& params,
  raft::device_matrix_view<const DataT> X,
  raft::device_matrix_view<const DataT> centroids,
  raft::device_vector_view<cub::KeyValuePair<IndexT, DataT>> minClusterAndDistance,
  raft::device_vector_view<DataT> L2NormX,
  rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,
  rmm::device_uvector<char>& workspace,
  raft::distance::DistanceType metric,
  cudaStream_t stream)
{
  auto n_samples          = X.extent(0);
  auto n_features         = X.extent(1);
  auto n_clusters         = centroids.extent(0);
  auto dataBatchSize      = getDataBatchSize(params, n_samples);
  auto centroidsBatchSize = getCentroidsBatchSize(params, n_clusters);

  if (metric == raft::distance::DistanceType::L2Expanded ||
      metric == raft::distance::DistanceType::L2SqrtExpanded) {
    L2NormBuf_OR_DistBuf.resize(n_clusters, stream);
    raft::linalg::rowNorm(L2NormBuf_OR_DistBuf.data(),
                          centroids.data(),
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
  auto centroidsNorm = raft::make_device_vector_view(L2NormBuf_OR_DistBuf.data(), n_clusters);
  // pairwiseDistance[ns x nc] - tensor wrapper around the distance buffer
  auto pairwiseDistance =
    raft::make_device_matrix_view(L2NormBuf_OR_DistBuf.data(), dataBatchSize, centroidsBatchSize);

  cub::KeyValuePair<IndexT, DataT> initial_value(0, std::numeric_limits<DataT>::max());

  thrust::fill(handle.get_thrust_policy(),
               minClusterAndDistance.data(),
               minClusterAndDistance.data() + minClusterAndDistance.size(),
               initial_value);

  // tile over the input dataset
  for (std::size_t dIdx = 0; dIdx < n_samples; dIdx += dataBatchSize) {
    // # of samples for the current batch
    auto ns = std::min(dataBatchSize, n_samples - dIdx);

    // datasetView [ns x n_features] - view representing the current batch of
    // input dataset
    auto datasetView =
      raft::make_device_matrix_view(X.data() + (dIdx * n_features), ns, n_features);
    // auto datasetView = X.template view<2>({ns, n_features}, {dIdx, 0});

    // minClusterAndDistanceView [ns x n_clusters]
    auto minClusterAndDistanceView =
      raft::make_device_vector_view(minClusterAndDistance.data() + dIdx, ns);

    auto L2NormXView = raft::make_device_vector_view(L2NormX.data() + dIdx, ns);

    // tile over the centroids
    for (std::size_t cIdx = 0; cIdx < n_clusters; cIdx += centroidsBatchSize) {
      // # of centroids for the current batch
      auto nc = std::min(centroidsBatchSize, n_clusters - cIdx);

      // centroidsView [nc x n_features] - view representing the current batch
      // of centroids
      auto centroidsView =
        raft::make_device_matrix_view(centroids.data() + (cIdx * n_features), nc, n_features);

      if (metric == raft::distance::DistanceType::L2Expanded ||
          metric == raft::distance::DistanceType::L2SqrtExpanded) {
        auto centroidsNormView = raft::make_device_vector_view(centroidsNorm.data() + cIdx, nc);
        workspace.resize((sizeof(int)) * ns, stream);

        FusedL2NNReduceOp<IndexT, DataT> redOp(cIdx);
        raft::distance::KVPMinReduce<IndexT, DataT> pairRedOp;

        raft::distance::fusedL2NN<DataT, cub::KeyValuePair<IndexT, DataT>, IndexT>(
          minClusterAndDistanceView.data(),
          datasetView.data(),
          centroidsView.data(),
          L2NormXView.data(),
          centroidsNormView.data(),
          ns,
          nc,
          n_features,
          (void*)workspace.data(),
          redOp,
          pairRedOp,
          (metric == raft::distance::DistanceType::L2Expanded) ? false : true,
          false,
          stream);
      } else {
        // pairwiseDistanceView [ns x nc] - view representing the pairwise
        // distance for current batch
        auto pairwiseDistanceView = raft::make_device_matrix_view(pairwiseDistance.data(), ns, nc);

        // calculate pairwise distance between current tile of cluster centroids
        // and input dataset
        pairwise_distance_kmeans<DataT, IndexT>(
          handle, datasetView, centroidsView, pairwiseDistanceView, workspace, metric, stream);

        // argmin reduction returning <index, value> pair
        // calculates the closest centroid and the distance to the closest
        // centroid
        raft::linalg::coalescedReduction(
          minClusterAndDistanceView.data(),
          pairwiseDistanceView.data(),
          pairwiseDistanceView.extent(1),
          pairwiseDistanceView.extent(0),
          initial_value,
          stream,
          true,
          [=] __device__(const DataT val, const IndexT i) {
            cub::KeyValuePair<IndexT, DataT> pair;
            pair.key   = cIdx + i;
            pair.value = val;
            return pair;
          },
          [=] __device__(cub::KeyValuePair<IndexT, DataT> a, cub::KeyValuePair<IndexT, DataT> b) {
            return (b.value < a.value) ? b : a;
          },
          [=] __device__(cub::KeyValuePair<IndexT, DataT> pair) { return pair; });
      }
    }
  }
}

template <typename DataT, typename IndexT>
void minClusterDistanceCompute(const raft::handle_t& handle,
                               const KMeansParams& params,
                               raft::device_matrix_view<const DataT> X,
                               raft::device_matrix_view<DataT> centroids,
                               raft::device_vector_view<DataT> minClusterDistance,
                               raft::device_vector_view<DataT> L2NormX,
                               rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,
                               rmm::device_uvector<char>& workspace,
                               raft::distance::DistanceType metric,
                               cudaStream_t stream)
{
  auto n_samples  = X.extent(0);
  auto n_features = X.extent(1);
  auto n_clusters = centroids.extent(0);

  auto dataBatchSize      = getDataBatchSize(params, n_samples);
  auto centroidsBatchSize = getCentroidsBatchSize(params, n_clusters);

  if (metric == raft::distance::DistanceType::L2Expanded ||
      metric == raft::distance::DistanceType::L2SqrtExpanded) {
    L2NormBuf_OR_DistBuf.resize(n_clusters, stream);
    raft::linalg::rowNorm(L2NormBuf_OR_DistBuf.data(),
                          centroids.data(),
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
    raft::make_device_vector_view<DataT>(L2NormBuf_OR_DistBuf.data(), n_clusters);
  // pairwiseDistance[ns x nc] - tensor wrapper around the distance buffer
  auto pairwiseDistance = raft::make_device_matrix_view<DataT>(
    L2NormBuf_OR_DistBuf.data(), dataBatchSize, centroidsBatchSize);

  thrust::fill(handle.get_thrust_policy(),
               minClusterDistance.data(),
               minClusterDistance.data() + minClusterDistance.size(),
               std::numeric_limits<DataT>::max());

  // tile over the input data and calculate distance matrix [n_samples x
  // n_clusters]
  for (std::size_t dIdx = 0; dIdx < n_samples; dIdx += dataBatchSize) {
    // # of samples for the current batch
    auto ns = std::min(dataBatchSize, n_samples - dIdx);

    // datasetView [ns x n_features] - view representing the current batch of
    // input dataset
    auto datasetView =
      raft::make_device_matrix_view<const DataT>(X.data() + dIdx * n_features, ns, n_features);

    // minClusterDistanceView [ns x n_clusters]
    auto minClusterDistanceView =
      raft::make_device_vector_view<DataT>(minClusterDistance.data() + dIdx, ns);

    auto L2NormXView = raft::make_device_vector_view<DataT>(L2NormX.data() + dIdx, ns);

    // tile over the centroids
    for (std::size_t cIdx = 0; cIdx < n_clusters; cIdx += centroidsBatchSize) {
      // # of centroids for the current batch
      auto nc = std::min(centroidsBatchSize, n_clusters - cIdx);

      // centroidsView [nc x n_features] - view representing the current batch
      // of centroids
      auto centroidsView =
        raft::make_device_matrix_view<DataT>(centroids.data() + cIdx * n_features, nc, n_features);

      if (metric == raft::distance::DistanceType::L2Expanded ||
          metric == raft::distance::DistanceType::L2SqrtExpanded) {
        auto centroidsNormView =
          raft::make_device_vector_view<DataT>(centroidsNorm.data() + cIdx, nc);
        workspace.resize((sizeof(IndexT)) * ns, stream);

        FusedL2NNReduceOp<IndexT, DataT> redOp(cIdx);
        raft::distance::KVPMinReduce<IndexT, DataT> pairRedOp;
        raft::distance::fusedL2NN<DataT, DataT, IndexT>(
          minClusterDistanceView.data(),
          datasetView.data(),
          centroidsView.data(),
          L2NormXView.data(),
          centroidsNormView.data(),
          ns,
          nc,
          n_features,
          (void*)workspace.data(),
          redOp,
          pairRedOp,
          (metric != raft::distance::DistanceType::L2Expanded),
          false,
          stream);
      } else {
        // pairwiseDistanceView [ns x nc] - view representing the pairwise
        // distance for current batch
        auto pairwiseDistanceView =
          raft::make_device_matrix_view<DataT>(pairwiseDistance.data(), ns, nc);

        // calculate pairwise distance between current tile of cluster centroids
        // and input dataset
        pairwise_distance_kmeans<DataT, IndexT>(
          handle, datasetView, centroidsView, pairwiseDistanceView, workspace, metric, stream);

        raft::linalg::coalescedReduction(
          minClusterDistanceView.data(),
          pairwiseDistanceView.data(),
          pairwiseDistanceView.extent(1),
          pairwiseDistanceView.extent(0),
          std::numeric_limits<DataT>::max(),
          stream,
          true,
          [=] __device__(DataT val, IndexT i) {  // MainLambda
            return val;
          },
          [=] __device__(DataT a, DataT b) {  // ReduceLambda
            return (b < a) ? b : a;
          },
          [=] __device__(DataT val) {  // FinalLambda
            return val;
          });
      }
    }
  }
}

template <typename DataT, typename IndexT>
void countSamplesInCluster(const raft::handle_t& handle,
                           const KMeansParams& params,
                           raft::device_matrix_view<const DataT> X,
                           raft::device_vector_view<DataT> L2NormX,
                           raft::device_matrix_view<DataT> centroids,
                           rmm::device_uvector<char>& workspace,
                           raft::distance::DistanceType metric,
                           raft::device_vector_view<DataT> sampleCountInCluster,
                           cudaStream_t stream)
{
  auto n_samples  = X.extent(0);
  auto n_features = X.extent(1);
  auto n_clusters = centroids.extent(0);

  // stores (key, value) pair corresponding to each sample where
  //   - key is the index of nearest cluster
  //   - value is the distance to the nearest cluster
  auto minClusterAndDistance =
    raft::make_device_vector<cub::KeyValuePair<IndexT, DataT>>(n_samples, stream);

  // temporary buffer to store distance matrix, destructor releases the resource
  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);

  // computes minClusterAndDistance[0:n_samples) where  minClusterAndDistance[i]
  // is a <key, value> pair where
  //   'key' is index to an sample in 'centroids' (index of the nearest
  //   centroid) and 'value' is the distance between the sample 'X[i]' and the
  //   'centroid[key]'
  minClusterAndDistanceCompute(handle,
                               params,
                               X,
                               (raft::device_matrix_view<const DataT>)centroids,
                               minClusterAndDistance.view(),
                               L2NormX,
                               L2NormBuf_OR_DistBuf,
                               workspace,
                               metric,
                               stream);

  // Using TransformInputIteratorT to dereference an array of cub::KeyValuePair
  // and converting them to just return the Key to be used in reduce_rows_by_key
  // prims
  KeyValueIndexOp<IndexT, DataT> conversion_op;
  cub::TransformInputIterator<IndexT,
                              KeyValueIndexOp<IndexT, DataT>,
                              cub::KeyValuePair<IndexT, DataT>*>
    itr(minClusterAndDistance.data(), conversion_op);

  // count # of samples in each cluster
  countLabels(handle,
              itr,
              sampleCountInCluster.data(),
              (IndexT)n_samples,
              (IndexT)n_clusters,
              workspace,
              stream);
}
}  // namespace detail
}  // namespace cluster
}  // namespace raft
