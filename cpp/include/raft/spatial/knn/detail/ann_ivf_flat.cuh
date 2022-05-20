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

#include "ann_ivf_flat_kernel.cuh"
#include "ann_kmeans_balanced.cuh"
#include "ann_utils.cuh"
#include "topk/radix_topk.cuh"

#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/distance/distance.hpp>
#include <raft/distance/distance_type.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/spatial/knn/ann_common.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>

namespace raft::spatial::knn::detail {

template <typename T>
void _ivfflat_interleaved(
  T* list_data, const T* dataset, uint32_t dim, size_t index, size_t prefix, uint32_t veclen)
{
  size_t group_id = index / WarpSize;
  size_t in_id    = (index % WarpSize) * veclen;
  list_data += (prefix + group_id * WarpSize) * dim + in_id;

  for (size_t i = 0; i < dim; i += veclen) {
    for (size_t j = 0; j < veclen; j++) {
      list_data[i * WarpSize + j] = dataset[i + j];
    }
  }
}

// This kernel intends to remove the dependency of having dataset in managed mem/host mem.
//
template <typename T>
__global__ void write_ivf_flat_interleaved_index(
  T* list_data, const T* dataset, uint32_t dim, size_t index, size_t prefix, uint32_t veclen)
{
  size_t group_id = index / WarpSize;
  size_t in_id    = (index % WarpSize) * veclen;
  list_data += (prefix + group_id * WarpSize) * dim + in_id;

  for (size_t i = 0; i < dim; i += veclen) {
    for (size_t j = 0; j < veclen; j++) {
      list_data[i * WarpSize + j] = dataset[i + j];
    }
  }
}

/* CUIVFL status type */
enum cuivflStatus_t : unsigned int {
  CUIVFL_STATUS_SUCCESS           = 0,
  CUIVFL_STATUS_ALLOC_FAILED      = 1,
  CUIVFL_STATUS_NOT_INITIALIZED   = 2,
  CUIVFL_STATUS_INVALID_VALUE     = 3,
  CUIVFL_STATUS_INTERNAL_ERROR    = 4,
  CUIVFL_STATUS_FILEIO_ERROR      = 5,
  CUIVFL_STATUS_CUDA_ERROR        = 6,
  CUIVFL_STATUS_CUBLAS_ERROR      = 7,
  CUIVFL_STATUS_INVALID_POINTER   = 8,
  CUIVFL_STATUS_VERSION_ERROR     = 9,
  CUIVFL_STATUS_UNSUPPORTED_DTYPE = 10,
  CUIVFL_STATUS_FAISS_ERROR       = 11,
  CUIVFL_STATUS_NOT_BUILD         = 12
};

template <typename T>
struct ivfflat_config {
};

template <>
struct ivfflat_config<float> {
  using value_t                   = float;
  static constexpr float kDivisor = 1.0;
};
template <>
struct ivfflat_config<uint8_t> {
  using value_t                   = uint32_t;
  static constexpr float kDivisor = 256.0;
};
template <>
struct ivfflat_config<int8_t> {
  using value_t                   = int32_t;
  static constexpr float kDivisor = 128.0;
};

template <typename T>
class cuivflHandle {
 public:
  cuivflHandle(const handle_t& handle,
               raft::distance::DistanceType metric_type,
               uint32_t dim,
               uint32_t nlist,
               uint32_t niter,
               uint32_t device);

  cuivflStatus_t cuivflBuildIndex(const T* dataset, T* trainset, uint32_t nrow, uint32_t nTrainset);

  cuivflStatus_t cuivflSetSearchParameters(const uint32_t nprobe,
                                           const uint32_t max_batch,
                                           const uint32_t max_k);

  cuivflStatus_t cuivflSearch(
    const T* queries, uint32_t batch_size, uint32_t k, size_t* neighbors, float* distances);

  cuivflStatus_t queryIVFFlatGridSize(const uint32_t nprobe,
                                      const uint32_t batch_size,
                                      const uint32_t k);
  uint32_t getDim() { return dim_; }

 private:
  const handle_t& handle_;
  const rmm::cuda_stream_view stream_;

  raft::distance::DistanceType metric_type_;
  bool greater_;
  uint32_t nlist_;           // The number of inverted lists= the number of centriods
  uint32_t niter_;           // The number of uint32_terations for kmeans to build the indexs
  uint32_t dim_;             // The dimension of vectors for input dataset
  uint32_t nprobe_;          // The number of clusters for searching
  uint32_t nrow_;            // The number of elements for input dataset
  size_t ninterleave_;       // The number of elements in 32 interleaved group for input dataset
  size_t buf_topk_size_;     // The size of buffer used for topk select.
  size_t float_query_size_;  // The size of float converted queries from int8_t/uint8_t
  uint32_t veclen_;          // The vectorization length of dataset in index.
  uint32_t grid_dim_x_;      // The number of blocks launched across nprobe.

  // device pointer
  //  The device memory pointer; inverted list for data; size [ninterleave_, dim_]
  rmm::device_uvector<T> list_data_dev_;
  // The device memory pointer; inverted list for index; size [ninterleave_]
  rmm::device_uvector<uint32_t> list_index_dev_;
  // The device memory pointer; Used for list_data_manage_ptr_; size [nlist_]
  rmm::device_uvector<uint32_t> list_prefix_interleaved_dev_;
  // The device memory pointer; the number of each cluster(list); size [nlist_]
  rmm::device_uvector<uint32_t> list_lengths_dev_;
  // The device memory pointer; centriod; size [nlist_, dim_]
  rmm::device_uvector<float> centriod_dev_;
  // The device memory pointer; centriod norm ; size [nlist_, dim_]
  rmm::device_uvector<float> centriod_norm_dev_;
  // The device memory; used for topk select.
  rmm::device_buffer select_workspace_dev_;

  // host pointer
  //  The host memory pointer; inverted list for data; size [ninterleave_, dim_]
  std::vector<T> list_data_host_;
  // The host memory pointer; inverted list for index; size [ninterleave_]
  std::vector<uint32_t> list_index_host_;
  // The host memory pointer; Used for list_data_manage_ptr_; size [nlist_]
  std::vector<uint32_t> list_prefix_interleaved_host_;
  // The host memory pointer; the number of each cluster(list); size [nlist_]
  std::vector<uint32_t> list_lengths_host_;

  cuivflStatus_t cuivflBuildOptimizedKmeans(float* centriod_managed_ptr,
                                            const T* dataset,
                                            T* trainset,
                                            uint32_t* clusterSize,
                                            uint32_t nrow,
                                            uint32_t ntrain);

  template <typename value_t>
  cuivflStatus_t cuivflSearchImpl(
    const T* queries, uint32_t batch_size, uint32_t k, size_t* neighbors, value_t* distances);
};

template <typename T>
cuivflHandle<T>::cuivflHandle(const handle_t& handle,
                              raft::distance::DistanceType metric_type,
                              uint32_t dim,
                              uint32_t nlist,
                              uint32_t niter,
                              uint32_t device)
  : handle_(handle),
    stream_(handle_.get_stream()),
    dim_(dim),
    nlist_(nlist),
    niter_(niter),
    metric_type_(metric_type),
    float_query_size_(0),
    grid_dim_x_(0),
    list_data_dev_(0, stream_),
    list_index_dev_(0, stream_),
    list_prefix_interleaved_dev_(0, stream_),
    list_lengths_dev_(0, stream_),
    centriod_dev_(0, stream_),
    centriod_norm_dev_(0, stream_),
    list_index_host_(0),
    list_prefix_interleaved_host_(0),
    list_lengths_host_(0),
    list_data_host_(0)
{
  veclen_ = 16 / sizeof(T);
  while (dim % veclen_ != 0) {
    veclen_ = veclen_ >> 1;
  }
}

/**
 * NB: `dataset` is accessed only by GPU code, `trainset` accessed by CPU and GPU.
 *
 */
template <typename T>
cuivflStatus_t cuivflHandle<T>::cuivflBuildOptimizedKmeans(float* centriod_managed_ptr,
                                                           const T* dataset,
                                                           T* trainset,
                                                           uint32_t* datasetLabels,
                                                           uint32_t nrow,
                                                           uint32_t ntrain)
{
  uint32_t numTrainset   = ntrain;
  uint32_t numClusters   = nlist_;
  uint32_t dimDataset    = dim_;
  uint32_t numIterations = niter_;

  rmm::device_uvector<uint32_t> trainsetLabels(numTrainset, stream_);

  float* clusterCenters = centriod_managed_ptr;

  uint32_t numMesoClusters = pow((double)(numClusters), (double)1.0 / 2.0) + 0.5;
  fprintf(stderr, "# numMesoClusters: %u\n", numMesoClusters);

  rmm::mr::managed_memory_resource managed_memory;
  rmm::device_uvector<float> mesoClusterCenters(
    numMesoClusters * dimDataset, stream_, &managed_memory);
  rmm::device_uvector<uint32_t> mesoClusterLabels(numTrainset, stream_, &managed_memory);
  rmm::device_uvector<uint32_t> mesoClusterSize_buf(numMesoClusters, stream_, &managed_memory);
  rmm::device_uvector<float> mesoClusterCentersTemp(
    numMesoClusters * dimDataset, stream_, &managed_memory);

  auto mesoClusterSize = mesoClusterSize_buf.data();

  size_t sizePredictWorkspace =
    _cuann_kmeans_predict_bufferSize(numMesoClusters,  // number of centers
                                     dimDataset,
                                     numTrainset  // number of vectors
    );
  rmm::device_buffer predictWorkspace(sizePredictWorkspace, stream_);
  // Training meso-clusters
  for (uint32_t iter = 0; iter < 2 * numIterations; iter += 2) {
    fprintf(stderr,
            "(%s) Training kmeans of meso-clusters: %.1f / %u    \r",
            __func__,
            (float)iter / 2,
            numIterations);
    _cuann_kmeans_predict(handle_,
                          mesoClusterCenters.data(),
                          numMesoClusters,
                          dimDataset,
                          trainset,
                          numTrainset,
                          mesoClusterLabels.data(),
                          metric_type_,
                          (iter != 0),
                          predictWorkspace.data(),
                          mesoClusterCentersTemp.data(),
                          mesoClusterSize);

    if (iter < 2 * (numIterations - 2)) {
      if (_cuann_kmeans_adjust_centers(mesoClusterCenters.data(),
                                       numMesoClusters,
                                       dimDataset,
                                       trainset,
                                       numTrainset,
                                       mesoClusterLabels.data(),
                                       metric_type_,
                                       mesoClusterSize,
                                       (float)1.0 / 4)) {
        iter -= 1;
      }  // end if _cuann_kmeans_adjust_centers
    }    // end if iter < 2 * (numIterations - 2)
  }      // end for (int iter = 0; iter < 2 * numIterations; iter += 2)

  fprintf(stderr, "\n");
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  std::vector<uint32_t> numFineClusters(numMesoClusters);
  std::vector<uint32_t> csumFineClusters(numMesoClusters + 1);
  csumFineClusters[0] = 0;

  uint32_t numClustersRemain  = numClusters;
  uint32_t numTrainsetRemain  = numTrainset;
  uint32_t mesoClusterSizeMax = 0;
  uint32_t mesoClusterSizeSum = 0;
  uint32_t numFineClustersSum = 0;  // checking
  uint32_t numFineClustersMax = 0;
  for (uint32_t i = 0; i < numMesoClusters; i++) {
    if (i < numMesoClusters - 1) {
      numFineClusters[i] = (double)numClustersRemain * mesoClusterSize[i] / numTrainsetRemain + .5;
    } else {
      numFineClusters[i] = numClustersRemain;
    }
    numClustersRemain -= numFineClusters[i];
    numTrainsetRemain -= mesoClusterSize[i];
    mesoClusterSizeMax = max(mesoClusterSizeMax, mesoClusterSize[i]);
    mesoClusterSizeSum += mesoClusterSize[i];
    numFineClustersSum += numFineClusters[i];
    numFineClustersMax      = max(numFineClustersMax, numFineClusters[i]);
    csumFineClusters[i + 1] = csumFineClusters[i] + numFineClusters[i];
  }  // end for (uint32_t i = 0; i < numMesoClusters; i++)
  // fprintf(stderr, "# mesoClusterSizeSum: %u\n", mesoClusterSizeSum);
  // fprintf(stderr, "# numFineClustersSum: %u\n", numFineClustersSum);
  assert(mesoClusterSizeSum == numTrainset);
  assert(numFineClustersSum == numClusters);
  assert(csumFineClusters[numMesoClusters] == numClusters);

  rmm::device_uvector<uint32_t> idsTrainset_buf(mesoClusterSizeMax, stream_, &managed_memory);
  rmm::device_uvector<float> subTrainset_buf(
    mesoClusterSizeMax * dimDataset, stream_, &managed_memory);
  auto idsTrainset = idsTrainset_buf.data();
  auto subTrainset = subTrainset_buf.data();

  sizePredictWorkspace = 0;
  for (uint32_t i = 0; i < numMesoClusters; i++) {
    sizePredictWorkspace =
      max(sizePredictWorkspace,
          _cuann_kmeans_predict_bufferSize(numFineClusters[i],  // number of centers
                                           dimDataset,
                                           mesoClusterSize[i]  // number of vectors
                                           ));
  }

  // label (cluster ID) of each vector
  rmm::device_uvector<uint32_t> labelsMP(mesoClusterSizeMax, stream_, &managed_memory);

  predictWorkspace.resize(sizePredictWorkspace, stream_);

  rmm::device_uvector<float> clusterCentersEach(
    numFineClustersMax * dimDataset, stream_, &managed_memory);
  rmm::device_uvector<float> clusterCentersMP(
    numFineClustersMax * dimDataset, stream_, &managed_memory);
  // number of vectors in each cluster
  rmm::device_uvector<uint32_t> clusterSizeMP(numFineClustersMax, stream_, &managed_memory);

  // Training clusters in each meso-clusters
  uint32_t numClustersDone = 0;
  for (uint32_t i = 0; i < numMesoClusters; i++) {
    uint32_t k = 0;
    for (uint32_t j = 0; j < numTrainset; j++) {
      if (mesoClusterLabels.data()[j] != i) continue;
      idsTrainset[k++] = j;
    }
    assert(k == mesoClusterSize[i]);

    utils::_cuann_copy_with_list<T>(mesoClusterSize[i],
                                    dimDataset,
                                    trainset,
                                    idsTrainset,
                                    dimDataset,
                                    subTrainset,
                                    dimDataset,
                                    ivfflat_config<T>::kDivisor);
    RAFT_CUDA_TRY(cudaDeviceSynchronize());

    for (uint32_t iter = 0; iter < 2 * numIterations; iter += 2) {
      fprintf(stderr,
              "(%s) Training kmeans of clusters in meso-cluster %u (numClusters: %u): "
              "%.1f / %u    \r",
              __func__,
              i,
              numFineClusters[i],
              (float)iter / 2,
              numIterations);

      _cuann_kmeans_predict(handle_,
                            clusterCentersEach.data(),
                            numFineClusters[i],
                            dimDataset,
                            subTrainset,
                            mesoClusterSize[i],
                            labelsMP.data(),
                            metric_type_,
                            (iter != 0),
                            predictWorkspace.data(),
                            clusterCentersMP.data(),
                            clusterSizeMP.data());

      if (iter < 2 * (numIterations - 2)) {
        if (_cuann_kmeans_adjust_centers(clusterCentersEach.data(),
                                         numFineClusters[i],
                                         dimDataset,
                                         subTrainset,
                                         mesoClusterSize[i],
                                         labelsMP.data(),
                                         metric_type_,
                                         clusterSizeMP.data(),
                                         (float)1.0 / 4)) {
          iter -= 1;
        }
      }
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
    }
    RAFT_CUDA_TRY(cudaMemcpy(clusterCenters + (dimDataset * csumFineClusters[i]),
                             clusterCentersEach.data(),
                             sizeof(float) * numFineClusters[i] * dimDataset,
                             cudaMemcpyDefault));
    numClustersDone += numFineClusters[i];
  }  // end for (uint32_t i = 0; i < numMesoClusters; i++)
  fprintf(stderr, "\n");
  assert(numClustersDone == numClusters);

  clusterCentersMP.resize(numClusters * dimDataset, stream_);
  clusterSizeMP.resize(numClusters, stream_);

  // [...]
  sizePredictWorkspace = _cuann_kmeans_predict_bufferSize(numClusters, dimDataset, numTrainset);
  predictWorkspace.resize(sizePredictWorkspace, stream_);

  // Fitting whole clusters using whole trainset.
  for (int iter = 0; iter < 2; iter++) {
    _cuann_kmeans_predict(handle_,
                          clusterCenters,
                          numClusters,
                          dimDataset,
                          trainset,
                          numTrainset,
                          trainsetLabels.data(),
                          metric_type_,
                          true,
                          predictWorkspace.data(),
                          clusterCentersMP.data(),
                          clusterSizeMP.data(),
                          true);
  }  // end for (int iter = 0; iter < 2; iter++)

  fprintf(stderr, "(%s) Final fitting\n", __func__);

  sizePredictWorkspace = _cuann_kmeans_predict_bufferSize(numClusters, dimDataset, nrow_);
  predictWorkspace.resize(sizePredictWorkspace, stream_);

  _cuann_kmeans_predict(handle_,
                        clusterCenters,
                        nlist_,
                        dim_,
                        dataset,
                        nrow_,
                        datasetLabels,
                        metric_type_,
                        true,
                        predictWorkspace.data(),
                        clusterCentersMP.data(),
                        clusterSizeMP.data(),
                        true);

  _cuann_kmeans_predict(handle_,
                        clusterCenters,
                        nlist_,
                        dim_,
                        dataset,
                        nrow_,
                        datasetLabels,
                        metric_type_,
                        true,
                        predictWorkspace.data(),
                        clusterCentersMP.data(),
                        clusterSizeMP.data(),
                        false);

  return cuivflStatus_t::CUIVFL_STATUS_SUCCESS;
}  // end func cuivflBuildOptimizedKmeans

template <typename T>
cuivflStatus_t cuivflHandle<T>::cuivflBuildIndex(const T* dataset,
                                                 T* trainset,
                                                 uint32_t nrow,
                                                 uint32_t ntrain)
{
  nrow_ = nrow;

  rmm::mr::managed_memory_resource managed_memory;
  rmm::device_uvector<float> centriod_managed_buf(nlist_ * dim_, stream_, &managed_memory);
  auto centriod_managed_ptr = centriod_managed_buf.data();

  if (this == NULL || nrow_ == 0) { return CUIVFL_STATUS_NOT_INITIALIZED; }
  if constexpr (!std::is_same_v<T, float> && !std::is_same_v<T, uint8_t> &&
                !std::is_same_v<T, int8_t>) {
    return CUIVFL_STATUS_UNSUPPORTED_DTYPE;
  }

  // Alloc manage memory for centriods, trainset and workspace
  rmm::device_uvector<uint32_t> datasetLabels_buf(nrow_, stream_, &managed_memory);  // [numDataset]
  auto datasetLabels = datasetLabels_buf.data();

  // Step 3: Predict labels of the whole dataset
  cuivflBuildOptimizedKmeans(centriod_managed_ptr, dataset, trainset, datasetLabels, nrow, ntrain);

  // Step 3.2: Calculate the L2 related result
  centriod_norm_dev_.resize(nlist_, stream_);

  if (metric_type_ == raft::distance::DistanceType::L2Expanded) {
    utils::_cuann_sqsum(nlist_, dim_, centriod_managed_ptr, centriod_norm_dev_.data());
#ifdef DEBUG_L2
    printDevPtr(centriod_norm_dev_.data(), 20, "centriod_norm_dev_");
#endif
  }

  // Step 4: Record the number of elements in each clusters
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  list_prefix_interleaved_host_.resize(nlist_);
  list_lengths_host_.assign(nlist_, 0);
  for (uint32_t i = 0; i < nrow_; i++) {
    uint32_t id_cluster = datasetLabels[i];
    list_lengths_host_[id_cluster] += 1;
  }

  ninterleave_ = 0;
  for (uint32_t i = 0; i < nlist_; i++) {
    list_prefix_interleaved_host_[i] = ninterleave_;
    ninterleave_ += ((list_lengths_host_[i] - 1) / WarpSize + 1) * WarpSize;
  }

  list_data_host_.assign(ninterleave_ * dim_, 0);
  list_index_host_.assign(ninterleave_, 0);
  list_lengths_host_.assign(nlist_, 0);

  for (size_t i = 0; i < nrow_; i++) {
    uint32_t id_cluster     = datasetLabels[i];
    uint32_t current_add    = list_lengths_host_[id_cluster];
    uint32_t interleave_add = list_prefix_interleaved_host_[id_cluster];
    _ivfflat_interleaved(
      list_data_host_.data(), dataset + i * dim_, dim_, current_add, interleave_add, veclen_);
    list_index_host_[interleave_add + current_add] = i;
    list_lengths_host_[id_cluster] += 1;
  }

  // Store index on GPU memory: temp WAR until we've entire index building buffers on device
  list_data_dev_.resize(ninterleave_ * dim_, stream_);
  list_index_dev_.resize(ninterleave_, stream_);
  list_prefix_interleaved_dev_.resize(nlist_, stream_);
  list_lengths_dev_.resize(nlist_, stream_);
  centriod_dev_.resize(nlist_ * dim_, stream_);

  // Step 3: Read the list
  copy(list_prefix_interleaved_dev_.data(), list_prefix_interleaved_host_.data(), nlist_, stream_);
  copy(list_lengths_dev_.data(), list_lengths_host_.data(), nlist_, stream_);
  copy(centriod_dev_.data(), centriod_managed_ptr, nlist_ * dim_, stream_);

  copy(list_data_dev_.data(), list_data_host_.data(), ninterleave_ * dim_, stream_);
  copy(list_index_dev_.data(), list_index_host_.data(), ninterleave_, stream_);

  return cuivflStatus_t::CUIVFL_STATUS_SUCCESS;
}  // end func cuivflBuildIndex

template <typename T>
cuivflStatus_t cuivflHandle<T>::queryIVFFlatGridSize(const uint32_t nprobe,
                                                     const uint32_t batch_size,
                                                     const uint32_t k)
{
  // query the gridDimX size to store probes topK output
  ivfflat_interleaved_scan<T, typename ivfflat_config<T>::value_t>(nullptr,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr,
                                                                   metric_type_,
                                                                   nprobe,
                                                                   k,
                                                                   batch_size,
                                                                   dim_,
                                                                   nullptr,
                                                                   nullptr,
                                                                   0,
                                                                   greater_,
                                                                   veclen_,
                                                                   grid_dim_x_);
  return cuivflStatus_t::CUIVFL_STATUS_SUCCESS;
}

template <typename T>
cuivflStatus_t cuivflHandle<T>::cuivflSetSearchParameters(const uint32_t nprobe,
                                                          const uint32_t max_batch,
                                                          const uint32_t max_k)
{
  nprobe_ = nprobe;
  if (nprobe_ <= 0) { return CUIVFL_STATUS_INVALID_VALUE; }
  // Set the greater_
  if (metric_type_ == raft::distance::DistanceType::L2Expanded ||
      metric_type_ == raft::distance::DistanceType::L2Unexpanded) {
    greater_ = false;
  } else {
    // Need to set this to true for inner product if need FAISS like behavior for inner product
    greater_ = false;
  }

  // Set buffer
  if constexpr (std::is_integral_v<T>) {
    float_query_size_ = sizeof(float) * max_batch * dim_;
  } else {
    float_query_size_ = 0;
  }

  size_t buf_coarse_size = 0;
  topk::radix_topk_11bits<float, uint32_t>(nullptr,
                                           buf_coarse_size,
                                           nullptr,
                                           (uint32_t)max_batch,
                                           (uint32_t)nlist_,
                                           (uint32_t)nprobe,
                                           nullptr,
                                           nullptr,
                                           greater_,
                                           0);

  size_t buf_refine_size = 0;
//#ifdef RADIX
#if 1
  topk::radix_topk_11bits<float, size_t>(nullptr,
                                         buf_refine_size,
                                         nullptr,
                                         nullptr,
                                         (size_t)max_batch,
                                         (size_t)max_k * nprobe,
                                         (size_t)max_k,
                                         nullptr,
                                         nullptr,
                                         greater_,
                                         0);
#else
  nv::warp_sort_topk<float, size_t>(nullptr,
                                    buf_refine_size,
                                    nullptr,
                                    nullptr,
                                    (size_t)max_batch,
                                    (size_t)(max_k * nprobe),
                                    (size_t)max_k,
                                    nullptr,
                                    nullptr,
                                    greater_,
                                    0);
#endif

  buf_topk_size_            = buf_coarse_size > buf_refine_size ? buf_coarse_size : buf_refine_size;
  uint32_t query_norm_size  = max_batch * sizeof(float);
  std::vector<size_t> sizes = {query_norm_size,
                               max_batch * nlist_ * sizeof(float),
                               max_batch * nprobe * sizeof(float),
                               max_batch * nprobe * sizeof(uint32_t),
                               max_batch * nprobe * max_k * sizeof(float),
                               max_batch * nprobe * max_k * sizeof(size_t),
                               buf_topk_size_,
                               float_query_size_};

  select_workspace_dev_.resize(utils::calc_aligned_size(sizes), stream_);
  return cuivflStatus_t::CUIVFL_STATUS_SUCCESS;
}

template <typename T>
cuivflStatus_t cuivflHandle<T>::cuivflSearch(const T* queries,  // [numQueries, dimDataset]
                                             uint32_t batch_size,
                                             uint32_t k,
                                             size_t* neighbors,  // [numQueries, topK]
                                             float* distances)
{
  cuivflSearchImpl<float>(queries, batch_size, k, neighbors, distances);
  return cuivflStatus_t::CUIVFL_STATUS_SUCCESS;
}  // end func cuivflSearch

template <typename T>
template <typename value_t>
cuivflStatus_t cuivflHandle<T>::cuivflSearchImpl(const T* queries,  // [numQueries, dimDataset]
                                                 uint32_t batch_size,
                                                 uint32_t k,
                                                 size_t* neighbors,  // [numQueries, topK]
                                                 value_t* distances)
{
  uint32_t nprobe = std::min(nprobe_, (uint32_t)nlist_);

  grid_dim_x_ = 0;
  queryIVFFlatGridSize(nprobe, batch_size, k);
  // Prepare the buffer for topk calculation
  uint32_t query_norm_size  = batch_size * sizeof(float);
  std::vector<size_t> sizes = {query_norm_size,
                               batch_size * nlist_ * sizeof(float),
                               batch_size * nprobe * sizeof(float),
                               batch_size * nprobe * sizeof(uint32_t),
                               batch_size * nprobe * k * sizeof(float),
                               batch_size * nprobe * k * sizeof(size_t),
                               buf_topk_size_,
                               float_query_size_};
  std::vector<void*> aligned_pointers =
    utils::calc_aligned_pointers(select_workspace_dev_.data(), sizes);

  // The norm of query [batch_size];
  float* query_norm_dev_ptr = static_cast<float*>(aligned_pointers[0]);
  // The distance value of cluster(list) and queries;[batch, nlist_]
  float* distance_buffer_dev_ptr = static_cast<float*>(aligned_pointers[1]);
  // The topk distance value of cluster(list) and queries;[batch, nprobe]
  float* coarse_distances_dev_ptr = static_cast<float*>(aligned_pointers[2]);
  // TODO:use float datatype here for now.
  // The topk  index of cluster(list) and queries;[batch, nprobe]
  uint32_t* coarse_indices_dev_ptr = static_cast<uint32_t*>(aligned_pointers[3]);
  // The topk distance value of candicate vectors from each cluster(list);[batch,k]
  value_t* refined_distances_dev_ptr = static_cast<value_t*>(aligned_pointers[4]);
  // The topk index of candicate vectors from each cluster(list);[batch, k]
  size_t* refined_indices_dev_ptr = static_cast<size_t*>(aligned_pointers[5]);
  void* buf_topk_dev_ptr          = static_cast<void*>(aligned_pointers[6]);
  float* convertedQueries         = static_cast<float*>(aligned_pointers[7]);

  if constexpr (std::is_same_v<T, float>) {
    convertedQueries = const_cast<float*>(queries);
  } else {
    utils::_cuann_copy<T, float>(batch_size,
                                 dim_,
                                 queries,
                                 dim_,
                                 convertedQueries,
                                 dim_,
                                 stream_,
                                 ivfflat_config<T>::kDivisor);
  }

  float alpha = 1.0f;
  float beta  = 0.0f;

  if (metric_type_ == raft::distance::DistanceType::L2Expanded) {
    alpha = -2.0f;
    beta  = 1.0f;
    utils::_cuann_sqsum(batch_size, dim_, convertedQueries, query_norm_dev_ptr);
    utils::_cuann_outer_add(
      query_norm_dev_ptr, batch_size, centriod_norm_dev_.data(), nlist_, distance_buffer_dev_ptr);
#ifdef DEBUG_L2
    utils::printDevPtr(centriod_norm_dev_.data(), 20, "centriod_norm_dev_");
    utils::printDevPtr(distance_buffer_dev_ptr, 20, "distance_buffer_dev_ptr");
#endif
  } else {
    alpha = 1.0f;
    beta  = 0.0f;
  }

  linalg::gemm(handle_,
               true,
               false,
               nlist_,
               batch_size,
               dim_,
               &alpha,
               centriod_dev_.data(),
               dim_,
               convertedQueries,
               dim_,
               &beta,
               distance_buffer_dev_ptr,
               nlist_,
               stream_);

#ifdef DEBUG_L2
  utils::printDevPtr(distance_buffer_dev_ptr, 20, "distance_buffer_dev_ptr");
#endif
  topk::radix_topk_11bits<value_t, uint32_t>(buf_topk_dev_ptr,
                                             buf_topk_size_,
                                             distance_buffer_dev_ptr,
                                             (uint32_t)batch_size,
                                             (uint32_t)nlist_,
                                             (uint32_t)nprobe,
                                             coarse_distances_dev_ptr,
                                             coarse_indices_dev_ptr,
                                             greater_,
                                             stream_);
#ifdef DEBUG_L2
  utils::printDevPtr(coarse_indices_dev_ptr, 1 * nprobe, "coarse_indices_dev_ptr");
  utils::printDevPtr(coarse_distances_dev_ptr, 1 * nprobe, "coarse_distances_dev_ptr");
#endif

  value_t* distances_dev_ptr = refined_distances_dev_ptr;
  size_t* indices_dev_ptr    = refined_indices_dev_ptr;
  if (nprobe == 1 || grid_dim_x_ == 1) {
    distances_dev_ptr = distances;
    indices_dev_ptr   = neighbors;
  }

  ivfflat_interleaved_scan<T, typename ivfflat_config<T>::value_t>(
    queries,
    coarse_indices_dev_ptr,
    list_index_dev_.data(),
    list_data_dev_.data(),
    list_lengths_dev_.data(),
    list_prefix_interleaved_dev_.data(),
    metric_type_,
    nprobe,
    k,
    batch_size,
    dim_,
    indices_dev_ptr,
    distances_dev_ptr,
    stream_,
    greater_,
    veclen_,
    grid_dim_x_);

#ifdef DEBUG_L2
  utils::printDevPtr(distances_dev_ptr, 2 * k, "distances_dev_ptr");
  utils::printDevPtr(indices_dev_ptr, 2 * k, "indices_dev_ptr");
#endif

  if (grid_dim_x_ > 1) {
//#ifdef RADIX
#if 1
    topk::radix_topk_11bits<value_t, size_t>(buf_topk_dev_ptr,
                                             buf_topk_size_,
                                             refined_distances_dev_ptr,
                                             refined_indices_dev_ptr,
                                             (size_t)batch_size,
                                             (size_t)k * grid_dim_x_,
                                             (size_t)k,
                                             distances,
                                             neighbors,
                                             greater_,
                                             stream_);
#else
    topk::warp_sort_topk<value_t, size_t>(buf_topk_dev_ptr,
                                          buf_topk_size_,
                                          refined_distances_dev_ptr,
                                          refined_indices_dev_ptr,
                                          (size_t)batch_size,
                                          (size_t)(k * grid_dim_x_),
                                          (size_t)k,
                                          distances,
                                          neighbors,
                                          greater_,
                                          stream_);
#endif
  }  // end if nprobe=1

  return cuivflStatus_t::CUIVFL_STATUS_SUCCESS;
}  // end func cuivflHandle::cuivflSearchImpl

}  // namespace raft::spatial::knn::detail
