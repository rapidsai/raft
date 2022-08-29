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

#include "../ivf_pq_types.hpp"
#include "ann_kmeans_balanced.cuh"
#include "ann_utils.cuh"

#include <raft/core/handle.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/cuda_utils.cuh>
#include <raft/distance/distance_type.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/pow2_utils.cuh>
#include <raft/stats/histogram.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

namespace raft::spatial::knn::ivf_pq::detail {

using namespace raft::spatial::knn::detail;  // NOLINT

template <typename D, typename S>
__global__ void kern_transpose_copy_3d(uint32_t num0,
                                       uint32_t num1,
                                       uint32_t num2,
                                       D* dst,  // [num2, ld1, ld0]
                                       uint32_t ld0,
                                       uint32_t ld1,
                                       const S* src,  // [...]
                                       uint32_t stride0,
                                       uint32_t stride1,
                                       uint32_t stride2)
{
  uint32_t tid = threadIdx.x + (blockDim.x * blockIdx.x);
  if (tid >= num0 * num1 * num2) return;
  uint32_t i0 = tid % num0;
  uint32_t i1 = (tid / num0) % num1;
  uint32_t i2 = (tid / num0) / num1;

  dst[i0 + (ld0 * i1) + (ld0 * ld1 * i2)] = src[(stride0 * i0) + (stride1 * i1) + (stride2 * i2)];
}

// transpose_copy_3d
template <typename D, typename S>
void _cuann_transpose_copy_3d(uint32_t num0,
                              uint32_t num1,
                              uint32_t num2,
                              D* dst,  // [num2, ld1, ld0]
                              uint32_t ld0,
                              uint32_t ld1,
                              const S* src,  // [...]
                              uint32_t stride0,
                              uint32_t stride1,
                              uint32_t stride2,
                              rmm::cuda_stream_view stream)
{
  uint32_t nThreads = 128;
  uint32_t nBlocks  = ((num0 * num1 * num2) + nThreads - 1) / nThreads;
  kern_transpose_copy_3d<D, S><<<nBlocks, nThreads, 0, stream>>>(
    num0, num1, num2, dst, ld0, ld1, src, stride0, stride1, stride2);
}

__device__ __host__ inline void ivfpq_encode_core(
  uint32_t ldDataset, uint32_t pq_dim, uint32_t pq_bits, const uint32_t* label, uint8_t* output)
{
  for (uint32_t j = 0; j < pq_dim; j++) {
    uint8_t code = label[(ldDataset * j)];
    if (pq_bits == 8) {
      uint8_t* ptrOutput = output + j;
      ptrOutput[0]       = code;
    } else if (pq_bits == 7) {
      uint8_t* ptrOutput = output + 7 * (j / 8);
      if (j % 8 == 0) {
        ptrOutput[0] |= code;
      } else if (j % 8 == 1) {
        ptrOutput[0] |= code << 7;
        ptrOutput[1] |= code >> 1;
      } else if (j % 8 == 2) {
        ptrOutput[1] |= code << 6;
        ptrOutput[2] |= code >> 2;
      } else if (j % 8 == 3) {
        ptrOutput[2] |= code << 5;
        ptrOutput[3] |= code >> 3;
      } else if (j % 8 == 4) {
        ptrOutput[3] |= code << 4;
        ptrOutput[4] |= code >> 4;
      } else if (j % 8 == 5) {
        ptrOutput[4] |= code << 3;
        ptrOutput[5] |= code >> 5;
      } else if (j % 8 == 6) {
        ptrOutput[5] |= code << 2;
        ptrOutput[6] |= code >> 6;
      } else if (j % 8 == 7) {
        ptrOutput[6] |= code << 1;
      }
    } else if (pq_bits == 6) {
      uint8_t* ptrOutput = output + 3 * (j / 4);
      if (j % 4 == 0) {
        ptrOutput[0] |= code;
      } else if (j % 4 == 1) {
        ptrOutput[0] |= code << 6;
        ptrOutput[1] |= code >> 2;
      } else if (j % 4 == 2) {
        ptrOutput[1] |= code << 4;
        ptrOutput[2] |= code >> 4;
      } else if (j % 4 == 3) {
        ptrOutput[2] |= code << 2;
      }
    } else if (pq_bits == 5) {
      uint8_t* ptrOutput = output + 5 * (j / 8);
      if (j % 8 == 0) {
        ptrOutput[0] |= code;
      } else if (j % 8 == 1) {
        ptrOutput[0] |= code << 5;
        ptrOutput[1] |= code >> 3;
      } else if (j % 8 == 2) {
        ptrOutput[1] |= code << 2;
      } else if (j % 8 == 3) {
        ptrOutput[1] |= code << 7;
        ptrOutput[2] |= code >> 1;
      } else if (j % 8 == 4) {
        ptrOutput[2] |= code << 4;
        ptrOutput[3] |= code >> 4;
      } else if (j % 8 == 5) {
        ptrOutput[3] |= code << 1;
      } else if (j % 8 == 6) {
        ptrOutput[3] |= code << 6;
        ptrOutput[4] |= code >> 2;
      } else if (j % 8 == 7) {
        ptrOutput[4] |= code << 3;
      }
    } else if (pq_bits == 4) {
      uint8_t* ptrOutput = output + (j / 2);
      if (j % 2 == 0) {
        ptrOutput[0] |= code;
      } else {
        ptrOutput[0] |= code << 4;
      }
    }
  }
}

//
__global__ void ivfpq_encode_kernel(uint32_t n_rows,
                                    uint32_t ldDataset,  // (*) ldDataset >= n_rows
                                    uint32_t pq_dim,
                                    uint32_t pq_bits,       // 4 <= pq_bits <= 8
                                    const uint32_t* label,  // [pq_dim, ldDataset]
                                    uint8_t* output         // [n_rows, pq_dim]
)
{
  uint32_t i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i >= n_rows) return;
  ivfpq_encode_core(ldDataset, pq_dim, pq_bits, label + i, output + (pq_dim * pq_bits / 8) * i);
}

//
inline void ivfpq_encode(uint32_t n_rows,
                         uint32_t ldDataset,  // (*) ldDataset >= n_rows
                         uint32_t pq_dim,
                         uint32_t pq_bits,       // 4 <= pq_bits <= 8
                         const uint32_t* label,  // [pq_dim, ldDataset]
                         uint8_t* output         // [n_rows, pq_dim]
)
{
#if 1
  // GPU
  dim3 iekThreads(128, 1, 1);
  dim3 iekBlocks((n_rows + iekThreads.x - 1) / iekThreads.x, 1, 1);
  ivfpq_encode_kernel<<<iekBlocks, iekThreads>>>(n_rows, ldDataset, pq_dim, pq_bits, label, output);
#else
  // CPU
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  for (uint32_t i = 0; i < n_rows; i++) {
    ivfpq_encode_core(ldDataset, pq_dim, pq_bits, label + i, output + (pq_dim * pq_bits / 8) * i);
  }
#endif
}

template <typename T>
int descending(const void* a, const void* b)
{
  T valA = ((T*)a)[0];
  T valB = ((T*)b)[0];
  if (valA > valB) return -1;
  if (valA < valB) return 1;
  return 0;
}

template <typename IdxT>
void _cuann_get_inclusiveSumSortedClusterSize(index<IdxT>& index)
{
  auto cluster_offsets     = index.list_offsets().data_handle();
  auto output              = index.inclusiveSumSortedClusterSize();
  index.numClustersSize0() = 0;
  for (uint32_t i = 0; i < index.n_lists(); i++) {
    output(i) = cluster_offsets[i + 1] - cluster_offsets[i];
    if (output(i) > 0) continue;

    index.numClustersSize0() += 1;
  }
  RAFT_LOG_DEBUG("Number of clusters of size zero: %d", index.numClustersSize0());
  // sort
  qsort(output.data_handle(), index.n_lists(), sizeof(uint32_t), descending<uint32_t>);
  // scan
  for (uint32_t i = 1; i < index.n_lists(); i++) {
    output(i) += output(i - 1);
  }
  RAFT_EXPECTS(output(index.n_lists() - 1) == index.size(), "cluster sizes do not add up");
}

template <typename T, typename X = T, typename Y = T>
T _cuann_dot(int n, const X* x, int incX, const Y* y, int incY)
{
  T val = 0;
  for (int i = 0; i < n; i++) {
    val += utils::mapping<T>{}(x[incX * i]) * utils::mapping<T>{}(y[incY * i]);
  }
  return val;
}

//
template <typename T>
T _cuann_rand()
{
  return (T)rand() / RAND_MAX;
}

// make rotation matrix
inline void _cuann_make_rotation_matrix(uint32_t nRows,
                                        uint32_t nCols,
                                        uint32_t pq_len,
                                        bool randomRotation,
                                        float* rotationMatrix  // [nRows, nCols]
)
{
  RAFT_EXPECTS(
    nRows >= nCols, "number of rows (%u) must be larger than number or cols (%u)", nRows, nCols);
  RAFT_EXPECTS(
    nRows % pq_len == 0, "number of rows (%u) must be a multiple of pq_len (%u)", nRows, pq_len);

  if (randomRotation) {
    RAFT_LOG_DEBUG("Creating a random rotation matrix.");
    double dot, norm;
    std::vector<double> matrix(nRows * nCols, 0.0);
    for (uint32_t i = 0; i < nRows * nCols; i++) {
      matrix[i] = _cuann_rand<double>() - 0.5;
    }
    for (uint32_t j = 0; j < nCols; j++) {
      // normalize the j-th col vector
      norm = sqrt(_cuann_dot<double>(nRows, &matrix[j], nCols, &matrix[j], nCols));
      for (uint32_t i = 0; i < nRows; i++) {
        matrix[j + (nCols * i)] /= norm;
      }
      // orthogonalize the j-th col vector with the previous col vectors
      for (uint32_t k = 0; k < j; k++) {
        dot = _cuann_dot<double>(nRows, &matrix[j], nCols, &matrix[k], nCols);
        for (uint32_t i = 0; i < nRows; i++) {
          matrix[j + (nCols * i)] -= dot * matrix[k + (nCols * i)];
        }
      }
      // normalize the j-th col vector again
      norm = sqrt(_cuann_dot<double>(nRows, &matrix[j], nCols, &matrix[j], nCols));
      for (uint32_t i = 0; i < nRows; i++) {
        matrix[j + (nCols * i)] /= norm;
      }
    }
    for (uint32_t i = 0; i < nRows * nCols; i++) {
      rotationMatrix[i] = (float)matrix[i];
    }
  } else {
    if (nRows == nCols) {
      memset(rotationMatrix, 0, sizeof(float) * nRows * nCols);
      for (uint32_t i = 0; i < nCols; i++) {
        rotationMatrix[i + (nCols * i)] = 1.0;
      }
    } else {
      memset(rotationMatrix, 0, sizeof(float) * nRows * nCols);
      uint32_t i = 0;
      for (uint32_t j = 0; j < nCols; j++) {
        rotationMatrix[j + (nCols * i)] = 1.0;
        i += pq_len;
        if (i >= nRows) { i = (i % nRows) + 1; }
      }
    }
  }
}

// show centers (for debuging)
inline void _cuann_kmeans_show_centers(const float* centers,  // [numCenters, dimCenters]
                                       uint32_t numCenters,
                                       uint32_t dimCenters,
                                       const uint32_t* centerSize,
                                       const uint32_t numShow = 5)
{
#if (RAFT_ACTIVE_LEVEL >= RAFT_LEVEL_DEBUG)
  for (uint64_t k = 0; k < numCenters; k++) {
    if ((numShow <= k) && (k < numCenters - numShow)) {
      if (k == numShow) fprintf(stderr, "...\n");
      continue;
    }
    fprintf(stderr, "# centers[%lu]:", k);
    for (uint64_t j = 0; j < dimCenters; j++) {
      if ((numShow <= j) && (j < dimCenters - numShow)) {
        if (j == numShow) fprintf(stderr, " ... ");
        continue;
      }
      fprintf(stderr, " %f,", centers[j + (dimCenters * k)]);
    }
    fprintf(stderr, " %d\n", centerSize[k]);
  }
#endif
}

//
uint32_t _get_num_trainset(uint32_t cluster_size, uint32_t pq_dim, uint32_t pq_bits)
{
  return min(cluster_size * pq_dim, 256 * max(1 << pq_bits, pq_dim));
}

//
template <typename T>
void _cuann_compute_PQ_code(const handle_t& handle,
                            uint32_t n_rows,
                            uint32_t data_dim,
                            uint32_t rot_dim,
                            uint32_t pq_dim,
                            uint32_t pq_len,
                            uint32_t pq_bits,
                            uint32_t n_clusters,
                            codebook_gen codebook_kind,
                            uint32_t maxClusterSize,
                            float* cluster_centers,           // [n_clusters, data_dim]
                            const float* rotationMatrix,      // [rot_dim, data_dim]
                            const T* data_vectors,            // [n_rows]
                            const uint32_t* data_indices,     // [n_rows]
                            const uint32_t* cluster_sizes,    // [n_clusters]
                            const uint32_t* cluster_offsets,  // [n_clusters + 1]
                            float* pqCenters,                 // [...]
                            uint32_t numIterations,
                            uint8_t* pqDataset,  // [n_rows, pq_dim * pq_bits / 8]
                            rmm::mr::device_memory_resource* managed_memory,
                            rmm::mr::device_memory_resource* device_memory)
{
  auto stream = handle.get_stream();

  //
  // Compute PQ code
  //
  utils::memzero(pqDataset, n_rows * pq_dim * pq_bits / 8, stream);

  rmm::device_uvector<float> res_vectors(maxClusterSize * data_dim, stream, managed_memory);
  rmm::device_uvector<float> rot_vectors(maxClusterSize * rot_dim, stream, managed_memory);
  rmm::device_uvector<float> sub_vectors(maxClusterSize * pq_dim * pq_len, stream, managed_memory);
  rmm::device_uvector<uint32_t> sub_vector_labels(maxClusterSize * pq_dim, stream, managed_memory);
  rmm::device_uvector<uint8_t> my_pq_dataset(
    maxClusterSize * pq_dim * pq_bits / 8 /* NB: pq_dim * bitPQ % 8 == 0 */,
    stream,
    managed_memory);
  rmm::device_uvector<uint32_t> rot_vector_labels(0, stream, managed_memory);
  rmm::device_uvector<uint32_t> pq_cluster_size(0, stream, managed_memory);
  rmm::device_uvector<float> my_pq_centers(0, stream, managed_memory);

  if ((numIterations > 0) && (codebook_kind == codebook_gen::PER_CLUSTER)) {
    utils::memzero(pqCenters, n_clusters * (1 << pq_bits) * pq_len, stream);
    rot_vector_labels.resize(maxClusterSize * pq_dim, stream);
    pq_cluster_size.resize((1 << pq_bits), stream);
    my_pq_centers.resize((1 << pq_bits) * pq_len, stream);
  }

  for (uint32_t l = 0; l < n_clusters; l++) {
    if (cluster_sizes[l] == 0) continue;

    //
    // Compute the residual vector of the new vector with its cluster
    // centroids.
    //   resVectors[..] = newVectors[..] - cluster_centers[..]
    //
    utils::copy_selected<float, T>(cluster_sizes[l],
                                   data_dim,
                                   data_vectors,
                                   data_indices + cluster_offsets[l],
                                   data_dim,
                                   res_vectors.data(),
                                   data_dim,
                                   stream);

    // substract centers from the vectors in the cluster.
    raft::matrix::linewiseOp(
      res_vectors.data(),
      res_vectors.data(),
      data_dim,
      cluster_sizes[l],
      true,
      [] __device__(float a, float b) { return a - b; },
      stream,
      cluster_centers + (uint64_t)l * data_dim);

    //
    // Rotate the residual vectors using a rotation matrix
    //
    float alpha = 1.0;
    float beta  = 0.0;
    linalg::gemm(handle,
                 true,
                 false,
                 rot_dim,
                 cluster_sizes[l],
                 data_dim,
                 &alpha,
                 rotationMatrix,
                 data_dim,
                 res_vectors.data(),
                 data_dim,
                 &beta,
                 rot_vectors.data(),
                 rot_dim,
                 stream);

    //
    // Training PQ codebook if codebook_gen::PER_CLUSTER
    // (*) PQ codebooks are trained for each cluster.
    //
    if ((numIterations > 0) && (codebook_kind == codebook_gen::PER_CLUSTER)) {
      uint32_t n_rows_train = _get_num_trainset(cluster_sizes[l], pq_dim, pq_bits);
      kmeans::build_clusters(handle,
                             numIterations,
                             pq_len,
                             rot_vectors.data(),
                             n_rows_train,
                             (1 << pq_bits),
                             my_pq_centers.data(),
                             rot_vector_labels.data(),
                             pq_cluster_size.data(),
                             raft::distance::DistanceType::L2Expanded,
                             stream,
                             device_memory);
      raft::copy(pqCenters + ((1 << pq_bits) * pq_len) * l,
                 my_pq_centers.data(),
                 (1 << pq_bits) * pq_len,
                 stream);
    }

    //
    // Change the order of the vector data to facilitate processing in
    // each vector subspace.
    //   input:  rot_vectors[cluster_sizes, rot_dim]
    //   output: sub_vectors[pq_dim, cluster_sizes, pq_len]
    //
    _cuann_transpose_copy_3d<float, float>(pq_len,
                                           cluster_sizes[l],
                                           pq_dim,
                                           sub_vectors.data(),
                                           pq_len,
                                           cluster_sizes[l],
                                           rot_vectors.data(),
                                           1,
                                           rot_dim,
                                           pq_len,
                                           stream);

    //
    // Find a label (cluster ID) for each vector subspace.
    //
    for (uint32_t j = 0; j < pq_dim; j++) {
      float* curPqCenters = nullptr;
      if (codebook_kind == codebook_gen::PER_SUBSPACE) {
        curPqCenters = pqCenters + ((1 << pq_bits) * pq_len) * j;
      } else if (codebook_kind == codebook_gen::PER_CLUSTER) {
        curPqCenters = pqCenters + ((1 << pq_bits) * pq_len) * l;
        if (numIterations > 0) { curPqCenters = my_pq_centers.data(); }
      }
      kmeans::predict(handle,
                      curPqCenters,
                      (1 << pq_bits),
                      pq_len,
                      sub_vectors.data() + j * (cluster_sizes[l] * pq_len),
                      cluster_sizes[l],
                      sub_vector_labels.data() + j * cluster_sizes[l],
                      raft::distance::DistanceType::L2Expanded,
                      stream,
                      device_memory);
    }
    handle.sync_stream();

    //
    // PQ encoding
    //
    ivfpq_encode(cluster_sizes[l],
                 cluster_sizes[l],
                 pq_dim,
                 pq_bits,
                 sub_vector_labels.data(),
                 my_pq_dataset.data());
    RAFT_CUDA_TRY(cudaMemcpy(pqDataset + ((uint64_t)cluster_offsets[l] * pq_dim * pq_bits / 8),
                             my_pq_dataset.data(),
                             sizeof(uint8_t) * cluster_sizes[l] * pq_dim * pq_bits / 8,
                             cudaMemcpyDeviceToHost));
  }
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
}

template <typename T, typename IdxT>
void cuannIvfPqBuildIndex(
  const handle_t& handle,
  index<IdxT>& index,
  const T* data_vectors, /* [index_size, data_dim] */
  double trainset_fraction,
  uint32_t numIterations, /* Number of iterations to train kmeans */
  bool randomRotation /* If true, rotate vectors with randamly created rotation matrix */)
{
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
                "Unsupported data type");
  if constexpr (!std::is_same_v<T, float>) {
    RAFT_EXPECTS(index.metric() != distance::DistanceType::InnerProduct,
                 "Unsupported data type (inner-product metric supports floats only)");
  }

  auto stream = handle.get_stream();

  auto trainset_ratio = std::max<size_t>(
    1, index.size() / std::max<size_t>(trainset_fraction * index.size(), index.n_lists()));
  auto n_rows_train = index.size() / trainset_ratio;

  rmm::mr::device_memory_resource* device_memory = nullptr;
  auto pool_guard = raft::get_pool_memory_resource(device_memory, 1024 * 1024);
  if (pool_guard) {
    RAFT_LOG_DEBUG("cuannIvfPqBuildIndex: using pool memory resource with initial size %zu bytes",
                   pool_guard->pool_size());
  }

  rmm::mr::managed_memory_resource managed_memory_upstream;
  rmm::mr::pool_memory_resource<rmm::mr::managed_memory_resource> managed_memory(
    &managed_memory_upstream, 1024 * 1024);

  // TODO: move to device_memory, blocked by _cuann_dot
  rmm::device_uvector<T> trainset(n_rows_train * index.dim(), stream, &managed_memory);
  // TODO: a proper sampling
  RAFT_CUDA_TRY(cudaMemcpy2DAsync(trainset.data(),
                                  sizeof(T) * index.dim(),
                                  data_vectors,
                                  sizeof(T) * index.dim() * trainset_ratio,
                                  sizeof(T) * index.dim(),
                                  n_rows_train,
                                  cudaMemcpyDefault,
                                  stream));

  // NB: here cluster_centers is used as if it is [n_clusters, data_dim] not [n_clusters, dim_ext]!
  auto cluster_centers   = index.centers().data_handle();
  auto pqCenters         = index.pq_centers().data_handle();
  auto pqDataset         = index.pq_dataset().data_handle();
  auto data_indices      = index.indices().data_handle();
  auto cluster_offsets   = index.list_offsets().data_handle();
  auto rotationMatrix    = index.rotation_matrix().data_handle();
  auto clusterRotCenters = index.centers_rot().data_handle();

  // Train balanced hierarchical kmeans clustering
  kmeans::build_hierarchical(handle,
                             numIterations,
                             index.dim(),
                             trainset.data(),
                             n_rows_train,
                             cluster_centers,
                             index.n_lists(),
                             index.metric(),
                             stream);

  rmm::device_uvector<uint32_t> dataset_labels(index.size(), stream, &managed_memory);
  rmm::device_uvector<uint32_t> cluster_sizes(index.n_lists(), stream, &managed_memory);

  //
  // Predict labels of whole data_vectors
  //
  kmeans::predict(handle,
                  cluster_centers,
                  index.n_lists(),
                  index.dim(),
                  data_vectors,
                  index.size(),
                  dataset_labels.data(),
                  index.metric(),
                  stream,
                  device_memory);
  kmeans::calc_centers_and_sizes(cluster_centers,
                                 cluster_sizes.data(),
                                 index.n_lists(),
                                 index.dim(),
                                 data_vectors,
                                 index.size(),
                                 dataset_labels.data(),
                                 true,
                                 stream);
  handle.sync_stream();

#if (RAFT_ACTIVE_LEVEL >= RAFT_LEVEL_DEBUG)
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  _cuann_kmeans_show_centers(cluster_centers, index.n_lists(), index.dim(), cluster_sizes.data());
#endif

  // Make rotation matrix
  RAFT_LOG_DEBUG("# data_dim: %u\n", index.dim());
  RAFT_LOG_DEBUG("# rot_dim: %u\n", index.rot_dim());
  RAFT_LOG_DEBUG("# randomRotation: %s\n", randomRotation ? "enabled" : "disabled");
  _cuann_make_rotation_matrix(
    index.rot_dim(), index.dim(), index.pq_len(), randomRotation, rotationMatrix);

  // Rotate cluster_centers
  float alpha = 1.0;
  float beta  = 0.0;
  linalg::gemm(handle,
               true,
               false,
               index.rot_dim(),
               index.n_lists(),
               index.dim(),
               &alpha,
               rotationMatrix,
               index.dim(),
               cluster_centers,
               index.dim(),
               &beta,
               clusterRotCenters,
               index.rot_dim(),
               stream);

  //
  // Make cluster_offsets, data_indices and pqDataset
  //
  uint32_t maxClusterSize = 0;
  // cluster_offsets
  cluster_offsets[0] = 0;
  for (uint32_t l = 0; l < index.n_lists(); l++) {
    cluster_offsets[l + 1] = cluster_offsets[l] + cluster_sizes.data()[l];
    if (maxClusterSize < cluster_sizes.data()[l]) { maxClusterSize = cluster_sizes.data()[l]; }
  }
  RAFT_EXPECTS(cluster_offsets[index.n_lists()] == index.size(), "Cluster sizes do not add up");

  // data_indices
  for (uint32_t i = 0; i < index.size(); i++) {
    uint32_t l                       = dataset_labels.data()[i];
    data_indices[cluster_offsets[l]] = i;
    cluster_offsets[l] += 1;
  }

  // Recover cluster_offsets
  for (uint32_t l = 0; l < index.n_lists(); l++) {
    cluster_offsets[l] -= cluster_sizes.data()[l];
  }

  rmm::device_uvector<uint32_t> pq_cluster_sizes(index.pq_width(), stream, &managed_memory);

  if (index.codebook_kind() == codebook_gen::PER_SUBSPACE) {
    //
    // Training PQ codebook (codebook_gen::PER_SUBSPACE)
    // (*) PQ codebooks are trained for each subspace.
    //
    rmm::device_uvector<uint32_t> trainset_labels(n_rows_train, stream, &managed_memory);

    // Predict label of trainset again
    kmeans::predict(handle,
                    cluster_centers,
                    index.n_lists(),
                    index.dim(),
                    trainset.data(),
                    n_rows_train,
                    trainset_labels.data(),
                    index.metric(),
                    stream,
                    device_memory);
    handle.sync_stream();

    // [pq_dim, n_rows_train, pq_len]
    std::vector<float> mod_trainset(index.pq_dim() * n_rows_train * index.pq_len(), 0.0f);

    // mod_trainset[] = transpose( rotate(trainset[]) - clusterRotCenters[] )
    for (size_t i = 0; i < n_rows_train; i++) {
      uint32_t l = trainset_labels.data()[i];
      for (size_t j = 0; j < index.rot_dim(); j++) {
        float val   = _cuann_dot<float>(index.dim(),
                                      trainset.data() + static_cast<size_t>(index.dim()) * i,
                                      1,
                                      rotationMatrix + static_cast<size_t>(index.dim()) * j,
                                      1);
        uint32_t j0 = j / (index.pq_len());  // 0 <= j0 < pq_dim
        uint32_t j1 = j % (index.pq_len());  // 0 <= j1 < pq_len
        uint64_t idx =
          j1 + ((uint64_t)(index.pq_len()) * i) + ((uint64_t)(index.pq_len()) * n_rows_train * j0);
        mod_trainset[idx] = val - clusterRotCenters[j + (index.rot_dim() * l)];
      }
    }

    rmm::device_uvector<float> sub_trainset(n_rows_train * index.pq_len(), stream, &managed_memory);
    rmm::device_uvector<uint32_t> sub_trainset_labels(n_rows_train, stream, &managed_memory);
    rmm::device_uvector<float> pq_centers(
      index.pq_width() * index.pq_len(), stream, &managed_memory);

    for (uint32_t j = 0; j < index.pq_dim(); j++) {
      float* curPqCenters = pqCenters + (index.pq_width() * index.pq_len()) * j;
      RAFT_CUDA_TRY(cudaMemcpy(sub_trainset.data(),
                               mod_trainset.data() + ((uint64_t)n_rows_train * index.pq_len() * j),
                               sizeof(float) * n_rows_train * index.pq_len(),
                               cudaMemcpyHostToDevice));
      // Train kmeans for each PQ
      kmeans::build_clusters(handle,
                             numIterations,
                             index.pq_len(),
                             sub_trainset.data(),
                             n_rows_train,
                             index.pq_width(),
                             pq_centers.data(),
                             sub_trainset_labels.data(),
                             pq_cluster_sizes.data(),
                             raft::distance::DistanceType::L2Expanded,
                             stream,
                             device_memory);
      raft::copy(curPqCenters, pq_centers.data(), index.pq_width() * index.pq_len(), stream);
      handle.sync_stream();
#if (RAFT_ACTIVE_LEVEL >= RAFT_LEVEL_DEBUG)
      if (j == 0) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());
        _cuann_kmeans_show_centers(
          curPqCenters, index.pq_width(), index.pq_len(), pq_cluster_size.data());
      }
#endif
    }
  }

  //
  // Compute PQ code for whole data_vectors
  //
  _cuann_compute_PQ_code<T>(handle,
                            index.size(),
                            index.dim(),
                            index.rot_dim(),
                            index.pq_dim(),
                            index.pq_len(),
                            index.pq_bits(),
                            index.n_lists(),
                            index.codebook_kind(),
                            maxClusterSize,
                            cluster_centers,
                            rotationMatrix,
                            data_vectors,
                            data_indices,
                            cluster_sizes.data(),
                            cluster_offsets,
                            pqCenters,
                            numIterations,
                            pqDataset,
                            &managed_memory,
                            device_memory);

  _cuann_get_inclusiveSumSortedClusterSize(index);

  auto center_norms = index.center_norms().data_handle();
  utils::dots_along_rows(
    index.n_lists(), index.dim(), cluster_centers, index.center_norms().data_handle(), stream);
  stream.synchronize();

  {
    // combine cluster_centers and their norms
    auto cluster_centers_tmp =
      make_host_mdarray<float>(make_extents<uint32_t>(index.n_lists(), index.dim()));
    for (uint32_t i = 0; i < index.n_lists() * index.dim(); i++) {
      cluster_centers_tmp.data_handle()[i] = cluster_centers[i];
    }
    for (uint32_t i = 0; i < index.n_lists(); i++) {
      for (uint32_t j = 0; j < index.dim(); j++) {
        cluster_centers[j + (index.dim_ext() * i)] = cluster_centers_tmp(i, j);
      }
      cluster_centers[index.dim() + (index.dim_ext() * i)] =
        cluster_sizes.data()[i] == 0 ? 1.0f : center_norms[i];
    }
  }
}

template <typename T, typename IdxT>
auto cuannIvfPqCreateNewIndexByAddingVectorsToOldIndex(
  const handle_t& handle,
  index<IdxT>& orig_index,
  const T* newVectors, /* [numNewVectors, data_dim] */
  uint32_t numNewVectors) -> index<IdxT>
{
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
                "Unsupported data type");
  if constexpr (!std::is_same_v<T, float>) {
    RAFT_EXPECTS(orig_index.metric() != distance::DistanceType::InnerProduct,
                 "Unsupported data type (inner-product metric supports floats only)");
  }

  rmm::mr::device_memory_resource* device_memory = nullptr;
  auto pool_guard = raft::get_pool_memory_resource(device_memory, 1024 * 1024);
  if (pool_guard) {
    RAFT_LOG_DEBUG(
      "cuannIvfPqCreateNewIndexByAddingVectorsToOldIndex: using pool memory resource with initial "
      "size %zu bytes",
      pool_guard->pool_size());
  }

  rmm::mr::managed_memory_resource managed_memory_upstream;
  rmm::mr::pool_memory_resource<rmm::mr::managed_memory_resource> managed_memory(
    &managed_memory_upstream, 1024 * 1024);

  RAFT_LOG_DEBUG("data_dim: %u", orig_index.dim());

  auto oldClusterCenters    = orig_index.centers().data_handle();
  auto oldPqCenters         = orig_index.pq_centers().data_handle();
  auto oldPqDataset         = orig_index.pq_dataset().data_handle();
  auto oldOriginalNumbers   = orig_index.indices().data_handle();
  auto old_cluster_offsets  = orig_index.list_offsets().data_handle();
  auto oldRotationMatrix    = orig_index.rotation_matrix().data_handle();
  auto oldClusterRotCenters = orig_index.centers_rot().data_handle();

  //
  // The cluster_centers stored in index contain data other than cluster
  // centroids to speed up the search. Here, only the cluster centroids
  // are extracted.
  //

  rmm::device_uvector<float> cluster_centers(
    orig_index.n_lists() * orig_index.dim(), handle.get_stream(), &managed_memory);
  for (uint32_t i = 0; i < orig_index.n_lists(); i++) {
    memcpy(cluster_centers.data() + (uint64_t)i * orig_index.dim(),
           oldClusterCenters + (uint64_t)i * orig_index.dim_ext(),
           sizeof(float) * orig_index.dim());
  }

  //
  // Use the existing cluster centroids to find the label (cluster ID)
  // of the vector to be added.
  //

  rmm::device_uvector<uint32_t> new_data_labels(
    numNewVectors, handle.get_stream(), &managed_memory);
  utils::memzero(new_data_labels.data(), numNewVectors, handle.get_stream());
  rmm::device_uvector<uint32_t> cluster_sizes(
    orig_index.n_lists(), handle.get_stream(), &managed_memory);
  utils::memzero(cluster_sizes.data(), orig_index.n_lists(), handle.get_stream());

  kmeans::predict(handle,
                  cluster_centers.data(),
                  orig_index.n_lists(),
                  orig_index.dim(),
                  newVectors,
                  numNewVectors,
                  new_data_labels.data(),
                  orig_index.metric(),
                  handle.get_stream());
  raft::stats::histogram<uint32_t, size_t>(raft::stats::HistTypeAuto,
                                           reinterpret_cast<int32_t*>(cluster_sizes.data()),
                                           size_t(orig_index.n_lists()),
                                           new_data_labels.data(),
                                           numNewVectors,
                                           1,
                                           handle.get_stream());
  handle.sync_stream();

#if (RAFT_ACTIVE_LEVEL >= RAFT_LEVEL_DEBUG)
  {
    const int _num_show = 10;
    fprintf(stderr, "# numNewVectors: %u\n", numNewVectors);
    fprintf(stderr, "# new_data_labels: ");
    for (uint32_t i = 0; i < numNewVectors; i++) {
      if ((i < _num_show) || (numNewVectors - i <= _num_show)) {
        fprintf(stderr, "%u, ", new_data_labels.data()[i]);
      } else if (i == _num_show) {
        fprintf(stderr, "..., ");
      }
    }
    fprintf(stderr, "\n");
  }
  {
    const int _num_show = 10;
    fprintf(stderr, "# orig_index.n_lists(): %u\n", orig_index.n_lists());
    fprintf(stderr, "# cluster_sizes: ");
    int _sum = 0;
    for (uint32_t i = 0; i < orig_index.n_lists(); i++) {
      _sum += cluster_sizes.data()[i];
      if ((i < _num_show) || (orig_index.n_lists() - i <= _num_show)) {
        fprintf(stderr, "%u, ", cluster_sizes.data()[i]);
      } else if (i == _num_show) {
        fprintf(stderr, "..., ");
      }
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "# _sum: %d\n", _sum);
  }
#endif

  //
  // Make cluster_offsets, data_indices
  //
  uint32_t maxClusterSize = 0;
  std::vector<uint32_t> cluster_offsets(orig_index.n_lists() + 1, 0);
  std::vector<uint32_t> data_indices(numNewVectors);
  // cluster_offsets
  cluster_offsets[0] = 0;
  for (uint32_t l = 0; l < orig_index.n_lists(); l++) {
    cluster_offsets[l + 1] = cluster_offsets[l] + cluster_sizes.data()[l];
    maxClusterSize         = max(maxClusterSize, cluster_sizes.data()[l]);
  }
  RAFT_EXPECTS(cluster_offsets[orig_index.n_lists()] == numNewVectors,
               "cluster sizes do not add up.");
  // data_indices
  for (uint32_t i = 0; i < numNewVectors; i++) {
    uint32_t l                       = new_data_labels.data()[i];
    data_indices[cluster_offsets[l]] = i;
    cluster_offsets[l] += 1;
  }
  // Recover cluster_offsets
  for (uint32_t l = 0; l < orig_index.n_lists(); l++) {
    cluster_offsets[l] -= cluster_sizes.data()[l];
  }

  //
  // Compute PQ code for new vectors
  //
  rmm::device_uvector<uint8_t> new_pq_codes(
    numNewVectors * orig_index.pq_dim() * orig_index.pq_bits() / 8,
    handle.get_stream(),
    &managed_memory);
  _cuann_compute_PQ_code<T>(handle,
                            numNewVectors,
                            orig_index.dim(),
                            orig_index.rot_dim(),
                            orig_index.pq_dim(),
                            orig_index.pq_len(),
                            orig_index.pq_bits(),
                            orig_index.n_lists(),
                            orig_index.codebook_kind(),
                            maxClusterSize,
                            cluster_centers.data(),
                            oldRotationMatrix,
                            newVectors,
                            data_indices.data(),
                            cluster_sizes.data(),
                            cluster_offsets.data(),
                            oldPqCenters,
                            0,
                            new_pq_codes.data(),
                            &managed_memory,
                            device_memory);

  //
  // Create descriptor for new index
  //
  ivf_pq::index<IdxT> new_index(handle,
                                orig_index.metric(),
                                orig_index.codebook_kind(),
                                orig_index.n_lists(),
                                orig_index.dim(),
                                orig_index.pq_bits(),
                                orig_index.pq_dim());
  new_index.allocate(handle, orig_index.size() + numNewVectors);
  RAFT_LOG_DEBUG("Index size: %u -> %u", orig_index.size(), new_index.size());

  auto newClusterCenters    = new_index.centers().data_handle();
  auto newPqCenters         = new_index.pq_centers().data_handle();
  auto newPqDataset         = new_index.pq_dataset().data_handle();
  auto newOriginalNumbers   = new_index.indices().data_handle();
  auto new_cluster_offsets  = new_index.list_offsets().data_handle();
  auto newRotationMatrix    = new_index.rotation_matrix().data_handle();
  auto newClusterRotCenters = new_index.centers_rot().data_handle();

  //
  // Copy the unchanged parts
  //
  copy(new_index.centers().data_handle(),
       orig_index.centers().data_handle(),
       orig_index.centers().size(),
       handle.get_stream());
  copy(new_index.centers_rot().data_handle(),
       orig_index.centers_rot().data_handle(),
       orig_index.centers_rot().size(),
       handle.get_stream());
  copy(new_index.pq_centers().data_handle(),
       orig_index.pq_centers().data_handle(),
       orig_index.pq_centers().size(),
       handle.get_stream());
  copy(new_index.rotation_matrix().data_handle(),
       orig_index.rotation_matrix().data_handle(),
       orig_index.rotation_matrix().size(),
       handle.get_stream());
  handle.sync_stream();

  //
  // Make new_cluster_offsets
  //
  maxClusterSize         = 0;
  new_cluster_offsets[0] = 0;
  for (uint32_t l = 0; l < new_index.n_lists(); l++) {
    uint32_t oldClusterSize    = old_cluster_offsets[l + 1] - old_cluster_offsets[l];
    new_cluster_offsets[l + 1] = new_cluster_offsets[l];
    new_cluster_offsets[l + 1] += oldClusterSize + cluster_sizes.data()[l];
    maxClusterSize = max(maxClusterSize, oldClusterSize + cluster_sizes.data()[l]);
  }

  //
  // Make newOriginalNumbers
  //
  for (uint32_t i = 0; i < numNewVectors; i++) {
    data_indices[i] += orig_index.size();
  }
  for (uint32_t l = 0; l < new_index.n_lists(); l++) {
    uint32_t oldClusterSize = old_cluster_offsets[l + 1] - old_cluster_offsets[l];
    memcpy(newOriginalNumbers + new_cluster_offsets[l],
           oldOriginalNumbers + old_cluster_offsets[l],
           sizeof(uint32_t) * oldClusterSize);
    memcpy(newOriginalNumbers + new_cluster_offsets[l] + oldClusterSize,
           data_indices.data() + cluster_offsets[l],
           sizeof(uint32_t) * cluster_sizes.data()[l]);
  }

  //
  // Make newPqDataset
  //
  size_t unitPqDataset = new_index.pq_dim() * new_index.pq_bits() / 8;
  for (uint32_t l = 0; l < new_index.n_lists(); l++) {
    uint32_t oldClusterSize = old_cluster_offsets[l + 1] - old_cluster_offsets[l];
    memcpy(newPqDataset + unitPqDataset * new_cluster_offsets[l],
           oldPqDataset + unitPqDataset * old_cluster_offsets[l],
           sizeof(uint8_t) * unitPqDataset * oldClusterSize);
    memcpy(newPqDataset + unitPqDataset * (new_cluster_offsets[l] + oldClusterSize),
           new_pq_codes.data() + unitPqDataset * cluster_offsets[l],
           sizeof(uint8_t) * unitPqDataset * cluster_sizes.data()[l]);
  }

  _cuann_get_inclusiveSumSortedClusterSize(new_index);

  return new_index;
}

/** See raft::spatial::knn::ivf_pq::extend docs */
template <typename T, typename IdxT>
inline auto extend(const handle_t& handle,
                   const index<IdxT>& orig_index,
                   const T* new_vectors,
                   const IdxT* new_indices,
                   IdxT n_rows) -> index<IdxT>
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_pq::extend(%zu, %u)", size_t(n_rows), orig_index.dim());

  if (new_indices != nullptr) {
    RAFT_LOG_WARN("Index input is ignored at the moment (non-null new_indices given).");
  }

  return ivf_pq::detail::cuannIvfPqCreateNewIndexByAddingVectorsToOldIndex(
    handle, const_cast<index<IdxT>&>(orig_index), new_vectors, n_rows);
}

/** See raft::spatial::knn::ivf_pq::build docs */
template <typename T, typename IdxT>
inline auto build(
  const handle_t& handle, const index_params& params, const T* dataset, IdxT n_rows, uint32_t dim)
  -> index<IdxT>
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_pq::build(%zu, %u)", size_t(n_rows), dim);
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
                "unsupported data type");
  RAFT_EXPECTS(n_rows > 0 && dim > 0, "empty dataset");

  ivf_pq::index<IdxT> index(handle, params, dim);
  index.allocate(handle, n_rows);

  // Build index
  ivf_pq::detail::cuannIvfPqBuildIndex(handle,
                                       index,
                                       dataset,
                                       params.kmeans_trainset_fraction,
                                       params.kmeans_n_iters,
                                       params.random_rotation);

  return index;
}

}  // namespace raft::spatial::knn::ivf_pq::detail
