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
#include "ann_utils.cuh"
#include "topk.cuh"
#include "topk/warpsort_topk.cuh"

#include <raft/common/device_loads_stores.cuh>
#include <raft/core/cudart_utils.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/cuda_utils.cuh>
#include <raft/device_atomics.cuh>
#include <raft/distance/distance.cuh>
#include <raft/distance/distance_type.hpp>
#include <raft/pow2_utils.cuh>
#include <raft/vectorized.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <cub/cub.cuh>
#include <thrust/sequence.h>

#include <cuda_fp16.h>

namespace raft::spatial::knn::ivf_pq::detail {

using namespace raft::spatial::knn::detail;  // NOLINT

template <unsigned expBitLen>
struct fp_8bit;

template <unsigned expBitLen>
__device__ __host__ fp_8bit<expBitLen> __float2fp_8bit(const float v);
template <unsigned expBitLen>
__device__ __host__ float __fp_8bit2float(const fp_8bit<expBitLen>& v);

template <unsigned expBitLen>
struct fp_8bit {
  uint8_t bitstring;

  __device__ __host__ fp_8bit(const uint8_t bs) { bitstring = bs; }
  __device__ __host__ fp_8bit(const float fp)
  {
    bitstring = __float2fp_8bit<expBitLen>(fp).bitstring;
  }
  __device__ __host__ fp_8bit<expBitLen>& operator=(const float fp)
  {
    bitstring = __float2fp_8bit<expBitLen>(fp).bitstring;
    return *this;
  }

  __device__ __host__ operator float() const { return __fp_8bit2float(*this); }
};

// Since __float_as_uint etc can not be used in host codes,
// these converters are needed for test.
union cvt_fp_32bit {
  float fp;
  uint32_t bs;
};
union cvt_fp_16bit {
  half fp;
  uint16_t bs;
};

// Type converters
template <unsigned expBitLen>
__device__ __host__ fp_8bit<expBitLen> __float2fp_8bit(const float v)
{
  if (v < 1. / (1u << ((1u << (expBitLen - 1)) - 1)))
    return fp_8bit<expBitLen>{static_cast<uint8_t>(0)};
  return fp_8bit<expBitLen>{static_cast<uint8_t>(
    (cvt_fp_32bit{.fp = v}.bs + (((1u << (expBitLen - 1)) - 1) << 23) - 0x3f800000u) >>
    (15 + expBitLen))};
}

template <unsigned expBitLen>
__device__ __host__ float __fp_8bit2float(const fp_8bit<expBitLen>& v)
{
  return cvt_fp_32bit{.bs = ((v.bitstring << (15 + expBitLen)) +
                             (0x3f800000u | (0x00400000u >> (8 - expBitLen))) -
                             (((1u << (expBitLen - 1)) - 1) << 23))}
    .fp;
}

__device__ inline uint32_t warp_scan(uint32_t x)
{
  uint32_t y;
  y = __shfl_up_sync(0xffffffff, x, 1);
  if (threadIdx.x % 32 >= 1) x += y;
  y = __shfl_up_sync(0xffffffff, x, 2);
  if (threadIdx.x % 32 >= 2) x += y;
  y = __shfl_up_sync(0xffffffff, x, 4);
  if (threadIdx.x % 32 >= 4) x += y;
  y = __shfl_up_sync(0xffffffff, x, 8);
  if (threadIdx.x % 32 >= 8) x += y;
  y = __shfl_up_sync(0xffffffff, x, 16);
  if (threadIdx.x % 32 >= 16) x += y;
  return x;
}

__device__ inline uint32_t thread_block_scan(uint32_t x, uint32_t* smem)
{
  x = warp_scan(x);
  __syncthreads();
  if (threadIdx.x % 32 == 31) { smem[threadIdx.x / 32] = x; }
  __syncthreads();
  if (threadIdx.x < 32) { smem[threadIdx.x] = warp_scan(smem[threadIdx.x]); }
  __syncthreads();
  if (threadIdx.x / 32 > 0) { x += smem[threadIdx.x / 32 - 1]; }
  __syncthreads();
  return x;
}

template <typename IdxT>
__global__ void ivfpq_make_chunk_index_ptr(
  uint32_t numProbes,
  uint32_t sizeBatch,
  const IdxT* cluster_offsets,            // [n_clusters + 1,]
  const uint32_t* _clusterLabelsToProbe,  // [sizeBatch, numProbes,]
  uint32_t* _chunkIndexPtr,               // [sizeBetch, numProbes,]
  uint32_t* numSamples                    // [sizeBatch,]
)
{
  __shared__ uint32_t smem_temp[32];
  __shared__ uint32_t smem_base[2];

  uint32_t iBatch = blockIdx.x;
  if (iBatch >= sizeBatch) return;
  const uint32_t* clusterLabelsToProbe = _clusterLabelsToProbe + (numProbes * iBatch);
  uint32_t* chunkIndexPtr              = _chunkIndexPtr + (numProbes * iBatch);

  //
  uint32_t j_end = (numProbes + 1024 - 1) / 1024;
  for (uint32_t j = 0; j < j_end; j++) {
    uint32_t i   = threadIdx.x + (1024 * j);
    uint32_t val = 0;
    if (i < numProbes) {
      uint32_t l = clusterLabelsToProbe[i];
      val        = static_cast<uint32_t>(cluster_offsets[l + 1] - cluster_offsets[l]);
    }
    val = thread_block_scan(val, smem_temp);

    if (i < numProbes) {
      if (j > 0) { val += smem_base[(j - 1) & 0x1]; }
      chunkIndexPtr[i] = val;
      if (i == numProbes - 1) { numSamples[iBatch] = val; }
    }

    if ((j < j_end - 1) && (threadIdx.x == 1023)) { smem_base[j & 0x1] = val; }
  }
}

template <typename IdxT>
__device__ void ivfpq_get_id_dataset(uint32_t iSample,
                                     uint32_t numProbes,
                                     const IdxT* cluster_offsets,     // [n_clusters + 1,]
                                     const uint32_t* cluster_labels,  // [numProbes,]
                                     const uint32_t* chunkIndexPtr,   // [numProbes,]
                                     uint32_t& iChunk,
                                     uint32_t& label,
                                     IdxT& iDataset)
{
  uint32_t minChunk = 0;
  uint32_t maxChunk = numProbes - 1;
  iChunk            = (minChunk + maxChunk) / 2;
  while (minChunk < maxChunk) {
    if (iSample >= chunkIndexPtr[iChunk]) {
      minChunk = iChunk + 1;
    } else {
      maxChunk = iChunk;
    }
    iChunk = (minChunk + maxChunk) / 2;
  }

  label                   = cluster_labels[iChunk];
  uint32_t iSampleInChunk = iSample;
  if (iChunk > 0) { iSampleInChunk -= chunkIndexPtr[iChunk - 1]; }
  iDataset = iSampleInChunk + cluster_offsets[label];
}

template <typename scoreDtype, typename IdxT>
__global__ void ivfpq_make_outputs(distance::DistanceType metric,
                                   uint32_t numProbes,
                                   uint32_t topk,
                                   uint32_t maxSamples,
                                   uint32_t sizeBatch,
                                   const IdxT* cluster_offsets,     // [n_clusters + 1]
                                   const IdxT* data_indices,        // [index_size]
                                   const uint32_t* cluster_labels,  // [sizeBatch, numProbes]
                                   const uint32_t* chunkIndexPtr,   // [sizeBatch, numProbes]
                                   const scoreDtype* scores,        // [sizeBatch, maxSamples] or
                                                                    // [sizeBatch, numProbes, topk]
                                   const uint32_t* scoreTopkIndex,  // [sizeBatch, numProbes, topk]
                                   const uint32_t* topkSampleIds,   // [sizeBatch, topk]
                                   IdxT* topkNeighbors,             // [sizeBatch, topk]
                                   float* topkScores                // [sizeBatch, topk]
)
{
  uint32_t i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i >= topk) return;
  uint32_t iBatch = blockIdx.y;
  if (iBatch >= sizeBatch) return;

  uint32_t iSample = topkSampleIds[i + (topk * iBatch)];
  float score;
  if (scoreTopkIndex == nullptr) {
    // 0 <= iSample < maxSamples
    score = scores[iSample + (maxSamples * iBatch)];
    uint32_t iChunk;
    uint32_t label;
    IdxT iDataset;
    ivfpq_get_id_dataset(iSample,
                         numProbes,
                         cluster_offsets,
                         cluster_labels + (numProbes * iBatch),
                         chunkIndexPtr + (numProbes * iBatch),
                         iChunk,
                         label,
                         iDataset);
    topkNeighbors[i + (topk * iBatch)] = data_indices[iDataset];
  } else {
    // 0 <= iSample < (numProbes * topk)
    score                              = scores[iSample + ((numProbes * topk) * iBatch)];
    IdxT iDataset                      = scoreTopkIndex[iSample + ((numProbes * topk) * iBatch)];
    topkNeighbors[i + (topk * iBatch)] = data_indices[iDataset];
  }
  switch (metric) {
    case distance::DistanceType::InnerProduct: {
      score = -score;
    } break;
    default: break;
  }
  topkScores[i + (topk * iBatch)] = score;
}

template <int pq_bits, int vecLen, typename T, typename IdxT, typename LutT = float>
__device__ float ivfpq_compute_score(uint32_t pq_dim,
                                     IdxT iDataset,
                                     const uint8_t* pqDataset,  // [n_rows, pq_dim * pq_bits / 8]
                                     const LutT* preCompScores  // [pq_dim, pq_width]
)
{
  float score             = 0.0;
  constexpr uint32_t bitT = sizeof(T) * 8;
  const T* headPqDataset  = (T*)(pqDataset + (uint64_t)iDataset * (pq_dim * pq_bits / 8));
  for (int j = 0; j < pq_dim / vecLen; j += 1) {
    T pqCode = headPqDataset[0];
    headPqDataset += 1;
    uint32_t bitLeft = bitT;
#pragma unroll vecLen
    for (int k = 0; k < vecLen; k += 1) {
      uint8_t code = pqCode;
      if (bitLeft > pq_bits) {
        // This condition is always true here (to make the compiler happy)
        if constexpr (bitT > pq_bits) { pqCode >>= pq_bits; }
        bitLeft -= pq_bits;
      } else {
        if (k < vecLen - 1) {
          pqCode = headPqDataset[0];
          headPqDataset += 1;
        }
        code |= (pqCode << bitLeft);
        pqCode >>= (pq_bits - bitLeft);
        bitLeft += (bitT - pq_bits);
      }
      code &= (1 << pq_bits) - 1;
      score += (float)preCompScores[code];
      preCompScores += (1 << pq_bits);
    }
  }
  return score;
}

//
// (*) Restrict the peak GPU occupancy up-to 50% by "__launch_bounds__(1024, 1)",
// as there were cases where performance dropped by a factor of two or more on V100
// when the peak GPU occupancy was set to more than 50%.
//
template <int pq_bits,
          int vecLen,
          typename T,
          typename IdxT,
          int depth,
          bool preCompBaseDiff,
          typename OutT,
          typename LutT,
          bool EnableSMemLut>
__launch_bounds__(1024, 1) __global__ void ivfpq_compute_similarity(
  uint32_t n_rows,
  uint32_t dim,
  uint32_t numProbes,
  uint32_t pq_dim,
  uint32_t sizeBatch,
  uint32_t maxSamples,
  distance::DistanceType metric,
  codebook_gen codebook_kind,
  uint32_t topk,
  const float* cluster_centers,    // [n_clusters, dim,]
  const float* pqCenters,          // [pq_dim, pq_width, pq_len,], or
                                   // [numClusetrs, pq_width, pq_len,]
  const uint8_t* pqDataset,        // [n_rows, pq_dim * pq_bits / 8]
  const IdxT* cluster_offsets,     // [n_clusters + 1,]
  const uint32_t* _clusterLabels,  // [sizeBatch, numProbes,]
  const uint32_t* _chunkIndexPtr,  // [sizeBatch, numProbes,]
  const float* _query,             // [sizeBatch, dim,]
  const uint32_t* indexList,       // [sizeBatch * numProbes]
  LutT* _preCompScores,            // [...]
  OutT* _output,                   // [sizeBatch, maxSamples,] or [sizeBatch, numProbes, topk]
  uint32_t* _topkIndex             // [sizeBatch, numProbes, topk]
)
{
  /* Shared memory:

    * preCompScores: lookup table (LUT) of size = `pq_dim << pq_bits`  (when EnableSMemLut)
    * baseDiff: size = dim  (which is equal to `pq_dim * pq_len`)
    * topk::block_sort: some amount of shared memory, but overlaps with the rest:
        block_sort only needs shared memory for `.done()` operation, which can come very last.
  */
  extern __shared__ __align__(256) uint8_t smem_buf[];

  const uint32_t pq_len = dim / pq_dim;

  LutT* preCompScores = EnableSMemLut ? reinterpret_cast<LutT*>(smem_buf)
                                      : (_preCompScores + (pq_dim << pq_bits) * blockIdx.x);
  float* baseDiff     = nullptr;
  if constexpr (preCompBaseDiff) {
    if constexpr (EnableSMemLut) {
      baseDiff = reinterpret_cast<float*>(preCompScores + (pq_dim << pq_bits));
    } else {
      baseDiff = reinterpret_cast<float*>(smem_buf);
    }
  }
  bool manageLocalTopk = false;
  if (_topkIndex != nullptr) { manageLocalTopk = true; }

  for (int ib = blockIdx.x; ib < sizeBatch * numProbes; ib += gridDim.x) {
    uint32_t iBatch;
    uint32_t iProbe;
    if (indexList == nullptr) {
      iBatch = ib % sizeBatch;
      iProbe = ib / sizeBatch;
    } else {
      iBatch = indexList[ib] / numProbes;
      iProbe = indexList[ib] % numProbes;
    }
    if (iBatch >= sizeBatch || iProbe >= numProbes) continue;

    const uint32_t* cluster_labels = _clusterLabels + (numProbes * iBatch);
    const uint32_t* chunkIndexPtr  = _chunkIndexPtr + (numProbes * iBatch);
    const float* query             = _query + (dim * iBatch);
    OutT* output;
    uint32_t* topkIndex = nullptr;
    if (manageLocalTopk) {
      // Store topk calculated distances to output (and its indices to topkIndex)
      output    = _output + (topk * (iProbe + (numProbes * iBatch)));
      topkIndex = _topkIndex + (topk * (iProbe + (numProbes * iBatch)));
    } else {
      // Store all calculated distances to output
      output = _output + (maxSamples * iBatch);
    }
    uint32_t label               = cluster_labels[iProbe];
    const float* myClusterCenter = cluster_centers + (dim * label);
    const float* myPqCenters;
    if (codebook_kind == codebook_gen::PER_SUBSPACE) {
      myPqCenters = pqCenters;
    } else {
      myPqCenters = pqCenters + (pq_len << pq_bits) * label;
    }

    if constexpr (preCompBaseDiff) {
      // Reduce computational complexity by pre-computing the difference
      // between the cluster centroid and the query.
      for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        baseDiff[i] = query[i] - myClusterCenter[i];
      }
      __syncthreads();
    }

    // Create a lookup table
    for (uint32_t i = threadIdx.x; i < (pq_dim << pq_bits); i += blockDim.x) {
      uint32_t iPq   = i >> pq_bits;
      uint32_t iCode = codebook_kind == codebook_gen::PER_CLUSTER ? i & ((1 << pq_bits) - 1) : i;
      float score    = 0.0;
      switch (metric) {
        case distance::DistanceType::L2Expanded: {
          for (uint32_t j = 0; j < pq_len; j++) {
            uint32_t k = j + (pq_len * iPq);
            float diff;
            if constexpr (preCompBaseDiff) {
              diff = baseDiff[k];
            } else {
              diff = query[k] - myClusterCenter[k];
            }
            diff -= myPqCenters[j + (pq_len * iCode)];
            score += diff * diff;
          }
        } break;
        case distance::DistanceType::InnerProduct: {
          for (uint32_t j = 0; j < pq_len; j++) {
            uint32_t k = j + (pq_len * iPq);
            score -= query[k] * (myClusterCenter[k] + myPqCenters[j + (pq_len * iCode)]);
          }
        } break;
      }
      preCompScores[i] = LutT(score);
    }

    uint32_t iSampleBase = 0;
    if (iProbe > 0) { iSampleBase = chunkIndexPtr[iProbe - 1]; }
    uint32_t nSamples            = chunkIndexPtr[iProbe] - iSampleBase;
    uint32_t nSamples32          = Pow2<32>::roundUp(nSamples);
    IdxT selected_cluster_offset = cluster_offsets[label];

    using block_sort_t =
      topk::block_sort<topk::warp_sort_immediate, depth * WarpSize, true, OutT, uint32_t>;
    block_sort_t block_topk(topk, reinterpret_cast<uint8_t*>(smem_buf));
    constexpr OutT limit = block_sort_t::queue_t::kDummy;

    // Compute a distance for each sample
    for (uint32_t i = threadIdx.x; i < nSamples32; i += blockDim.x) {
      float score = limit;
      if (i < nSamples) {
        score = ivfpq_compute_score<pq_bits, vecLen, T, IdxT, LutT>(
          pq_dim, selected_cluster_offset + i, pqDataset, preCompScores);
      }
      if (manageLocalTopk) {
        block_topk.add(score, selected_cluster_offset + i);
      } else {
        if (i < nSamples) { output[i + iSampleBase] = score; }
      }
    }
    __syncthreads();
    if (manageLocalTopk) {
      // sync threads before and after the topk merging operation, because we reuse smem_buf
      block_topk.done();
      block_topk.store(output, topkIndex);
      __syncthreads();
    } else {
      // fill in the rest of the output with dummy values
      if (iProbe + 1 == numProbes) {
        for (uint32_t i = threadIdx.x + iSampleBase + nSamples; i < maxSamples; i += blockDim.x) {
          output[i] = limit;
        }
      }
    }
  }
}

// search
template <typename scoreDtype, typename LutT, typename IdxT>
void ivfpq_search(const handle_t& handle,
                  const index<IdxT>& index,
                  uint32_t n_probes,
                  uint32_t max_batch_size,
                  uint32_t topK,
                  uint32_t preferred_thread_block_size,
                  uint32_t n_queries,
                  const float* cluster_centers,          // [index_size, rot_dim]
                  const float* pqCenters,                // [pq_dim, pq_width, pq_len]
                  const uint8_t* pqDataset,              // [index_size, pq_dim * pq_bits / 8]
                  const IdxT* data_indices,              // [index_size]
                  const IdxT* cluster_offsets,           // [n_clusters + 1]
                  const uint32_t* clusterLabelsToProbe,  // [n_queries, numProbes]
                  const float* query,                    // [n_queries, rot_dim]
                  IdxT* topkNeighbors,                   // [n_queries, topK]
                  float* topkDistances,                  // [n_queries, topK]
                  rmm::mr::device_memory_resource* mr)
{
  RAFT_EXPECTS(n_queries <= max_batch_size,
               "number of queries (%u) must be smaller the max batch size (%u)",
               n_queries,
               max_batch_size);
  auto stream = handle.get_stream();

  auto max_samples = Pow2<128>::roundUp(index.inclusiveSumSortedClusterSize()(n_probes - 1));

  bool manage_local_topk =
    raft::ceildiv<int>(topK, 32) <= 4    // depth is not too large
    && n_probes >= 16                    // not too few clusters looked up
    && max_batch_size * n_probes >= 256  // overall amount of work is not too small
    ;
  auto topk_len = manage_local_topk ? n_probes * topK : max_samples;

  rmm::device_uvector<uint32_t> cluster_labels_out(max_batch_size * n_probes, stream, mr);
  rmm::device_uvector<uint32_t> index_list_sorted_buf(0, stream, mr);
  uint32_t* index_list_sorted = nullptr;
  rmm::device_uvector<uint32_t> num_samples(max_batch_size, stream, mr);
  rmm::device_uvector<uint32_t> chunk_index(max_batch_size * n_probes, stream, mr);
  rmm::device_uvector<uint32_t> topk_sids(max_batch_size * topK, stream, mr);
  // [maxBatchSize, maxSamples] or  [maxBatchSize, numProbes, topk]
  rmm::device_uvector<scoreDtype> scores_buf(max_batch_size * topk_len, stream, mr);
  rmm::device_uvector<uint32_t> topk_index_buf(0, stream, mr);
  uint32_t* topk_index = nullptr;
  if (manage_local_topk) {
    topk_index_buf.resize(max_batch_size * topk_len, stream);
    topk_index = topk_index_buf.data();
  }

  dim3 mc_threads(1024, 1, 1);  // DO NOT CHANGE
  dim3 mc_blocks(n_queries, 1, 1);
  ivfpq_make_chunk_index_ptr<<<mc_blocks, mc_threads, 0, stream>>>(n_probes,
                                                                   n_queries,
                                                                   cluster_offsets,
                                                                   clusterLabelsToProbe,
                                                                   chunk_index.data(),
                                                                   num_samples.data());

  if (n_queries * n_probes > 256) {
    // Sorting index by cluster number (label).
    // The goal is to incrase the L2 cache hit rate to read the vectors
    // of a cluster by processing the cluster at the same time as much as
    // possible.
    index_list_sorted_buf.resize(max_batch_size * n_probes, stream);
    rmm::device_uvector<uint32_t> index_list_buf(max_batch_size * n_probes, stream, mr);
    auto index_list   = index_list_buf.data();
    index_list_sorted = index_list_sorted_buf.data();
    thrust::sequence(handle.get_thrust_policy(),
                     thrust::device_pointer_cast(index_list),
                     thrust::device_pointer_cast(index_list + n_queries * n_probes));

    int begin_bit             = 0;
    int end_bit               = sizeof(uint32_t) * 8;
    size_t cub_workspace_size = 0;
    cub::DeviceRadixSort::SortPairs(nullptr,
                                    cub_workspace_size,
                                    clusterLabelsToProbe,
                                    cluster_labels_out.data(),
                                    index_list,
                                    index_list_sorted,
                                    n_queries * n_probes,
                                    begin_bit,
                                    end_bit,
                                    stream);
    rmm::device_buffer cub_workspace(cub_workspace_size, stream, mr);
    cub::DeviceRadixSort::SortPairs(cub_workspace.data(),
                                    cub_workspace_size,
                                    clusterLabelsToProbe,
                                    cluster_labels_out.data(),
                                    index_list,
                                    index_list_sorted,
                                    n_queries * n_probes,
                                    begin_bit,
                                    end_bit,
                                    stream);
  }

  // Select a GPU kernel for distance calculation
#define SET_KERNEL1(B, V, T, D)                                                             \
  do {                                                                                      \
    static_assert((B * V) % (sizeof(T) * 8) == 0);                                          \
    kernel_no_basediff =                                                                    \
      ivfpq_compute_similarity<B, V, T, IdxT, D, false, scoreDtype, LutT, true>;            \
    kernel_fast = ivfpq_compute_similarity<B, V, T, IdxT, D, true, scoreDtype, LutT, true>; \
    kernel_no_smem_lut =                                                                    \
      ivfpq_compute_similarity<B, V, T, IdxT, D, true, scoreDtype, LutT, false>;            \
  } while (0)

#define SET_KERNEL2(B, M, D)                                                     \
  do {                                                                           \
    RAFT_EXPECTS(index.pq_dim() % M == 0, "pq_dim must be a multiple of %u", M); \
    if (index.pq_dim() % (M * 8) == 0) {                                         \
      SET_KERNEL1(B, (M * 8), uint64_t, D);                                      \
    } else if (index.pq_dim() % (M * 4) == 0) {                                  \
      SET_KERNEL1(B, (M * 4), uint32_t, D);                                      \
    } else if (index.pq_dim() % (M * 2) == 0) {                                  \
      SET_KERNEL1(B, (M * 2), uint16_t, D);                                      \
    } else if (index.pq_dim() % (M * 1) == 0) {                                  \
      SET_KERNEL1(B, (M * 1), uint8_t, D);                                       \
    }                                                                            \
  } while (0)

#define SET_KERNEL3(D)                     \
  do {                                     \
    switch (index.pq_bits()) {             \
      case 4: SET_KERNEL2(4, 2, D); break; \
      case 5: SET_KERNEL2(5, 8, D); break; \
      case 6: SET_KERNEL2(6, 4, D); break; \
      case 7: SET_KERNEL2(7, 8, D); break; \
      case 8: SET_KERNEL2(8, 1, D); break; \
    }                                      \
  } while (0)

  typedef void (*kernel_t)(uint32_t,
                           uint32_t,
                           uint32_t,
                           uint32_t,
                           uint32_t,
                           uint32_t,
                           distance::DistanceType,
                           codebook_gen,
                           uint32_t,
                           const float*,
                           const float*,
                           const uint8_t*,
                           const IdxT*,
                           const uint32_t*,
                           const uint32_t*,
                           const float*,
                           const uint32_t*,
                           LutT*,
                           scoreDtype*,
                           uint32_t*);
  kernel_t kernel_no_basediff;
  kernel_t kernel_fast;
  kernel_t kernel_no_smem_lut;
  uint32_t depth = 1;
  if (manage_local_topk) {
    while (depth * WarpSize < topK) {
      depth *= 2;
    }
  }
  switch (depth) {
    case 1: SET_KERNEL3(1); break;
    case 2: SET_KERNEL3(2); break;
    case 4: SET_KERNEL3(4); break;
    default: RAFT_FAIL("ivf_pq::search(k = %u): depth value is too big (%d)", topK, depth);
  }
  RAFT_LOG_DEBUG("ivf_pq::search(k = %u, depth = %d, dim = %u/%u/%u)",
                 topK,
                 depth,
                 index.dim(),
                 index.rot_dim(),
                 index.pq_dim());
  constexpr size_t thresholdSmem = 48 * 1024;
  size_t sizeSmem                = sizeof(LutT) * index.pq_dim() * index.pq_width();
  size_t sizeSmemBaseDiff        = sizeof(float) * index.rot_dim();

  uint32_t numCTAs = n_queries * n_probes;
  int numThreads   = 1024;
  // preferred_thread_block_size == 0 means using auto thread block size calculation
  // mode
  if (preferred_thread_block_size == 0) {
    constexpr int minThreads = 256;
    while (numThreads > minThreads) {
      if (numCTAs < uint32_t(getMultiProcessorCount() * (1024 / (numThreads / 2)))) { break; }
      if (handle.get_device_properties().sharedMemPerMultiprocessor * 2 / 3 <
          sizeSmem * (1024 / (numThreads / 2))) {
        break;
      }
      numThreads /= 2;
    }
  } else {
    numThreads = preferred_thread_block_size;
  }
  size_t sizeSmemForLocalTopk =
    topk::template calc_smem_size_for_block_wide<float, uint32_t>(numThreads / WarpSize, topK);
  sizeSmem = max(sizeSmem, sizeSmemForLocalTopk);

  kernel_t kernel = kernel_no_basediff;

  bool kernel_no_basediff_available = true;
  bool use_smem_lut                 = true;
  if (sizeSmem > thresholdSmem) {
    cudaError_t cudaError = cudaFuncSetAttribute(
      kernel_no_basediff, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeSmem);
    if (cudaError != cudaSuccess) {
      RAFT_EXPECTS(
        cudaError == cudaGetLastError(),
        "Tried to reset the expected cuda error code, but it didn't match the expectation");
      kernel_no_basediff_available = false;

      // Use "kernel_no_smem_lut" which just uses small amount of shared memory.
      kernel       = kernel_no_smem_lut;
      use_smem_lut = false;
      numThreads   = 1024;
      size_t sizeSmemForLocalTopk =
        topk::calc_smem_size_for_block_wide<float, uint32_t>(numThreads / WarpSize, topK);
      sizeSmem = max(sizeSmemBaseDiff, sizeSmemForLocalTopk);
      numCTAs  = getMultiProcessorCount();
    }
  }
  if (kernel_no_basediff_available) {
    bool kernel_fast_available = true;
    if (sizeSmem + sizeSmemBaseDiff > thresholdSmem) {
      cudaError_t cudaError = cudaFuncSetAttribute(
        kernel_fast, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeSmem + sizeSmemBaseDiff);
      if (cudaError != cudaSuccess) {
        RAFT_EXPECTS(
          cudaError == cudaGetLastError(),
          "Tried to reset the expected cuda error code, but it didn't match the expectation");
        kernel_fast_available = false;
      }
    }
    if (kernel_fast_available) {
      int numBlocks_kernel_no_basediff = 0;
      RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks_kernel_no_basediff, kernel_no_basediff, numThreads, sizeSmem));

      int numBlocks_kernel_fast = 0;
      RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks_kernel_fast, kernel_fast, numThreads, sizeSmem + sizeSmemBaseDiff));

      // Use "kernel_fast" only if GPU occupancy does not drop
      if (numBlocks_kernel_no_basediff == numBlocks_kernel_fast) {
        kernel = kernel_fast;
        sizeSmem += sizeSmemBaseDiff;
      }
    }
  }

  rmm::device_uvector<LutT> precomp_scores(
    use_smem_lut ? 0 : numCTAs * index.pq_dim() * index.pq_width(), stream, mr);
  dim3 cta_threads(numThreads, 1, 1);
  dim3 cta_blocks(numCTAs, 1, 1);
  kernel<<<cta_blocks, cta_threads, sizeSmem, stream>>>(index.size(),
                                                        index.rot_dim(),
                                                        n_probes,
                                                        index.pq_dim(),
                                                        n_queries,
                                                        max_samples,
                                                        index.metric(),
                                                        index.codebook_kind(),
                                                        topK,
                                                        cluster_centers,
                                                        pqCenters,
                                                        pqDataset,
                                                        cluster_offsets,
                                                        clusterLabelsToProbe,
                                                        chunk_index.data(),
                                                        query,
                                                        index_list_sorted,
                                                        precomp_scores.data(),
                                                        scores_buf.data(),
                                                        topk_index);

  {
    // Select topk vectors for each query
    rmm::device_uvector<scoreDtype> topk_dists(n_queries * topK, stream, mr);
    select_topk<scoreDtype, uint32_t>(scores_buf.data(),
                                      nullptr,
                                      n_queries,
                                      topk_len,
                                      topK,
                                      topk_dists.data(),
                                      topk_sids.data(),
                                      true,
                                      stream,
                                      mr);
  }

  dim3 mo_threads(128, 1, 1);
  dim3 mo_blocks(raft::ceildiv<uint32_t>(topK, mo_threads.x), n_queries, 1);
  ivfpq_make_outputs<scoreDtype><<<mo_blocks, mo_threads, 0, stream>>>(index.metric(),
                                                                       n_probes,
                                                                       topK,
                                                                       max_samples,
                                                                       n_queries,
                                                                       cluster_offsets,
                                                                       data_indices,
                                                                       clusterLabelsToProbe,
                                                                       chunk_index.data(),
                                                                       scores_buf.data(),
                                                                       topk_index,
                                                                       topk_sids.data(),
                                                                       topkNeighbors,
                                                                       topkDistances);
}

/** See raft::spatial::knn::ivf_pq::search docs */
template <typename T, typename IdxT>
inline void search(const handle_t& handle,
                   const search_params& params,
                   const index<IdxT>& index,
                   const T* queries,
                   uint32_t n_queries,
                   uint32_t k,
                   IdxT* neighbors,
                   float* distances,
                   rmm::mr::device_memory_resource* mr = nullptr)
{
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
                "Unsupported element type.");
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_pq::search(k = %u, n_queries = %u, dim = %zu)", k, n_queries, index.dim());

  RAFT_EXPECTS(
    params.internal_distance_dtype == CUDA_R_16F || params.internal_distance_dtype == CUDA_R_32F,
    "internal_distance_dtype must be either CUDA_R_16F or CUDA_R_32F");
  RAFT_EXPECTS(params.smem_lut_dtype == CUDA_R_16F || params.smem_lut_dtype == CUDA_R_32F ||
                 params.smem_lut_dtype == CUDA_R_8U,
               "smem_lut_dtype must be CUDA_R_16F, CUDA_R_32F or CUDA_R_8U");
  RAFT_EXPECTS(
    params.preferred_thread_block_size == 256 || params.preferred_thread_block_size == 512 ||
      params.preferred_thread_block_size == 1024 || params.preferred_thread_block_size == 0,
    "preferred_thread_block_size must be 0, 256, 512 or 1024, but %u is given.",
    params.preferred_thread_block_size);
  RAFT_EXPECTS(k > 0, "parameter `k` in top-k must be positive.");
  RAFT_EXPECTS(
    k <= index.size(),
    "parameter `k` (%u) in top-k must not be larger that the total size of the index (%zu)",
    k,
    static_cast<uint64_t>(index.size()));
  RAFT_EXPECTS(params.n_probes > 0,
               "n_probes (number of clusters to probe in the search) must be positive.");
  auto n_probes = std::min<uint32_t>(params.n_probes, index.n_lists());
  {
    IdxT n_samples_worst_case = index.size();
    if (n_probes < index.n_lists()) {
      n_samples_worst_case =
        index.size() - index.inclusiveSumSortedClusterSize()(
                         std::max<IdxT>(index.numClustersSize0(), index.n_lists() - 1 - n_probes) -
                         index.numClustersSize0());
    }
    if (IdxT{k} > n_samples_worst_case) {
      RAFT_LOG_WARN(
        "n_probes is too small to get top-k results reliably (n_probes: %u, k: %u, "
        "n_samples_worst_case: %zu).",
        n_probes,
        k,
        static_cast<uint64_t>(n_samples_worst_case));
    }
  }

  auto pool_guard = raft::get_pool_memory_resource(mr, n_queries * n_probes * k * 16);
  if (pool_guard) {
    RAFT_LOG_DEBUG("ivf_pq::search: using pool memory resource with initial size %zu bytes",
                   pool_guard->pool_size());
  }

  // Maximum number of query vectors to search at the same time.
  uint32_t batch_size = std::min<uint32_t>(n_queries, 32768);
  auto max_queries    = min(max(batch_size, 1), 4096);
  auto max_batch_size = max_queries;
  {
    // TODO: copied from {legacy}; figure this out.
    // Adjust max_batch_size to improve GPU occupancy of topk kernel.
    uint32_t numCta_total    = getMultiProcessorCount() * 2;
    uint32_t numCta_perBatch = numCta_total / max_batch_size;
    float utilization        = (float)numCta_perBatch * max_batch_size / numCta_total;
    if (numCta_perBatch > 1 || (numCta_perBatch == 1 && utilization < 0.6)) {
      uint32_t numCta_perBatch_1 = numCta_perBatch + 1;
      uint32_t maxBatchSize_1    = numCta_total / numCta_perBatch_1;
      float utilization_1        = (float)numCta_perBatch_1 * maxBatchSize_1 / numCta_total;
      if (utilization < utilization_1) { max_batch_size = maxBatchSize_1; }
    }
  }

  auto stream = handle.get_stream();

  auto cluster_centers   = index.centers().data_handle();
  auto pqCenters         = index.pq_centers().data_handle();
  auto pqDataset         = index.pq_dataset().data_handle();
  auto data_indices      = index.indices().data_handle();
  auto cluster_offsets   = index.list_offsets().data_handle();
  auto rotationMatrix    = index.rotation_matrix().data_handle();
  auto clusterRotCenters = index.centers_rot().data_handle();

  //
  rmm::device_uvector<T> dev_queries(max_queries * index.dim_ext(), stream, mr);
  rmm::device_uvector<float> cur_queries(max_queries * index.dim_ext(), stream, mr);
  rmm::device_uvector<float> rot_queries(max_queries * index.rot_dim(), stream, mr);
  rmm::device_uvector<uint32_t> clusters_to_probe(max_queries * params.n_probes, stream, mr);
  rmm::device_uvector<float> qc_distances(max_queries * index.n_lists(), stream, mr);

  void (*_ivfpq_search)(const handle_t&,
                        const ivf_pq::index<IdxT>&,
                        uint32_t,
                        uint32_t,
                        uint32_t,
                        uint32_t,
                        uint32_t,
                        const float*,
                        const float*,
                        const uint8_t*,
                        const IdxT*,
                        const IdxT*,
                        const uint32_t*,
                        const float*,
                        IdxT*,
                        float*,
                        rmm::mr::device_memory_resource*);
  if (params.internal_distance_dtype == CUDA_R_16F) {
    if (params.smem_lut_dtype == CUDA_R_16F) {
      _ivfpq_search = ivfpq_search<half, half>;
    } else if (params.smem_lut_dtype == CUDA_R_8U) {
      _ivfpq_search = ivfpq_search<half, fp_8bit<5>>;
    } else {
      _ivfpq_search = ivfpq_search<half, float>;
    }
  } else {
    if (params.smem_lut_dtype == CUDA_R_16F) {
      _ivfpq_search = ivfpq_search<float, half>;
    } else if (params.smem_lut_dtype == CUDA_R_8U) {
      _ivfpq_search = ivfpq_search<float, fp_8bit<5>>;
    } else {
      _ivfpq_search = ivfpq_search<float, float>;
    }
  }

  switch (utils::check_pointer_residency(queries, neighbors, distances)) {
    case utils::pointer_residency::device_only:
    case utils::pointer_residency::host_and_device: break;
    default: RAFT_FAIL("all pointers must be accessible from the device.");
  }

  for (uint32_t i = 0; i < n_queries; i += max_queries) {
    uint32_t nQueries = min(max_queries, n_queries - i);

    /* NOTE[qc_distances]

      We compute query-center distances to choose the clusters to probe.
      We accomplish that with just one GEMM operation thanks to some preprocessing:

        L2 distance:
          cluster_centers[i, dim()] contains the squared norm of the center vector i;
          we extend the dimension K of the GEMM to compute it together with all the dot products:

          `cq_distances[i, j] = 0.5 |luster_centers[j]|^2 - (queries[i], cluster_centers[j])`

          This is a monotonous mapping of the proper L2 distance.

        IP distance:
          `cq_distances[i, j] = - (queries[i], cluster_centers[j])`

          This is a negative inner-product distance. We minimize it to find the similar clusters.

          NB: cq_distances is NOT used further in ivfpq_search.
     */
    float norm_factor;
    switch (index.metric()) {
      case raft::distance::DistanceType::L2Expanded: norm_factor = 1.0 / -2.0; break;
      case raft::distance::DistanceType::InnerProduct: norm_factor = 0.0; break;
      default: RAFT_FAIL("Unsupported distance type %d.", int(index.metric()));
    }
    utils::copy_fill(nQueries,
                     index.dim(),
                     queries + static_cast<size_t>(index.dim()) * i,
                     index.dim(),
                     cur_queries.data(),
                     index.dim_ext(),
                     norm_factor,
                     stream);

    float alpha;
    float beta;
    uint32_t gemmK = index.dim();
    switch (index.metric()) {
      case raft::distance::DistanceType::L2Expanded: {
        alpha = -2.0;
        beta  = 0.0;
        gemmK = index.dim() + 1;
        RAFT_EXPECTS(gemmK <= index.dim_ext(), "unexpected gemmK or dim_ext");
      } break;
      case raft::distance::DistanceType::InnerProduct: {
        alpha = -1.0;
        beta  = 0.0;
      } break;
      default: RAFT_FAIL("Unsupported distance type %d.", int(index.metric()));
    }
    linalg::gemm(handle,
                 true,
                 false,
                 index.n_lists(),
                 nQueries,
                 gemmK,
                 &alpha,
                 cluster_centers,
                 index.dim_ext(),
                 cur_queries.data(),
                 index.dim_ext(),
                 &beta,
                 qc_distances.data(),
                 index.n_lists(),
                 stream);

    // Rotate queries
    alpha = 1.0;
    beta  = 0.0;
    linalg::gemm(handle,
                 true,
                 false,
                 index.rot_dim(),
                 nQueries,
                 index.dim(),
                 &alpha,
                 rotationMatrix,
                 index.dim(),
                 cur_queries.data(),
                 index.dim_ext(),
                 &beta,
                 rot_queries.data(),
                 index.rot_dim(),
                 stream);

    {
      // Select neighbor clusters for each query.
      rmm::device_uvector<float> cluster_dists(max_queries * params.n_probes, stream, mr);
      select_topk<float, uint32_t>(qc_distances.data(),
                                   nullptr,
                                   nQueries,
                                   index.n_lists(),
                                   params.n_probes,
                                   cluster_dists.data(),
                                   clusters_to_probe.data(),
                                   true,
                                   stream,
                                   mr);
    }

    for (uint32_t j = 0; j < nQueries; j += max_batch_size) {
      uint32_t batchSize = min(max_batch_size, nQueries - j);
      /* The distance calculation is done in the rotated/transformed space;
         as long as `index.rotation_matrix()` is orthogonal, the distances and thus results are
         preserved.
       */
      _ivfpq_search(handle,
                    index,
                    params.n_probes,
                    max_batch_size,
                    k,
                    params.preferred_thread_block_size,
                    batchSize,
                    clusterRotCenters,
                    pqCenters,
                    pqDataset,
                    data_indices,
                    cluster_offsets,
                    clusters_to_probe.data() + ((uint64_t)(params.n_probes) * j),
                    rot_queries.data() + ((uint64_t)(index.rot_dim()) * j),
                    neighbors + ((uint64_t)(k) * (i + j)),
                    distances + ((uint64_t)(k) * (i + j)),
                    mr);
    }
  }
}

}  // namespace raft::spatial::knn::ivf_pq::detail
