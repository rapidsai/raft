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

template <unsigned ExpBits>
struct fp_8bit {
 public:
  uint8_t bitstring;

  HDI explicit fp_8bit(uint8_t bs) : bitstring(bs) {}
  HDI explicit fp_8bit(float fp) : fp_8bit(float2fp_8bit(fp).bitstring) {}
  HDI auto operator=(float fp) -> fp_8bit<ExpBits>&
  {
    bitstring = float2fp_8bit(fp).bitstring;
    return *this;
  }
  HDI explicit operator float() const { return fp_8bit2float(*this); }

 private:
  static HDI auto float2fp_8bit(float v) -> fp_8bit<ExpBits>
  {
    if (v < 1. / (1u << ((1u << (ExpBits - 1)) - 1))) {
      return fp_8bit<ExpBits>{static_cast<uint8_t>(0)};
    }
    return fp_8bit<ExpBits>{static_cast<uint8_t>(
      (*reinterpret_cast<uint32_t*>(&v) + (((1u << (ExpBits - 1)) - 1) << 23) - 0x3f800000u) >>
      (15 + ExpBits))};
  }

  static HDI auto fp_8bit2float(const fp_8bit<ExpBits>& v) -> float
  {
    float r;
    *reinterpret_cast<uint32_t*>(&r) =
      ((v.bitstring << (15 + ExpBits)) + (0x3f800000u | (0x00400000u >> (8 - ExpBits))) -
       (((1u << (ExpBits - 1)) - 1) << 23));
    return r;
  }
};

__device__ inline auto warp_scan(uint32_t x) -> uint32_t
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

__device__ inline auto thread_block_scan(uint32_t x, uint32_t* smem) -> uint32_t
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
  uint32_t n_probes,
  uint32_t batch_size,
  const IdxT* cluster_offsets,        // [n_clusters + 1,]
  const uint32_t* clusters_to_probe,  // [batch_size, n_probes,]
  uint32_t* chunk_indices,            // [sizeBetch, n_probes,]
  uint32_t* numSamples                // [batch_size,]
)
{
  __shared__ uint32_t smem_temp[32];  // NOLINT
  __shared__ uint32_t smem_base[2];   // NOLINT

  uint32_t batch_ix = blockIdx.x;
  if (batch_ix >= batch_size) return;
  clusters_to_probe += n_probes * batch_ix;
  chunk_indices += n_probes * batch_ix;

  //
  uint32_t j_end = raft::ceildiv(n_probes, 1024u);
  for (uint32_t j = 0; j < j_end; j++) {
    uint32_t i   = threadIdx.x + (1024 * j);
    uint32_t val = 0;
    if (i < n_probes) {
      uint32_t l = clusters_to_probe[i];
      val        = static_cast<uint32_t>(cluster_offsets[l + 1] - cluster_offsets[l]);
    }
    val = thread_block_scan(val, smem_temp);

    if (i < n_probes) {
      if (j > 0) { val += smem_base[(j - 1) & 0x1]; }
      chunk_indices[i] = val;
      if (i == n_probes - 1) { numSamples[batch_ix] = val; }
    }

    if ((j < j_end - 1) && (threadIdx.x == 1023)) { smem_base[j & 0x1] = val; }
  }
}

template <typename IdxT>
__device__ void ivfpq_get_id_dataset(uint32_t sample_ix,
                                     uint32_t n_probes,
                                     const IdxT* cluster_offsets,     // [n_clusters + 1,]
                                     const uint32_t* cluster_labels,  // [n_probes,]
                                     const uint32_t* chunk_indices,   // [n_probes,]
                                     uint32_t& chunk_ix,              // NOLINT
                                     uint32_t& label,                 // NOLINT
                                     IdxT& data_ix)
{
  uint32_t ix_min = 0;
  uint32_t ix_max = n_probes - 1;
  chunk_ix        = (ix_min + ix_max) / 2;
  while (ix_min < ix_max) {
    if (sample_ix >= chunk_indices[chunk_ix]) {
      ix_min = chunk_ix + 1;
    } else {
      ix_max = chunk_ix;
    }
    chunk_ix = (ix_min + ix_max) / 2;
  }

  label = cluster_labels[chunk_ix];
  if (chunk_ix > 0) { sample_ix -= chunk_indices[chunk_ix - 1]; }
  data_ix = sample_ix + cluster_offsets[label];
}

template <typename ScoreT, typename IdxT>
__global__ void ivfpq_make_outputs(distance::DistanceType metric,
                                   uint32_t n_probes,
                                   uint32_t topk,
                                   uint32_t max_samples,
                                   uint32_t batch_size,
                                   const IdxT* cluster_offsets,     // [n_clusters + 1]
                                   const IdxT* data_indices,        // [index_size]
                                   const uint32_t* cluster_labels,  // [batch_size, n_probes]
                                   const uint32_t* chunk_indices,   // [batch_size, n_probes]
                                   const ScoreT* scores,            // [batch_size, max_samples] or
                                                                    // [batch_size, n_probes, topk]
                                   const uint32_t* scoreTopkIndex,  // [batch_size, n_probes, topk]
                                   const uint32_t* topkSampleIds,   // [batch_size, topk]
                                   IdxT* topkNeighbors,             // [batch_size, topk]
                                   float* topkScores                // [batch_size, topk]
)
{
  uint32_t i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i >= topk) return;
  uint32_t batch_ix = blockIdx.y;
  if (batch_ix >= batch_size) return;

  uint32_t sample_ix = topkSampleIds[i + (topk * batch_ix)];
  float score;
  if (scoreTopkIndex == nullptr) {
    // 0 <= sample_ix < max_samples
    score = scores[sample_ix + (max_samples * batch_ix)];
    uint32_t chunk_ix;
    uint32_t label;
    IdxT data_ix;
    ivfpq_get_id_dataset(sample_ix,
                         n_probes,
                         cluster_offsets,
                         cluster_labels + (n_probes * batch_ix),
                         chunk_indices + (n_probes * batch_ix),
                         chunk_ix,
                         label,
                         data_ix);
    topkNeighbors[i + (topk * batch_ix)] = data_indices[data_ix];
  } else {
    // 0 <= sample_ix < (n_probes * topk)
    score        = scores[sample_ix + ((n_probes * topk) * batch_ix)];
    IdxT data_ix = scoreTopkIndex[sample_ix + ((n_probes * topk) * batch_ix)];
    topkNeighbors[i + (topk * batch_ix)] = data_indices[data_ix];
  }
  switch (metric) {
    case distance::DistanceType::InnerProduct: {
      score = -score;
    } break;
    default: break;
  }
  topkScores[i + (topk * batch_ix)] = score;
}

template <int PqBits, int VecLen, typename PqT, typename IdxT, typename LutT = float>
__device__ auto ivfpq_compute_score(uint32_t pq_dim,
                                    IdxT data_ix,
                                    const uint8_t* pq_dataset,  // [n_rows, pq_dim * PqBits / 8]
                                    const LutT* lut_scores      // [pq_dim, pq_width]
                                    ) -> float
{
  float score                   = 0.0;
  constexpr uint32_t kBitsTotal = 8 * sizeof(PqT);
  const PqT* pq_head =
    reinterpret_cast<const PqT*>(pq_dataset + uint64_t(data_ix) * (pq_dim * PqBits / 8));
  for (int j = 0; j < pq_dim / VecLen; j += 1) {
    PqT pq_code = pq_head[0];
    pq_head += 1;
    auto bits_left = kBitsTotal;
#pragma unroll VecLen
    for (int k = 0; k < VecLen; k += 1) {
      uint8_t code = pq_code;
      if (bits_left > PqBits) {
        // This condition is always true here (to make the compiler happy)
        if constexpr (kBitsTotal > PqBits) { pq_code >>= PqBits; }
        bits_left -= PqBits;
      } else {
        if (k < VecLen - 1) {
          pq_code = pq_head[0];
          pq_head += 1;
        }
        code |= (pq_code << bits_left);
        pq_code >>= (PqBits - bits_left);
        bits_left += (kBitsTotal - PqBits);
      }
      code &= (1 << PqBits) - 1;
      score += float(lut_scores[code]);
      lut_scores += (1 << PqBits);
    }
  }
  return score;
}

/**
 * The main kernel that computes the top-k scores across multiple queries and probes.
 * If the index output pointer is provided, it also selects top K candidates for each query and
 * probe.
 *
 * @tparam PqBits
 *   The bit length of an encoded vector element after compression by PQ
 *   (NB: pq_width = 1 << PqBits).
 * @tparam VecLen
 *   The size of the PQ vector used solely in `ivfpq_compute_score`;
 *   It'd defined such that `PqBits * VecLen = 8 * sizeof(PqT)`.
 * @tparam PqT
 *   The carrier integral type used solely in `ivfpq_compute_score` to represent PQ codes.
 * @tparam IdxT
 *   The type of data indices
 * @tparam Capacity
 *   Power-of-two; the maximum possible `k` in top-k.
 * @tparam PrecompBaseDiff
 *   Defines whether we should precompute part of the distance and keep it in shared memory
 *   before the main part (score calculation) to increase memory usage efficiency in the latter.
 *   For L2, this is the distance between the query and the cluster center.
 * @tparam OutT
 *   The output type - distances.
 * @tparam LutT
 *   The lookup table element type (lut_scores).
 * @tparam EnableSMemLut
 *   Defines whether to use the shared memory for the lookup table (`lut_scores`).
 *   Setting this to `false` allows to reduce the shared memory usage (and maximum data dim)
 *   at the cost of reducing global memory reading throughput.
 *
 * @param n_rows the number of records in the dataset
 * @param dim the dimensionality of the data (NB: after rotation transform, i.e. `index.rot_dim()`).
 * @param n_probes the number of clusters to search for each query
 * @param pq_dim
 *   The dimensionality of an encoded vector after compression by PQ.
 * @param batch_size the number of queries.
 * @param max_samples
 *   The maximum number of samples could be retrieved for the given query when all scores are to be
 *   returned rather than only the top k (`manage_local_topk = false`).
 * @param metric the distance type.
 * @param codebook_kind Defines the way PQ codebooks have been trained.
 * @param topk the `k` in the select top-k.
 * @param cluster_centers
 *   The device pointer to the cluster centers in the original space (NB: after rotation)
 *   [n_clusters, dim].
 * @param pq_centers
 *   The device pointer to the cluster centers in the PQ space
 *   [pq_dim, pq_width, pq_len] or [n_clusters, pq_width, pq_len,].
 * @param pq_dataset
 *   The device pointer to the PQ index (data) [n_rows, pq_dim * PqBits / 8].
 * @param cluster_offsets
 *   The device pointer to the cluster offsets [n_clusters + 1].
 * @param _cluster_labels
 *   The device pointer to the labels (clusters) for each query and probe [batch_size, n_probes].
 * @param _chunk_indices
 *   The device pointer to the data offsets for each query and probe [batch_size, n_probes].
 * @param queries
 *   The device pointer to the queries (NB: after rotation) [batch_size, dim].
 * @param index_list
 *   An optional device pointer to the enforced order of search [batch_size, n_probes].
 *   One can pass reordered indices here to try to improve data reading locality.
 * @param lut_scores
 *   The device pointer for storing the lookup table globally [gridDim.x, pq_dim << PqBits].
 *   Ignored when `EnableSMemLut == true`.
 * @param _out_scores
 *   The device pointer to the output scores
 *   [batch_size, max_samples] or [batch_size, n_probes, topk].
 * @param _out_indices
 *   The device pointer to the output indices [batch_size, n_probes, topk].
 *   Ignored  when `manage_local_topk = false`.
 */
template <int PqBits,
          int VecLen,
          typename PqT,
          typename IdxT,
          int Capacity,
          bool PrecompBaseDiff,
          typename OutT,
          typename LutT,
          bool EnableSMemLut>
__launch_bounds__(1024, 1) __global__ void ivfpq_compute_similarity(uint32_t n_rows,
                                                                    uint32_t dim,
                                                                    uint32_t n_probes,
                                                                    uint32_t pq_dim,
                                                                    uint32_t batch_size,
                                                                    uint32_t max_samples,
                                                                    distance::DistanceType metric,
                                                                    codebook_gen codebook_kind,
                                                                    uint32_t topk,
                                                                    const float* cluster_centers,
                                                                    const float* pq_centers,
                                                                    const uint8_t* pq_dataset,
                                                                    const IdxT* cluster_offsets,
                                                                    const uint32_t* _cluster_labels,
                                                                    const uint32_t* _chunk_indices,
                                                                    const float* queries,
                                                                    const uint32_t* index_list,
                                                                    LutT* lut_scores,
                                                                    OutT* _out_scores,
                                                                    uint32_t* _out_indices)
{
  /* Shared memory:

    * lut_scores: lookup table (LUT) of size = `pq_dim << PqBits`  (when EnableSMemLut)
    * base_diff: size = dim  (which is equal to `pq_dim * pq_len`)
    * topk::block_sort: some amount of shared memory, but overlaps with the rest:
        block_sort only needs shared memory for `.done()` operation, which can come very last.
  */
  extern __shared__ __align__(256) uint8_t smem_buf[];  // NOLINT

  const uint32_t pq_len = dim / pq_dim;

  if constexpr (EnableSMemLut) {
    lut_scores = reinterpret_cast<LutT*>(smem_buf);
  } else {
    lut_scores += (pq_dim << PqBits) * blockIdx.x;
  }

  float* base_diff = nullptr;
  if constexpr (PrecompBaseDiff) {
    if constexpr (EnableSMemLut) {
      base_diff = reinterpret_cast<float*>(lut_scores + (pq_dim << PqBits));
    } else {
      base_diff = reinterpret_cast<float*>(smem_buf);
    }
  }
  bool manage_local_topk = false;
  if (_out_indices != nullptr) { manage_local_topk = true; }

  for (int ib = blockIdx.x; ib < batch_size * n_probes; ib += gridDim.x) {
    uint32_t batch_ix;
    uint32_t probe_ix;
    if (index_list == nullptr) {
      batch_ix = ib % batch_size;
      probe_ix = ib / batch_size;
    } else {
      batch_ix = index_list[ib] / n_probes;
      probe_ix = index_list[ib] % n_probes;
    }
    if (batch_ix >= batch_size || probe_ix >= n_probes) continue;

    const uint32_t* cluster_labels = _cluster_labels + (n_probes * batch_ix);
    const uint32_t* chunk_indices  = _chunk_indices + (n_probes * batch_ix);
    const float* query             = queries + (dim * batch_ix);
    OutT* out_scores;
    uint32_t* out_indices = nullptr;
    if (manage_local_topk) {
      // Store topk calculated distances to out_scores (and its indices to out_indices)
      out_scores  = _out_scores + (topk * (probe_ix + (n_probes * batch_ix)));
      out_indices = _out_indices + (topk * (probe_ix + (n_probes * batch_ix)));
    } else {
      // Store all calculated distances to out_scores
      out_scores = _out_scores + (max_samples * batch_ix);
    }
    uint32_t label              = cluster_labels[probe_ix];
    const float* cluster_center = cluster_centers + (dim * label);
    const float* pq_center;
    if (codebook_kind == codebook_gen::PER_SUBSPACE) {
      pq_center = pq_centers;
    } else {
      pq_center = pq_centers + (pq_len << PqBits) * label;
    }

    if constexpr (PrecompBaseDiff) {
      // Reduce computational complexity by pre-computing the difference
      // between the cluster centroid and the query.
      for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        base_diff[i] = query[i] - cluster_center[i];
      }
      __syncthreads();
    }

    // Create a lookup table
    for (uint32_t i = threadIdx.x; i < (pq_dim << PqBits); i += blockDim.x) {
      uint32_t i_pq   = i >> PqBits;
      uint32_t i_code = codebook_kind == codebook_gen::PER_CLUSTER ? i & ((1 << PqBits) - 1) : i;
      float score     = 0.0;
      switch (metric) {
        case distance::DistanceType::L2Expanded: {
          for (uint32_t j = 0; j < pq_len; j++) {
            uint32_t k = j + (pq_len * i_pq);
            float diff;
            if constexpr (PrecompBaseDiff) {
              diff = base_diff[k];
            } else {
              diff = query[k] - cluster_center[k];
            }
            diff -= pq_center[j + pq_len * i_code];
            score += diff * diff;
          }
        } break;
        case distance::DistanceType::InnerProduct: {
          for (uint32_t j = 0; j < pq_len; j++) {
            uint32_t k = j + (pq_len * i_pq);
            score -= query[k] * (cluster_center[k] + pq_center[j + pq_len * i_code]);
          }
        } break;
      }
      lut_scores[i] = LutT(score);
    }

    uint32_t sample_offset = 0;
    if (probe_ix > 0) { sample_offset = chunk_indices[probe_ix - 1]; }
    uint32_t n_samples   = chunk_indices[probe_ix] - sample_offset;
    uint32_t n_samples32 = Pow2<32>::roundUp(n_samples);
    IdxT cluster_offset  = cluster_offsets[label];

    using block_sort_t =
      topk::block_sort<topk::warp_sort_immediate, Capacity, true, OutT, uint32_t>;
    block_sort_t block_topk(topk, reinterpret_cast<uint8_t*>(smem_buf));
    constexpr OutT kLimit = block_sort_t::queue_t::kDummy;

    // Compute a distance for each sample
    for (uint32_t i = threadIdx.x; i < n_samples32; i += blockDim.x) {
      float score = kLimit;
      if (i < n_samples) {
        score = ivfpq_compute_score<PqBits, VecLen, PqT, IdxT, LutT>(
          pq_dim, cluster_offset + i, pq_dataset, lut_scores);
      }
      if (manage_local_topk) {
        block_topk.add(score, cluster_offset + i);
      } else {
        if (i < n_samples) { out_scores[i + sample_offset] = score; }
      }
    }
    __syncthreads();
    if (manage_local_topk) {
      // sync threads before and after the topk merging operation, because we reuse smem_buf
      block_topk.done();
      block_topk.store(out_scores, out_indices);
      __syncthreads();
    } else {
      // fill in the rest of the out_scores with dummy values
      if (probe_ix + 1 == n_probes) {
        for (uint32_t i = threadIdx.x + sample_offset + n_samples; i < max_samples;
             i += blockDim.x) {
          out_scores[i] = kLimit;
        }
      }
    }
  }
}

/**
 * The "main part" of the search, which assumes that outer-level `search` has already:
 *
 *   1. computed the closest clusters to probe (`clusters_to_probe`);
 *   2. transformed input queries into the rotated space (rot_dim);
 *   3. split the query batch into smaller chunks, so that the device workspace
 *      is guaranteed to fit into GPU memory.
 */
template <typename ScoreT, typename LutT, typename IdxT>
void ivfpq_search(const handle_t& handle,
                  const index<IdxT>& index,
                  uint32_t n_probes,
                  uint32_t max_batch_size,
                  uint32_t topK,
                  uint32_t preferred_thread_block_size,
                  uint32_t n_queries,
                  const float* cluster_centers,       // [index_size, rot_dim]
                  const float* pq_centers,            // [pq_dim, pq_width, pq_len]
                  const uint8_t* pq_dataset,          // [index_size, pq_dim * pq_bits / 8]
                  const IdxT* data_indices,           // [index_size]
                  const IdxT* cluster_offsets,        // [n_clusters + 1]
                  const uint32_t* clusters_to_probe,  // [n_queries, n_probes]
                  const float* query,                 // [n_queries, rot_dim]
                  IdxT* topkNeighbors,                // [n_queries, topK]
                  float* topkDistances,               // [n_queries, topK]
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
  // [maxBatchSize, max_samples] or  [maxBatchSize, n_probes, topk]
  rmm::device_uvector<ScoreT> scores_buf(max_batch_size * topk_len, stream, mr);
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
                                                                   clusters_to_probe,
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
                                    clusters_to_probe,
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
                                    clusters_to_probe,
                                    cluster_labels_out.data(),
                                    index_list,
                                    index_list_sorted,
                                    n_queries * n_probes,
                                    begin_bit,
                                    end_bit,
                                    stream);
  }

  // Select a GPU kernel for distance calculation
#define SET_KERNEL1(B, V, T, D)                                                                    \
  do {                                                                                             \
    static_assert((B * V) % (sizeof(T) * 8) == 0);                                                 \
    kernel_no_basediff =                                                                           \
      ivfpq_compute_similarity<B, V, T, IdxT, D * WarpSize, false, ScoreT, LutT, true>;            \
    kernel_fast = ivfpq_compute_similarity<B, V, T, IdxT, D * WarpSize, true, ScoreT, LutT, true>; \
    kernel_no_smem_lut =                                                                           \
      ivfpq_compute_similarity<B, V, T, IdxT, D * WarpSize, true, ScoreT, LutT, false>;            \
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

  using kernel_t = void (*)(uint32_t,
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
                            ScoreT*,
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
  const size_t smem_threshold = 48 * 1024;
  size_t smem_size            = sizeof(LutT) * index.pq_dim() * index.pq_width();
  size_t smem_size_base_diff  = sizeof(float) * index.rot_dim();

  uint32_t n_ctas = n_queries * n_probes;
  int n_threads   = 1024;
  // preferred_thread_block_size == 0 means using auto thread block size calculation
  // mode
  if (preferred_thread_block_size == 0) {
    const int thread_min = 256;
    while (n_threads > thread_min) {
      if (n_ctas < uint32_t(getMultiProcessorCount() * (1024 / (n_threads / 2)))) { break; }
      if (handle.get_device_properties().sharedMemPerMultiprocessor * 2 / 3 <
          smem_size * (1024 / (n_threads / 2))) {
        break;
      }
      n_threads /= 2;
    }
  } else {
    n_threads = preferred_thread_block_size;
  }
  size_t smem_size_local_topk =
    topk::template calc_smem_size_for_block_wide<float, uint32_t>(n_threads / WarpSize, topK);
  smem_size = max(smem_size, smem_size_local_topk);

  kernel_t kernel = kernel_no_basediff;

  bool kernel_no_basediff_available = true;
  bool use_smem_lut                 = true;
  if (smem_size > smem_threshold) {
    cudaError_t cuda_status = cudaFuncSetAttribute(
      kernel_no_basediff, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (cuda_status != cudaSuccess) {
      RAFT_EXPECTS(
        cuda_status == cudaGetLastError(),
        "Tried to reset the expected cuda error code, but it didn't match the expectation");
      kernel_no_basediff_available = false;

      // Use "kernel_no_smem_lut" which just uses small amount of shared memory.
      kernel       = kernel_no_smem_lut;
      use_smem_lut = false;
      n_threads    = 1024;
      size_t smem_size_local_topk =
        topk::calc_smem_size_for_block_wide<float, uint32_t>(n_threads / WarpSize, topK);
      smem_size = max(smem_size_base_diff, smem_size_local_topk);
      n_ctas    = getMultiProcessorCount();
    }
  }
  if (kernel_no_basediff_available) {
    bool kernel_fast_available = true;
    if (smem_size + smem_size_base_diff > smem_threshold) {
      cudaError_t cuda_status = cudaFuncSetAttribute(
        kernel_fast, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size + smem_size_base_diff);
      if (cuda_status != cudaSuccess) {
        RAFT_EXPECTS(
          cuda_status == cudaGetLastError(),
          "Tried to reset the expected cuda error code, but it didn't match the expectation");
        kernel_fast_available = false;
      }
    }
    if (kernel_fast_available) {
      int kernel_no_basediff_n_blocks = 0;
      RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &kernel_no_basediff_n_blocks, kernel_no_basediff, n_threads, smem_size));

      int kernel_fast_n_blocks = 0;
      RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &kernel_fast_n_blocks, kernel_fast, n_threads, smem_size + smem_size_base_diff));

      // Use "kernel_fast" only if GPU occupancy does not drop
      if (kernel_no_basediff_n_blocks == kernel_fast_n_blocks) {
        kernel = kernel_fast;
        smem_size += smem_size_base_diff;
      }
    }
  }

  rmm::device_uvector<LutT> precomp_scores(
    use_smem_lut ? 0 : n_ctas * index.pq_dim() * index.pq_width(), stream, mr);
  dim3 cta_threads(n_threads, 1, 1);
  dim3 cta_blocks(n_ctas, 1, 1);
  kernel<<<cta_blocks, cta_threads, smem_size, stream>>>(index.size(),
                                                         index.rot_dim(),
                                                         n_probes,
                                                         index.pq_dim(),
                                                         n_queries,
                                                         max_samples,
                                                         index.metric(),
                                                         index.codebook_kind(),
                                                         topK,
                                                         cluster_centers,
                                                         pq_centers,
                                                         pq_dataset,
                                                         cluster_offsets,
                                                         clusters_to_probe,
                                                         chunk_index.data(),
                                                         query,
                                                         index_list_sorted,
                                                         precomp_scores.data(),
                                                         scores_buf.data(),
                                                         topk_index);

  {
    // Select topk vectors for each query
    rmm::device_uvector<ScoreT> topk_dists(n_queries * topK, stream, mr);
    select_topk<ScoreT, uint32_t>(scores_buf.data(),
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
  ivfpq_make_outputs<ScoreT><<<mo_blocks, mo_threads, 0, stream>>>(index.metric(),
                                                                   n_probes,
                                                                   topK,
                                                                   max_samples,
                                                                   n_queries,
                                                                   cluster_offsets,
                                                                   data_indices,
                                                                   clusters_to_probe,
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
  RAFT_EXPECTS(params.lut_dtype == CUDA_R_16F || params.lut_dtype == CUDA_R_32F ||
                 params.lut_dtype == CUDA_R_8U,
               "lut_dtype must be CUDA_R_16F, CUDA_R_32F or CUDA_R_8U");
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
    uint32_t n_ctas_total           = getMultiProcessorCount() * 2;
    uint32_t n_ctas_total_per_batch = n_ctas_total / max_batch_size;
    float utilization               = float(n_ctas_total_per_batch * max_batch_size) / n_ctas_total;
    if (n_ctas_total_per_batch > 1 || (n_ctas_total_per_batch == 1 && utilization < 0.6)) {
      uint32_t n_ctas_total_per_batch_1 = n_ctas_total_per_batch + 1;
      uint32_t max_batch_size_1         = n_ctas_total / n_ctas_total_per_batch_1;
      float utilization_1 = float(n_ctas_total_per_batch_1 * max_batch_size_1) / n_ctas_total;
      if (utilization < utilization_1) { max_batch_size = max_batch_size_1; }
    }
  }

  auto stream = handle.get_stream();

  rmm::device_uvector<T> dev_queries(max_queries * index.dim_ext(), stream, mr);
  rmm::device_uvector<float> cur_queries(max_queries * index.dim_ext(), stream, mr);
  rmm::device_uvector<float> rot_queries(max_queries * index.rot_dim(), stream, mr);
  rmm::device_uvector<uint32_t> clusters_to_probe(max_queries * params.n_probes, stream, mr);
  rmm::device_uvector<float> qc_distances(max_queries * index.n_lists(), stream, mr);

  void (*_ivfpq_search)(const handle_t&,  // NOLINT
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
    if (params.lut_dtype == CUDA_R_16F) {
      _ivfpq_search = ivfpq_search<half, half>;
    } else if (params.lut_dtype == CUDA_R_8U) {
      _ivfpq_search = ivfpq_search<half, fp_8bit<5>>;
    } else {
      _ivfpq_search = ivfpq_search<half, float>;
    }
  } else {
    if (params.lut_dtype == CUDA_R_16F) {
      _ivfpq_search = ivfpq_search<float, half>;
    } else if (params.lut_dtype == CUDA_R_8U) {
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
    uint32_t queries_batch = min(max_queries, n_queries - i);

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
    utils::copy_fill(queries_batch,
                     index.dim(),
                     queries + static_cast<size_t>(index.dim()) * i,
                     index.dim(),
                     cur_queries.data(),
                     index.dim_ext(),
                     norm_factor,
                     stream);

    float alpha;
    float beta;
    uint32_t gemm_k = index.dim();
    switch (index.metric()) {
      case raft::distance::DistanceType::L2Expanded: {
        alpha  = -2.0;
        beta   = 0.0;
        gemm_k = index.dim() + 1;
        RAFT_EXPECTS(gemm_k <= index.dim_ext(), "unexpected gemm_k or dim_ext");
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
                 queries_batch,
                 gemm_k,
                 &alpha,
                 index.centers().data_handle(),
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
                 queries_batch,
                 index.dim(),
                 &alpha,
                 index.rotation_matrix().data_handle(),
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
                                   queries_batch,
                                   index.n_lists(),
                                   params.n_probes,
                                   cluster_dists.data(),
                                   clusters_to_probe.data(),
                                   true,
                                   stream,
                                   mr);
    }

    for (uint32_t j = 0; j < queries_batch; j += max_batch_size) {
      uint32_t batch_size = min(max_batch_size, queries_batch - j);
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
                    batch_size,
                    index.centers_rot().data_handle(),
                    index.pq_centers().data_handle(),
                    index.pq_dataset().data_handle(),
                    index.indices().data_handle(),
                    index.list_offsets().data_handle(),
                    clusters_to_probe.data() + ((uint64_t)(params.n_probes) * j),
                    rot_queries.data() + ((uint64_t)(index.rot_dim()) * j),
                    neighbors + ((uint64_t)(k) * (i + j)),
                    distances + ((uint64_t)(k) * (i + j)),
                    mr);
    }
  }
}

}  // namespace raft::spatial::knn::ivf_pq::detail
