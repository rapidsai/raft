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

#include <raft/core/cudart_utils.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/device_atomics.cuh>
#include <raft/util/device_loads_stores.cuh>
#include <raft/util/pow2_utils.cuh>
#include <raft/util/vectorized.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <cub/cub.cuh>
#include <thrust/sequence.h>

#include <cuda_fp16.h>

namespace raft::spatial::knn::ivf_pq::detail {

/**
 * Maximum value of k for the fused calculate & select in ivfpq.
 *
 * If runtime value of k is larger than this, the main search operation
 * is split into two kernels (per batch, first calculate distance, then select top-k).
 */
static constexpr int kMaxCapacity = 128;
static_assert((kMaxCapacity >= 32) && !(kMaxCapacity & (kMaxCapacity - 1)),
              "kMaxCapacity must be a power of two, not smaller than the WarpSize.");

using namespace raft::spatial::knn::detail;  // NOLINT

/** 8-bit floating-point type. */
template <uint32_t ExpBits, bool Signed>
struct fp_8bit {
  static_assert(ExpBits + uint8_t{Signed} <= 8, "The type does not fit in 8 bits.");
  constexpr static uint32_t ExpMask = (1u << (ExpBits - 1u)) - 1u;  // NOLINT
  constexpr static uint32_t ValBits = 8u - ExpBits;                 // NOLINT

 public:
  uint8_t bitstring;

  HDI explicit fp_8bit(uint8_t bs) : bitstring(bs) {}
  HDI explicit fp_8bit(float fp) : fp_8bit(float2fp_8bit(fp).bitstring) {}
  HDI auto operator=(float fp) -> fp_8bit<ExpBits, Signed>&
  {
    bitstring = float2fp_8bit(fp).bitstring;
    return *this;
  }
  HDI explicit operator float() const { return fp_8bit2float(*this); }

 private:
  static constexpr float kMin = 1.0f / float(1u << ExpMask);
  static constexpr float kMax = float(1u << (ExpMask + 1)) * (2.0f - 1.0f / float(1u << ValBits));

  static HDI auto float2fp_8bit(float v) -> fp_8bit<ExpBits, Signed>
  {
    if constexpr (Signed) {
      auto u = fp_8bit<ExpBits, false>(std::abs(v)).bitstring;
      u      = (u & 0xfeu) | uint8_t{v < 0};  // set the sign bit
      return fp_8bit<ExpBits, true>(u);
    } else {
      // sic! all small and negative numbers are truncated to zero.
      if (v < kMin) { return fp_8bit<ExpBits, false>{static_cast<uint8_t>(0)}; }
      // protect from overflow
      if (v >= kMax) { return fp_8bit<ExpBits, false>{static_cast<uint8_t>(0xffu)}; }
      // the rest of possible float values should be within the normalized range
      return fp_8bit<ExpBits, false>{static_cast<uint8_t>(
        (*reinterpret_cast<uint32_t*>(&v) + (ExpMask << 23u) - 0x3f800000u) >> (15u + ExpBits))};
    }
  }

  static HDI auto fp_8bit2float(const fp_8bit<ExpBits, Signed>& v) -> float
  {
    uint32_t u = v.bitstring;
    if constexpr (Signed) {
      u &= ~1;  // zero the sign bit
    }
    float r;
    *reinterpret_cast<uint32_t*>(&r) =
      ((u << (15u + ExpBits)) + (0x3f800000u | (0x00400000u >> ValBits)) - (ExpMask << 23));
    if constexpr (Signed) {  // recover the sign bit
      if (v.bitstring & 1) { r = -r; }
    }
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

/** An unsinged integer type that is used for bit operations on multiple PQ codes at once. */
template <int TotalBits>
struct code_carrier_t {
  static_assert(TotalBits != TotalBits, "There's no carrier type for this bitsize.");
};

template <>
struct code_carrier_t<64> {
  using value = uint64_t;
};

template <>
struct code_carrier_t<32> {
  using value = uint32_t;
};

template <>
struct code_carrier_t<16> {
  using value = uint16_t;
};

template <>
struct code_carrier_t<8> {
  using value = uint8_t;
};

template <int PqBits, int VecLen, typename IdxT, typename LutT = float>
__device__ auto ivfpq_compute_score(uint32_t pq_dim,
                                    IdxT data_ix,
                                    const uint8_t* pq_dataset,  // [n_rows, pq_dim * PqBits / 8]
                                    const LutT* lut_scores      // [pq_dim, pq_width]
                                    ) -> float
{
  float score                   = 0.0;
  using pq_t                    = typename code_carrier_t<gcd(PqBits * VecLen, 64)>::value;
  constexpr uint32_t kBitsTotal = 8 * sizeof(pq_t);
  const pq_t* pq_head =
    reinterpret_cast<const pq_t*>(pq_dataset + uint64_t(data_ix) * (pq_dim * PqBits / 8));
  for (int j = 0; j < pq_dim / VecLen; j += 1) {
    pq_t pq_code = pq_head[0];
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

template <typename T>
struct dummy_block_sort_t {
  using queue_t = topk::warp_sort_immediate<WarpSize, true, T, uint32_t>;
  __device__ dummy_block_sort_t(int k, uint8_t* smem_buf){};
};

template <int Capacity, typename T>
struct pq_block_sort : topk::block_sort<topk::warp_sort_immediate, Capacity, true, T, uint32_t> {
  using type = topk::block_sort<topk::warp_sort_immediate, Capacity, true, T, uint32_t>;
};

template <typename T>
struct pq_block_sort<0, T> : dummy_block_sort_t<T> {
  using type = dummy_block_sort_t<T>;
};

template <int Capacity, typename T>
using block_sort_t = typename pq_block_sort<Capacity, T>::type;

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
 *   It'd defined such that
 *     1. `PqBits * VecLen % 8 * sizeof(PqT) == 0`.
 *     2. `pq_dim % VecLen == 0`
 *   `PqT` is a carrier integer type selected to maximize throughput.
 * @tparam IdxT
 *   The type of data indices
 * @tparam Capacity
 *   Power-of-two; the maximum possible `k` in top-k.
 * @tparam OutT
 *   The output type - distances.
 * @tparam LutT
 *   The lookup table element type (lut_scores).
 * @tparam PrecompBaseDiff
 *   Defines whether we should precompute part of the distance and keep it in shared memory
 *   before the main part (score calculation) to increase memory usage efficiency in the latter.
 *   For L2, this is the distance between the query and the cluster center.
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
 *   returned rather than only the top k (`kManageLocalTopK = false`).
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
 *   Ignored  when `kManageLocalTopK = false`.
 */
template <int PqBits,
          int VecLen,
          typename IdxT,
          int Capacity,
          typename OutT,
          typename LutT,
          bool PrecompBaseDiff,
          bool EnableSMemLut>
__launch_bounds__(1024, 1) __global__
  void ivfpq_compute_similarity_kernel(uint32_t n_rows,
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
  constexpr bool kManageLocalTopK = Capacity > 0;

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
    if constexpr (kManageLocalTopK) {
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
            score      = query[k] * (cluster_center[k] + pq_center[j + pq_len * i_code]);
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

    block_sort_t<Capacity, OutT> block_topk(topk, smem_buf);

    // Ensure lut_scores is written by all threads before using it in ivfpq_compute_score
    __threadfence_block();
    __syncthreads();

    // Compute a distance for each sample
    for (uint32_t i = threadIdx.x; i < n_samples32; i += blockDim.x) {
      OutT score = block_sort_t<Capacity, OutT>::queue_t::kDummy;
      if (i < n_samples) {
        float fscore = ivfpq_compute_score<PqBits, VecLen, IdxT, LutT>(
          pq_dim, cluster_offset + i, pq_dataset, lut_scores);
        switch (metric) {
          // For similarity metrics,
          // we negate the scores as we hardcoded select-topk to always take the minimum
          case distance::DistanceType::InnerProduct: fscore = -fscore; break;
          default: break;
        }
        if (fscore < float(score)) { score = OutT{fscore}; }
      }
      if constexpr (kManageLocalTopK) {
        block_topk.add(score, cluster_offset + i);
      } else {
        if (i < n_samples) { out_scores[i + sample_offset] = score; }
      }
    }
    __syncthreads();
    if constexpr (kManageLocalTopK) {
      // sync threads before and after the topk merging operation, because we reuse smem_buf
      block_topk.done();
      block_topk.store(out_scores, out_indices);
      __syncthreads();
    } else {
      // fill in the rest of the out_scores with dummy values
      if (probe_ix + 1 == n_probes) {
        for (uint32_t i = threadIdx.x + sample_offset + n_samples; i < max_samples;
             i += blockDim.x) {
          out_scores[i] = block_sort_t<Capacity, OutT>::queue_t::kDummy;
        }
      }
    }
  }
}

/**
 * This structure selects configurable template parameters (instance) based on
 * the search/index parameters at runtime.
 *
 * This is done by means of recusively iterating through a small set of possible
 * values for every parameter.
 */
template <typename IdxT, typename OutT, typename LutT>
struct ivfpq_compute_similarity {
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
                            OutT*,
                            uint32_t*);

  template <bool PrecompBaseDiff, bool EnableSMemLut>
  struct configured {
   public:
    /**
     * Select a proper kernel instance based on the runtime parameters.
     *
     * @param pq_bits
     * @param pq_dim
     * @param k_max
     */
    static auto kernel(uint32_t pq_bits, uint32_t pq_dim, uint32_t k_max) -> kernel_t
    {
      return kernel_base(pq_bits, pq_dim, k_max);
    }

   private:
    template <int PqBits, int VecLen, int Capacity>
    static auto kernel_try_capacity(uint32_t k_max) -> kernel_t
    {
      if constexpr (Capacity > 0) {
        if (k_max == 0 || k_max > Capacity) {
          return kernel_try_capacity<PqBits, VecLen, 0>(k_max);
        }
      }
      if constexpr (Capacity > 32) {
        if (k_max * 2 <= Capacity) {
          return kernel_try_capacity<PqBits, VecLen, (Capacity / 2)>(k_max);
        }
      }
      return ivfpq_compute_similarity_kernel<PqBits,
                                             VecLen,
                                             IdxT,
                                             Capacity,
                                             OutT,
                                             LutT,
                                             PrecompBaseDiff,
                                             EnableSMemLut>;
    }

    template <int PqBits, int VecLen>
    static auto kernel_fixed_bits_try_veclen(uint32_t pq_dim, uint32_t k_max) -> kernel_t
    {
      if (pq_dim % VecLen == 0) { return kernel_try_capacity<PqBits, VecLen, kMaxCapacity>(k_max); }
      if constexpr (VecLen > 1 && (PqBits * VecLen) % 16 == 0) {
        return kernel_fixed_bits_try_veclen<PqBits, (VecLen / 2)>(pq_dim, k_max);
      } else {
        RAFT_FAIL("pq_dim must be a multiple of %d", VecLen);
      }
    }

    template <int PqBits>
    static auto kernel_fixed_bits(uint32_t pq_dim, uint32_t k_max) -> kernel_t
    {
      return kernel_fixed_bits_try_veclen<PqBits, 64 / gcd(PqBits, 64)>(pq_dim, k_max);
    }

    static auto kernel_base(uint32_t pq_bits, uint32_t pq_dim, uint32_t k_max) -> kernel_t
    {
      switch (pq_bits) {
        case 4: return kernel_fixed_bits<4>(pq_dim, k_max);
        case 5: return kernel_fixed_bits<5>(pq_dim, k_max);
        case 6: return kernel_fixed_bits<6>(pq_dim, k_max);
        case 7: return kernel_fixed_bits<7>(pq_dim, k_max);
        case 8: return kernel_fixed_bits<8>(pq_dim, k_max);
        default: RAFT_FAIL("Unsupported pq_bits = %u", pq_bits);
      }
    }
  };
};

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
    topK <= kMaxCapacity                 // depth is not too large
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

  using run_t            = ivfpq_compute_similarity<IdxT, ScoreT, LutT>;
  using kernel_t         = typename run_t::kernel_t;
  using conf_fast        = typename run_t::configured<true, true>;
  using conf_no_basediff = typename run_t::configured<false, true>;
  using conf_no_smem_lut = typename run_t::configured<true, false>;

  kernel_t kernel_fast =
    conf_fast::kernel(index.pq_bits(), index.pq_dim(), manage_local_topk ? topK : 0u);
  kernel_t kernel_no_basediff =
    conf_no_basediff::kernel(index.pq_bits(), index.pq_dim(), manage_local_topk ? topK : 0u);
  kernel_t kernel_no_smem_lut =
    conf_no_smem_lut::kernel(index.pq_bits(), index.pq_dim(), manage_local_topk ? topK : 0u);

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
    manage_local_topk
      ? topk::template calc_smem_size_for_block_wide<ScoreT, uint32_t>(n_threads / WarpSize, topK)
      : 0;
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
      kernel               = kernel_no_smem_lut;
      use_smem_lut         = false;
      n_threads            = 1024;
      smem_size_local_topk = manage_local_topk
                               ? topk::template calc_smem_size_for_block_wide<ScoreT, uint32_t>(
                                   n_threads / WarpSize, topK)
                               : 0;
      smem_size            = max(smem_size_base_diff, smem_size_local_topk);
      n_ctas               = getMultiProcessorCount();
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

  bool signed_metric = false;
  switch (index.metric()) {
    case raft::distance::DistanceType::InnerProduct: signed_metric = true; break;
    default: break;
  }

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
      if (signed_metric) {
        _ivfpq_search = ivfpq_search<half, fp_8bit<5, true>>;
      } else {
        _ivfpq_search = ivfpq_search<half, fp_8bit<5, false>>;
      }
    } else {
      _ivfpq_search = ivfpq_search<half, float>;
    }
  } else {
    if (params.lut_dtype == CUDA_R_16F) {
      _ivfpq_search = ivfpq_search<float, half>;
    } else if (params.lut_dtype == CUDA_R_8U) {
      if (signed_metric) {
        _ivfpq_search = ivfpq_search<float, fp_8bit<5, true>>;
      } else {
        _ivfpq_search = ivfpq_search<float, fp_8bit<5, false>>;
      }
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
