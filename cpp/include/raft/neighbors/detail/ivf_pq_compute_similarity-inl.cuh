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

#include <raft/distance/distance_types.hpp>  // raft::distance::DistanceType
#include <raft/matrix/detail/select_warpsort.cuh>  // matrix::detail::select::warpsort::warp_sort_distributed
#include <raft/neighbors/detail/ivf_common.cuh>    // dummy_block_sort_t
#include <raft/neighbors/ivf_pq_types.hpp>         // codebook_gen
#include <raft/neighbors/sample_filter_types.hpp>  // none_ivf_sample_filter
#include <raft/util/cuda_rt_essentials.hpp>        // RAFT_CUDA_TRY
#include <raft/util/device_atomics.cuh>            // raft::atomicMin
#include <raft/util/pow2_utils.cuh>                // raft::Pow2
#include <raft/util/vectorized.cuh>                // raft::TxN_t

#include <rmm/cuda_stream_view.hpp>  // rmm::cuda_stream_view

namespace raft::neighbors::ivf_pq::detail {

/**
 * Maximum value of k for the fused calculate & select in ivfpq.
 *
 * If runtime value of k is larger than this, the main search operation
 * is split into two kernels (per batch, first calculate distance, then select top-k).
 */
static constexpr int kMaxCapacity = 128;
static_assert((kMaxCapacity >= 32) && !(kMaxCapacity & (kMaxCapacity - 1)),
              "kMaxCapacity must be a power of two, not smaller than the WarpSize.");

// using weak attribute here, because it may be compiled multiple times.
auto RAFT_WEAK_FUNCTION is_local_topk_feasible(uint32_t k, uint32_t n_probes, uint32_t n_queries)
  -> bool
{
  if (k > kMaxCapacity) { return false; }            // warp_sort not possible
  if (n_queries * n_probes <= 16) { return false; }  // overall amount of work is too small
  return true;
}

template <int Capacity, typename T, typename IdxT>
struct pq_block_sort {
  using type = matrix::detail::select::warpsort::block_sort<
    matrix::detail::select::warpsort::warp_sort_distributed_ext,
    Capacity,
    true,
    T,
    IdxT>;

  static auto get_mem_required(uint32_t k_max)
  {
    if (k_max == 0 || k_max > Capacity) {
      return pq_block_sort<0, T, IdxT>::get_mem_required(k_max);
    }
    if constexpr (Capacity > 1) {
      if (k_max * 2 <= Capacity) {
        return pq_block_sort<(Capacity / 2), T, IdxT>::get_mem_required(k_max);
      }
    }
    return type::queue_t::mem_required;
  }
};

template <typename T, typename IdxT>
struct pq_block_sort<0, T, IdxT> : ivf::detail::dummy_block_sort_t<T, IdxT> {
  using type = ivf::detail::dummy_block_sort_t<T, IdxT>;
  static auto mem_required(uint32_t) -> size_t { return 0; }
  static auto get_mem_required(uint32_t) { return mem_required; }
};

template <int Capacity, typename T, typename IdxT>
using block_sort_t = typename pq_block_sort<Capacity, T, IdxT>::type;

/**
 * Estimate a carveout value as expected by `cudaFuncAttributePreferredSharedMemoryCarveout`
 * (which does not take into account `reservedSharedMemPerBlock`),
 * given by a desired schmem-L1 split and a per-block memory requirement in bytes.
 *
 * NB: As per the programming guide, the memory carveout setting is just a hint for the driver; it's
 * free to choose any shmem-L1 configuration it deems appropriate. For example, if you set the
 * carveout to zero, it will choose a non-zero config that will allow to run at least one active
 * block per SM.
 *
 * @param shmem_fraction
 *   a fraction representing a desired split (shmem / (shmem + L1)) [0, 1].
 * @param shmem_per_block
 *   a shared memory usage per block (dynamic + static shared memory sizes), in bytes.
 * @param dev_props
 *   device properties.
 * @return
 *   a carveout value in percents [0, 100].
 */
constexpr inline auto estimate_carveout(double shmem_fraction,
                                        size_t shmem_per_block,
                                        const cudaDeviceProp& dev_props) -> int
{
  using shmem_unit = Pow2<128>;
  size_t m         = shmem_unit::roundUp(shmem_per_block);
  size_t r         = dev_props.reservedSharedMemPerBlock;
  size_t s         = dev_props.sharedMemPerMultiprocessor;
  return (size_t(100 * s * m * shmem_fraction) - (m - 1) * r) / (s * (m + r));
}

/* Manually unrolled loop over a chunk of pq_dataset that fits into one VecT. */
template <typename OutT,
          typename LutT,
          typename VecT,
          bool CheckBounds,
          uint32_t PqBits,
          uint32_t BitsLeft = 0,
          uint32_t Ix       = 0>
__device__ __forceinline__ void ivfpq_compute_chunk(OutT& score /* NOLINT */,
                                                    typename VecT::math_t& pq_code,
                                                    const VecT& pq_codes,
                                                    const LutT*& lut_head,
                                                    const LutT*& lut_end)
{
  if constexpr (CheckBounds) {
    if (lut_head >= lut_end) { return; }
  }
  constexpr uint32_t kTotalBits = 8 * sizeof(typename VecT::math_t);
  constexpr uint32_t kPqShift   = 1u << PqBits;
  constexpr uint32_t kPqMask    = kPqShift - 1u;
  if constexpr (BitsLeft >= PqBits) {
    uint8_t code = pq_code & kPqMask;
    pq_code >>= PqBits;
    score += OutT(lut_head[code]);
    lut_head += kPqShift;
    return ivfpq_compute_chunk<OutT, LutT, VecT, CheckBounds, PqBits, BitsLeft - PqBits, Ix>(
      score, pq_code, pq_codes, lut_head, lut_end);
  } else if constexpr (Ix < VecT::Ratio) {
    uint8_t code                = pq_code;
    pq_code                     = pq_codes.val.data[Ix];
    constexpr uint32_t kRemBits = PqBits - BitsLeft;
    constexpr uint32_t kRemMask = (1u << kRemBits) - 1u;
    code |= (pq_code & kRemMask) << BitsLeft;
    pq_code >>= kRemBits;
    score += OutT(lut_head[code]);
    lut_head += kPqShift;
    return ivfpq_compute_chunk<OutT,
                               LutT,
                               VecT,
                               CheckBounds,
                               PqBits,
                               kTotalBits - kRemBits,
                               Ix + 1>(score, pq_code, pq_codes, lut_head, lut_end);
  }
}

/* Compute the similarity for one vector in the pq_dataset */
template <typename OutT, typename LutT, typename VecT, uint32_t PqBits>
__device__ auto ivfpq_compute_score(uint32_t pq_dim,
                                    const typename VecT::io_t* pq_head,
                                    const LutT* lut_scores,
                                    OutT early_stop_limit) -> OutT
{
  constexpr uint32_t kChunkSize = sizeof(VecT) * 8u / PqBits;
  auto lut_head                 = lut_scores;
  auto lut_end                  = lut_scores + (pq_dim << PqBits);
  VecT pq_codes;
  OutT score{0};
  for (; pq_dim >= kChunkSize; pq_dim -= kChunkSize) {
    *pq_codes.vectorized_data() = *pq_head;
    pq_head += kIndexGroupSize;
    typename VecT::math_t pq_code = 0;
    ivfpq_compute_chunk<OutT, LutT, VecT, false, PqBits>(
      score, pq_code, pq_codes, lut_head, lut_end);
    // Early stop when it makes sense (otherwise early_stop_limit is kDummy/infinity).
    if (score >= early_stop_limit) { return score; }
  }
  if (pq_dim > 0) {
    *pq_codes.vectorized_data()   = *pq_head;
    typename VecT::math_t pq_code = 0;
    ivfpq_compute_chunk<OutT, LutT, VecT, true, PqBits>(
      score, pq_code, pq_codes, lut_head, lut_end);
  }
  return score;
}

/**
 * The main kernel that computes similarity scores across multiple queries and probes.
 * When `Capacity > 0`, it also selects top K candidates for each query and probe
 * (which need to be merged across probes afterwards).
 *
 * Each block processes a (query, probe) pair: it calculates the distance between the single query
 * vector and all the dataset vector in the cluster that we are probing.
 *
 * @tparam OutT
 *   The output type - distances.
 * @tparam LutT
 *   The lookup table element type (lut_scores).
 * @tparam PqBits
 *   The bit length of an encoded vector element after compression by PQ
 *   (NB: pq_book_size = 1 << PqBits).
 * @tparam Capacity
 *   Power-of-two; the maximum possible `k` in top-k. Value zero disables fused top-k search.
 * @tparam PrecompBaseDiff
 *   Defines whether we should precompute part of the distance and keep it in shared memory
 *   before the main part (score calculation) to increase memory usage efficiency in the latter.
 *   For L2, this is the distance between the query and the cluster center.
 * @tparam EnableSMemLut
 *   Defines whether to use the shared memory for the lookup table (`lut_scores`).
 *   Setting this to `false` allows to reduce the shared memory usage (and maximum data dim)
 *   at the cost of reducing global memory reading throughput.
 *
 * @param dim the dimensionality of the data (NB: after rotation transform, i.e. `index.rot_dim()`).
 * @param n_probes the number of clusters to search for each query
 * @param pq_dim
 *   The dimensionality of an encoded vector after compression by PQ.
 * @param n_queries the number of queries.
 * @param queries_offset
 *   An offset of the current query batch. It is used for feeding sample_filter with the
 *   correct query index.
 * @param metric the distance type.
 * @param codebook_kind Defines the way PQ codebooks have been trained.
 * @param topk the `k` in the select top-k.
 * @param max_samples the size of the output for a single query.
 * @param cluster_centers
 *   The device pointer to the cluster centers in the original space (NB: after rotation)
 *   [n_clusters, dim].
 * @param pq_centers
 *   The device pointer to the cluster centers in the PQ space
 *   [pq_dim, pq_book_size, pq_len] or [n_clusters, pq_book_size, pq_len].
 * @param pq_dataset
 *   The device pointer to the PQ index (data) [n_rows, ...].
 * @param cluster_labels
 *   The device pointer to the labels (clusters) for each query and probe [n_queries, n_probes].
 * @param _chunk_indices
 *   The device pointer to the data offsets for each query and probe [n_queries, n_probes].
 * @param queries
 *   The device pointer to the queries (NB: after rotation) [n_queries, dim].
 * @param index_list
 *   An optional device pointer to the enforced order of search [n_queries, n_probes].
 *   One can pass reordered indices here to try to improve data reading locality.
 * @param query_kth
 *   query_kths keep the current state of the filtering - atomically updated distances to the
 *   k-th closest neighbors for each query [n_queries].
 * @param sample_filter
 *   A filter that selects samples for a given query. Use an instance of none_ivf_sample_filter to
 *   provide a green light for every sample.
 * @param lut_scores
 *   The device pointer for storing the lookup table globally [gridDim.x, pq_dim << PqBits].
 *   Ignored when `EnableSMemLut == true`.
 * @param _out_scores
 *   The device pointer to the output scores
 *   [n_queries, max_samples] or [n_queries, n_probes, topk].
 * @param _out_indices
 *   The device pointer to the output indices [n_queries, n_probes, topk].
 *   These are the indices of the records as they appear in the database view formed by the probed
 *   clusters / defined by the `_chunk_indices`.
 *   The indices can have values within the range [0, max_samples).
 *   Ignored  when `Capacity == 0`.
 */
template <typename OutT,
          typename LutT,
          typename IvfSampleFilterT,
          uint32_t PqBits,
          int Capacity,
          bool PrecompBaseDiff,
          bool EnableSMemLut>
RAFT_KERNEL compute_similarity_kernel(uint32_t dim,
                                      uint32_t n_probes,
                                      uint32_t pq_dim,
                                      uint32_t n_queries,
                                      uint32_t queries_offset,
                                      distance::DistanceType metric,
                                      codebook_gen codebook_kind,
                                      uint32_t topk,
                                      uint32_t max_samples,
                                      const float* cluster_centers,
                                      const float* pq_centers,
                                      const uint8_t* const* pq_dataset,
                                      const uint32_t* cluster_labels,
                                      const uint32_t* _chunk_indices,
                                      const float* queries,
                                      const uint32_t* index_list,
                                      float* query_kths,
                                      IvfSampleFilterT sample_filter,
                                      LutT* lut_scores,
                                      OutT* _out_scores,
                                      uint32_t* _out_indices)
{
  /* Shared memory:

    * lut_scores: lookup table (LUT) of size = `pq_dim << PqBits`  (when EnableSMemLut)
    * lut_end+:
       * base_diff: size = dim (which is equal to `pq_dim * pq_len`)  or dim*2
       * topk::warp_sort::mem_required - local topk temporary buffer (if necessary)
    * topk::block_sort: some amount of shared memory, but overlaps with the rest:
        block_sort only needs shared memory for `.done()` operation, which can come very last.
  */
  extern __shared__ __align__(256) uint8_t smem_buf[];  // NOLINT
  constexpr bool kManageLocalTopK = Capacity > 0;

  constexpr uint32_t PqShift = 1u << PqBits;  // NOLINT
  constexpr uint32_t PqMask  = PqShift - 1u;  // NOLINT

  const uint32_t pq_len   = dim / pq_dim;
  const uint32_t lut_size = pq_dim * PqShift;

  if constexpr (EnableSMemLut) {
    lut_scores = reinterpret_cast<LutT*>(smem_buf);
  } else {
    lut_scores += lut_size * blockIdx.x;
  }

  uint8_t* lut_end = nullptr;
  if constexpr (EnableSMemLut) {
    lut_end = reinterpret_cast<uint8_t*>(lut_scores + lut_size);
  } else {
    lut_end = smem_buf;
  }

  for (int ib = blockIdx.x; ib < n_queries * n_probes; ib += gridDim.x) {
    if (ib >= gridDim.x) {
      // sync shared memory accesses on the second and further iterations
      __syncthreads();
    }
    uint32_t query_ix;
    uint32_t probe_ix;
    if (index_list == nullptr) {
      query_ix = ib % n_queries;
      probe_ix = ib / n_queries;
    } else {
      auto ordered_ix = index_list[ib];
      query_ix        = ordered_ix / n_probes;
      probe_ix        = ordered_ix % n_probes;
    }

    const uint32_t* chunk_indices = _chunk_indices + (n_probes * query_ix);
    const float* query            = queries + (dim * query_ix);
    OutT* out_scores;
    uint32_t* out_indices = nullptr;
    if constexpr (kManageLocalTopK) {
      // Store topk calculated distances to out_scores (and its indices to out_indices)
      const uint64_t out_offset = probe_ix + n_probes * query_ix;
      out_scores                = _out_scores + out_offset * topk;
      out_indices               = _out_indices + out_offset * topk;
    } else {
      // Store all calculated distances to out_scores
      out_scores = _out_scores + uint64_t(max_samples) * query_ix;
    }
    uint32_t label              = cluster_labels[n_probes * query_ix + probe_ix];
    const float* cluster_center = cluster_centers + dim * label;
    const float* pq_center;
    if (codebook_kind == codebook_gen::PER_SUBSPACE) {
      pq_center = pq_centers;
    } else {
      pq_center = pq_centers + (pq_len << PqBits) * label;
    }

    if constexpr (PrecompBaseDiff) {
      // Reduce number of memory reads later by pre-computing parts of the score
      switch (metric) {
        case distance::DistanceType::L2SqrtExpanded:
        case distance::DistanceType::L2Expanded: {
          for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
            reinterpret_cast<float*>(lut_end)[i] = query[i] - cluster_center[i];
          }
        } break;
        case distance::DistanceType::InnerProduct: {
          float2 pvals;
          for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
            pvals.x                               = query[i];
            pvals.y                               = cluster_center[i] * pvals.x;
            reinterpret_cast<float2*>(lut_end)[i] = pvals;
          }
        } break;
        default: __builtin_unreachable();
      }
      __syncthreads();
    }

    {
      // Create a lookup table
      // For each subspace, the lookup table stores the distance between the actual query vector
      // (projected into the subspace) and all possible pq vectors in that subspace.
      for (uint32_t i = threadIdx.x; i < lut_size; i += blockDim.x) {
        const uint32_t i_pq  = i >> PqBits;
        uint32_t j           = i_pq * pq_len;
        const uint32_t j_end = pq_len + j;
        auto cur_pq_center   = pq_center + (i & PqMask) +
                             (codebook_kind == codebook_gen::PER_SUBSPACE ? j * PqShift : 0u);
        float score = 0.0;
        do {
          float pq_c = *cur_pq_center;
          cur_pq_center += PqShift;
          switch (metric) {
            case distance::DistanceType::L2SqrtExpanded:
            case distance::DistanceType::L2Expanded: {
              float diff;
              if constexpr (PrecompBaseDiff) {
                diff = reinterpret_cast<float*>(lut_end)[j];
              } else {
                diff = query[j] - cluster_center[j];
              }
              diff -= pq_c;
              score += diff * diff;
            } break;
            case distance::DistanceType::InnerProduct: {
              // NB: we negate the scores as we hardcoded select-topk to always compute the minimum
              float q;
              if constexpr (PrecompBaseDiff) {
                float2 pvals = reinterpret_cast<float2*>(lut_end)[j];
                q            = pvals.x;
                score -= pvals.y;
              } else {
                q = query[j];
                score -= q * cluster_center[j];
              }
              score -= q * pq_c;
            } break;
            default: __builtin_unreachable();
          }
        } while (++j < j_end);
        lut_scores[i] = LutT(score);
      }
    }

    // Define helper types for efficient access to the pq_dataset, which is stored in an interleaved
    // format. The chunks of PQ data are stored in kIndexGroupVecLen-bytes-long chunks, interleaved
    // in groups of kIndexGroupSize elems (which is normally equal to the warp size) for the fastest
    // possible access by thread warps.
    //
    // Consider one record in the pq_dataset is `pq_dim * pq_bits`-bit-long.
    // Assuming `kIndexGroupVecLen = 16`, one chunk of data read by a thread at once is 128-bits.
    // Then, such a chunk contains `chunk_size = 128 / pq_bits` record elements, and the record
    // consists of `ceildiv(pq_dim, chunk_size)` chunks. The chunks are interleaved in groups of 32,
    // so that the warp can achieve the best coalesced read throughput.
    using group_align  = Pow2<kIndexGroupSize>;
    using vec_align    = Pow2<kIndexGroupVecLen>;
    using local_topk_t = block_sort_t<Capacity, OutT, uint32_t>;
    using op_t         = uint32_t;
    using vec_t        = TxN_t<op_t, kIndexGroupVecLen / sizeof(op_t)>;

    uint32_t sample_offset = 0;
    if (probe_ix > 0) { sample_offset = chunk_indices[probe_ix - 1]; }
    uint32_t n_samples            = chunk_indices[probe_ix] - sample_offset;
    uint32_t n_samples_aligned    = group_align::roundUp(n_samples);
    constexpr uint32_t kChunkSize = (kIndexGroupVecLen * 8u) / PqBits;
    uint32_t pq_line_width        = div_rounding_up_unsafe(pq_dim, kChunkSize) * kIndexGroupVecLen;
    auto pq_thread_data = pq_dataset[label] + group_align::roundDown(threadIdx.x) * pq_line_width +
                          group_align::mod(threadIdx.x) * vec_align::Value;
    pq_line_width *= blockDim.x;

    constexpr OutT kDummy = upper_bound<OutT>();
    OutT query_kth        = kDummy;
    if constexpr (kManageLocalTopK) { query_kth = OutT(query_kths[query_ix]); }
    OutT early_stop_limit = kDummy;
    switch (metric) {
      // If the metric is non-negative, we can use the query_kth approximation as an early stop
      // threshold to skip some iterations when computing the score. Add such metrics here.
      case distance::DistanceType::L2SqrtExpanded:
      case distance::DistanceType::L2Expanded: {
        early_stop_limit = query_kth;
      } break;
      default: break;
    }

    // Ensure lut_scores is written by all threads before using it in ivfpq-compute-score
    __threadfence_block();
    __syncthreads();
    local_topk_t block_topk(topk, lut_end, query_kth);

    // Compute a distance for each sample
    for (uint32_t i = threadIdx.x; i < n_samples_aligned;
         i += blockDim.x, pq_thread_data += pq_line_width) {
      OutT score = kDummy;
      bool valid = i < n_samples;
      // Check bounds and that the sample is acceptable for the query
      if (valid && sample_filter(queries_offset + query_ix, label, i)) {
        score = ivfpq_compute_score<OutT, LutT, vec_t, PqBits>(
          pq_dim,
          reinterpret_cast<const vec_t::io_t*>(pq_thread_data),
          lut_scores,
          early_stop_limit);
      }
      if constexpr (kManageLocalTopK) {
        block_topk.add(score, sample_offset + i);
      } else {
        if (valid) { out_scores[sample_offset + i] = score; }
      }
    }
    if constexpr (kManageLocalTopK) {
      // sync threads before the topk merging operation, because we reuse smem_buf
      __syncthreads();
      block_topk.done(smem_buf);
      block_topk.store(out_scores, out_indices);
      if (threadIdx.x == 0) { atomicMin(query_kths + query_ix, float(out_scores[topk - 1])); }
    } else {
      // fill in the rest of the out_scores with dummy values
      if (probe_ix + 1 == n_probes) {
        for (uint32_t i = threadIdx.x + sample_offset + n_samples; i < max_samples;
             i += blockDim.x) {
          out_scores[i] = kDummy;
        }
      }
    }
  }
}

// The signature of the kernel defined by a minimal set of template parameters
template <typename OutT,
          typename LutT,
          typename IvfSampleFilterT = raft::neighbors::filtering::none_ivf_sample_filter>
using compute_similarity_kernel_t =
  decltype(&compute_similarity_kernel<OutT, LutT, IvfSampleFilterT, 8, 0, true, true>);

// The config struct lifts the runtime parameters to the template parameters
template <typename OutT,
          typename LutT,
          bool PrecompBaseDiff,
          bool EnableSMemLut,
          typename IvfSampleFilterT = raft::neighbors::filtering::none_ivf_sample_filter>
struct compute_similarity_kernel_config {
 public:
  static auto get(uint32_t pq_bits, uint32_t k_max)
    -> compute_similarity_kernel_t<OutT, LutT, IvfSampleFilterT>
  {
    return kernel_choose_bits(pq_bits, k_max);
  }

 private:
  static auto kernel_choose_bits(uint32_t pq_bits, uint32_t k_max)
    -> compute_similarity_kernel_t<OutT, LutT, IvfSampleFilterT>
  {
    switch (pq_bits) {
      case 4: return kernel_try_capacity<4, kMaxCapacity>(k_max);
      case 5: return kernel_try_capacity<5, kMaxCapacity>(k_max);
      case 6: return kernel_try_capacity<6, kMaxCapacity>(k_max);
      case 7: return kernel_try_capacity<7, kMaxCapacity>(k_max);
      case 8: return kernel_try_capacity<8, kMaxCapacity>(k_max);
      default: RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 8]", pq_bits);
    }
  }

  template <uint32_t PqBits, int Capacity>
  static auto kernel_try_capacity(uint32_t k_max)
    -> compute_similarity_kernel_t<OutT, LutT, IvfSampleFilterT>
  {
    if constexpr (Capacity > 0) {
      if (k_max == 0 || k_max > Capacity) { return kernel_try_capacity<PqBits, 0>(k_max); }
    }
    if constexpr (Capacity > 1) {
      if (k_max * 2 <= Capacity) { return kernel_try_capacity<PqBits, (Capacity / 2)>(k_max); }
    }
    return compute_similarity_kernel<OutT,
                                     LutT,
                                     IvfSampleFilterT,
                                     PqBits,
                                     Capacity,
                                     PrecompBaseDiff,
                                     EnableSMemLut>;
  }
};

// A standalone accessor function was necessary to make sure template
// instantiation work correctly. This accessor function is not used anymore and
// may be removed.
template <typename OutT,
          typename LutT,
          bool PrecompBaseDiff,
          bool EnableSMemLut,
          typename IvfSampleFilterT = raft::neighbors::filtering::none_ivf_sample_filter>
auto get_compute_similarity_kernel(uint32_t pq_bits, uint32_t k_max)
  -> compute_similarity_kernel_t<OutT, LutT, IvfSampleFilterT>
{
  return compute_similarity_kernel_config<OutT,
                                          LutT,
                                          PrecompBaseDiff,
                                          EnableSMemLut,
                                          IvfSampleFilterT>::get(pq_bits, k_max);
}

/** Estimate the occupancy for the given kernel on the given device. */
template <typename OutT, typename LutT, typename IvfSampleFilterT>
struct occupancy_t {
  using shmem_unit = Pow2<128>;

  int blocks_per_sm = 0;
  double occupancy  = 0.0;
  double shmem_use  = 1.0;

  inline occupancy_t() = default;
  inline occupancy_t(size_t smem,
                     uint32_t n_threads,
                     compute_similarity_kernel_t<OutT, LutT, IvfSampleFilterT> kernel,
                     const cudaDeviceProp& dev_props)
  {
    RAFT_CUDA_TRY(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, n_threads, smem));
    occupancy = double(blocks_per_sm * n_threads) / double(dev_props.maxThreadsPerMultiProcessor);
    shmem_use = double(shmem_unit::roundUp(smem) * blocks_per_sm) /
                double(dev_props.sharedMemPerMultiprocessor);
  }
};

template <typename OutT, typename LutT, typename IvfSampleFilterT>
struct selected {
  compute_similarity_kernel_t<OutT, LutT, IvfSampleFilterT> kernel;
  dim3 grid_dim;
  dim3 block_dim;
  size_t smem_size;
  size_t device_lut_size;
};

template <typename OutT,
          typename LutT,
          typename IvfSampleFilterT = raft::neighbors::filtering::none_ivf_sample_filter>
void compute_similarity_run(selected<OutT, LutT, IvfSampleFilterT> s,
                            rmm::cuda_stream_view stream,
                            uint32_t dim,
                            uint32_t n_probes,
                            uint32_t pq_dim,
                            uint32_t n_queries,
                            uint32_t queries_offset,
                            distance::DistanceType metric,
                            codebook_gen codebook_kind,
                            uint32_t topk,
                            uint32_t max_samples,
                            const float* cluster_centers,
                            const float* pq_centers,
                            const uint8_t* const* pq_dataset,
                            const uint32_t* cluster_labels,
                            const uint32_t* _chunk_indices,
                            const float* queries,
                            const uint32_t* index_list,
                            float* query_kths,
                            IvfSampleFilterT sample_filter,
                            LutT* lut_scores,
                            OutT* _out_scores,
                            uint32_t* _out_indices)
{
  s.kernel<<<s.grid_dim, s.block_dim, s.smem_size, stream>>>(dim,
                                                             n_probes,
                                                             pq_dim,
                                                             n_queries,
                                                             queries_offset,
                                                             metric,
                                                             codebook_kind,
                                                             topk,
                                                             max_samples,
                                                             cluster_centers,
                                                             pq_centers,
                                                             pq_dataset,
                                                             cluster_labels,
                                                             _chunk_indices,
                                                             queries,
                                                             index_list,
                                                             query_kths,
                                                             sample_filter,
                                                             lut_scores,
                                                             _out_scores,
                                                             _out_indices);
  RAFT_CHECK_CUDA(stream);
}

/**
 * Use heuristics to choose an optimal instance of the search kernel.
 * It selects among a few kernel variants (with/out using shared mem for
 * lookup tables / precomputed distances) and tries to choose the block size
 * to maximize kernel occupancy.
 *
 * @param manage_local_topk
 *    whether use the fused calculate+select or just calculate the distances for each
 *    query and probed cluster.
 *
 * @param locality_hint
 *    beyond this limit do not consider increasing the number of active blocks per SM
 *    would improve locality anymore.
 */
template <typename OutT,
          typename LutT,
          typename IvfSampleFilterT = raft::neighbors::filtering::none_ivf_sample_filter>
auto compute_similarity_select(const cudaDeviceProp& dev_props,
                               bool manage_local_topk,
                               int locality_hint,
                               double preferred_shmem_carveout,
                               uint32_t pq_bits,
                               uint32_t pq_dim,
                               uint32_t precomp_data_count,
                               uint32_t n_queries,
                               uint32_t n_probes,
                               uint32_t topk) -> selected<OutT, LutT, IvfSampleFilterT>
{
  // Shared memory for storing the lookup table
  size_t lut_mem = sizeof(LutT) * (pq_dim << pq_bits);
  // Shared memory for storing pre-computed pieces to speedup the lookup table construction
  // (e.g. the distance between a cluster center and the query for L2).
  size_t bdf_mem = sizeof(float) * precomp_data_count;

  // Shared memory used by the fused top-k during cluster scanning;
  // may overlap with the precomputed distance array
  struct ltk_add_mem_t {
    size_t (*mem_required)(uint32_t);

    ltk_add_mem_t(bool manage_local_topk, uint32_t topk)
      : mem_required(pq_block_sort<kMaxCapacity, OutT, uint32_t>::get_mem_required(
          manage_local_topk ? topk : 0))
    {
    }

    [[nodiscard]] auto operator()(uint32_t n_threads) const -> size_t
    {
      return mem_required(n_threads);
    }
  } ltk_add_mem{manage_local_topk, topk};

  // Shared memory for the fused top-k component;
  // may overlap with all other uses of shared memory
  struct ltk_reduce_mem_t {
    uint32_t subwarp_size;
    uint32_t topk;
    bool manage_local_topk;
    ltk_reduce_mem_t(bool manage_local_topk, uint32_t topk)
      : manage_local_topk(manage_local_topk), topk(topk)
    {
      subwarp_size = WarpSize;
      while (topk * 2 <= subwarp_size) {
        subwarp_size /= 2;
      }
    }

    [[nodiscard]] auto operator()(uint32_t n_threads) const -> size_t
    {
      return manage_local_topk
               ? matrix::detail::select::warpsort::template calc_smem_size_for_block_wide<OutT,
                                                                                          uint32_t>(
                   n_threads / subwarp_size, topk)
               : 0;
    }
  } ltk_reduce_mem{manage_local_topk, topk};

  struct total_shared_mem_t {
    ltk_add_mem_t& ltk_add_mem;
    ltk_reduce_mem_t& ltk_reduce_mem;
    size_t lut_mem;
    size_t bdf_mem;
    [[nodiscard]] auto operator()(uint32_t n_threads) const -> size_t
    {
      return std::max(ltk_reduce_mem(n_threads),
                      lut_mem + std::max(bdf_mem, ltk_add_mem(n_threads)));
    }
  };

  // Total amount of work; should be enough to occupy the GPU.
  uint32_t n_blocks = n_queries * n_probes;

  // The minimum block size we may want:
  //   1. It's a power-of-two for efficient L1 caching of pq_centers values
  //      (multiples of `1 << pq_bits`).
  //   2. It should be large enough to fully utilize an SM.
  uint32_t n_threads_min = WarpSize;
  while (dev_props.maxBlocksPerMultiProcessor * int(n_threads_min) <
         dev_props.maxThreadsPerMultiProcessor) {
    n_threads_min *= 2;
  }
  // Further increase the minimum block size to make sure full device occupancy
  // (NB: this may lead to `n_threads_min` being larger than the kernel's maximum)
  while (int(n_blocks * n_threads_min) <
           dev_props.multiProcessorCount * dev_props.maxThreadsPerMultiProcessor &&
         int(n_threads_min) < dev_props.maxThreadsPerBlock) {
    n_threads_min *= 2;
  }
  // Even further, increase it to allow less blocks per SM if there not enough queries.
  // With this, we reduce the chance of different clusters being processed by two blocks
  // on the same SM and thus improve the data locality for L1 caching.
  while (int(n_queries * n_threads_min) < dev_props.maxThreadsPerMultiProcessor &&
         int(n_threads_min) < dev_props.maxThreadsPerBlock) {
    n_threads_min *= 2;
  }

  // Granularity of changing the number of threads when computing the maximum block size.
  // It's good to have it multiple of the PQ book width.
  uint32_t n_threads_gty = round_up_safe<uint32_t>(1u << pq_bits, WarpSize);

  /*
   Shared memory / L1 cache balance is the main limiter of this kernel.
   The more blocks per SM we launch, the more shared memory we need. Besides that, we have
   three versions of the kernel varying in performance and shmem usage.

   We try the most demanding and the fastest kernel first, trying to maximize occupancy with
   the minimum number of blocks (just one, really). Then, we tweak the `n_threads` to further
   optimize occupancy and data locality for the L1 cache.
   */
  auto conf_fast        = get_compute_similarity_kernel<OutT, LutT, true, true, IvfSampleFilterT>;
  auto conf_no_basediff = get_compute_similarity_kernel<OutT, LutT, false, true, IvfSampleFilterT>;
  auto conf_no_smem_lut = get_compute_similarity_kernel<OutT, LutT, true, false, IvfSampleFilterT>;
  auto topk_or_zero     = manage_local_topk ? topk : 0u;
  std::array candidates{
    std::make_tuple(conf_fast(pq_bits, topk_or_zero),
                    total_shared_mem_t{ltk_add_mem, ltk_reduce_mem, lut_mem, bdf_mem},
                    true),
    std::make_tuple(conf_no_basediff(pq_bits, topk_or_zero),
                    total_shared_mem_t{ltk_add_mem, ltk_reduce_mem, lut_mem, 0},
                    true),
    std::make_tuple(conf_no_smem_lut(pq_bits, topk_or_zero),
                    total_shared_mem_t{ltk_add_mem, ltk_reduce_mem, 0, bdf_mem},
                    false)};

  // we may allow slightly lower than 100% occupancy;
  constexpr double kTargetOccupancy = 0.75;
  // This struct is used to select the better candidate
  occupancy_t<OutT, LutT, IvfSampleFilterT> selected_perf{};
  selected<OutT, LutT, IvfSampleFilterT> selected_config;
  for (auto [kernel, smem_size_f, lut_is_in_shmem] : candidates) {
    if (smem_size_f(WarpSize) > dev_props.sharedMemPerBlockOptin) {
      // Even a single block cannot fit into an SM due to shmem requirements. Skip the candidate.
      continue;
    }

    // First, we set the carveout hint to the preferred value. The driver will increase this if
    // needed to run at least one block per SM. At the same time, if more blocks fit into one SM,
    // this carveout value will limit the calculated occupancy. When we're done selecting the best
    // launch configuration, we will tighten the carveout once more, based on the final memory
    // usage and occupancy.
    const int max_carveout =
      estimate_carveout(preferred_shmem_carveout, smem_size_f(WarpSize), dev_props);
    RAFT_CUDA_TRY(
      cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, max_carveout));

    // Get the theoretical maximum possible number of threads per block
    cudaFuncAttributes kernel_attrs;
    RAFT_CUDA_TRY(cudaFuncGetAttributes(&kernel_attrs, kernel));
    uint32_t n_threads = round_down_safe<uint32_t>(kernel_attrs.maxThreadsPerBlock, n_threads_gty);

    // Actual required shmem depens on the number of threads
    size_t smem_size = smem_size_f(n_threads);

    // Make sure the kernel can get enough shmem.
    cudaError_t cuda_status =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (cuda_status != cudaSuccess) {
      RAFT_EXPECTS(
        cuda_status == cudaGetLastError(),
        "Tried to reset the expected cuda error code, but it didn't match the expectation");
      // Failed to request enough shmem for the kernel. Skip the candidate.
      continue;
    }

    occupancy_t<OutT, LutT, IvfSampleFilterT> cur(smem_size, n_threads, kernel, dev_props);
    if (cur.blocks_per_sm <= 0) {
      // For some reason, we still cannot make this kernel run. Skip the candidate.
      continue;
    }

    {
      // Try to reduce the number of threads to increase occupancy and data locality
      auto n_threads_tmp = n_threads_min;
      while (n_threads_tmp * 2 < n_threads) {
        n_threads_tmp *= 2;
      }
      if (n_threads_tmp < n_threads) {
        while (n_threads_tmp >= n_threads_min) {
          auto smem_size_tmp = smem_size_f(n_threads_tmp);
          occupancy_t<OutT, LutT, IvfSampleFilterT> tmp(
            smem_size_tmp, n_threads_tmp, kernel, dev_props);
          bool select_it = false;
          if (lut_is_in_shmem && locality_hint >= tmp.blocks_per_sm) {
            // Normally, the smaller the block the better for L1 cache hit rate.
            // Hence, the occupancy should be "just good enough"
            select_it = tmp.occupancy >= min(kTargetOccupancy, cur.occupancy);
          } else if (lut_is_in_shmem) {
            // If we don't have enough repeating probes (locality_hint < tmp.blocks_per_sm),
            // the locality is not going to improve with increasing the number of blocks per SM.
            // Hence, the only metric here is the occupancy.
            bool improves_occupancy = tmp.occupancy > cur.occupancy;
            // Otherwise, the performance still improves with a smaller block size,
            // given there is enough work to do
            bool improves_parallelism =
              tmp.occupancy == cur.occupancy &&
              7u * tmp.blocks_per_sm * dev_props.multiProcessorCount <= n_blocks;
            select_it = improves_occupancy || improves_parallelism;
          } else {
            // If we don't use shared memory for the lookup table, increasing the number of blocks
            // is very taxing on the global memory usage.
            // In this case, the occupancy must increase a lot to make it worth the cost.
            select_it = tmp.occupancy >= min(1.0, cur.occupancy / kTargetOccupancy);
          }
          if (select_it) {
            n_threads = n_threads_tmp;
            smem_size = smem_size_tmp;
            cur       = tmp;
          }
          n_threads_tmp /= 2;
        }
      }
    }

    {
      if (selected_perf.occupancy <= 0.0  // no candidate yet
          || (selected_perf.occupancy < cur.occupancy * kTargetOccupancy &&
              selected_perf.shmem_use >= cur.shmem_use)  // much improved occupancy
      ) {
        selected_perf = cur;
        if (lut_is_in_shmem) {
          selected_config = {
            kernel, dim3(n_blocks, 1, 1), dim3(n_threads, 1, 1), smem_size, size_t(0)};
        } else {
          // When the global memory is used for the lookup table, we need to minimize the grid
          // size; otherwise, the kernel may quickly run out of memory.
          auto n_blocks_min =
            std::min<uint32_t>(n_blocks, cur.blocks_per_sm * dev_props.multiProcessorCount);
          selected_config = {kernel,
                             dim3(n_blocks_min, 1, 1),
                             dim3(n_threads, 1, 1),
                             smem_size,
                             size_t(n_blocks_min) * size_t(pq_dim << pq_bits)};
        }
        // Actual shmem/L1 split wildly rounds up the specified preferred carveout, so we set here
        // a rather conservative bar; most likely, the kernel gets more shared memory than this,
        // and the occupancy doesn't get hurt.
        auto carveout = std::min<int>(max_carveout, std::ceil(100.0 * cur.shmem_use));
        RAFT_CUDA_TRY(
          cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, carveout));
        if (cur.occupancy >= kTargetOccupancy) { break; }
      } else if (selected_perf.occupancy > 0.0) {
        // If we found a reasonable candidate on a previous iteration, and this one is not better,
        // then don't try any more candidates because they are much slower anyway.
        break;
      }
    }
  }

  RAFT_EXPECTS(selected_perf.occupancy > 0.0,
               "Couldn't determine a working kernel launch configuration.");

  return selected_config;
}

}  // namespace raft::neighbors::ivf_pq::detail
