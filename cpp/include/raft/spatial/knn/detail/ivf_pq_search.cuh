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
#include <raft/core/device_mdarray.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/logger.hpp>
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

/** 8-bit floating-point storage type.
 *
 * This is a custom type for the current IVF-PQ implementation. No arithmetic operations defined
 * only conversion to and from fp32. This type is unrelated to the proposed FP8 specification.
 */
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

/**
 * Select the clusters to probe and, as a side-effect, translate the queries type `T -> float`
 *
 * Assuming the number of clusters is not that big (a few thousands), we do a plain GEMM
 * followed by select_topk to select the clusters to probe. There's no need to return the similarity
 * scores here.
 */
template <typename T>
void select_clusters(const handle_t& handle,
                     uint32_t* clusters_to_probe,  // [n_queries, n_probes]
                     float* float_queries,         // [n_queries, dim_ext]
                     uint32_t n_queries,
                     uint32_t n_probes,
                     uint32_t n_lists,
                     uint32_t dim,
                     uint32_t dim_ext,
                     raft::distance::DistanceType metric,
                     const T* queries,              // [n_queries, dim]
                     const float* cluster_centers,  // [n_lists, dim_ext]
                     rmm::mr::device_memory_resource* mr)
{
  auto stream = handle.get_stream();
  rmm::device_uvector<float> qc_distances(n_queries * n_lists, stream, mr);
  /* NOTE[qc_distances]

  We compute query-center distances to choose the clusters to probe.
  We accomplish that with just one GEMM operation thanks to some preprocessing:

    L2 distance:
      cluster_centers[i, dim()] contains the squared norm of the center vector i;
      we extend the dimension K of the GEMM to compute it together with all the dot products:

      `cq_distances[i, j] = |cluster_centers[j]|^2 - 2 * (queries[i], cluster_centers[j])`

      This is a monotonous mapping of the proper L2 distance.

    IP distance:
      `cq_distances[i, j] = - (queries[i], cluster_centers[j])`

      This is a negative inner-product distance. We minimize it to find the similar clusters.

      NB: cq_distances is NOT used further in ivfpq_search.
 */
  float norm_factor;
  switch (metric) {
    case raft::distance::DistanceType::L2Expanded: norm_factor = 1.0 / -2.0; break;
    case raft::distance::DistanceType::InnerProduct: norm_factor = 0.0; break;
    default: RAFT_FAIL("Unsupported distance type %d.", int(metric));
  }
  linalg::writeOnlyUnaryOp(
    float_queries,
    dim_ext * n_queries,
    [queries, dim, dim_ext, norm_factor] __device__(float* out, uint32_t ix) {
      uint32_t col = ix % dim_ext;
      uint32_t row = ix / dim_ext;
      *out         = col < dim ? utils::mapping<float>{}(queries[col + dim * row]) : norm_factor;
    },
    stream);

  float alpha;
  float beta;
  uint32_t gemm_k = dim;
  switch (metric) {
    case raft::distance::DistanceType::L2Expanded: {
      alpha  = -2.0;
      beta   = 0.0;
      gemm_k = dim + 1;
      RAFT_EXPECTS(gemm_k <= dim_ext, "unexpected gemm_k or dim_ext");
    } break;
    case raft::distance::DistanceType::InnerProduct: {
      alpha = -1.0;
      beta  = 0.0;
    } break;
    default: RAFT_FAIL("Unsupported distance type %d.", int(metric));
  }
  linalg::gemm(handle,
               true,
               false,
               n_lists,
               n_queries,
               gemm_k,
               &alpha,
               cluster_centers,
               dim_ext,
               float_queries,
               dim_ext,
               &beta,
               qc_distances.data(),
               n_lists,
               stream);

  // Select neighbor clusters for each query.
  rmm::device_uvector<float> cluster_dists(n_queries * n_probes, stream, mr);
  select_topk<float, uint32_t>(qc_distances.data(),
                               nullptr,
                               n_queries,
                               n_lists,
                               n_probes,
                               cluster_dists.data(),
                               clusters_to_probe,
                               true,
                               stream,
                               mr);
}

/**
 * For each query, we calculate a cumulative sum of the cluster sizes that we probe, and return that
 * in chunk_indices. Essentially this is a segmented inclusive scan of the cluster sizes. The total
 * number of samples per query (sum of the cluster sizes that we probe) is returned in n_samples.
 */
template <int BlockDim, typename IdxT>
__launch_bounds__(BlockDim) __global__
  void calc_chunk_indices_kernel(uint32_t n_probes,
                                 const IdxT* cluster_offsets,        // [n_clusters + 1]
                                 const uint32_t* clusters_to_probe,  // [n_queries, n_probes]
                                 uint32_t* chunk_indices,            // [n_queries, n_probes]
                                 uint32_t* n_samples                 // [n_queries]
  )
{
  using block_scan = cub::BlockScan<uint32_t, BlockDim>;
  __shared__ typename block_scan::TempStorage shm;

  // locate the query data
  clusters_to_probe += n_probes * blockIdx.x;
  chunk_indices += n_probes * blockIdx.x;

  // block scan
  const uint32_t n_probes_aligned = Pow2<BlockDim>::roundUp(n_probes);
  uint32_t total                  = 0;
  for (uint32_t probe_ix = threadIdx.x; probe_ix < n_probes_aligned; probe_ix += BlockDim) {
    auto label = probe_ix < n_probes ? clusters_to_probe[probe_ix] : 0u;
    auto chunk = probe_ix < n_probes
                   ? static_cast<uint32_t>(cluster_offsets[label + 1] - cluster_offsets[label])
                   : 0u;
    if (threadIdx.x == 0) { chunk += total; }
    block_scan(shm).InclusiveSum(chunk, chunk, total);
    __syncthreads();
    if (probe_ix < n_probes) { chunk_indices[probe_ix] = chunk; }
  }
  // save the total size
  if (threadIdx.x == 0) { n_samples[blockIdx.x] = total; }
}

template <typename IdxT>
struct calc_chunk_indices {
 public:
  struct configured {
    void* kernel;
    dim3 block_dim;
    dim3 grid_dim;
    uint32_t n_probes;

    void operator()(const IdxT* cluster_offsets,
                    const uint32_t* clusters_to_probe,
                    uint32_t* chunk_indices,
                    uint32_t* n_samples,
                    rmm::cuda_stream_view stream)
    {
      void* args[] =  // NOLINT
        {&n_probes, &cluster_offsets, &clusters_to_probe, &chunk_indices, &n_samples};
      RAFT_CUDA_TRY(cudaLaunchKernel(kernel, grid_dim, block_dim, args, 0, stream));
    }
  };

  static auto configure(uint32_t n_probes, uint32_t n_queries) -> configured
  {
    return try_block_dim<1024>(n_probes, n_queries);
  }

 private:
  template <int BlockDim>
  static auto try_block_dim(uint32_t n_probes, uint32_t n_queries) -> configured
  {
    if constexpr (BlockDim >= WarpSize * 2) {
      if (BlockDim >= n_probes * 2) { return try_block_dim<(BlockDim / 2)>(n_probes, n_queries); }
    }
    return {reinterpret_cast<void*>(calc_chunk_indices_kernel<BlockDim, IdxT>),
            dim3(BlockDim, 1, 1),
            dim3(n_queries, 1, 1),
            n_probes};
  }
};

/**
 * Look up the dataset index that corresponds to a sample index.
 *
 * Each query vector was compared to all the vectors from n_probes clusters, and sample_ix is one of
 * such vector. This function looks up which cluster sample_ix belongs to, and returns the original
 * dataset index for that vector.
 *
 * @return whether the input index is in a valid range
 *    (the opposite can happen if there is not enough data to output in the selected clusters).
 */
template <typename IdxT>
__device__ auto find_db_row(IdxT& x,  // NOLINT
                            uint32_t n_probes,
                            const IdxT* cluster_offsets,     // [n_clusters + 1,]
                            const uint32_t* cluster_labels,  // [n_probes,]
                            const uint32_t* chunk_indices    // [n_probes,]
                            ) -> bool
{
  uint32_t ix_min = 0;
  uint32_t ix_max = n_probes;
  do {
    uint32_t i = (ix_min + ix_max) / 2;
    if (IdxT(chunk_indices[i]) < x) {
      ix_min = i + 1;
    } else {
      ix_max = i;
    }
  } while (ix_min < ix_max);
  if (ix_min == n_probes) { return false; }
  if (ix_min > 0) { x -= chunk_indices[ix_min - 1]; }
  x += cluster_offsets[cluster_labels[ix_min]];
  return true;
}

template <int BlockDim, typename IdxT>
__launch_bounds__(BlockDim) __global__
  void postprocess_neighbors_kernel(IdxT* neighbors,                    // [n_queries, topk]
                                    const IdxT* db_indices,             // [n_rows]
                                    const IdxT* cluster_offsets,        // [n_clusters + 1]
                                    const uint32_t* clusters_to_probe,  // [n_queries, n_probes]
                                    const uint32_t* chunk_indices,      // [n_queries, n_probes]
                                    uint32_t n_queries,
                                    uint32_t n_probes,
                                    uint32_t topk)
{
  uint64_t i        = threadIdx.x + BlockDim * uint64_t(blockIdx.x);
  uint32_t query_ix = i / uint64_t(topk);
  if (query_ix >= n_queries) { return; }
  uint32_t k = i % uint64_t(topk);
  neighbors += query_ix * topk;
  IdxT data_ix = neighbors[k];
  // backtrace the index if we don't have local top-k
  bool valid = true;
  if (n_probes > 0) {
    valid = find_db_row(data_ix,
                        n_probes,
                        cluster_offsets,
                        clusters_to_probe + n_probes * query_ix,
                        chunk_indices + n_probes * query_ix);
  }
  neighbors[k] = valid ? db_indices[data_ix] : std::numeric_limits<IdxT>::max();
}

/**
 * Transform found neighbor indices into the corresponding database indices
 * (as stored in index.indices()).
 *
 * When the main kernel runs with a fused top-k (`manage_local_topk == true`), this function simply
 * fetches the index values  by the returned row ids. Otherwise, the found neighbors require extra
 * pre-processing (performed by `find_db_row`).
 */
template <typename IdxT>
void postprocess_neighbors(IdxT* neighbors,  // [n_queries, topk]
                           bool manage_local_topk,
                           const IdxT* db_indices,             // [n_rows]
                           const IdxT* cluster_offsets,        // [n_clusters + 1]
                           const uint32_t* clusters_to_probe,  // [n_queries, n_probes]
                           const uint32_t* chunk_indices,      // [n_queries, n_probes]
                           uint32_t n_queries,
                           uint32_t n_probes,
                           uint32_t topk,
                           rmm::cuda_stream_view stream)
{
  constexpr int kPNThreads = 256;
  const int pn_blocks      = raft::div_rounding_up_unsafe<size_t>(n_queries * topk, kPNThreads);
  postprocess_neighbors_kernel<kPNThreads, IdxT>
    <<<pn_blocks, kPNThreads, 0, stream>>>(neighbors,
                                           db_indices,
                                           cluster_offsets,
                                           clusters_to_probe,
                                           chunk_indices,
                                           n_queries,
                                           manage_local_topk ? 0u : n_probes,
                                           topk);
}

/**
 * Post-process the scores depending on the metric type;
 * translate the element type if necessary.
 */
template <typename ScoreT>
void postprocess_distances(float* out,        // [n_queries, topk]
                           const ScoreT* in,  // [n_queries, topk]
                           distance::DistanceType metric,
                           uint32_t n_queries,
                           uint32_t topk,
                           float scaling_factor,
                           rmm::cuda_stream_view stream)
{
  size_t len = size_t(n_queries) * size_t(topk);
  switch (metric) {
    case distance::DistanceType::L2Unexpanded:
    case distance::DistanceType::L2Expanded: {
      linalg::unaryOp(
        out,
        in,
        len,
        [scaling_factor] __device__(ScoreT x) -> float {
          return scaling_factor * scaling_factor * float(x);
        },
        stream);
    } break;
    case distance::DistanceType::L2SqrtUnexpanded:
    case distance::DistanceType::L2SqrtExpanded: {
      linalg::unaryOp(
        out,
        in,
        len,
        [scaling_factor] __device__(ScoreT x) -> float { return scaling_factor * sqrtf(float(x)); },
        stream);
    } break;
    case distance::DistanceType::InnerProduct: {
      linalg::unaryOp(
        out,
        in,
        len,
        [scaling_factor] __device__(ScoreT x) -> float {
          return -scaling_factor * scaling_factor * float(x);
        },
        stream);
    } break;
    default: RAFT_FAIL("Unexpected metric.");
  }
}

/**
 * @brief Compute the similarity score between a vector from `pq_dataset` and a query vector.
 *
 * @tparam OpT an unsigned integer type that is used for bit operations on multiple PQ codes
 *   at once; it's selected to maximize throughput while matching criteria:
 *     1. `pq_bits * vec_len % 8 * sizeof(OpT) == 0`.
 *     2. `pq_dim % vec_len == 0`
 *
 * @tparam LutT type of the elements in the lookup table.
 *
 * @param pq_bits The bit length of an encoded vector element after compression by PQ
 * @param vec_len == 8 * sizeof(OpT) / gcd(8 * sizeof(OpT), pq_bits)
 * @param pq_dim
 * @param[in] pq_code_ptr
 *   a device pointer to the dataset at the indexed position (`pq_dim * pq_bits` bits-wide)
 * @param[in] lut_scores
 *   a device or shared memory pointer to the lookup table [pq_dim, pq_book_size]
 *
 * @return the score for the entry `data_ix` in the `pq_dataset`.
 */
template <typename OpT, typename LutT>
__device__ auto ivfpq_compute_score(
  uint32_t pq_bits, uint32_t vec_len, uint32_t pq_dim, const OpT* pq_head, const LutT* lut_scores)
  -> float
{
  float score                   = 0.0;
  constexpr uint32_t kBitsTotal = 8 * sizeof(OpT);
  for (; pq_dim > 0; pq_dim -= vec_len) {
    OpT pq_code = pq_head[0];
    pq_head++;
    auto bits_left = kBitsTotal;
    for (uint32_t k = 0; k < vec_len; k++) {
      uint8_t code = pq_code;
      if (bits_left > pq_bits) {
        pq_code >>= pq_bits;
        bits_left -= pq_bits;
      } else {
        if (k < vec_len - 1) {
          pq_code = pq_head[0];
          pq_head++;
        }
        code |= (pq_code << bits_left);
        pq_code >>= (pq_bits - bits_left);
        bits_left += (kBitsTotal - pq_bits);
      }
      code &= (1 << pq_bits) - 1;
      score += float(lut_scores[code]);
      lut_scores += (1 << pq_bits);
    }
  }
  return score;
}

template <typename T, typename IdxT>
struct dummy_block_sort_t {
  using queue_t = topk::warp_sort_immediate<WarpSize, true, T, IdxT>;
  __device__ dummy_block_sort_t(int k, uint8_t* smem_buf){};
};

template <int Capacity, typename T, typename IdxT>
struct pq_block_sort {
  using type = topk::block_sort<topk::warp_sort_immediate, Capacity, true, T, IdxT>;
};

template <typename T, typename IdxT>
struct pq_block_sort<0, T, IdxT> : dummy_block_sort_t<T, IdxT> {
  using type = dummy_block_sort_t<T, IdxT>;
};

template <int Capacity, typename T, typename IdxT>
using block_sort_t = typename pq_block_sort<Capacity, T, IdxT>::type;

/**
 * The main kernel that computes similarity scores across multiple queries and probes.
 * When `Capacity > 0`, it also selects top K candidates for each query and probe
 * (which need to be merged across probes afterwards).
 *
 * Each block processes a (query, probe) pair: it calculates the distance between the single query
 * vector and all the dataset vector in the cluster that we are probing.
 *
 * @tparam OpT is a carrier integer type selected to maximize throughput;
 *   Used solely in `ivfpq_compute_score`;
 * @tparam IdxT
 *   The type of data indices
 * @tparam OutT
 *   The output type - distances.
 * @tparam LutT
 *   The lookup table element type (lut_scores).
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
 * @param n_rows the number of records in the dataset
 * @param dim the dimensionality of the data (NB: after rotation transform, i.e. `index.rot_dim()`).
 * @param n_probes the number of clusters to search for each query
 * @param pq_bits the bit length of an encoded vector element after compression by PQ
 *   (NB: pq_book_size = 1 << pq_bits).
 * @param pq_dim
 *   The dimensionality of an encoded vector after compression by PQ.
 * @param n_queries the number of queries.
 * @param metric the distance type.
 * @param codebook_kind Defines the way PQ codebooks have been trained.
 * @param topk the `k` in the select top-k.
 * @param cluster_centers
 *   The device pointer to the cluster centers in the original space (NB: after rotation)
 *   [n_clusters, dim].
 * @param pq_centers
 *   The device pointer to the cluster centers in the PQ space
 *   [pq_dim, pq_book_size, pq_len] or [n_clusters, pq_book_size, pq_len,].
 * @param pq_dataset
 *   The device pointer to the PQ index (data) [n_rows, pq_dim * pq_bits / 8].
 * @param cluster_offsets
 *   The device pointer to the cluster offsets [n_clusters + 1].
 * @param cluster_labels
 *   The device pointer to the labels (clusters) for each query and probe [n_queries, n_probes].
 * @param _chunk_indices
 *   The device pointer to the data offsets for each query and probe [n_queries, n_probes].
 * @param queries
 *   The device pointer to the queries (NB: after rotation) [n_queries, dim].
 * @param index_list
 *   An optional device pointer to the enforced order of search [n_queries, n_probes].
 *   One can pass reordered indices here to try to improve data reading locality.
 * @param lut_scores
 *   The device pointer for storing the lookup table globally [gridDim.x, pq_dim << pq_bits].
 *   Ignored when `EnableSMemLut == true`.
 * @param _out_scores
 *   The device pointer to the output scores
 *   [n_queries, max_samples] or [n_queries, n_probes, topk].
 * @param _out_indices
 *   The device pointer to the output indices [n_queries, n_probes, topk].
 *   Ignored  when `Capacity == 0`.
 */
template <typename OpT,
          typename IdxT,
          typename OutT,
          typename LutT,
          int Capacity,
          bool PrecompBaseDiff,
          bool EnableSMemLut>
__launch_bounds__(1024) __global__
  void ivfpq_compute_similarity_kernel(uint32_t n_rows,
                                       uint32_t dim,
                                       uint32_t n_probes,
                                       uint32_t pq_bits,
                                       uint32_t pq_dim,
                                       uint32_t n_queries,
                                       distance::DistanceType metric,
                                       codebook_gen codebook_kind,
                                       uint32_t topk,
                                       const float* cluster_centers,
                                       const float* pq_centers,
                                       const uint8_t* pq_dataset,
                                       const IdxT* cluster_offsets,
                                       const uint32_t* cluster_labels,
                                       const uint32_t* _chunk_indices,
                                       const float* queries,
                                       const uint32_t* index_list,
                                       LutT* lut_scores,
                                       OutT* _out_scores,
                                       IdxT* _out_indices)
{
  /* Shared memory:

    * lut_scores: lookup table (LUT) of size = `pq_dim << pq_bits`  (when EnableSMemLut)
    * base_diff: size = dim  (which is equal to `pq_dim * pq_len`)
    * topk::block_sort: some amount of shared memory, but overlaps with the rest:
        block_sort only needs shared memory for `.done()` operation, which can come very last.
  */
  extern __shared__ __align__(256) uint8_t smem_buf[];  // NOLINT
  constexpr bool kManageLocalTopK = Capacity > 0;
  constexpr uint32_t kOpBits      = 8 * sizeof(OpT);

  const uint32_t pq_len  = dim / pq_dim;
  const uint32_t vec_len = kOpBits / gcd<uint32_t>(kOpBits, pq_bits);

  if constexpr (EnableSMemLut) {
    lut_scores = reinterpret_cast<LutT*>(smem_buf);
  } else {
    lut_scores += (pq_dim << pq_bits) * blockIdx.x;
  }

  float* base_diff = nullptr;
  if constexpr (PrecompBaseDiff) {
    if constexpr (EnableSMemLut) {
      base_diff = reinterpret_cast<float*>(lut_scores + (pq_dim << pq_bits));
    } else {
      base_diff = reinterpret_cast<float*>(smem_buf);
    }
  }

  for (int ib = blockIdx.x; ib < n_queries * n_probes; ib += gridDim.x) {
    uint32_t query_ix;
    uint32_t probe_ix;
    if (index_list == nullptr) {
      query_ix = ib % n_queries;
      probe_ix = ib / n_queries;
    } else {
      query_ix = index_list[ib] / n_probes;
      probe_ix = index_list[ib] % n_probes;
    }
    if (query_ix >= n_queries || probe_ix >= n_probes) continue;

    const uint32_t* chunk_indices = _chunk_indices + (n_probes * query_ix);
    const float* query            = queries + (dim * query_ix);
    OutT* out_scores;
    IdxT* out_indices = nullptr;
    if constexpr (kManageLocalTopK) {
      // Store topk calculated distances to out_scores (and its indices to out_indices)
      out_scores  = _out_scores + topk * (probe_ix + (n_probes * query_ix));
      out_indices = _out_indices + topk * (probe_ix + (n_probes * query_ix));
    } else {
      // Store all calculated distances to out_scores
      auto max_samples = Pow2<128>::roundUp(cluster_offsets[n_probes]);
      out_scores       = _out_scores + max_samples * query_ix;
    }
    uint32_t label              = cluster_labels[n_probes * query_ix + probe_ix];
    const float* cluster_center = cluster_centers + (dim * label);
    const float* pq_center;
    if (codebook_kind == codebook_gen::PER_SUBSPACE) {
      pq_center = pq_centers;
    } else {
      pq_center = pq_centers + (pq_len << pq_bits) * label;
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
    // For each subspace, the lookup table stores the distance between the actual query vector
    // (projected into the subspace) and all possible pq vectors in that subspace.
    for (uint32_t i = threadIdx.x; i < (pq_dim << pq_bits); i += blockDim.x) {
      uint32_t i_pq   = i >> pq_bits;
      uint32_t i_code = codebook_kind == codebook_gen::PER_CLUSTER ? i & ((1 << pq_bits) - 1) : i;
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
            score += query[k] * (cluster_center[k] + pq_center[j + pq_len * i_code]);
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

    using local_topk_t = block_sort_t<Capacity, OutT, IdxT>;
    local_topk_t block_topk(topk, smem_buf);

    // Ensure lut_scores is written by all threads before using it in ivfpq_compute_score
    __threadfence_block();
    __syncthreads();

    // Compute a distance for each sample
    const uint32_t pq_line_width = pq_dim * pq_bits / 8;
    for (uint32_t i = threadIdx.x; i < n_samples32; i += blockDim.x) {
      OutT score = local_topk_t::queue_t::kDummy;
      if (i < n_samples) {
        auto pq_ptr =
          reinterpret_cast<const OpT*>(pq_dataset + uint64_t(pq_line_width) * (cluster_offset + i));
        float fscore = ivfpq_compute_score<OpT, LutT>(pq_bits, vec_len, pq_dim, pq_ptr, lut_scores);
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
      uint32_t max_samples = uint32_t(Pow2<128>::roundUp(cluster_offsets[n_probes]));
      if (probe_ix + 1 == n_probes) {
        for (uint32_t i = threadIdx.x + sample_offset + n_samples; i < max_samples;
             i += blockDim.x) {
          out_scores[i] = local_topk_t::queue_t::kDummy;
        }
      }
    }
  }
}

/**
 * This structure selects configurable template parameters (instance) based on
 * the search/index parameters at runtime.
 *
 * This is done by means of recursively iterating through a small set of possible
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
                            IdxT*);

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
    template <typename OpT, int Capacity>
    static auto kernel_try_capacity(uint32_t k_max) -> kernel_t
    {
      if constexpr (Capacity > 0) {
        if (k_max == 0 || k_max > Capacity) { return kernel_try_capacity<OpT, 0>(k_max); }
      }
      if constexpr (Capacity > 32) {
        if (k_max * 2 <= Capacity) { return kernel_try_capacity<OpT, (Capacity / 2)>(k_max); }
      }
      return ivfpq_compute_similarity_kernel<OpT,
                                             IdxT,
                                             OutT,
                                             LutT,
                                             Capacity,
                                             PrecompBaseDiff,
                                             EnableSMemLut>;
    }

    static auto kernel_base(uint32_t pq_bits, uint32_t pq_dim, uint32_t k_max) -> kernel_t
    {
      switch (gcd<uint32_t>(pq_bits * pq_dim, 64)) {
        case 64: return kernel_try_capacity<uint64_t, kMaxCapacity>(k_max);
        case 32: return kernel_try_capacity<uint32_t, kMaxCapacity>(k_max);
        case 16: return kernel_try_capacity<uint16_t, kMaxCapacity>(k_max);
        case 8: return kernel_try_capacity<uint8_t, kMaxCapacity>(k_max);
        default:
          RAFT_FAIL("`pq_bits * pq_dim` must be a multiple of 8 (pq_bits = %u, pq_dim = %u).",
                    pq_bits,
                    pq_dim);
      }
    }
  };

  struct selected {
    void* kernel;
    dim3 grid_dim;
    dim3 block_dim;
    size_t smem_size;
    size_t device_lut_size;

    template <typename... Args>
    void operator()(rmm::cuda_stream_view stream, Args... args)
    {
      void* xs[] = {&args...};  // NOLINT
      RAFT_CUDA_TRY(cudaLaunchKernel(kernel, grid_dim, block_dim, xs, smem_size, stream));
    }
  };

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
   */
  static inline auto select(bool manage_local_topk,
                            uint32_t pq_bits,
                            uint32_t pq_dim,
                            uint32_t rot_dim,
                            uint32_t preferred_thread_block_size,
                            uint32_t n_queries,
                            uint32_t n_probes,
                            uint32_t topk) -> selected
  {
    using conf_fast        = configured<true, true>;
    using conf_no_basediff = configured<false, true>;
    using conf_no_smem_lut = configured<true, false>;

    kernel_t kernel_fast = conf_fast::kernel(pq_bits, pq_dim, manage_local_topk ? topk : 0u);
    kernel_t kernel_no_basediff =
      conf_no_basediff::kernel(pq_bits, pq_dim, manage_local_topk ? topk : 0u);
    kernel_t kernel_no_smem_lut =
      conf_no_smem_lut::kernel(pq_bits, pq_dim, manage_local_topk ? topk : 0u);

    const size_t smem_threshold = 48 * 1024;
    size_t smem_size            = sizeof(LutT) * (pq_dim << pq_bits);
    size_t smem_size_base_diff  = sizeof(float) * rot_dim;

    uint32_t n_blocks  = n_queries * n_probes;
    uint32_t n_threads = 1024;
    // preferred_thread_block_size == 0 means using auto thread block size calculation mode
    if (preferred_thread_block_size == 0) {
      const uint32_t thread_min = 256;
      int cur_dev;
      cudaDeviceProp dev_props;
      RAFT_CUDA_TRY(cudaGetDevice(&cur_dev));
      RAFT_CUDA_TRY(cudaGetDeviceProperties(&dev_props, cur_dev));
      while (n_threads > thread_min) {
        if (n_blocks < uint32_t(getMultiProcessorCount() * (1024 / (n_threads / 2)))) { break; }
        if (dev_props.sharedMemPerMultiprocessor * 2 / 3 < smem_size * (1024 / (n_threads / 2))) {
          break;
        }
        n_threads /= 2;
      }
    } else {
      n_threads = preferred_thread_block_size;
    }
    size_t smem_size_local_topk =
      manage_local_topk
        ? topk::template calc_smem_size_for_block_wide<OutT, IdxT>(n_threads / WarpSize, topk)
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
        RAFT_LOG_DEBUG(
          "Non-shared-mem look-up table kernel is selected, because it wouldn't fit shmem "
          "required: "
          "%zu bytes)",
          smem_size);
        kernel       = kernel_no_smem_lut;
        use_smem_lut = false;
        n_threads    = 1024;
        smem_size_local_topk =
          manage_local_topk
            ? topk::template calc_smem_size_for_block_wide<OutT, IdxT>(n_threads / WarpSize, topk)
            : 0;
        smem_size = max(smem_size_base_diff, smem_size_local_topk);
        n_blocks  = getMultiProcessorCount();
      }
    }
    if (kernel_no_basediff_available) {
      bool kernel_fast_available = true;
      if (smem_size + smem_size_base_diff > smem_threshold) {
        cudaError_t cuda_status = cudaFuncSetAttribute(kernel_fast,
                                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                       smem_size + smem_size_base_diff);
        if (cuda_status != cudaSuccess) {
          RAFT_EXPECTS(
            cuda_status == cudaGetLastError(),
            "Tried to reset the expected cuda error code, but it didn't match the expectation");
          kernel_fast_available = false;
          RAFT_LOG_DEBUG(
            "No-precomputed-basediff kernel is selected, because the basediff wouldn't fit (shmem "
            "required: %zu bytes)",
            smem_size + smem_size_base_diff);
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

    uint32_t device_lut_size = use_smem_lut ? 0u : n_blocks * (pq_dim << pq_bits);
    return {reinterpret_cast<void*>(kernel),
            dim3(n_blocks, 1, 1),
            dim3(n_threads, 1, 1),
            smem_size,
            device_lut_size};
  }
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
void ivfpq_search_worker(const handle_t& handle,
                         const index<IdxT>& index,
                         uint32_t max_samples,
                         uint32_t n_probes,
                         uint32_t topK,
                         uint32_t preferred_thread_block_size,
                         uint32_t n_queries,
                         const uint32_t* clusters_to_probe,  // [n_queries, n_probes]
                         const float* query,                 // [n_queries, rot_dim]
                         IdxT* neighbors,                    // [n_queries, topK]
                         float* distances,                   // [n_queries, topK]
                         float scaling_factor,
                         rmm::mr::device_memory_resource* mr)
{
  auto stream = handle.get_stream();

  auto pq_centers      = index.pq_centers().data_handle();
  auto pq_dataset      = index.pq_dataset().data_handle();
  auto data_indices    = index.indices().data_handle();
  auto cluster_centers = index.centers_rot().data_handle();
  auto cluster_offsets = index.list_offsets().data_handle();

  bool manage_local_topk = topK <= kMaxCapacity  // depth is not too large
                           && n_probes >= 16     // not too few clusters looked up
                           &&
                           n_queries * n_probes >= 256  // overall amount of work is not too small
    ;
  auto topk_len = manage_local_topk ? n_probes * topK : max_samples;
  if (manage_local_topk) {
    RAFT_LOG_DEBUG("Fused version of the search kernel is selected (manage_local_topk == true)");
  } else {
    RAFT_LOG_DEBUG(
      "Non-fused version of the search kernel is selected (manage_local_topk == false)");
  }

  rmm::device_uvector<uint32_t> index_list_sorted_buf(0, stream, mr);
  uint32_t* index_list_sorted = nullptr;
  rmm::device_uvector<uint32_t> num_samples(n_queries, stream, mr);
  rmm::device_uvector<uint32_t> chunk_index(n_queries * n_probes, stream, mr);
  // [maxBatchSize, max_samples] or  [maxBatchSize, n_probes, topk]
  rmm::device_uvector<ScoreT> distances_buf(n_queries * topk_len, stream, mr);
  rmm::device_uvector<IdxT> neighbors_buf(0, stream, mr);
  IdxT* neighbors_ptr = nullptr;
  if (manage_local_topk) {
    neighbors_buf.resize(n_queries * topk_len, stream);
    neighbors_ptr = neighbors_buf.data();
  }

  calc_chunk_indices<IdxT>::configure(n_probes, n_queries)(
    cluster_offsets, clusters_to_probe, chunk_index.data(), num_samples.data(), stream);

  if (n_queries * n_probes > 256) {
    // Sorting index by cluster number (label).
    // The goal is to incrase the L2 cache hit rate to read the vectors
    // of a cluster by processing the cluster at the same time as much as
    // possible.
    index_list_sorted_buf.resize(n_queries * n_probes, stream);
    rmm::device_uvector<uint32_t> index_list_buf(n_queries * n_probes, stream, mr);
    rmm::device_uvector<uint32_t> cluster_labels_out(n_queries * n_probes, stream, mr);
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

  // select and run the main search kernel
  auto search_instance =
    ivfpq_compute_similarity<IdxT, ScoreT, LutT>::select(manage_local_topk,
                                                         index.pq_bits(),
                                                         index.pq_dim(),
                                                         index.rot_dim(),
                                                         preferred_thread_block_size,
                                                         n_queries,
                                                         n_probes,
                                                         topK);

  rmm::device_uvector<LutT> device_lut(search_instance.device_lut_size, stream, mr);
  search_instance(stream,
                  index.size(),
                  index.rot_dim(),
                  n_probes,
                  index.pq_bits(),
                  index.pq_dim(),
                  n_queries,
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
                  device_lut.data(),
                  distances_buf.data(),
                  neighbors_ptr);

  // Select topk vectors for each query
  rmm::device_uvector<ScoreT> topk_dists(n_queries * topK, stream, mr);
  select_topk<ScoreT, IdxT>(distances_buf.data(),
                            neighbors_ptr,
                            n_queries,
                            topk_len,
                            topK,
                            topk_dists.data(),
                            neighbors,
                            true,
                            stream,
                            mr);

  // Postprocessing
  postprocess_distances(
    distances, topk_dists.data(), index.metric(), n_queries, topK, scaling_factor, stream);
  postprocess_neighbors(neighbors,
                        manage_local_topk,
                        data_indices,
                        cluster_offsets,
                        clusters_to_probe,
                        chunk_index.data(),
                        n_queries,
                        n_probes,
                        topK,
                        stream);
}

/**
 * This structure helps selecting a proper instance of the worker search function,
 * which contains a few template parameters.
 */
template <typename IdxT>
struct ivfpq_search {
 public:
  using fun_t = void (*)(const handle_t&,
                         const ivf_pq::index<IdxT>&,
                         uint32_t,
                         uint32_t,
                         uint32_t,
                         uint32_t,
                         uint32_t,
                         const uint32_t*,
                         const float*,
                         IdxT*,
                         float*,
                         float,
                         rmm::mr::device_memory_resource*);

  /**
   * Select an instance of the ivf-pq search function based on search tuning parameters,
   * such as the look-up data type or the internal score type.
   */
  static auto fun(const search_params& params, distance::DistanceType metric) -> fun_t
  {
    return fun_try_score_t(params, metric);
  }

 private:
  template <typename ScoreT>
  static auto fun_try_lut_t(const search_params& params, distance::DistanceType metric) -> fun_t
  {
    bool signed_metric = false;
    switch (metric) {
      case raft::distance::DistanceType::InnerProduct: signed_metric = true; break;
      default: break;
    }

    switch (params.lut_dtype) {
      case CUDA_R_32F: return ivfpq_search_worker<ScoreT, float, IdxT>;
      case CUDA_R_16F: return ivfpq_search_worker<ScoreT, half, IdxT>;
      case CUDA_R_8U:
      case CUDA_R_8I:
        if (signed_metric) {
          return ivfpq_search_worker<float, fp_8bit<5, true>, IdxT>;
        } else {
          return ivfpq_search_worker<float, fp_8bit<5, false>, IdxT>;
        }
      default: RAFT_FAIL("Unexpected lut_dtype (%d)", int(params.lut_dtype));
    }
  }

  static auto fun_try_score_t(const search_params& params, distance::DistanceType metric) -> fun_t
  {
    switch (params.internal_distance_dtype) {
      case CUDA_R_32F: return fun_try_lut_t<float>(params, metric);
      case CUDA_R_16F: return fun_try_lut_t<half>(params, metric);
      default:
        RAFT_FAIL("Unexpected internal_distance_dtype (%d)", int(params.internal_distance_dtype));
    }
  }
};

/**
 * A heuristic for bounding the number of queries per batch, to improve GPU utilization.
 * (based on the number of SMs and the work size).
 *
 * @param n_queries number of queries hoped to be processed at once.
 *                  (maximum value for the returned batch size)
 *
 * @return maximum recommended batch size.
 */
inline auto get_max_batch_size(uint32_t n_queries) -> uint32_t
{
  uint32_t max_batch_size         = n_queries;
  uint32_t n_ctas_total           = getMultiProcessorCount() * 2;
  uint32_t n_ctas_total_per_batch = n_ctas_total / max_batch_size;
  float utilization               = float(n_ctas_total_per_batch * max_batch_size) / n_ctas_total;
  if (n_ctas_total_per_batch > 1 || (n_ctas_total_per_batch == 1 && utilization < 0.6)) {
    uint32_t n_ctas_total_per_batch_1 = n_ctas_total_per_batch + 1;
    uint32_t max_batch_size_1         = n_ctas_total / n_ctas_total_per_batch_1;
    float utilization_1 = float(n_ctas_total_per_batch_1 * max_batch_size_1) / n_ctas_total;
    if (utilization < utilization_1) { max_batch_size = max_batch_size_1; }
  }
  return max_batch_size;
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

  switch (utils::check_pointer_residency(queries, neighbors, distances)) {
    case utils::pointer_residency::device_only:
    case utils::pointer_residency::host_and_device: break;
    default: RAFT_FAIL("all pointers must be accessible from the device.");
  }

  auto stream = handle.get_stream();

  auto dim      = index.dim();
  auto dim_ext  = index.dim_ext();
  auto n_probes = std::min<uint32_t>(params.n_probes, index.n_lists());

  IdxT max_samples = 0;
  {
    IdxT offset_worst_case = 0;
    auto cluster_offsets   = index.list_offsets().data_handle();
    copy(&max_samples, cluster_offsets + n_probes, 1, stream);
    if (n_probes < index.n_nonempty_lists()) {
      copy(&offset_worst_case, cluster_offsets + index.n_nonempty_lists() - n_probes, 1, stream);
    }
    handle.sync_stream();
    max_samples      = Pow2<128>::roundUp(max_samples);
    IdxT min_samples = index.size() - offset_worst_case;
    if (IdxT{k} > min_samples) {
      RAFT_LOG_WARN(
        "n_probes is too small to get top-k results reliably (n_probes: %u, k: %u, n_samples "
        "(worst_case): %zu).",
        n_probes,
        k,
        static_cast<uint64_t>(min_samples));
    }
    RAFT_EXPECTS(max_samples <= IdxT(std::numeric_limits<uint32_t>::max()),
                 "The maximum sample size is too big.");
  }

  auto pool_guard = raft::get_pool_memory_resource(mr, n_queries * n_probes * k * 16);
  if (pool_guard) {
    RAFT_LOG_DEBUG("ivf_pq::search: using pool memory resource with initial size %zu bytes",
                   pool_guard->pool_size());
  }

  // Maximum number of query vectors to search at the same time.
  const auto max_queries = std::min<uint32_t>(std::max<uint32_t>(n_queries, 1), 4096);
  auto max_batch_size    = get_max_batch_size(max_queries);

  rmm::device_uvector<float> float_queries(max_queries * dim_ext, stream, mr);
  rmm::device_uvector<float> rot_queries(max_queries * index.rot_dim(), stream, mr);
  rmm::device_uvector<uint32_t> clusters_to_probe(max_queries * params.n_probes, stream, mr);

  auto search_instance = ivfpq_search<IdxT>::fun(params, index.metric());

  for (uint32_t offset_q = 0; offset_q < n_queries; offset_q += max_queries) {
    uint32_t queries_batch = min(max_queries, n_queries - offset_q);

    select_clusters(handle,
                    clusters_to_probe.data(),
                    float_queries.data(),
                    queries_batch,
                    params.n_probes,
                    index.n_lists(),
                    dim,
                    dim_ext,
                    index.metric(),
                    queries + static_cast<size_t>(dim) * offset_q,
                    index.centers().data_handle(),
                    mr);

    // Rotate queries
    float alpha = 1.0;
    float beta  = 0.0;
    linalg::gemm(handle,
                 true,
                 false,
                 index.rot_dim(),
                 queries_batch,
                 dim,
                 &alpha,
                 index.rotation_matrix().data_handle(),
                 dim,
                 float_queries.data(),
                 dim_ext,
                 &beta,
                 rot_queries.data(),
                 index.rot_dim(),
                 stream);

    for (uint32_t offset_b = 0; offset_b < queries_batch; offset_b += max_batch_size) {
      uint32_t batch_size = min(max_batch_size, queries_batch - offset_b);
      /* The distance calculation is done in the rotated/transformed space;
         as long as `index.rotation_matrix()` is orthogonal, the distances and thus results are
         preserved.
       */
      search_instance(handle,
                      index,
                      max_samples,
                      params.n_probes,
                      k,
                      params.preferred_thread_block_size,
                      batch_size,
                      clusters_to_probe.data() + uint64_t(params.n_probes) * offset_b,
                      rot_queries.data() + uint64_t(index.rot_dim()) * offset_b,
                      neighbors + uint64_t(k) * (offset_q + offset_b),
                      distances + uint64_t(k) * (offset_q + offset_b),
                      utils::config<T>::kDivisor / utils::config<float>::kDivisor,
                      mr);
    }
  }
}

}  // namespace raft::spatial::knn::ivf_pq::detail
