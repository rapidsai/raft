/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <raft/spatial/knn/detail/ann_utils.cuh>

#include <raft/neighbors/ivf_pq_types.hpp>

#include <raft/core/cudart_utils.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/operators.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/detail/select_k.cuh>
#include <raft/matrix/detail/select_warpsort.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/device_atomics.cuh>
#include <raft/util/device_loads_stores.cuh>
#include <raft/util/pow2_utils.cuh>
#include <raft/util/vectorized.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <cub/cub.cuh>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <cuda_fp16.h>

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
  HDI explicit operator half() const { return half(fp_8bit2float(*this)); }

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
 * followed by select_k to select the clusters to probe. There's no need to return the similarity
 * scores here.
 */
template <typename T>
void select_clusters(raft::device_resources const& handle,
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
  /* NOTE[qc_distances]

  We compute query-center distances to choose the clusters to probe.
  We accomplish that with just one GEMM operation thanks to some preprocessing:

    L2 distance:
      cluster_centers[i, dim()] contains the squared norm of the center vector i;
      we extend the dimension K of the GEMM to compute it together with all the dot products:

      `qc_distances[i, j] = |cluster_centers[j]|^2 - 2 * (queries[i], cluster_centers[j])`

      This is a monotonous mapping of the proper L2 distance.

    IP distance:
      `qc_distances[i, j] = - (queries[i], cluster_centers[j])`

      This is a negative inner-product distance. We minimize it to find the similar clusters.

      NB: qc_distances is NOT used further in ivfpq_search.
 */
  float norm_factor;
  switch (metric) {
    case raft::distance::DistanceType::L2SqrtExpanded:
    case raft::distance::DistanceType::L2Expanded: norm_factor = 1.0 / -2.0; break;
    case raft::distance::DistanceType::InnerProduct: norm_factor = 0.0; break;
    default: RAFT_FAIL("Unsupported distance type %d.", int(metric));
  }
  auto float_queries_view =
    raft::make_device_vector_view<float, uint32_t>(float_queries, dim_ext * n_queries);
  linalg::map_offset(
    handle, float_queries_view, [queries, dim, dim_ext, norm_factor] __device__(uint32_t ix) {
      uint32_t col = ix % dim_ext;
      uint32_t row = ix / dim_ext;
      return col < dim ? utils::mapping<float>{}(queries[col + dim * row]) : norm_factor;
    });

  float alpha;
  float beta;
  uint32_t gemm_k = dim;
  switch (metric) {
    case raft::distance::DistanceType::L2SqrtExpanded:
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
  rmm::device_uvector<float> qc_distances(n_queries * n_lists, stream, mr);
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
  matrix::detail::select_k<float, uint32_t>(qc_distances.data(),
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
template <int BlockDim>
__launch_bounds__(BlockDim) __global__
  void calc_chunk_indices_kernel(uint32_t n_probes,
                                 const uint32_t* cluster_sizes,      // [n_clusters]
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
    auto chunk = probe_ix < n_probes ? cluster_sizes[label] : 0u;
    if (threadIdx.x == 0) { chunk += total; }
    block_scan(shm).InclusiveSum(chunk, chunk, total);
    __syncthreads();
    if (probe_ix < n_probes) { chunk_indices[probe_ix] = chunk; }
  }
  // save the total size
  if (threadIdx.x == 0) { n_samples[blockIdx.x] = total; }
}

struct calc_chunk_indices {
 public:
  struct configured {
    void* kernel;
    dim3 block_dim;
    dim3 grid_dim;
    uint32_t n_probes;

    inline void operator()(const uint32_t* cluster_sizes,
                           const uint32_t* clusters_to_probe,
                           uint32_t* chunk_indices,
                           uint32_t* n_samples,
                           rmm::cuda_stream_view stream)
    {
      void* args[] =  // NOLINT
        {&n_probes, &cluster_sizes, &clusters_to_probe, &chunk_indices, &n_samples};
      RAFT_CUDA_TRY(cudaLaunchKernel(kernel, grid_dim, block_dim, args, 0, stream));
    }
  };

  static inline auto configure(uint32_t n_probes, uint32_t n_queries) -> configured
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
    return {reinterpret_cast<void*>(calc_chunk_indices_kernel<BlockDim>),
            dim3(BlockDim, 1, 1),
            dim3(n_queries, 1, 1),
            n_probes};
  }
};

/**
 * Look up the chunk id corresponding to the sample index.
 *
 * Each query vector was compared to all the vectors from n_probes clusters, and sample_ix is an
 * ordered number of one of such vectors. This function looks up to which chunk it belongs,
 * and returns the index within the chunk (which is also an index within a cluster).
 *
 * @param[inout] sample_ix
 *   input: the offset of the sample in the batch;
 *   output: the offset inside the chunk (probe) / selected cluster.
 * @param[in] n_probes number of probes
 * @param[in] chunk_indices offsets of the chunks within the batch [n_probes]
 * @return chunk index (== n_probes when the input index is not in the valid range,
 *    which can happen if there is not enough data to output in the selected clusters).
 */
__device__ inline auto find_chunk_ix(uint32_t& sample_ix,  // NOLINT
                                     uint32_t n_probes,
                                     const uint32_t* chunk_indices) -> uint32_t
{
  uint32_t ix_min = 0;
  uint32_t ix_max = n_probes;
  do {
    uint32_t i = (ix_min + ix_max) / 2;
    if (chunk_indices[i] <= sample_ix) {
      ix_min = i + 1;
    } else {
      ix_max = i;
    }
  } while (ix_min < ix_max);
  if (ix_min > 0) { sample_ix -= chunk_indices[ix_min - 1]; }
  return ix_min;
}

template <int BlockDim, typename IdxT>
__launch_bounds__(BlockDim) __global__
  void postprocess_neighbors_kernel(IdxT* neighbors_out,                // [n_queries, topk]
                                    const uint32_t* neighbors_in,       // [n_queries, topk]
                                    const IdxT* const* db_indices,      // [n_clusters][..]
                                    const uint32_t* clusters_to_probe,  // [n_queries, n_probes]
                                    const uint32_t* chunk_indices,      // [n_queries, n_probes]
                                    uint32_t n_queries,
                                    uint32_t n_probes,
                                    uint32_t topk)
{
  const uint64_t i        = threadIdx.x + BlockDim * uint64_t(blockIdx.x);
  const uint32_t query_ix = i / uint64_t(topk);
  if (query_ix >= n_queries) { return; }
  const uint32_t k = i % uint64_t(topk);
  neighbors_in += query_ix * topk;
  neighbors_out += query_ix * topk;
  chunk_indices += query_ix * n_probes;
  clusters_to_probe += query_ix * n_probes;
  uint32_t data_ix        = neighbors_in[k];
  const uint32_t chunk_ix = find_chunk_ix(data_ix, n_probes, chunk_indices);
  const bool valid        = chunk_ix < n_probes;
  neighbors_out[k] =
    valid ? db_indices[clusters_to_probe[chunk_ix]][data_ix] : ivf_pq::kOutOfBoundsRecord<IdxT>;
}

/**
 * Transform found sample indices into the corresponding database indices
 * (as stored in index.indices()).
 * The sample indices are the record indices as they appear in the database view formed by the
 * probed clusters / defined by the `chunk_indices`.
 * We assume the searched sample sizes (for a single query) fit into `uint32_t`.
 */
template <typename IdxT>
void postprocess_neighbors(IdxT* neighbors_out,                // [n_queries, topk]
                           const uint32_t* neighbors_in,       // [n_queries, topk]
                           const IdxT* const* db_indices,      // [n_clusters][..]
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
    <<<pn_blocks, kPNThreads, 0, stream>>>(neighbors_out,
                                           neighbors_in,
                                           db_indices,
                                           clusters_to_probe,
                                           chunk_indices,
                                           n_queries,
                                           n_probes,
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
      linalg::unaryOp(out,
                      in,
                      len,
                      raft::compose_op(raft::mul_const_op<float>{scaling_factor * scaling_factor},
                                       raft::cast_op<float>{}),
                      stream);
    } break;
    case distance::DistanceType::L2SqrtUnexpanded:
    case distance::DistanceType::L2SqrtExpanded: {
      linalg::unaryOp(
        out,
        in,
        len,
        raft::compose_op{
          raft::mul_const_op<float>{scaling_factor}, raft::sqrt_op{}, raft::cast_op<float>{}},
        stream);
    } break;
    case distance::DistanceType::InnerProduct: {
      linalg::unaryOp(out,
                      in,
                      len,
                      raft::compose_op(raft::mul_const_op<float>{-scaling_factor * scaling_factor},
                                       raft::cast_op<float>{}),
                      stream);
    } break;
    default: RAFT_FAIL("Unexpected metric.");
  }
}

template <typename T, typename IdxT>
struct dummy_block_sort_t {
  using queue_t = matrix::detail::select::warpsort::warp_sort_distributed<WarpSize, true, T, IdxT>;
  template <typename... Args>
  __device__ dummy_block_sort_t(int k, Args...){};
};

template <int Capacity, typename T, typename IdxT>
struct pq_block_sort {
  using type = matrix::detail::select::warpsort::
    block_sort<matrix::detail::select::warpsort::warp_sort_distributed, Capacity, true, T, IdxT>;
};

template <typename T, typename IdxT>
struct pq_block_sort<0, T, IdxT> : dummy_block_sort_t<T, IdxT> {
  using type = dummy_block_sort_t<T, IdxT>;
};

template <int Capacity, typename T, typename IdxT>
using block_sort_t = typename pq_block_sort<Capacity, T, IdxT>::type;

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
 * @param n_rows the number of records in the dataset
 * @param dim the dimensionality of the data (NB: after rotation transform, i.e. `index.rot_dim()`).
 * @param n_probes the number of clusters to search for each query
 * @param pq_dim
 *   The dimensionality of an encoded vector after compression by PQ.
 * @param n_queries the number of queries.
 * @param metric the distance type.
 * @param codebook_kind Defines the way PQ codebooks have been trained.
 * @param topk the `k` in the select top-k.
 * @param max_samples the size of the output for a single query.
 * @param cluster_centers
 *   The device pointer to the cluster centers in the original space (NB: after rotation)
 *   [n_clusters, dim].
 * @param pq_centers
 *   The device pointer to the cluster centers in the PQ space
 *   [pq_dim, pq_book_size, pq_len] or [n_clusters, pq_book_size, pq_len,].
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
          uint32_t PqBits,
          int Capacity,
          bool PrecompBaseDiff,
          bool EnableSMemLut>
__global__ void compute_similarity_kernel(uint32_t n_rows,
                                          uint32_t dim,
                                          uint32_t n_probes,
                                          uint32_t pq_dim,
                                          uint32_t n_queries,
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
                                          LutT* lut_scores,
                                          OutT* _out_scores,
                                          uint32_t* _out_indices)
{
  /* Shared memory:

    * lut_scores: lookup table (LUT) of size = `pq_dim << PqBits`  (when EnableSMemLut)
    * base_diff: size = dim (which is equal to `pq_dim * pq_len`)  or dim*2
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

  float* base_diff = nullptr;
  if constexpr (PrecompBaseDiff) {
    if constexpr (EnableSMemLut) {
      base_diff = reinterpret_cast<float*>(lut_scores + lut_size);
    } else {
      base_diff = reinterpret_cast<float*>(smem_buf);
    }
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
      out_scores  = _out_scores + topk * (probe_ix + (n_probes * query_ix));
      out_indices = _out_indices + topk * (probe_ix + (n_probes * query_ix));
    } else {
      // Store all calculated distances to out_scores
      out_scores = _out_scores + max_samples * query_ix;
    }
    uint32_t label              = cluster_labels[n_probes * query_ix + probe_ix];
    const float* cluster_center = cluster_centers + (dim * label);
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
            base_diff[i] = query[i] - cluster_center[i];
          }
        } break;
        case distance::DistanceType::InnerProduct: {
          float2 pvals;
          for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
            pvals.x                                 = query[i];
            pvals.y                                 = cluster_center[i] * pvals.x;
            reinterpret_cast<float2*>(base_diff)[i] = pvals;
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
                diff = base_diff[j];
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
                float2 pvals = reinterpret_cast<float2*>(base_diff)[j];
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
    local_topk_t block_topk(topk, nullptr, query_kth);
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

    // Compute a distance for each sample
    for (uint32_t i = threadIdx.x; i < n_samples_aligned;
         i += blockDim.x, pq_thread_data += pq_line_width) {
      OutT score = kDummy;
      bool valid = i < n_samples;
      if (valid) {
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
template <typename OutT, typename LutT>
using compute_similarity_kernel_t =
  decltype(&compute_similarity_kernel<OutT, LutT, 8, 0, true, true>);

// The config struct lifts the runtime parameters to the template parameters
template <typename OutT, typename LutT, bool PrecompBaseDiff, bool EnableSMemLut>
struct compute_similarity_kernel_config {
 public:
  static auto get(uint32_t pq_bits, uint32_t k_max) -> compute_similarity_kernel_t<OutT, LutT>
  {
    return kernel_choose_bits(pq_bits, k_max);
  }

 private:
  static auto kernel_choose_bits(uint32_t pq_bits, uint32_t k_max)
    -> compute_similarity_kernel_t<OutT, LutT>
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
  static auto kernel_try_capacity(uint32_t k_max) -> compute_similarity_kernel_t<OutT, LutT>
  {
    if constexpr (Capacity > 0) {
      if (k_max == 0 || k_max > Capacity) { return kernel_try_capacity<PqBits, 0>(k_max); }
    }
    if constexpr (Capacity > 1) {
      if (k_max * 2 <= Capacity) { return kernel_try_capacity<PqBits, (Capacity / 2)>(k_max); }
    }
    return compute_similarity_kernel<OutT, LutT, PqBits, Capacity, PrecompBaseDiff, EnableSMemLut>;
  }
};

// A standalone accessor function is necessary to make sure template specializations work correctly
// (we "extern template" this function)
template <typename OutT, typename LutT, bool PrecompBaseDiff, bool EnableSMemLut>
auto get_compute_similarity_kernel(uint32_t pq_bits, uint32_t k_max)
  -> compute_similarity_kernel_t<OutT, LutT>
{
  return compute_similarity_kernel_config<OutT, LutT, PrecompBaseDiff, EnableSMemLut>::get(pq_bits,
                                                                                           k_max);
}

/**
 * An approximation to the number of times each cluster appears in a batched sample.
 *
 * If the pairs (probe_ix, query_ix) are sorted by the probe_ix, there is a good chance that
 * the same probe_ix (cluster) is processed by several blocks on a single SM. This greatly
 * increases the L1 cache hit rate (i.e. increases the data locality).
 *
 * This function gives an estimate of how many times a specific cluster may appear in the
 * batch. Thus, it gives a practical limit to how many blocks should be active on the same SM
 * to improve the L1 cache hit rate.
 */
constexpr inline auto expected_probe_coresidency(uint32_t n_clusters,
                                                 uint32_t n_probes,
                                                 uint32_t n_queries) -> uint32_t
{
  /*
    Let say:
      n = n_clusters
      k = n_probes
      m = n_queries
      r = # of times a specific block appears in the batched sample.

    Then, r has the Binomial distribution (p = k / n):
      P(r) = C(m,r) * k^r * (n - k)^(m - r) / n^m
      E[r] = m * k / n
      E[r | r > 0] = m * k / n / (1 - (1 - k/n)^m)

    The latter can be approximated by a much simpler formula, assuming (k / n) -> 0:
      E[r | r > 0] = 1 + (m - 1) * k / (2 * n) + O( (k/n)^2 )
   */
  return 1 + (n_queries - 1) * n_probes / (2 * n_clusters);
}

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

/** Select an appropriate kernel instance and launch parameters. */
template <typename OutT, typename LutT>
struct compute_similarity {
  /** Estimate the occupancy for the given kernel on the given device. */
  struct occupancy_t {
    using shmem_unit = Pow2<128>;

    int blocks_per_sm = 0;
    double occupancy  = 0.0;
    double shmem_use  = 1.0;

    inline occupancy_t() = default;
    inline occupancy_t(size_t smem,
                       uint32_t n_threads,
                       compute_similarity_kernel_t<OutT, LutT> kernel,
                       const cudaDeviceProp& dev_props)
    {
      RAFT_CUDA_TRY(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, n_threads, smem));
      occupancy = double(blocks_per_sm * n_threads) / double(dev_props.maxThreadsPerMultiProcessor);
      shmem_use = double(shmem_unit::roundUp(smem) * blocks_per_sm) /
                  double(dev_props.sharedMemPerMultiprocessor);
    }
  };

  struct selected {
    compute_similarity_kernel_t<OutT, LutT> kernel;
    dim3 grid_dim;
    dim3 block_dim;
    size_t smem_size;
    size_t device_lut_size;

    template <typename... Args>
    void operator()(rmm::cuda_stream_view stream, Args... args)
    {
      kernel<<<grid_dim, block_dim, smem_size, stream>>>(args...);
      RAFT_CHECK_CUDA(stream);
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
   * @param locality_hint
   *    beyond this limit do not consider increasing the number of active blocks per SM
   *    would improve locality anymore.
   */
  static inline auto select(const cudaDeviceProp& dev_props,
                            bool manage_local_topk,
                            int locality_hint,
                            double preferred_shmem_carveout,
                            uint32_t pq_bits,
                            uint32_t pq_dim,
                            uint32_t precomp_data_count,
                            uint32_t n_queries,
                            uint32_t n_probes,
                            uint32_t topk) -> selected
  {
    // Shared memory for storing the lookup table
    size_t lut_mem = sizeof(LutT) * (pq_dim << pq_bits);
    // Shared memory for storing pre-computed pieces to speedup the lookup table construction
    // (e.g. the distance between a cluster center and the query for L2).
    size_t bdf_mem = sizeof(float) * precomp_data_count;
    // Shared memory for the fused top-k component; it may overlap with the other uses of shared
    // memory and depends on the number of threads.
    struct ltk_mem_t {
      uint32_t subwarp_size;
      uint32_t topk;
      bool manage_local_topk;
      ltk_mem_t(bool manage_local_topk, uint32_t topk)
        : manage_local_topk(manage_local_topk), topk(topk)
      {
        subwarp_size = WarpSize;
        while (topk * 2 <= subwarp_size) {
          subwarp_size /= 2;
        }
      }

      [[nodiscard]] auto operator()(uint32_t n_threads) const -> size_t
      {
        return manage_local_topk ? matrix::detail::select::warpsort::
                                     template calc_smem_size_for_block_wide<OutT, uint32_t>(
                                       n_threads / subwarp_size, topk)
                                 : 0;
      }
    } ltk_mem{manage_local_topk, topk};

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
    auto conf_fast        = get_compute_similarity_kernel<OutT, LutT, true, true>;
    auto conf_no_basediff = get_compute_similarity_kernel<OutT, LutT, false, true>;
    auto conf_no_smem_lut = get_compute_similarity_kernel<OutT, LutT, true, false>;
    auto topk_or_zero     = manage_local_topk ? topk : 0u;
    std::array candidates{
      std::make_tuple(conf_fast(pq_bits, topk_or_zero), lut_mem + bdf_mem, true),
      std::make_tuple(conf_no_basediff(pq_bits, topk_or_zero), lut_mem, true),
      std::make_tuple(conf_no_smem_lut(pq_bits, topk_or_zero), bdf_mem, false)};

    // we may allow slightly lower than 100% occupancy;
    constexpr double kTargetOccupancy = 0.75;
    // This struct is used to select the better candidate
    occupancy_t selected_perf{};
    selected selected_config;
    for (auto [kernel, smem_size_const, lut_is_in_shmem] : candidates) {
      if (smem_size_const > dev_props.sharedMemPerBlockOptin) {
        // Even a single block cannot fit into an SM due to shmem requirements. Skip the candidate.
        continue;
      }

      // First, we set the carveout hint to the preferred value. The driver will increase this if
      // needed to run at least one block per SM. At the same time, if more blocks fit into one SM,
      // this carveout value will limit the calculated occupancy. When we're done selecting the best
      // launch configuration, we will tighten the carveout once more, based on the final memory
      // usage and occupancy.
      const int max_carveout =
        estimate_carveout(preferred_shmem_carveout, smem_size_const, dev_props);
      RAFT_CUDA_TRY(
        cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, max_carveout));

      // Get the theoretical maximum possible number of threads per block
      cudaFuncAttributes kernel_attrs;
      RAFT_CUDA_TRY(cudaFuncGetAttributes(&kernel_attrs, kernel));
      uint32_t n_threads =
        round_down_safe<uint32_t>(kernel_attrs.maxThreadsPerBlock, n_threads_gty);

      // Actual required shmem depens on the number of threads
      size_t smem_size = max(smem_size_const, ltk_mem(n_threads));

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

      occupancy_t cur(smem_size, n_threads, kernel, dev_props);
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
            auto smem_size_tmp = max(smem_size_const, ltk_mem(n_threads_tmp));
            occupancy_t tmp(smem_size_tmp, n_threads_tmp, kernel, dev_props);
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
};

inline auto is_local_topk_feasible(uint32_t k, uint32_t n_probes, uint32_t n_queries) -> bool
{
  if (k > kMaxCapacity) { return false; }             // warp_sort not possible
  if (n_probes <= 16) { return false; }               // too few clusters
  if (n_queries * n_probes <= 256) { return false; }  // overall amount of work is too small
  return true;
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
void ivfpq_search_worker(raft::device_resources const& handle,
                         const index<IdxT>& index,
                         uint32_t max_samples,
                         uint32_t n_probes,
                         uint32_t topK,
                         uint32_t n_queries,
                         const uint32_t* clusters_to_probe,  // [n_queries, n_probes]
                         const float* query,                 // [n_queries, rot_dim]
                         IdxT* neighbors,                    // [n_queries, topK]
                         float* distances,                   // [n_queries, topK]
                         float scaling_factor,
                         double preferred_shmem_carveout,
                         rmm::mr::device_memory_resource* mr)
{
  auto stream = handle.get_stream();

  bool manage_local_topk = is_local_topk_feasible(topK, n_probes, n_queries);
  auto topk_len          = manage_local_topk ? n_probes * topK : max_samples;
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
  rmm::device_uvector<uint32_t> neighbors_buf(0, stream, mr);
  uint32_t* neighbors_ptr = nullptr;
  if (manage_local_topk) {
    neighbors_buf.resize(n_queries * topk_len, stream);
    neighbors_ptr = neighbors_buf.data();
  }
  rmm::device_uvector<uint32_t> neighbors_uint32_buf(0, stream, mr);
  uint32_t* neighbors_uint32 = nullptr;
  if constexpr (sizeof(IdxT) == sizeof(uint32_t)) {
    neighbors_uint32 = reinterpret_cast<uint32_t*>(neighbors);
  } else {
    neighbors_uint32_buf.resize(n_queries * topK, stream);
    neighbors_uint32 = neighbors_uint32_buf.data();
  }

  calc_chunk_indices::configure(n_probes, n_queries)(index.list_sizes().data_handle(),
                                                     clusters_to_probe,
                                                     chunk_index.data(),
                                                     num_samples.data(),
                                                     stream);

  auto coresidency = expected_probe_coresidency(index.n_lists(), n_probes, n_queries);

  if (coresidency > 1) {
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
  uint32_t precomp_data_count = 0;
  switch (index.metric()) {
    case distance::DistanceType::L2SqrtExpanded:
    case distance::DistanceType::L2SqrtUnexpanded:
    case distance::DistanceType::L2Unexpanded:
    case distance::DistanceType::L2Expanded: {
      // stores basediff (query[i] - center[i])
      precomp_data_count = index.rot_dim();
    } break;
    case distance::DistanceType::InnerProduct: {
      // stores two components (query[i] * center[i], query[i] * center[i])
      precomp_data_count = index.rot_dim() * 2;
    } break;
    default: {
      RAFT_FAIL("Unsupported metric");
    } break;
  }

  auto search_instance = compute_similarity<ScoreT, LutT>::select(handle.get_device_properties(),
                                                                  manage_local_topk,
                                                                  coresidency,
                                                                  preferred_shmem_carveout,
                                                                  index.pq_bits(),
                                                                  index.pq_dim(),
                                                                  precomp_data_count,
                                                                  n_queries,
                                                                  n_probes,
                                                                  topK);

  rmm::device_uvector<LutT> device_lut(search_instance.device_lut_size, stream, mr);
  rmm::device_uvector<float> query_kths(0, stream, mr);
  if (manage_local_topk) {
    query_kths.resize(n_queries, stream);
    thrust::fill_n(handle.get_thrust_policy(),
                   query_kths.data(),
                   n_queries,
                   float(dummy_block_sort_t<ScoreT, IdxT>::queue_t::kDummy));
  }
  search_instance(stream,
                  index.size(),
                  index.rot_dim(),
                  n_probes,
                  index.pq_dim(),
                  n_queries,
                  index.metric(),
                  index.codebook_kind(),
                  topK,
                  max_samples,
                  index.centers_rot().data_handle(),
                  index.pq_centers().data_handle(),
                  index.data_ptrs().data_handle(),
                  clusters_to_probe,
                  chunk_index.data(),
                  query,
                  index_list_sorted,
                  query_kths.data(),
                  device_lut.data(),
                  distances_buf.data(),
                  neighbors_ptr);

  // Select topk vectors for each query
  rmm::device_uvector<ScoreT> topk_dists(n_queries * topK, stream, mr);
  matrix::detail::select_k<ScoreT, uint32_t>(distances_buf.data(),
                                             neighbors_ptr,
                                             n_queries,
                                             topk_len,
                                             topK,
                                             topk_dists.data(),
                                             neighbors_uint32,
                                             true,
                                             stream,
                                             mr);

  // Postprocessing
  postprocess_distances(
    distances, topk_dists.data(), index.metric(), n_queries, topK, scaling_factor, stream);
  postprocess_neighbors(neighbors,
                        neighbors_uint32,
                        index.inds_ptrs().data_handle(),
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
  using fun_t = decltype(&ivfpq_search_worker<float, float, IdxT>);

  /**
   * Select an instance of the ivf-pq search function based on search tuning parameters,
   * such as the look-up data type or the internal score type.
   */
  static auto fun(const search_params& params, distance::DistanceType metric) -> fun_t
  {
    return fun_try_score_t(params, metric);
  }

 private:
  template <typename ScoreT, typename LutT>
  static auto filter_reasonable_instances(const search_params& params) -> fun_t
  {
    if constexpr (sizeof(ScoreT) >= sizeof(LutT)) {
      return ivfpq_search_worker<ScoreT, LutT, IdxT>;
    } else {
      RAFT_FAIL(
        "Unexpected lut_dtype / internal_distance_dtype combination (%d, %d). "
        "Size of the internal_distance_dtype should be not smaller than the size of the lut_dtype.",
        int(params.lut_dtype),
        int(params.internal_distance_dtype));
    }
  }

  template <typename ScoreT>
  static auto fun_try_lut_t(const search_params& params, distance::DistanceType metric) -> fun_t
  {
    bool signed_metric = false;
    switch (metric) {
      case raft::distance::DistanceType::InnerProduct: signed_metric = true; break;
      default: break;
    }

    switch (params.lut_dtype) {
      case CUDA_R_32F: return filter_reasonable_instances<ScoreT, float>(params);
      case CUDA_R_16F: return filter_reasonable_instances<ScoreT, half>(params);
      case CUDA_R_8U:
      case CUDA_R_8I:
        if (signed_metric) {
          return filter_reasonable_instances<ScoreT, fp_8bit<5, true>>(params);
        } else {
          return filter_reasonable_instances<ScoreT, fp_8bit<5, false>>(params);
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
 * @param k top-k
 * @param n_probes number of selected clusters per query
 * @param n_queries number of queries hoped to be processed at once.
 *                  (maximum value for the returned batch size)
 * @param max_samples maximum possible number of samples to be processed for the given `n_probes`
 *
 * @return maximum recommended batch size.
 */
inline auto get_max_batch_size(uint32_t k,
                               uint32_t n_probes,
                               uint32_t n_queries,
                               uint32_t max_samples) -> uint32_t
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
  // Check in the tmp distance buffer is not too big
  auto ws_size = [k, n_probes, max_samples](uint32_t bs) -> uint64_t {
    return uint64_t(is_local_topk_feasible(k, n_probes, bs) ? k * n_probes : max_samples) * bs;
  };
  constexpr uint64_t kMaxWsSize = 1024 * 1024 * 1024;
  if (ws_size(max_batch_size) > kMaxWsSize) {
    uint32_t smaller_batch_size = bound_by_power_of_two(max_batch_size);
    // gradually reduce the batch size until we fit into the max size limit.
    while (smaller_batch_size > 1 && ws_size(smaller_batch_size) > kMaxWsSize) {
      smaller_batch_size >>= 1;
    }
    return smaller_batch_size;
  }
  return max_batch_size;
}

/** See raft::spatial::knn::ivf_pq::search docs */
template <typename T, typename IdxT>
inline void search(raft::device_resources const& handle,
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
    "ivf_pq::search(n_queries = %u, n_probes = %u, k = %u, dim = %zu)",
    n_queries,
    params.n_probes,
    k,
    index.dim());

  RAFT_EXPECTS(
    params.internal_distance_dtype == CUDA_R_16F || params.internal_distance_dtype == CUDA_R_32F,
    "internal_distance_dtype must be either CUDA_R_16F or CUDA_R_32F");
  RAFT_EXPECTS(params.lut_dtype == CUDA_R_16F || params.lut_dtype == CUDA_R_32F ||
                 params.lut_dtype == CUDA_R_8U,
               "lut_dtype must be CUDA_R_16F, CUDA_R_32F or CUDA_R_8U");
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

  uint32_t max_samples = 0;
  {
    IdxT ms = Pow2<128>::roundUp(index.accum_sorted_sizes()(n_probes));
    RAFT_EXPECTS(ms <= IdxT(std::numeric_limits<uint32_t>::max()),
                 "The maximum sample size is too big.");
    max_samples = ms;
  }

  auto pool_guard = raft::get_pool_memory_resource(mr, n_queries * n_probes * k * 16);
  if (pool_guard) {
    RAFT_LOG_DEBUG("ivf_pq::search: using pool memory resource with initial size %zu bytes",
                   pool_guard->pool_size());
  }

  // Maximum number of query vectors to search at the same time.
  const auto max_queries = std::min<uint32_t>(std::max<uint32_t>(n_queries, 1), 4096);
  auto max_batch_size    = get_max_batch_size(k, n_probes, max_queries, max_samples);

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
                      batch_size,
                      clusters_to_probe.data() + uint64_t(params.n_probes) * offset_b,
                      rot_queries.data() + uint64_t(index.rot_dim()) * offset_b,
                      neighbors + uint64_t(k) * (offset_q + offset_b),
                      distances + uint64_t(k) * (offset_q + offset_b),
                      utils::config<T>::kDivisor / utils::config<float>::kDivisor,
                      params.preferred_shmem_carveout,
                      mr);
    }
  }
}

}  // namespace raft::neighbors::ivf_pq::detail
