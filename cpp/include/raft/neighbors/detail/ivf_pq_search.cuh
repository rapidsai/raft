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

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/spatial/knn/detail/ann_utils.cuh>

#include <raft/neighbors/detail/ivf_pq_compute_similarity.cuh>
#include <raft/neighbors/detail/ivf_pq_dummy_block_sort.cuh>
#include <raft/neighbors/detail/ivf_pq_fp_8bit.cuh>
#include <raft/neighbors/ivf_pq_types.hpp>
#include <raft/neighbors/sample_filter_types.hpp>

#include <raft/core/cudart_utils.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/detail/device_memory_resource.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>
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

#include <cuda_fp16.h>

#include <optional>

namespace raft::neighbors::ivf_pq::detail {

using namespace raft::spatial::knn::detail;  // NOLINT

/**
 * Select the clusters to probe and, as a side-effect, translate the queries type `T -> float`
 *
 * Assuming the number of clusters is not that big (a few thousands), we do a plain GEMM
 * followed by select_k to select the clusters to probe. There's no need to return the similarity
 * scores here.
 */
template <typename T>
void select_clusters(raft::resources const& handle,
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
  auto stream = resource::get_cuda_stream(handle);
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
  matrix::detail::select_k<float, uint32_t>(handle,
                                            qc_distances.data(),
                                            nullptr,
                                            n_queries,
                                            n_lists,
                                            n_probes,
                                            cluster_dists.data(),
                                            clusters_to_probe,
                                            true,
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
 * The "main part" of the search, which assumes that outer-level `search` has already:
 *
 *   1. computed the closest clusters to probe (`clusters_to_probe`);
 *   2. transformed input queries into the rotated space (rot_dim);
 *   3. split the query batch into smaller chunks, so that the device workspace
 *      is guaranteed to fit into GPU memory.
 */
template <typename ScoreT, typename LutT, typename IvfSampleFilterT, typename IdxT>
void ivfpq_search_worker(raft::resources const& handle,
                         const index<IdxT>& index,
                         uint32_t max_samples,
                         uint32_t n_probes,
                         uint32_t topK,
                         uint32_t n_queries,
                         uint32_t queries_offset,            // needed for filtering
                         const uint32_t* clusters_to_probe,  // [n_queries, n_probes]
                         const float* query,                 // [n_queries, rot_dim]
                         IdxT* neighbors,                    // [n_queries, topK]
                         float* distances,                   // [n_queries, topK]
                         float scaling_factor,
                         double preferred_shmem_carveout,
                         IvfSampleFilterT sample_filter)
{
  auto stream = resource::get_cuda_stream(handle);
  auto mr     = resource::get_workspace_resource(handle);

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
    auto index_list_buf =
      make_device_mdarray<uint32_t>(handle, mr, make_extents<uint32_t>(n_queries * n_probes));
    rmm::device_uvector<uint32_t> cluster_labels_out(n_queries * n_probes, stream, mr);
    auto index_list   = index_list_buf.data_handle();
    index_list_sorted = index_list_sorted_buf.data();

    linalg::map_offset(handle, index_list_buf.view(), identity_op{});

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

  auto search_instance = compute_similarity_select<ScoreT, LutT, IvfSampleFilterT>(
    resource::get_device_properties(handle),
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
  std::optional<device_vector<float>> query_kths_buf{std::nullopt};
  float* query_kths = nullptr;
  if (manage_local_topk) {
    query_kths_buf.emplace(
      make_device_mdarray<float>(handle, mr, make_extents<uint32_t>(n_queries)));
    linalg::map(handle,
                query_kths_buf->view(),
                raft::const_op<float>{dummy_block_sort_t<ScoreT, IdxT>::queue_t::kDummy});
    query_kths = query_kths_buf->data_handle();
  }
  compute_similarity_run(search_instance,
                         stream,
                         index.size(),
                         index.rot_dim(),
                         n_probes,
                         index.pq_dim(),
                         n_queries,
                         queries_offset,
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
                         query_kths,
                         sample_filter,
                         device_lut.data(),
                         distances_buf.data(),
                         neighbors_ptr);

  // Select topk vectors for each query
  rmm::device_uvector<ScoreT> topk_dists(n_queries * topK, stream, mr);
  matrix::detail::select_k<ScoreT, uint32_t>(handle,
                                             distances_buf.data(),
                                             neighbors_ptr,
                                             n_queries,
                                             topk_len,
                                             topK,
                                             topk_dists.data(),
                                             neighbors_uint32,
                                             true,
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
template <typename IdxT, typename IvfSampleFilterT>
struct ivfpq_search {
 public:
  using fun_t = decltype(&ivfpq_search_worker<float, float, IvfSampleFilterT, IdxT>);

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
      return ivfpq_search_worker<ScoreT, LutT, IvfSampleFilterT, IdxT>;
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
 * @param res is used to query the workspace size
 * @param k top-k
 * @param n_probes number of selected clusters per query
 * @param n_queries number of queries hoped to be processed at once.
 *                  (maximum value for the returned batch size)
 * @param max_samples maximum possible number of samples to be processed for the given `n_probes`
 *
 * @return maximum recommended batch size.
 */
inline auto get_max_batch_size(raft::resources const& res,
                               uint32_t k,
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
  auto max_ws_size = resource::get_workspace_free_bytes(res);
  if (ws_size(max_batch_size) > max_ws_size) {
    uint32_t smaller_batch_size = bound_by_power_of_two(max_batch_size);
    // gradually reduce the batch size until we fit into the max size limit.
    while (smaller_batch_size > 1 && ws_size(smaller_batch_size) > max_ws_size) {
      smaller_batch_size >>= 1;
    }
    return smaller_batch_size;
  }
  return max_batch_size;
}

/** See raft::spatial::knn::ivf_pq::search docs */
template <typename T,
          typename IdxT,
          typename IvfSampleFilterT = raft::neighbors::filtering::none_ivf_sample_filter>
inline void search(raft::resources const& handle,
                   const search_params& params,
                   const index<IdxT>& index,
                   const T* queries,
                   uint32_t n_queries,
                   uint32_t k,
                   IdxT* neighbors,
                   float* distances,
                   IvfSampleFilterT sample_filter = IvfSampleFilterT())
{
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
                "Unsupported element type.");
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_pq::search(n_queries = %u, n_probes = %u, k = %u, dim = %zu)",
    n_queries,
    params.n_probes,
    k,
    index.dim());
  resource::detail::warn_non_pool_workspace(handle, "raft::ivf_pq::search");

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

  auto stream = resource::get_cuda_stream(handle);

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

  auto mr = resource::get_workspace_resource(handle);

  // Maximum number of query vectors to search at the same time.
  const auto max_queries = std::min<uint32_t>(std::max<uint32_t>(n_queries, 1), 4096);
  auto max_batch_size    = get_max_batch_size(handle, k, n_probes, max_queries, max_samples);

  rmm::device_uvector<float> float_queries(max_queries * dim_ext, stream, mr);
  rmm::device_uvector<float> rot_queries(max_queries * index.rot_dim(), stream, mr);
  rmm::device_uvector<uint32_t> clusters_to_probe(max_queries * n_probes, stream, mr);

  auto search_instance = ivfpq_search<IdxT, IvfSampleFilterT>::fun(params, index.metric());

  for (uint32_t offset_q = 0; offset_q < n_queries; offset_q += max_queries) {
    uint32_t queries_batch = min(max_queries, n_queries - offset_q);

    select_clusters(handle,
                    clusters_to_probe.data(),
                    float_queries.data(),
                    queries_batch,
                    n_probes,
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
                      n_probes,
                      k,
                      batch_size,
                      offset_q + offset_b,
                      clusters_to_probe.data() + uint64_t(n_probes) * offset_b,
                      rot_queries.data() + uint64_t(index.rot_dim()) * offset_b,
                      neighbors + uint64_t(k) * (offset_q + offset_b),
                      distances + uint64_t(k) * (offset_q + offset_b),
                      utils::config<T>::kDivisor / utils::config<float>::kDivisor,
                      params.preferred_shmem_carveout,
                      sample_filter);
    }
  }
}

}  // namespace raft::neighbors::ivf_pq::detail
