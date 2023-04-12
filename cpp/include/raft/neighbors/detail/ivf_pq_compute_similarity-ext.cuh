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

#include <raft/neighbors/detail/ivf_pq_fp_8bit.cuh>
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
#include <raft/util/raft_explicit.hpp>  // RAFT_EXPLICIT
#include <raft/util/vectorized.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <cub/cub.cuh>

#include <cuda_fp16.h>

#include <optional>

#ifdef RAFT_EXPLICIT_INSTANTIATE

namespace raft::neighbors::ivf_pq::detail {

// is_local_topk_feasible is not inline here, because we would have to define it
// here as well. That would run the risk of the definitions here and in the
// -inl.cuh header diverging.
auto is_local_topk_feasible(uint32_t k, uint32_t n_probes, uint32_t n_queries) -> bool;

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
                                          uint32_t* _out_indices) RAFT_EXPLICIT;

// The signature of the kernel defined by a minimal set of template parameters
template <typename OutT, typename LutT>
using compute_similarity_kernel_t =
  decltype(&compute_similarity_kernel<OutT, LutT, 8, 0, true, true>);

template <typename OutT, typename LutT>
struct occupancy_t {
  using shmem_unit = Pow2<128>;

  int blocks_per_sm = 0;
  double occupancy  = 0.0;
  double shmem_use  = 1.0;

  inline occupancy_t() = default;
  inline occupancy_t(size_t smem,
                     uint32_t n_threads,
                     compute_similarity_kernel_t<OutT, LutT> kernel,
                     const cudaDeviceProp& dev_props) RAFT_EXPLICIT;
};

template <typename OutT, typename LutT>
struct selected {
  compute_similarity_kernel_t<OutT, LutT> kernel;
  dim3 grid_dim;
  dim3 block_dim;
  size_t smem_size;
  size_t device_lut_size;

  template <typename... Args>
  void operator()(rmm::cuda_stream_view stream, Args... args);
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
template <typename OutT, typename LutT>
auto compute_similarity_select(const cudaDeviceProp& dev_props,
                               bool manage_local_topk,
                               int locality_hint,
                               double preferred_shmem_carveout,
                               uint32_t pq_bits,
                               uint32_t pq_dim,
                               uint32_t precomp_data_count,
                               uint32_t n_queries,
                               uint32_t n_probes,
                               uint32_t topk) -> selected<OutT, LutT> RAFT_EXPLICIT;

}  // namespace raft::neighbors::ivf_pq::detail

#endif  // RAFT_EXPLICIT_INSTANTIATE

#define instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(OutT, LutT)         \
  extern template auto raft::neighbors::ivf_pq::detail::compute_similarity_select<OutT, LutT>( \
    const cudaDeviceProp& dev_props,                                                           \
    bool manage_local_topk,                                                                    \
    int locality_hint,                                                                         \
    double preferred_shmem_carveout,                                                           \
    uint32_t pq_bits,                                                                          \
    uint32_t pq_dim,                                                                           \
    uint32_t precomp_data_count,                                                               \
    uint32_t n_queries,                                                                        \
    uint32_t n_probes,                                                                         \
    uint32_t topk)                                                                             \
    ->raft::neighbors::ivf_pq::detail::selected<OutT, LutT>;

#define COMMA ,
instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(
  half, raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA false>);
instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(
  half, raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA true>);
instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(half, half);
instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(float, half);
instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(float, float);
instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(
  float, raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA false>);
instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(
  float, raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA true>);

#undef COMMA

#undef instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select
