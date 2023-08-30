/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <numeric>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/core/resources.hpp>

#include <vector>

#include "bitonic.hpp"
#include "compute_distance.hpp"
#include "device_common.hpp"
#include "hashmap.hpp"
#include "search_multi_cta_kernel.cuh"
#include "search_plan.cuh"
#include "topk_for_cagra/topk_core.cuh"  // TODO replace with raft topk if possible
#include "utils.hpp"
#include <raft/core/logger.hpp>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cudart_utils.hpp>  // RAFT_CUDA_TRY_NOT_THROW is used TODO(tfeher): consider moving this to cuda_rt_essentials.hpp

namespace raft::neighbors::cagra::detail {
namespace multi_cta_search {

template <unsigned TEAM_SIZE,
          unsigned MAX_DATASET_DIM,
          typename DATA_T,
          typename INDEX_T,
          typename DISTANCE_T>

struct search : public search_plan_impl<DATA_T, INDEX_T, DISTANCE_T> {
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::max_queries;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::itopk_size;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::algo;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::team_size;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::search_width;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::min_iterations;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::max_iterations;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::thread_block_size;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::hashmap_mode;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::hashmap_min_bitlen;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::hashmap_max_fill_rate;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::num_random_samplings;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::rand_xor_mask;

  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::max_dim;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::dim;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::graph_degree;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::topk;

  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::hash_bitlen;

  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::small_hash_bitlen;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::small_hash_reset_interval;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::hashmap_size;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::dataset_size;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::result_buffer_size;

  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::smem_size;

  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::hashmap;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::num_executed_iterations;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::dev_seed;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::num_seeds;

  uint32_t num_cta_per_query;
  rmm::device_uvector<INDEX_T> intermediate_indices;
  rmm::device_uvector<float> intermediate_distances;
  size_t topk_workspace_size;
  rmm::device_uvector<uint32_t> topk_workspace;

  search(raft::resources const& res,
         search_params params,
         int64_t dim,
         int64_t graph_degree,
         uint32_t topk)
    : search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>(res, params, dim, graph_degree, topk),
      intermediate_indices(0, resource::get_cuda_stream(res)),
      intermediate_distances(0, resource::get_cuda_stream(res)),
      topk_workspace(0, resource::get_cuda_stream(res))

  {
    set_params(res, params);
  }

  void set_params(raft::resources const& res, const search_params& params)
  {
    this->itopk_size   = 32;
    search_width       = 1;
    num_cta_per_query  = max(params.search_width, params.itopk_size / 32);
    result_buffer_size = itopk_size + search_width * graph_degree;
    typedef raft::Pow2<32> AlignBytes;
    unsigned result_buffer_size_32 = AlignBytes::roundUp(result_buffer_size);
    // constexpr unsigned max_result_buffer_size = 256;
    RAFT_EXPECTS(result_buffer_size_32 <= 256, "Result buffer size cannot exceed 256");

    smem_size = sizeof(float) * max_dim +
                (sizeof(INDEX_T) + sizeof(DISTANCE_T)) * result_buffer_size_32 +
                sizeof(uint32_t) * search_width + sizeof(uint32_t);
    RAFT_LOG_DEBUG("# smem_size: %u", smem_size);

    //
    // Determine the thread block size
    //
    constexpr unsigned min_block_size = 64;
    constexpr unsigned max_block_size = 1024;
    uint32_t block_size               = thread_block_size;
    if (block_size == 0) {
      block_size = min_block_size;

      // Increase block size according to shared memory requirements.
      // If block size is 32, upper limit of shared memory size per
      // thread block is set to 4096. This is GPU generation dependent.
      constexpr unsigned ulimit_smem_size_cta32 = 4096;
      while (smem_size > ulimit_smem_size_cta32 / 32 * block_size) {
        block_size *= 2;
      }

      // Increase block size to improve GPU occupancy when total number of
      // CTAs (= num_cta_per_query * max_queries) is small.
      cudaDeviceProp deviceProp = resource::get_device_properties(res);
      RAFT_LOG_DEBUG("# multiProcessorCount: %d", deviceProp.multiProcessorCount);
      while ((block_size < max_block_size) &&
             (graph_degree * search_width * team_size >= block_size * 2) &&
             (num_cta_per_query * max_queries <=
              (1024 / (block_size * 2)) * deviceProp.multiProcessorCount)) {
        block_size *= 2;
      }
    }
    RAFT_LOG_DEBUG("# thread_block_size: %u", block_size);
    RAFT_EXPECTS(block_size >= min_block_size,
                 "block_size cannot be smaller than min_block size, %u",
                 min_block_size);
    RAFT_EXPECTS(block_size <= max_block_size,
                 "block_size cannot be larger than max_block size %u",
                 max_block_size);
    thread_block_size = block_size;

    //
    // Allocate memory for intermediate buffer and workspace.
    //
    uint32_t num_intermediate_results = num_cta_per_query * itopk_size;
    intermediate_indices.resize(num_intermediate_results * max_queries,
                                resource::get_cuda_stream(res));
    intermediate_distances.resize(num_intermediate_results * max_queries,
                                  resource::get_cuda_stream(res));

    hashmap.resize(hashmap_size, resource::get_cuda_stream(res));

    topk_workspace_size = _cuann_find_topk_bufferSize(
      topk, max_queries, num_intermediate_results, utils::get_cuda_data_type<DATA_T>());
    RAFT_LOG_DEBUG("# topk_workspace_size: %lu", topk_workspace_size);
    topk_workspace.resize(topk_workspace_size, resource::get_cuda_stream(res));
  }

  void check(const uint32_t topk) override
  {
    RAFT_EXPECTS(num_cta_per_query * 32 >= topk,
                 "`num_cta_per_query` (%u) * 32 must be equal to or greater than "
                 "`topk` (%u) when 'search_mode' is \"multi-cta\". "
                 "(`num_cta_per_query`=max(`search_width`, `itopk_size`/32))",
                 num_cta_per_query,
                 topk);
  }

  ~search() {}

  void operator()(raft::resources const& res,
                  raft::device_matrix_view<const DATA_T, int64_t, layout_stride> dataset,
                  raft::device_matrix_view<const INDEX_T, int64_t, row_major> graph,
                  INDEX_T* const topk_indices_ptr,          // [num_queries, topk]
                  DISTANCE_T* const topk_distances_ptr,     // [num_queries, topk]
                  const DATA_T* const queries_ptr,          // [num_queries, dataset_dim]
                  const uint32_t num_queries,
                  const INDEX_T* dev_seed_ptr,              // [num_queries, num_seeds]
                  uint32_t* const num_executed_iterations,  // [num_queries,]
                  uint32_t topk)
  {
    cudaStream_t stream = resource::get_cuda_stream(res);

    select_and_run<TEAM_SIZE, MAX_DATASET_DIM, DATA_T, INDEX_T, DISTANCE_T>(
      dataset,
      graph,
      intermediate_indices.data(),
      intermediate_distances.data(),
      queries_ptr,
      num_queries,
      dev_seed_ptr,
      num_executed_iterations,
      topk,
      thread_block_size,
      result_buffer_size,
      smem_size,
      hash_bitlen,
      hashmap.data(),
      num_cta_per_query,
      num_random_samplings,
      rand_xor_mask,
      num_seeds,
      itopk_size,
      search_width,
      min_iterations,
      max_iterations,
      stream);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    // Select the top-k results from the intermediate results
    const uint32_t num_intermediate_results = num_cta_per_query * itopk_size;
    _cuann_find_topk(topk,
                     num_queries,
                     num_intermediate_results,
                     intermediate_distances.data(),
                     num_intermediate_results,
                     intermediate_indices.data(),
                     num_intermediate_results,
                     topk_distances_ptr,
                     topk,
                     topk_indices_ptr,
                     topk,
                     topk_workspace.data(),
                     true,
                     NULL,
                     stream);
  }
};

}  // namespace multi_cta_search
}  // namespace raft::neighbors::cagra::detail
