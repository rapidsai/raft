/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include "../common.hpp"
#include "../coo_spmv_kernel.cuh"
#include "../utils.cuh"
#include "coo_mask_row_iterators.cuh"

#include <raft/core/resource/cuda_stream.hpp>

#include <rmm/device_uvector.hpp>

namespace raft {
namespace sparse {
namespace distance {
namespace detail {

template <typename value_idx, typename value_t, int tpb>
class coo_spmv_strategy {
 public:
  coo_spmv_strategy(const distances_config_t<value_idx, value_t>& config_) : config(config_)
  {
    smem = raft::getSharedMemPerBlock();
  }

  template <typename strategy_t,
            typename indptr_it,
            typename product_f,
            typename accum_f,
            typename write_f>
  void _dispatch_base(strategy_t& strategy,
                      int smem_dim,
                      indptr_it& a_indptr,
                      value_t* out_dists,
                      value_idx* coo_rows_b,
                      product_f product_func,
                      accum_f accum_func,
                      write_f write_func,
                      int chunk_size,
                      int n_blocks,
                      int n_blocks_per_row)
  {
    RAFT_CUDA_TRY(cudaFuncSetCacheConfig(balanced_coo_generalized_spmv_kernel<strategy_t,
                                                                              indptr_it,
                                                                              value_idx,
                                                                              value_t,
                                                                              false,
                                                                              tpb,
                                                                              product_f,
                                                                              accum_f,
                                                                              write_f>,
                                         cudaFuncCachePreferShared));

    balanced_coo_generalized_spmv_kernel<strategy_t, indptr_it, value_idx, value_t, false, tpb>
      <<<n_blocks, tpb, smem, resource::get_cuda_stream(config.handle)>>>(strategy,
                                                                          a_indptr,
                                                                          config.a_indices,
                                                                          config.a_data,
                                                                          config.a_nnz,
                                                                          coo_rows_b,
                                                                          config.b_indices,
                                                                          config.b_data,
                                                                          config.a_nrows,
                                                                          config.b_nrows,
                                                                          smem_dim,
                                                                          config.b_nnz,
                                                                          out_dists,
                                                                          n_blocks_per_row,
                                                                          chunk_size,
                                                                          config.b_ncols,
                                                                          product_func,
                                                                          accum_func,
                                                                          write_func);
  }

  template <typename strategy_t,
            typename indptr_it,
            typename product_f,
            typename accum_f,
            typename write_f>
  void _dispatch_base_rev(strategy_t& strategy,
                          int smem_dim,
                          indptr_it& b_indptr,
                          value_t* out_dists,
                          value_idx* coo_rows_a,
                          product_f product_func,
                          accum_f accum_func,
                          write_f write_func,
                          int chunk_size,
                          int n_blocks,
                          int n_blocks_per_row)
  {
    RAFT_CUDA_TRY(cudaFuncSetCacheConfig(balanced_coo_generalized_spmv_kernel<strategy_t,
                                                                              indptr_it,
                                                                              value_idx,
                                                                              value_t,
                                                                              true,
                                                                              tpb,
                                                                              product_f,
                                                                              accum_f,
                                                                              write_f>,
                                         cudaFuncCachePreferShared));

    balanced_coo_generalized_spmv_kernel<strategy_t, indptr_it, value_idx, value_t, true, tpb>
      <<<n_blocks, tpb, smem, resource::get_cuda_stream(config.handle)>>>(strategy,
                                                                          b_indptr,
                                                                          config.b_indices,
                                                                          config.b_data,
                                                                          config.b_nnz,
                                                                          coo_rows_a,
                                                                          config.a_indices,
                                                                          config.a_data,
                                                                          config.b_nrows,
                                                                          config.a_nrows,
                                                                          smem_dim,
                                                                          config.a_nnz,
                                                                          out_dists,
                                                                          n_blocks_per_row,
                                                                          chunk_size,
                                                                          config.a_ncols,
                                                                          product_func,
                                                                          accum_func,
                                                                          write_func);
  }

 protected:
  int smem;
  const distances_config_t<value_idx, value_t>& config;
};

}  // namespace detail
}  // namespace distance
}  // namespace sparse
}  // namespace raft
