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

#include "../../csr.hpp"
#include "../../detail/utils.h"
#include "common.hpp"
#include "coo_spmv_strategies/dense_smem_strategy.cuh"
#include "coo_spmv_strategies/hash_strategy.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cusparse_v2.h>
#include <limits.h>

#include <nvfunctional>

namespace raft {
namespace sparse {
namespace distance {
namespace detail {

template <typename value_idx,
          typename value_t,
          int threads_per_block = 1024,
          typename product_f,
          typename accum_f,
          typename write_f,
          typename strategy_t>
inline void balanced_coo_pairwise_generalized_spmv(
  value_t* out_dists,
  const distances_config_t<value_idx, value_t>& config_,
  value_idx* coo_rows_b,
  product_f product_func,
  accum_f accum_func,
  write_f write_func,
  strategy_t strategy,
  int chunk_size = 500000)
{
  uint64_t n = (uint64_t)sizeof(value_t) * (uint64_t)config_.a_nrows * (uint64_t)config_.b_nrows;
  RAFT_CUDA_TRY(cudaMemsetAsync(out_dists, 0, n, resource::get_cuda_stream(config_.handle)));

  strategy.dispatch(out_dists, coo_rows_b, product_func, accum_func, write_func, chunk_size);
};

/**
 * Performs generalized sparse-matrix-sparse-matrix multiplication via a
 * sparse-matrix-sparse-vector layout `out=A*B` where generalized product()
 * and sum() operations can be used in place of the standard sum and product:
 *
 * out_ij = sum_k(product(A_ik, B_ik)) The sum goes through values of
 * k=0..n_cols-1 where B_kj is nonzero.
 *
 * The product and sum operations shall form a semiring algebra with the
 * following properties:
 * 1. {+, 0} is a commutative sum reduction monoid with identity element 0
 * 2. {*, 1} is a product monoid with identity element 1
 * 3. Multiplication by 0 annihilates x. e.g. product(x, 0) = 0
 *
 * Each vector of A is loaded into shared memory in dense form and the
 * non-zeros of B load balanced across the threads of each block.
 * @tparam value_idx index type
 * @tparam value_t value type
 * @tparam threads_per_block block size
 * @tparam product_f semiring product() function
 * @tparam accum_f semiring sum() function
 * @tparam write_f atomic semiring sum() function
 * @param[out] out_dists dense array of out distances of size m * n in row-major
 *             format.
 * @param[in] config_ distance config object
 * @param[in] coo_rows_b coo row array for B
 * @param[in] product_func semiring product() function
 * @param[in] accum_func semiring sum() function
 * @param[in] write_func atomic semiring sum() function
 * @param[in] chunk_size number of nonzeros of B to process for each row of A
 *            this value was found through profiling and represents a reasonable
 *            setting for both large and small densities
 */
template <typename value_idx,
          typename value_t,
          int threads_per_block = 1024,
          typename product_f,
          typename accum_f,
          typename write_f>
inline void balanced_coo_pairwise_generalized_spmv(
  value_t* out_dists,
  const distances_config_t<value_idx, value_t>& config_,
  value_idx* coo_rows_b,
  product_f product_func,
  accum_f accum_func,
  write_f write_func,
  int chunk_size = 500000)
{
  uint64_t n = (uint64_t)sizeof(value_t) * (uint64_t)config_.a_nrows * (uint64_t)config_.b_nrows;
  RAFT_CUDA_TRY(cudaMemsetAsync(out_dists, 0, n, resource::get_cuda_stream(config_.handle)));

  int max_cols = max_cols_per_block<value_idx, value_t>();

  if (max_cols > config_.a_ncols) {
    dense_smem_strategy<value_idx, value_t, threads_per_block> strategy(config_);
    strategy.dispatch(out_dists, coo_rows_b, product_func, accum_func, write_func, chunk_size);
  } else {
    hash_strategy<value_idx, value_t, threads_per_block> strategy(config_);
    strategy.dispatch(out_dists, coo_rows_b, product_func, accum_func, write_func, chunk_size);
  }
};

template <typename value_idx,
          typename value_t,
          int threads_per_block = 1024,
          typename product_f,
          typename accum_f,
          typename write_f,
          typename strategy_t>
inline void balanced_coo_pairwise_generalized_spmv_rev(
  value_t* out_dists,
  const distances_config_t<value_idx, value_t>& config_,
  value_idx* coo_rows_a,
  product_f product_func,
  accum_f accum_func,
  write_f write_func,
  strategy_t strategy,
  int chunk_size = 500000)
{
  strategy.dispatch_rev(out_dists, coo_rows_a, product_func, accum_func, write_func, chunk_size);
};

/**
 * Used for computing distances where the reduction (e.g. product()) function
 * requires an implicit union (product(x, 0) = x) to capture the difference A-B.
 * This is necessary in some applications because the standard semiring algebra
 * endowed with the default multiplication product monoid will only
 * compute the intersection & B-A.
 *
 * This particular function is meant to accompany the function
 * `balanced_coo_pairwise_generalized_spmv` and executes the product operation
 * on only those columns that exist in B and not A.
 *
 * The product and sum operations shall enable the computation of a
 * non-annihilating semiring algebra with the following properties:
 * 1. {+, 0} is a commutative sum reduction monoid with identity element 0
 * 2. {*, 0} is a product monoid with identity element 0
 * 3. Multiplication by 0 does not annihilate x. e.g. product(x, 0) = x
 *
 * Manattan distance sum(abs(x_k-y_k)) is a great example of when this type of
 * execution pattern is necessary.
 *
 * @tparam value_idx index type
 * @tparam value_t value type
 * @tparam threads_per_block block size
 * @tparam product_f semiring product() function
 * @tparam accum_f semiring sum() function
 * @tparam write_f atomic semiring sum() function
 * @param[out] out_dists dense array of out distances of size m * n
 * @param[in] config_ distance config object
 * @param[in] coo_rows_a coo row array for A
 * @param[in] product_func semiring product() function
 * @param[in] accum_func semiring sum() function
 * @param[in] write_func atomic semiring sum() function
 * @param[in] chunk_size number of nonzeros of B to process for each row of A
 *            this value was found through profiling and represents a reasonable
 *            setting for both large and small densities
 */
template <typename value_idx,
          typename value_t,
          int threads_per_block = 1024,
          typename product_f,
          typename accum_f,
          typename write_f>
inline void balanced_coo_pairwise_generalized_spmv_rev(
  value_t* out_dists,
  const distances_config_t<value_idx, value_t>& config_,
  value_idx* coo_rows_a,
  product_f product_func,
  accum_f accum_func,
  write_f write_func,
  int chunk_size = 500000)
{
  // try dense first
  int max_cols = max_cols_per_block<value_idx, value_t>();

  if (max_cols > config_.b_ncols) {
    dense_smem_strategy<value_idx, value_t, threads_per_block> strategy(config_);
    strategy.dispatch_rev(out_dists, coo_rows_a, product_func, accum_func, write_func, chunk_size);
  } else {
    hash_strategy<value_idx, value_t, threads_per_block> strategy(config_);
    strategy.dispatch_rev(out_dists, coo_rows_a, product_func, accum_func, write_func, chunk_size);
  }
};

}  // namespace detail
}  // namespace distance
}  // namespace sparse
};  // namespace raft
