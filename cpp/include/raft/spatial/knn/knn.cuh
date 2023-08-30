/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <raft/core/device_mdspan.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/detail/select_radix.cuh>
#include <raft/matrix/detail/select_warpsort.cuh>
#include <raft/neighbors/detail/knn_brute_force.cuh>
#include <raft/neighbors/detail/selection_faiss.cuh>

namespace raft::spatial::knn {

/**
 * Performs a k-select across row partitioned index/distance
 * matrices formatted like the following:
 * row1: k0, k1, k2
 * row2: k0, k1, k2
 * row3: k0, k1, k2
 * row1: k0, k1, k2
 * row2: k0, k1, k2
 * row3: k0, k1, k2
 *
 * etc...
 *
 * @tparam idx_t
 * @tparam value_t
 * @param in_keys
 * @param in_values
 * @param out_keys
 * @param out_values
 * @param n_samples
 * @param n_parts
 * @param k
 * @param stream
 * @param translations
 */
template <typename idx_t = int64_t, typename value_t = float>
inline void knn_merge_parts(const value_t* in_keys,
                            const idx_t* in_values,
                            value_t* out_keys,
                            idx_t* out_values,
                            size_t n_samples,
                            int n_parts,
                            int k,
                            cudaStream_t stream,
                            idx_t* translations)
{
  raft::neighbors::detail::knn_merge_parts(
    in_keys, in_values, out_keys, out_values, n_samples, n_parts, k, stream, translations);
}

/** Choose an implementation for the select-top-k, */
enum class SelectKAlgo {
  /** Adapted from the faiss project. Result: sorted (not stable). */
  FAISS,
  /** Incomplete series of radix sort passes, comparing 8 bits per pass. Result: unsorted. */
  RADIX_8_BITS,
  /** Incomplete series of radix sort passes, comparing 11 bits per pass. Result: unsorted. */
  RADIX_11_BITS,
  /** Filtering with a bitonic-sort-based priority queue. Result: sorted (not stable). */
  WARP_SORT
};

/**
 * Select k smallest or largest key/values from each row in the input data.
 *
 * If you think of the input data `in_keys` as a row-major matrix with input_len columns and
 * n_inputs rows, then this function selects k smallest/largest values in each row and fills
 * in the row-major matrix `out_keys` of size (n_inputs, k).
 *
 * Note, depending on the selected algorithm, the values within rows of `out_keys` are not
 * necessarily sorted. See the `SelectKAlgo` enumeration for more details.
 *
 * Note: This call is deprecated, please use `raft/matrix/select_k.cuh`
 *
 * @tparam idx_t
 *   the payload type (what is being selected together with the keys).
 * @tparam value_t
 *   the type of the keys (what is being compared).
 *
 * @param[in] in_keys
 *   contiguous device array of inputs of size (input_len * n_inputs);
 *   these are compared and selected.
 * @param[in] in_values
 *   contiguous device array of inputs of size (input_len * n_inputs);
 *   typically, these are indices of the corresponding in_keys.
 *   You can pass `NULL` as an argument here; this would imply `in_values` is a homogeneous array
 *   of indices from `0` to `input_len - 1` for every input and reduce the usage of memory
 *   bandwidth.
 * @param[in] n_inputs
 *   number of input rows, i.e. the batch size.
 * @param[in] input_len
 *   length of a single input array (row); also sometimes referred as n_cols.
 *   Invariant: input_len >= k.
 * @param[out] out_keys
 *   contiguous device array of outputs of size (k * n_inputs);
 *   the k smallest/largest values from each row of the `in_keys`.
 * @param[out] out_values
 *   contiguous device array of outputs of size (k * n_inputs);
 *   the payload selected together with `out_keys`.
 * @param[in] select_min
 *   whether to select k smallest (true) or largest (false) keys.
 * @param[in] k
 *   the number of outputs to select in each input row.
 * @param[in] stream
 * @param[in] algo
 *   the implementation of the algorithm
 */
template <typename idx_t = int, typename value_t = float>
[[deprecated("Use function `select_k` from `raft/matrix/select_k.cuh`")]] inline void select_k(
  const value_t* in_keys,
  const idx_t* in_values,
  size_t n_inputs,
  size_t input_len,
  value_t* out_keys,
  idx_t* out_values,
  bool select_min,
  int k,
  cudaStream_t stream,
  SelectKAlgo algo = SelectKAlgo::FAISS)
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope("select-%s-%d (%zu, %zu) algo-%d",
                                                            select_min ? "min" : "max",
                                                            k,
                                                            n_inputs,
                                                            input_len,
                                                            int(algo));
  ASSERT(size_t(input_len) >= size_t(k),
         "Size of the input (input_len = %zu) must be not smaller than the selection (k = %zu).",
         size_t(input_len),
         size_t(k));

  switch (algo) {
    case SelectKAlgo::FAISS:
      neighbors::detail::select_k(
        in_keys, in_values, n_inputs, input_len, out_keys, out_values, select_min, k, stream);
      break;

    case SelectKAlgo::RADIX_8_BITS:
      matrix::detail::select::radix::select_k<value_t, idx_t, 8, 512>(
        in_keys, in_values, n_inputs, input_len, k, out_keys, out_values, select_min, true, stream);
      break;

    case SelectKAlgo::RADIX_11_BITS:
      matrix::detail::select::radix::select_k<value_t, idx_t, 11, 512>(
        in_keys, in_values, n_inputs, input_len, k, out_keys, out_values, select_min, true, stream);
      break;

    case SelectKAlgo::WARP_SORT: {
      raft::resources res;
      resource::set_cuda_stream(res, stream);
      matrix::detail::select::warpsort::select_k<value_t, idx_t>(
        res, in_keys, in_values, n_inputs, input_len, k, out_keys, out_values, select_min, stream);
    } break;

    default: ASSERT(false, "Unknown algorithm (id = %d)", int(algo));
  }
}

/**
 * @brief Flat C++ API function to perform a brute force knn on
 * a series of input arrays and combine the results into a single
 * output array for indexes and distances.
 *
 * @param[in] handle the cuml handle to use
 * @param[in] input vector of pointers to the input arrays
 * @param[in] sizes vector of sizes of input arrays
 * @param[in] D the dimensionality of the arrays
 * @param[in] search_items array of items to search of dimensionality D
 * @param[in] n number of rows in search_items
 * @param[out] res_I the resulting index array of size n * k
 * @param[out] res_D the resulting distance array of size n * k
 * @param[in] k the number of nearest neighbors to return
 * @param[in] rowMajorIndex are the index arrays in row-major order?
 * @param[in] rowMajorQuery are the query arrays in row-major order?
 * @param[in] metric distance metric to use. Euclidean (L2) is used by
 * 			   default
 * @param[in] metric_arg the value of `p` for Minkowski (l-p) distances. This
 * 					 is ignored if the metric_type is not Minkowski.
 * @param[in] translations starting offsets for partitions. should be the same size
 *            as input vector.
 */
template <typename idx_t = std::int64_t, typename value_t = float, typename value_int = int>
void brute_force_knn(raft::resources const& handle,
                     std::vector<value_t*>& input,
                     std::vector<value_int>& sizes,
                     value_int D,
                     value_t* search_items,
                     value_int n,
                     idx_t* res_I,
                     value_t* res_D,
                     value_int k,
                     bool rowMajorIndex               = true,
                     bool rowMajorQuery               = true,
                     std::vector<idx_t>* translations = nullptr,
                     distance::DistanceType metric    = distance::DistanceType::L2Unexpanded,
                     float metric_arg                 = 2.0f)
{
  ASSERT(input.size() == sizes.size(), "input and sizes vectors must be the same size");

  raft::neighbors::detail::brute_force_knn_impl(handle,
                                                input,
                                                sizes,
                                                D,
                                                search_items,
                                                n,
                                                res_I,
                                                res_D,
                                                k,
                                                rowMajorIndex,
                                                rowMajorQuery,
                                                translations,
                                                metric,
                                                metric_arg);
}

}  // namespace raft::spatial::knn
