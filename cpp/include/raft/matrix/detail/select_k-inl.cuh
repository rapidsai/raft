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

#include "select_radix.cuh"
#include "select_warpsort.cuh"

#include <raft/core/nvtx.hpp>

#include <raft/neighbors/detail/selection_faiss.cuh>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace raft::matrix::detail {

// this is a subset of algorithms, chosen by running the algorithm_selection
// notebook in cpp/scripts/heuristics/select_k
enum class Algo { kRadix11bits, kWarpDistributedShm, kFaissBlockSelect };

/**
 * Predict the fastest select_k algorithm based on the number of rows/cols/k
 *
 * The body of this method is automatically generated, using a DecisionTree
 * to predict the fastest algorithm based off of thousands of trial runs
 * on different values of rows/cols/k. The decision tree is converted to c++
 * code, which is cut and paste below.
 *
 * NOTE: The code to generate is in cpp/scripts/heuristics/select_k, running the
 * 'generate_heuristic' notebook there will replace the body of this function
 * with the latest learned heuristic
 */
inline Algo choose_select_k_algorithm(size_t rows, size_t cols, int k)
{
  if (k > 134) {
    if (k > 256) {
      if (k > 809) {
        return Algo::kRadix11bits;
      } else {
        if (rows > 124) {
          if (cols > 63488) {
            return Algo::kFaissBlockSelect;
          } else {
            return Algo::kRadix11bits;
          }
        } else {
          return Algo::kRadix11bits;
        }
      }
    } else {
      if (cols > 678736) {
        return Algo::kWarpDistributedShm;
      } else {
        return Algo::kRadix11bits;
      }
    }
  } else {
    if (cols > 13776) {
      if (rows > 335) {
        if (k > 1) {
          if (rows > 546) {
            return Algo::kWarpDistributedShm;
          } else {
            if (k > 17) {
              return Algo::kWarpDistributedShm;
            } else {
              return Algo::kFaissBlockSelect;
            }
          }
        } else {
          return Algo::kFaissBlockSelect;
        }
      } else {
        if (k > 44) {
          if (cols > 1031051) {
            return Algo::kWarpDistributedShm;
          } else {
            if (rows > 22) {
              return Algo::kWarpDistributedShm;
            } else {
              return Algo::kRadix11bits;
            }
          }
        } else {
          return Algo::kWarpDistributedShm;
        }
      }
    } else {
      if (k > 1) {
        if (rows > 188) {
          return Algo::kWarpDistributedShm;
        } else {
          if (k > 72) {
            return Algo::kRadix11bits;
          } else {
            return Algo::kWarpDistributedShm;
          }
        }
      } else {
        return Algo::kFaissBlockSelect;
      }
    }
  }
}

/**
 * Select k smallest or largest key/values from each row in the input data.
 *
 * If you think of the input data `in_val` as a row-major matrix with `len` columns and
 * `batch_size` rows, then this function selects `k` smallest/largest values in each row and fills
 * in the row-major matrix `out_val` of size (batch_size, k).
 *
 * @tparam T
 *   the type of the keys (what is being compared).
 * @tparam IdxT
 *   the index type (what is being selected together with the keys).
 *
 * @param[in] in_val
 *   contiguous device array of inputs of size (len * batch_size);
 *   these are compared and selected.
 * @param[in] in_idx
 *   contiguous device array of inputs of size (len * batch_size);
 *   typically, these are indices of the corresponding in_val.
 * @param batch_size
 *   number of input rows, i.e. the batch size.
 * @param len
 *   length of a single input array (row); also sometimes referred as n_cols.
 *   Invariant: len >= k.
 * @param k
 *   the number of outputs to select in each input row.
 * @param[out] out_val
 *   contiguous device array of outputs of size (k * batch_size);
 *   the k smallest/largest values from each row of the `in_val`.
 * @param[out] out_idx
 *   contiguous device array of outputs of size (k * batch_size);
 *   the payload selected together with `out_val`.
 * @param select_min
 *   whether to select k smallest (true) or largest (false) keys.
 * @param stream
 * @param mr an optional memory resource to use across the calls (you can provide a large enough
 *           memory pool here to avoid memory allocations within the call).
 */
template <typename T, typename IdxT>
void select_k(const T* in_val,
              const IdxT* in_idx,
              size_t batch_size,
              size_t len,
              int k,
              T* out_val,
              IdxT* out_idx,
              bool select_min,
              rmm::cuda_stream_view stream,
              rmm::mr::device_memory_resource* mr = nullptr)
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "matrix::select_k(batch_size = %zu, len = %zu, k = %d)", batch_size, len, k);

  auto algo = choose_select_k_algorithm(batch_size, len, k);
  switch (algo) {
    case Algo::kRadix11bits:
      return detail::select::radix::select_k<T, IdxT, 11, 512>(in_val,
                                                               in_idx,
                                                               batch_size,
                                                               len,
                                                               k,
                                                               out_val,
                                                               out_idx,
                                                               select_min,
                                                               true,  // fused_last_filter
                                                               stream);
    case Algo::kWarpDistributedShm:
      return detail::select::warpsort::
        select_k_impl<T, IdxT, detail::select::warpsort::warp_sort_distributed_ext>(
          in_val, in_idx, batch_size, len, k, out_val, out_idx, select_min, stream);
    case Algo::kFaissBlockSelect:
      return neighbors::detail::select_k(
        in_val, in_idx, batch_size, len, out_val, out_idx, select_min, k, stream);
  }
}
}  // namespace raft::matrix::detail
