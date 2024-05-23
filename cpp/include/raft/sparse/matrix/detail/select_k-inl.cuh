/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/linalg/map.cuh>
#include <raft/matrix/detail/select_k-inl.cuh>
#include <raft/matrix/select_k_types.hpp>

#include <cub/cub.cuh>

#include <type_traits>

namespace raft::sparse::matrix::detail {

using namespace raft::matrix::detail;
using raft::matrix::SelectAlgo;

/**
 * Selects the k smallest or largest keys/values from each row of the input CSR matrix.
 *
 * This function operates on a CSR matrix `in_val` with a logical dense shape of [batch_size, len],
 * selecting the k smallest or largest elements from each row. The selected elements are then stored
 * in a row-major output matrix `out_val` with dimensions `batch_size` x k.
 *
 * @tparam T
 *   Type of the elements being compared (keys).
 * @tparam IdxT
 *   Type of the indices associated with the keys.
 * @tparam NZType
 *   Type representing non-zero elements of `in_val`.
 *
 * @param[in] handle
 *   Container for managing reusable resources.
 * @param[in] in_val
 *   Input matrix in CSR format with a logical dense shape of [batch_size, len],
 *   containing the elements to be compared and selected.
 * @param[in] in_idx
 *   Optional input indices [in_val.nnz] associated with `in_val.values`.
 *   If `in_idx` is `std::nullopt`, it defaults to a contiguous array from 0 to len-1.
 * @param[out] out_val
 *   Output matrix [in_val.get_n_row(), k] storing the selected k smallest/largest elements
 *   from each row of `in_val`.
 * @param[out] out_idx
 *   Output indices [in_val.get_n_row(), k] corresponding to the selected elements in `out_val`.
 * @param[in] select_min
 *   Flag indicating whether to select the k smallest (true) or largest (false) elements.
 * @param[in] sorted
 *   whether to make sure selected pairs are sorted by value
 * @param[in] algo
 *   the selection algorithm to use
 */
template <typename T, typename IdxT>
void select_k(raft::resources const& handle,
              raft::device_csr_matrix_view<const T, IdxT, IdxT, IdxT> in_val,
              std::optional<raft::device_vector_view<const IdxT, IdxT>> in_idx,
              raft::device_matrix_view<T, IdxT, raft::row_major> out_val,
              raft::device_matrix_view<IdxT, IdxT, raft::row_major> out_idx,
              bool select_min,
              bool sorted     = false,
              SelectAlgo algo = SelectAlgo::kAuto)
{
  auto csr_view = in_val.structure_view();
  auto nnz      = csr_view.get_nnz();

  if (nnz == 0) return;

  auto batch_size = csr_view.get_n_rows();
  auto len        = csr_view.get_n_cols();
  auto k          = IdxT(out_val.extent(1));

  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "sparse::matrix::select_k(batch_size = %zu, len = %zu, k = %d)", batch_size, len, k);

  RAFT_EXPECTS(out_val.extent(1) <= int64_t(std::numeric_limits<int>::max()),
               "output k must fit the int type.");

  RAFT_EXPECTS(batch_size == out_val.extent(0), "batch sizes must be equal");
  RAFT_EXPECTS(batch_size == out_idx.extent(0), "batch sizes must be equal");

  if (in_idx.has_value()) {
    RAFT_EXPECTS(size_t(nnz) == in_idx->size(),
                 "nnz of in_val must be equal to the length of in_idx");
  }
  RAFT_EXPECTS(IdxT(k) == out_idx.extent(1), "value and index output lengths must be equal");

  if (algo == SelectAlgo::kAuto) { algo = choose_select_k_algorithm(batch_size, len, k); }

  auto indptr = csr_view.get_indptr().data();

  switch (algo) {
    case SelectAlgo::kRadix8bits:
    case SelectAlgo::kRadix11bits:
    case SelectAlgo::kRadix11bitsExtraPass: {
      if (algo == SelectAlgo::kRadix8bits) {
        select::radix::select_k<T, IdxT, 8, 512, false>(
          handle,
          in_val.get_elements().data(),
          (in_idx.has_value() ? in_idx->data_handle() : csr_view.get_indices().data()),
          batch_size,
          len,
          k,
          out_val.data_handle(),
          out_idx.data_handle(),
          select_min,
          true,
          indptr);
      } else {
        bool fused_last_filter = algo == SelectAlgo::kRadix11bits;
        select::radix::select_k<T, IdxT, 11, 512, false>(
          handle,
          in_val.get_elements().data(),
          (in_idx.has_value() ? in_idx->data_handle() : csr_view.get_indices().data()),
          batch_size,
          len,
          k,
          out_val.data_handle(),
          out_idx.data_handle(),
          select_min,
          fused_last_filter,
          indptr);
      }

      if (sorted) {
        auto offsets = make_device_mdarray<IdxT, IdxT>(
          handle, resource::get_workspace_resource(handle), make_extents<IdxT>(batch_size + 1));
        raft::linalg::map_offset(handle, offsets.view(), mul_const_op<IdxT>(k));

        auto keys =
          raft::make_device_vector_view<T, IdxT>(out_val.data_handle(), (IdxT)(batch_size * k));
        auto vals =
          raft::make_device_vector_view<IdxT, IdxT>(out_idx.data_handle(), (IdxT)(batch_size * k));

        segmented_sort_by_key<T, IdxT>(
          handle, raft::make_const_mdspan(offsets.view()), keys, vals, select_min);
      }

      return;
    }
    case SelectAlgo::kWarpDistributed:
      return select::warpsort::select_k_impl<T, IdxT, select::warpsort::warp_sort_distributed>(
        handle,
        in_val.get_elements().data(),
        (in_idx.has_value() ? in_idx->data_handle() : csr_view.get_indices().data()),
        batch_size,
        len,
        k,
        out_val.data_handle(),
        out_idx.data_handle(),
        select_min,
        indptr);
    case SelectAlgo::kWarpDistributedShm:
      return select::warpsort::select_k_impl<T, IdxT, select::warpsort::warp_sort_distributed_ext>(
        handle,
        in_val.get_elements().data(),
        (in_idx.has_value() ? in_idx->data_handle() : csr_view.get_indices().data()),
        batch_size,
        len,
        k,
        out_val.data_handle(),
        out_idx.data_handle(),
        select_min,
        indptr);
    case SelectAlgo::kWarpAuto:
      return select::warpsort::select_k<T, IdxT>(
        handle,
        in_val.get_elements().data(),
        (in_idx.has_value() ? in_idx->data_handle() : csr_view.get_indices().data()),
        batch_size,
        len,
        k,
        out_val.data_handle(),
        out_idx.data_handle(),
        select_min,
        indptr);
    case SelectAlgo::kWarpImmediate:
      return select::warpsort::select_k_impl<T, IdxT, select::warpsort::warp_sort_immediate>(
        handle,
        in_val.get_elements().data(),
        (in_idx.has_value() ? in_idx->data_handle() : csr_view.get_indices().data()),
        batch_size,
        len,
        k,
        out_val.data_handle(),
        out_idx.data_handle(),
        select_min,
        indptr);
    case SelectAlgo::kWarpFiltered:
      return select::warpsort::select_k_impl<T, IdxT, select::warpsort::warp_sort_filtered>(
        handle,
        in_val.get_elements().data(),
        (in_idx.has_value() ? in_idx->data_handle() : csr_view.get_indices().data()),
        batch_size,
        len,
        k,
        out_val.data_handle(),
        out_idx.data_handle(),
        select_min,
        indptr);
    default: RAFT_FAIL("K-selection Algorithm not supported.");
  }

  return;
}

}  // namespace raft::sparse::matrix::detail
