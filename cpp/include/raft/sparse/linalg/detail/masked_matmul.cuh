/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain A copy of the License at
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

#include <raft/core/bitmap.cuh>
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/distance/detail/utils.cuh>
#include <raft/sparse/linalg/sddmm.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/device_uvector.hpp>

namespace raft {
namespace sparse {
namespace linalg {
namespace detail {

template <typename value_t, typename index_t, typename nnz_t, typename bitmap_t>
void masked_matmul(raft::resources const& handle,
                   raft::device_matrix_view<const value_t, index_t, raft::row_major>& A,
                   raft::device_matrix_view<const value_t, index_t, raft::row_major>& B,
                   raft::core::bitmap_view<const bitmap_t, index_t>& mask,
                   raft::device_csr_matrix_view<value_t, index_t, index_t, nnz_t>& C,
                   std::optional<raft::host_scalar_view<value_t>> alpha,
                   std::optional<raft::host_scalar_view<value_t>> beta)
{
  index_t m   = A.extent(0);
  index_t n   = B.extent(0);
  index_t dim = A.extent(1);

  auto compressed_C_view = C.structure_view();

  RAFT_EXPECTS(A.extent(1) == B.extent(1), "The dim of A must be equal to the dim of B.");
  RAFT_EXPECTS(A.extent(0) == compressed_C_view.get_n_rows(),
               "Number of rows in C must match the number of rows in A.");
  RAFT_EXPECTS(B.extent(0) == compressed_C_view.get_n_cols(),
               "Number of columns in C must match the number of columns in B.");

  auto stream = raft::resource::get_cuda_stream(handle);

  auto C_matrix = raft::make_device_csr_matrix<value_t, index_t>(handle, compressed_C_view);

  // fill C
  raft::sparse::convert::bitmap_to_csr(handle, mask, C_matrix);

  if (m > 10 || alpha.has_value() || beta.has_value()) {
    auto C_view = raft::make_device_csr_matrix_view<value_t, index_t, index_t, index_t>(
      C.get_elements().data(), compressed_C_view);

    // create B col_major view
    auto B_col_major = raft::make_device_matrix_view<const value_t, index_t, raft::col_major>(
      B.data_handle(), dim, n);

    value_t default_alpha = static_cast<value_t>(1.0f);
    value_t default_beta  = static_cast<value_t>(0.0f);

    if (!alpha.has_value()) { alpha = raft::make_host_scalar_view<value_t>(&default_alpha); }
    if (!beta.has_value()) { beta = raft::make_host_scalar_view<value_t>(&default_beta); }

    raft::sparse::linalg::sddmm(handle,
                                A,
                                B_col_major,
                                C_view,
                                raft::linalg::Operation::NON_TRANSPOSE,
                                raft::linalg::Operation::NON_TRANSPOSE,
                                *alpha,
                                *beta);
  } else {
    raft::sparse::distance::detail::faster_dot_on_csr(handle,
                                                      C.get_elements().data(),
                                                      compressed_C_view.get_nnz(),
                                                      compressed_C_view.get_indptr().data(),
                                                      compressed_C_view.get_indices().data(),
                                                      A.data_handle(),
                                                      B.data_handle(),
                                                      compressed_C_view.get_n_rows(),
                                                      dim);
  }
}

}  // namespace detail
}  // namespace linalg
}  // namespace sparse
}  // namespace raft
