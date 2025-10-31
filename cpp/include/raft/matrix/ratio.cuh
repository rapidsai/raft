/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/detail/matrix.cuh>

namespace raft::matrix {

/**
 * @defgroup matrix_ratio Matrix ratio operations
 * @{
 */

/**
 * @brief ratio of every element over sum of input vector is calculated
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam idx_t integer type used for indexing
 * @tparam layout layout of the matrix data (must be row or col major)
 * @param[in] handle
 * @param[in] src: input matrix
 * @param[out] dest: output matrix. The result is stored in the dest matrix
 */
template <typename math_t, typename idx_t, typename layout>
void ratio(raft::resources const& handle,
           raft::device_matrix_view<const math_t, idx_t, layout> src,
           raft::device_matrix_view<math_t, idx_t, layout> dest)
{
  RAFT_EXPECTS(src.size() == dest.size(), "Input and output matrices must be the same size.");
  detail::ratio(
    handle, src.data_handle(), dest.data_handle(), src.size(), resource::get_cuda_stream(handle));
}

/**
 * @brief ratio of every element over sum of input vector is calculated
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam idx_t integer type used for indexing
 * @tparam layout layout of the matrix data (must be row or col major)
 * @param[in] handle
 * @param[inout] inout: input matrix
 */
template <typename math_t, typename idx_t, typename layout>
void ratio(raft::resources const& handle, raft::device_matrix_view<math_t, idx_t, layout> inout)
{
  detail::ratio(handle,
                inout.data_handle(),
                inout.data_handle(),
                inout.size(),
                resource::get_cuda_stream(handle));
}

/** @} */  // end group matrix_ratio

}  // namespace raft::matrix
