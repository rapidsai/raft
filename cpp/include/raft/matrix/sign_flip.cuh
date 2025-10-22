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
 * @defgroup matrix_sign_flip Matrix sign flip operations
 * @{
 */

/**
 * @brief sign flip stabilizes the sign of col major eigen vectors.
 * The sign is flipped if the column has negative |max|.
 * @tparam math_t floating point type used for matrix elements
 * @tparam idx_t integer type used for indexing
 * @param[in] handle: raft handle
 * @param[inout] inout: input matrix. Result also stored in this parameter
 */
template <typename math_t, typename idx_t>
void sign_flip(raft::resources const& handle,
               raft::device_matrix_view<math_t, idx_t, col_major> inout)
{
  detail::signFlip(
    inout.data_handle(), inout.extent(0), inout.extent(1), resource::get_cuda_stream(handle));
}

/** @} */  // end group matrix_sign_flip
}  // namespace raft::matrix
