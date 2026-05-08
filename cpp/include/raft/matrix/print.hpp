/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/matrix/detail/print.hpp>
#include <raft/matrix/matrix_types.hpp>

namespace RAFT_EXPORT raft {
namespace matrix {

/**
 * @brief Prints the data stored in CPU memory
 * @param[in] in: input matrix with column-major layout
 * @param[in] separators: horizontal and vertical separator characters
 */
template <typename m_t, typename idx_t>
void print(raft::host_matrix_view<const m_t, idx_t, col_major> in, print_separators& separators)
{
  detail::printHost(
    in.data_handle(), in.extent(0), in.extent(1), separators.horizontal, separators.vertical);
}
}  // namespace matrix
}  // namespace RAFT_EXPORT raft
