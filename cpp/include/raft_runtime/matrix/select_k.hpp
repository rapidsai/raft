/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <optional>

namespace raft::runtime::matrix {
void select_k(const resources& handle,
              raft::device_matrix_view<const float, int64_t, row_major> in_val,
              std::optional<raft::device_matrix_view<const int64_t, int64_t, row_major>> in_idx,
              raft::device_matrix_view<float, int64_t, row_major> out_val,
              raft::device_matrix_view<int64_t, int64_t, row_major> out_idx,
              bool select_min);

}  // namespace raft::runtime::matrix
