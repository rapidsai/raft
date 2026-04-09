/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>

namespace raft::sparse::solver {

/**
 * @brief Configuration parameters for the sparse randomized SVD solver
 *
 * @tparam ValueTypeT Data type for values (float or double)
 */
template <typename ValueTypeT>
struct sparse_svd_config {
  /** @brief Number of singular values/vectors to compute */
  int n_components;

  /** @brief Number of extra random vectors for better approximation.
   *  Total subspace dimension is n_components + n_oversamples. */
  int n_oversamples = 10;

  /** @brief Number of power iteration passes. More iterations improve accuracy
   *  for matrices with slowly decaying singular values. */
  int n_power_iters = 2;

  /** @brief Random seed for reproducibility */
  std::optional<uint64_t> seed = std::nullopt;
};

}  // namespace raft::sparse::solver
