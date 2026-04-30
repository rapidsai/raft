/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>

#include <cstdint>

namespace RAFT_EXPORT raft {
namespace linalg {

/**
 * @brief Solver algorithm for PCA/TSVD eigen decomposition.
 *
 * @param COV_EIG_DQ covariance + divide-and-conquer eigen decomposition for symmetric matrices
 * @param COV_EIG_JACOBI covariance + Jacobi eigen decomposition for symmetric matrices
 */
enum class solver : int {
  COV_EIG_DQ,
  COV_EIG_JACOBI,
};

/** @brief Parameters for TSVD (and base for PCA). */
struct paramsTSVD {
  float tol             = 0.0;
  uint64_t n_iterations = 15;
  solver algorithm      = solver::COV_EIG_DQ;
};

/** @brief Parameters for PCA (extends TSVD with whitening / copy controls). */
struct paramsPCA : paramsTSVD {
  bool copy   = true;
  bool whiten = false;
};

};  // end namespace linalg
};  // end namespace RAFT_EXPORT raft
