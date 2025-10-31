/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace raft::sparse::solver {

/**
 * @enum LANCZOS_WHICH
 * @brief Enumeration specifying which eigenvalues to compute in the Lanczos algorithm
 */
enum LANCZOS_WHICH {
  /** @brief LA: Largest (algebraic) eigenvalues */
  LA,
  /** @brief LM: Largest (in magnitude) eigenvalues */
  LM,
  /** @brief SA: Smallest (algebraic) eigenvalues */
  SA,
  /** @brief SM: Smallest (in magnitude) eigenvalues */
  SM
};

/**
 * @brief Configuration parameters for the Lanczos eigensolver
 *
 * This structure encapsulates all configuration parameters needed to run the
 * Lanczos algorithm for computing eigenvalues and eigenvectors of large sparse matrices.
 *
 * @tparam ValueTypeT Data type for values (float or double)
 */
template <typename ValueTypeT>
struct lanczos_solver_config {
  /** @brief The number of eigenvalues and eigenvectors to compute
   *  @note Must be 1 <= n_components < n, where n is the matrix dimension
   */
  int n_components;

  /** @brief Maximum number of iterations allowed for the algorithm to converge */
  int max_iterations;

  /** @brief The number of Lanczos vectors to generate
   *  @note Must satisfy n_components + 1 < ncv < n, where n is the matrix dimension
   */
  int ncv;

  /** @brief Convergence tolerance for residuals
   *  @note Used to determine when to stop iteration based on ||Ax - wx|| < tolerance
   */
  ValueTypeT tolerance;

  /** @brief Specifies which eigenvalues to compute in the Lanczos algorithm
   *  @see LANCZOS_WHICH for possible values (SA, LA, SM, LM)
   */
  LANCZOS_WHICH which;

  /** @brief Random seed for initialization of the algorithm
   *  @note Controls reproducibility of results
   */
  uint64_t seed;
};

}  // namespace raft::sparse::solver
