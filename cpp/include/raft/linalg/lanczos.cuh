/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * This file is deprecated and will be removed in release 22.06.
 * Please use the cuh version instead.
 */

/**
 * DISCLAIMER: this file is deprecated: use lanczos.cuh instead
 */

#pragma once

#ifndef RAFT_HIDE_DEPRECATION_WARNINGS
#pragma message(__FILE__                                                  \
                " is deprecated and will be removed in a future release." \
                " Please use the sparse solvers version instead.")
#endif

#include <raft/sparse/solver/lanczos.cuh>

namespace raft::linalg {
using raft::sparse::solver::computeLargestEigenvectors;
using raft::sparse::solver::computeSmallestEigenvectors;
}  // namespace raft::linalg
