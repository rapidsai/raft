/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * This file is deprecated and will be removed in release 22.06.
 * Please use the cuh version instead.
 */

/**
 * DISCLAIMER: this file is deprecated: use epsilon_neighborhood.cuh instead
 */

#pragma once

#ifndef RAFT_HIDE_DEPRECATION_WARNINGS
#pragma message(__FILE__                                                  \
                " is deprecated and will be removed in a future release." \
                " Please use the raft::neighbors version instead.")
#endif

#include <raft/neighbors/epsilon_neighborhood.cuh>

namespace raft::spatial::knn {

using raft::neighbors::epsilon_neighborhood::eps_neighbors_l2sq;
using raft::neighbors::epsilon_neighborhood::epsUnexpL2SqNeighborhood;

}  // namespace raft::spatial::knn
