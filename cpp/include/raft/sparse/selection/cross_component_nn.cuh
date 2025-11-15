/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * This file is deprecated and will be removed in release 22.06.
 * Please use the cuh version instead.
 */

/**
 * DISCLAIMER: this file is deprecated: use cross_component_nn.cuh instead
 */

#pragma once

#ifndef RAFT_HIDE_DEPRECATION_WARNINGS
#pragma message(__FILE__                                                  \
                " is deprecated and will be removed in a future release." \
                " Please use the sparse/spatial version instead.")
#endif

#include <raft/sparse/neighbors/cross_component_nn.cuh>

namespace raft::linkage {
using raft::sparse::neighbors::cross_component_nn;
using raft::sparse::neighbors::FixConnectivitiesRedOp;
using raft::sparse::neighbors::get_n_components;
}  // namespace raft::linkage
