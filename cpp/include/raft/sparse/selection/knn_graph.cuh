/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * This file is deprecated and will be removed in release 22.06.
 * Please use the cuh version instead.
 */

/**
 * DISCLAIMER: this file is deprecated: use knn_graph.cuh instead
 */

#pragma once

#ifndef RAFT_HIDE_DEPRECATION_WARNINGS
#pragma message(__FILE__                                                  \
                " is deprecated and will be removed in a future release." \
                " Please use the sparse/spatial version instead.")
#endif

#include <raft/sparse/neighbors/knn_graph.cuh>

namespace raft::sparse::selection {
using raft::sparse::neighbors::knn_graph;
}
