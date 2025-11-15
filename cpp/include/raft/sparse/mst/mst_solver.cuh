
/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * This file is deprecated and will be removed in release 22.06.
 * Please use the cuh version instead.
 */
#pragma once

#ifndef RAFT_HIDE_DEPRECATION_WARNINGS
#pragma message(__FILE__                                                  \
                " is deprecated and will be removed in a future release." \
                " Please use the raft/sparse/solver version instead.")
#endif

#include <raft/sparse/solver/mst_solver.cuh>

namespace raft {
using raft::sparse::solver::Graph_COO;
}

namespace raft::mst {
using raft::sparse::solver::MST_solver;
}
