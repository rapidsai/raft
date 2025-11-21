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

#include <raft/neighbors/ivf_pq.cuh>

namespace raft::spatial::knn::ivf_pq {

using raft::neighbors::ivf_pq::build;
using raft::neighbors::ivf_pq::extend;
using raft::neighbors::ivf_pq::search;

}  // namespace raft::spatial::knn::ivf_pq
