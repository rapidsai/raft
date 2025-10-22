/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/distance/distance_types.hpp>

namespace raft::runtime::distance {

/**
 * @defgroup pairwise_distance_runtime Pairwise Distances Runtime API
 * @{
 */

void pairwise_distance(raft::resources const& handle,
                       float* x,
                       float* y,
                       float* dists,
                       int m,
                       int n,
                       int k,
                       raft::distance::DistanceType metric,
                       bool isRowMajor,
                       float metric_arg);

void pairwise_distance(raft::resources const& handle,
                       double* x,
                       double* y,
                       double* dists,
                       int m,
                       int n,
                       int k,
                       raft::distance::DistanceType metric,
                       bool isRowMajor,
                       float metric_arg);

/** @} */  // end group pairwise_distance_runtime

}  // namespace raft::runtime::distance
