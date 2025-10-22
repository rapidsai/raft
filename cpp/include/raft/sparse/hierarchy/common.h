/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
                " Please use raft/cluster/single_linkage_types.hpp instead.")
#endif

#include <raft/cluster/single_linkage_types.hpp>

namespace raft::hierarchy {
using raft::cluster::linkage_output;
using raft::cluster::linkage_output_int;
using raft::cluster::linkage_output_int64;
using raft::cluster::LinkageDistance;
}  // namespace raft::hierarchy
