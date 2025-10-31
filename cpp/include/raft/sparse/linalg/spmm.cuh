/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#ifndef RAFT_HIDE_DEPRECATION_WARNINGS
#pragma message(__FILE__                                                  \
                " is deprecated and will be removed in a future release." \
                " Please use the spmm.hpp at the same path instead.")
#endif

#include <raft/sparse/linalg/detail/spmm.hpp>
