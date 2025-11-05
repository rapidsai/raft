/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#ifndef RAFT_DISABLE_CUDA
#pragma message(__FILE__                                             \
                " should only be used in CUDA-disabled RAFT builds." \
                " Please use equivalent .cuh header instead.")
#else
// It is safe to include this cuh file in an hpp header because all CUDA code
// is ifdef'd out for CUDA-disabled builds.
#include <raft/core/mdbuffer.cuh>
#endif
