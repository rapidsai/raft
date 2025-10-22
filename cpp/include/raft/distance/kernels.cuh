/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/distance/detail/kernels/gram_matrix.cuh>
#include <raft/distance/detail/kernels/kernel_factory.cuh>
#include <raft/distance/distance.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/util/cuda_utils.cuh>

namespace raft::distance::kernels {

// TODO: Need to expose formal APIs for this that are more consistent w/ other APIs in RAFT
using raft::distance::kernels::detail::GramMatrixBase;
using raft::distance::kernels::detail::KernelFactory;

};  // end namespace raft::distance::kernels
