/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
