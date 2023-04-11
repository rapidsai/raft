/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

/* This file has two responsibilities:
 *
 * 1. Dispatch to the correct implementation of a kernel based on the
 *    architecture of the device on which the kernel will be launched. For
 *    instance, the cosine distance has a CUTLASS-based implementation that can
 *    be used on SM80+ and the normal implementation that is used on older
 *    architectures.
 *
 * 2. Provide concise function templates that can be instantiated in
 *    src/distance/distance/specializations/detail/. Previously,
 *    raft::distance::detail::distance was instantiated. The function
 *    necessarily required a large set of include files, which slowed down the
 *    build. The raft::distance::detail::pairwise_matrix_arch_dispatch functions
 *    do not require as large an include files set, which speeds up the build.
 */

#if !defined(RAFT_EXPLICIT_INSTANTIATE)
#include "dispatch-inl.cuh"
#endif

#ifdef RAFT_COMPILED
#include "dispatch-ext.cuh"
#endif
