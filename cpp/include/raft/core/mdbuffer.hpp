/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#ifndef RAFT_DISABLE_CUDA
#pragma message(__FILE__                                               \
                  " should only be used in CUDA-disabled RAFT builds." \
                  " Please use equivalent .cuh header instead.")
#else
// It is safe to include this cuh file in an hpp header because all CUDA code
// is ifdef'd out for CUDA-disabled builds.
#include <raft/core/mdbuffer.cuh>
#endif
