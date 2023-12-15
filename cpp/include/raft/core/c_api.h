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

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uintptr_t raftResources_t;

typedef enum raftError_t { RAFT_ERROR, RAFT_SUCCESS } raftError_t;

raftError_t raftCreateResources(raftResources_t* res);

raftError_t raftDestroyResources(raftResources_t res);

raftError_t raftSetStream(raftResources_t res, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
