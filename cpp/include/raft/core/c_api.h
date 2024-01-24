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

#include <stdint.h>

#include <cuda_runtime.h>

/**
 * @defgroup c_api C API Core Types and Functions
 * @{
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief An opaque C handle for C++ type `raft::resources`
 *
 */
typedef uintptr_t raftResources_t;

/**
 * @brief An enum denoting return values for function calls
 *
 */
typedef enum raftError_t { RAFT_ERROR, RAFT_SUCCESS } raftError_t;

/**
 * @brief Create an Initialized opaque C handle for C++ type `raft::resources`
 *
 * @param[in] res raftResources_t opaque C handle
 * @return raftError_t
 */
raftError_t raftCreateResources(raftResources_t* res);

/**
 * @brief Destroy and de-allocate opaque C handle for C++ type `raft::resources`
 *
 * @param[in] res raftResources_t opaque C handle
 * @return raftError_t
 */
raftError_t raftDestroyResources(raftResources_t res);

/**
 * @brief Set cudaStream_t on raftResources_t to queue CUDA kernels on APIs
 *        that accept a raftResources_t handle
 *
 * @param[in] res raftResources_t opaque C handle
 * @param[in] stream cudaStream_t stream to queue CUDA kernels
 * @return raftError_t
 */
raftError_t raftSetStream(raftResources_t res, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

/** @} */
