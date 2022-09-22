/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#ifndef _RAFT_HAS_CUDA
#if defined(__CUDACC__)
#define _RAFT_HAS_CUDA __CUDACC__
#endif
#endif

#ifndef _RAFT_HOST_DEVICE
#if defined(_RAFT_HAS_CUDA)
#define _RAFT_HOST_DEVICE __host__ __device__
#else
#define _RAFT_HOST_DEVICE
#endif
#endif

#ifndef RAFT_INLINE_FUNCTION
#define RAFT_INLINE_FUNCTION inline _RAFT_HOST_DEVICE
#endif
