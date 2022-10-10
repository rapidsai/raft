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
#define _RAFT_DEVICE      __device__
#define _RAFT_HOST        __host__
#define _RAFT_FORCEINLINE __forceinline__
#else
#define _RAFT_DEVICE
#define _RAFT_HOST
#define _RAFT_FORCEINLINE inline
#endif
#endif

#define _RAFT_HOST_DEVICE _RAFT_HOST _RAFT_DEVICE

#ifndef RAFT_INLINE_FUNCTION
#define RAFT_INLINE_FUNCTION _RAFT_HOST_DEVICE _RAFT_FORCEINLINE
#endif

/**
 * Some macro magic to remove optional parentheses of a macro argument.
 * See https://stackoverflow.com/a/62984543
 */
#ifndef RAFT_DEPAREN_MAGICRAFT_DEPAREN_H1
#define RAFT_DEPAREN(X)      RAFT_DEPAREN_H2(RAFT_DEPAREN_H1 X)
#define RAFT_DEPAREN_H1(...) RAFT_DEPAREN_H1 __VA_ARGS__
#define RAFT_DEPAREN_H2(...) RAFT_DEPAREN_H3(__VA_ARGS__)
#define RAFT_DEPAREN_H3(...) RAFT_DEPAREN_MAGIC##__VA_ARGS__
#define RAFT_DEPAREN_MAGICRAFT_DEPAREN_H1
#endif

#ifndef RAFT_STRINGIFY
#define RAFT_STRINGIFY_DETAIL(...) #__VA_ARGS__
#define RAFT_STRINGIFY(...)        RAFT_STRINGIFY_DETAIL(__VA_ARGS__)
#endif
