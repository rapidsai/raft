/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#if defined(_RAFT_HAS_CUDA)
#define CUDA_CONDITION_ELSE_TRUE(condition)  condition
#define CUDA_CONDITION_ELSE_FALSE(condition) condition
#else
#define CUDA_CONDITION_ELSE_TRUE(condition)  true
#define CUDA_CONDITION_ELSE_FALSE(condition) false
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

#ifndef RAFT_DEVICE_INLINE_FUNCTION
#define RAFT_DEVICE_INLINE_FUNCTION _RAFT_DEVICE _RAFT_FORCEINLINE
#endif

// The RAFT_INLINE_CONDITIONAL is a conditional inline specifier that removes
// the inline specification when RAFT_COMPILED is defined.
//
// When RAFT_COMPILED is not defined, functions may be defined in multiple
// translation units and we do not want that to lead to linker errors.
//
// When RAFT_COMPILED is defined, this serves two purposes:
//
// 1. It triggers a multiple definition error message when memory_pool-inl.hpp
// (for instance) is accidentally included in multiple translation units.
//
// 2. We function definitions to be non-inline, because non-inline functions
// symbols are always exported in the object symbol table. For inline functions,
// the compiler may elide the external symbol, which results in linker errors.
#ifdef RAFT_COMPILED
#define RAFT_INLINE_CONDITIONAL
#else
#define RAFT_INLINE_CONDITIONAL inline
#endif  // RAFT_COMPILED

// The RAFT_WEAK_FUNCTION specificies that:
//
// 1. A function may be defined in multiple translation units (like inline)
//
// 2. Must still emit an external symbol (unlike inline). This enables declaring
// a function signature in an `-ext` header and defining it in a source file.
//
// From
// https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#Common-Function-Attributes:
//
// "The weak attribute causes a declaration of an external symbol to be emitted
// as a weak symbol rather than a global."
#define RAFT_WEAK_FUNCTION __attribute__((weak))

// The RAFT_HIDDEN_FUNCTION specificies that the function will be hidden
// and therefore not callable by consumers of raft when compiled as
// a shared library.
//
// Hidden visibility also ensures that the linker doesn't de-duplicate the
// symbol across multiple `.so`. This allows multiple libraries to embed raft
// without issue
#define RAFT_HIDDEN_FUNCTION __attribute__((visibility("hidden")))

// The RAFT_KERNEL specificies that a kernel has hidden visibility
//
// Raft needs to ensure that the visibility of its __global__ function
// templates have hidden visibility ( default is weak visibility).
//
// When kernls have weak visibility it means that if two dynamic libraries
// both contain identical instantiations of a RAFT template, then the linker
// will discard one of the two instantiations and use only one of them.
//
// Do to unique requirements of how the CUDA works this de-deduplication
// can lead to the wrong kernels being called ( SM version being wrong ),
// silently no kernel being called at all, or cuda runtime errors being
// thrown.
//
// https://github.com/rapidsai/raft/issues/1722
#if defined(__CUDACC_RDC__)
#define RAFT_KERNEL RAFT_HIDDEN_FUNCTION __global__ void
#elif defined(_RAFT_HAS_CUDA)
#define RAFT_KERNEL static __global__ void
#else
#define RAFT_KERNEL static void
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
