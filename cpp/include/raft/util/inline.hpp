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
