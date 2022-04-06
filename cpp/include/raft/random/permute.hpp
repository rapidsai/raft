/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
/**
 * This file is deprecated and will be removed in release 22.06.
 * Please use the cuh version instead.
 */

#ifndef __PERMUTE_H
#define __PERMUTE_H

#pragma once

#include "detail/permute.cuh"

namespace raft::random {

/**
 * @brief Generate permutations of the input array. Pretty useful primitive for
 * shuffling the input datasets in ML algos. See note at the end for some of its
 * limitations!
 * @tparam Type Data type of the array to be shuffled
 * @tparam IntType Integer type used for ther perms array
 * @tparam IdxType Integer type used for addressing indices
 * @tparam TPB threads per block
 * @param perms the output permutation indices. Typically useful only when
 * one wants to refer back. If you don't need this, pass a nullptr
 * @param out the output shuffled array. Pass nullptr if you don't want this to
 * be written. For eg: when you only want the perms array to be filled.
 * @param in input array (in-place is not supported due to race conditions!)
 * @param D number of columns of the input array
 * @param N length of the input array (or number of rows)
 * @param rowMajor whether the input/output matrices are row or col major
 * @param stream cuda stream where to launch the work
 *
 * @note This is NOT a uniform permutation generator! In fact, it only generates
 * very small percentage of permutations. If your application really requires a
 * high quality permutation generator, it is recommended that you pick
 * Knuth Shuffle.
 */
template <typename Type, typename IntType = int, typename IdxType = int, int TPB = 256>
void permute(IntType* perms,
             Type* out,
             const Type* in,
             IntType D,
             IntType N,
             bool rowMajor,
             cudaStream_t stream)
{
  detail::permute<Type, IntType, IdxType, TPB>(perms, out, in, D, N, rowMajor, stream);
}

};  // end namespace raft::random

#endif