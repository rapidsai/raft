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

#include "detail/normalize.cuh"

namespace raft {
namespace linalg {

/**
 * @brief Divide rows by their L2 norm.
 *
 * Note that the implementation is efficient for matrices with a large number of rows, not "thick"
 * matrices with few long rows.
 *
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @param out the output matrix (row-major)
 * @param in the input matrix (row-major)
 * @param D number of columns of data
 * @param N number of rows of data
 * @param rowMajor whether the input is row-major or not
 * @param stream cuda stream where to launch work
 */
template <typename Type, typename IdxType = int>
void rowNormalize(Type* out, const Type* in, IdxType D, IdxType N, cudaStream_t stream)
{
  detail::coalescedNormalize(out, in, D, N, stream);
}

// todo(lsugy): mdspan API

}  // namespace linalg
}  // namespace raft
