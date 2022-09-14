/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

#ifndef __SUM_H
#define __SUM_H

#pragma once

#include "detail/sum.cuh"

#include <raft/util/cudart_utils.hpp>

namespace raft {
namespace stats {

/**
 * @brief Compute sum of the input matrix
 *
 * Sum operation is assumed to be performed on a given column.
 *
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @param output the output mean vector
 * @param input the input matrix
 * @param D number of columns of data
 * @param N number of rows of data
 * @param rowMajor whether the input data is row or col major
 * @param stream cuda stream where to launch work
 */
template <typename Type, typename IdxType = int>
void sum(Type* output, const Type* input, IdxType D, IdxType N, bool rowMajor, cudaStream_t stream)
{
  detail::sum(output, input, D, N, rowMajor, stream);
}

};  // end namespace stats
};  // end namespace raft

#endif