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
/**
 * @warning This file is deprecated and will be removed in release 22.06.
 * Please use the cuh version instead.
 */

#ifndef __MEAN_CENTER_H
#define __MEAN_CENTER_H

#pragma once

#include "detail/mean_center.cuh"

namespace raft {
namespace stats {

/**
 * @brief Center the input matrix wrt its mean
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @tparam TPB threads per block of the cuda kernel launched
 * @param out the output mean-centered matrix
 * @param data input matrix
 * @param mu the mean vector
 * @param D number of columns of data
 * @param N number of rows of data
 * @param rowMajor whether input is row or col major
 * @param bcastAlongRows whether to broadcast vector along rows or columns
 * @param stream cuda stream where to launch work
 */
template <typename Type, typename IdxType = int, int TPB = 256>
void meanCenter(Type* out,
                const Type* data,
                const Type* mu,
                IdxType D,
                IdxType N,
                bool rowMajor,
                bool bcastAlongRows,
                cudaStream_t stream)
{
  detail::meanCenter<Type, IdxType, TPB>(out, data, mu, D, N, rowMajor, bcastAlongRows, stream);
}

/**
 * @brief Add the input matrix wrt its mean
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @tparam TPB threads per block of the cuda kernel launched
 * @param out the output mean-added matrix
 * @param data input matrix
 * @param mu the mean vector
 * @param D number of columns of data
 * @param N number of rows of data
 * @param rowMajor whether input is row or col major
 * @param bcastAlongRows whether to broadcast vector along rows or columns
 * @param stream cuda stream where to launch work
 */
template <typename Type, typename IdxType = int, int TPB = 256>
void meanAdd(Type* out,
             const Type* data,
             const Type* mu,
             IdxType D,
             IdxType N,
             bool rowMajor,
             bool bcastAlongRows,
             cudaStream_t stream)
{
  detail::meanAdd<Type, IdxType, TPB>(out, data, mu, D, N, rowMajor, bcastAlongRows, stream);
}

};  // end namespace stats
};  // end namespace raft

#endif