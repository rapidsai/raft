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

#pragma once

#include <raft/linalg/reduce.cuh>
#include <raft/stats/sum.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

namespace raft {
namespace stats {
namespace detail {

/**
 * @brief Compute the row-wise weighted mean of the input matrix with a
 * vector of weights
 *
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @param mu the output mean vector
 * @param data the input matrix
 * @param weights weight of size D if along_row is true, else of size N
 * @param D number of columns of data
 * @param N number of rows of data
 * @param row_major data input matrix is row-major or not
 * @param along_rows whether to reduce along rows or columns
 * @param stream cuda stream to launch work on
 */
template <typename Type, typename IdxType = int>
void weightedMean(Type* mu,
                  const Type* data,
                  const Type* weights,
                  IdxType D,
                  IdxType N,
                  bool row_major,
                  bool along_rows,
                  cudaStream_t stream)
{
  // sum the weights & copy back to CPU
  auto weight_size = along_rows ? D : N;
  Type WS          = 0;
  raft::stats::sum(mu, weights, (IdxType)1, weight_size, false, stream);
  raft::update_host(&WS, mu, 1, stream);

  raft::linalg::reduce(
    mu,
    data,
    D,
    N,
    (Type)0,
    row_major,
    along_rows,
    stream,
    false,
    [weights] __device__(Type v, IdxType i) { return v * weights[i]; },
    raft::add_op{},
    raft::div_const_op<Type>(WS));
}
};  // end namespace detail
};  // end namespace stats
};  // end namespace raft