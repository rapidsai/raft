/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <raft/core/operators.hpp>
#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/util/cuda_utils.cuh>

#include <cub/cub.cuh>

namespace raft {
namespace stats {
namespace detail {

/**
 * @brief Compute stddev of the input matrix
 *
 * Stddev operation is assumed to be performed on a given column.
 *
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @param std the output stddev vector
 * @param data the input matrix
 * @param mu the mean vector
 * @param D number of columns of data
 * @param N number of rows of data
 * @param sample whether to evaluate sample stddev or not. In other words,
 * whether
 *  to normalize the output using N-1 or N, for true or false, respectively
 * @param rowMajor whether the input data is row or col major
 * @param stream cuda stream where to launch work
 */
template <typename Type, typename IdxType = int>
void stddev(Type* std,
            const Type* data,
            const Type* mu,
            IdxType D,
            IdxType N,
            bool sample,
            bool rowMajor,
            cudaStream_t stream)
{
  raft::linalg::reduce(
    std, data, D, N, Type(0), rowMajor, false, stream, false, [mu] __device__(Type a, IdxType i) {
      return a * a;
    });
  Type ratio      = Type(1) / ((sample) ? Type(N - 1) : Type(N));
  Type ratio_mean = sample ? ratio * Type(N) : Type(1);
  raft::linalg::binaryOp(std,
                         std,
                         mu,
                         D,
                         raft::compose_op(raft::sqrt_op(),
                                          raft::abs_op(),
                                          [ratio, ratio_mean] __device__(Type a, Type b) {
                                            return a * ratio - b * b * ratio_mean;
                                          }),
                         stream);
}

/**
 * @brief Compute variance of the input matrix
 *
 * Variance operation is assumed to be performed on a given column.
 *
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @param var the output stddev vector
 * @param data the input matrix
 * @param mu the mean vector
 * @param D number of columns of data
 * @param N number of rows of data
 * @param sample whether to evaluate sample stddev or not. In other words,
 * whether
 *  to normalize the output using N-1 or N, for true or false, respectively
 * @param rowMajor whether the input data is row or col major
 * @param stream cuda stream where to launch work
 */
template <typename Type, typename IdxType = int>
void vars(Type* var,
          const Type* data,
          const Type* mu,
          IdxType D,
          IdxType N,
          bool sample,
          bool rowMajor,
          cudaStream_t stream)
{
  raft::linalg::reduce(
    var, data, D, N, Type(0), rowMajor, false, stream, false, [mu] __device__(Type a, IdxType i) {
      return a * a;
    });
  Type ratio      = Type(1) / ((sample) ? Type(N - 1) : Type(N));
  Type ratio_mean = sample ? ratio * Type(N) : Type(1);
  raft::linalg::binaryOp(var,
                         var,
                         mu,
                         D,
                         raft::compose_op(raft::abs_op(),
                                          [ratio, ratio_mean] __device__(Type a, Type b) {
                                            return a * ratio - b * b * ratio_mean;
                                          }),
                         stream);
}

}  // namespace detail
}  // namespace stats
}  // namespace raft