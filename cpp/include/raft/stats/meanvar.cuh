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
#ifndef __MEANVAR_H
#define __MEANVAR_H

#pragma once

#include <raft/core/mdarray.hpp>
#include <raft/stats/detail/meanvar.cuh>

namespace raft::stats {

/**
 * @brief Compute mean and variance for each column of a given matrix.
 *
 * The operation is performed in a single sweep. Consider using it when you need to compute
 * both mean and variance, or when you need to compute variance but don't have the mean.
 * It's almost twice faster than running `mean` and `vars` sequentially, because all three
 * kernels are memory-bound.
 *
 * @tparam Type the data type
 * @tparam IdxType Integer type used for addressing
 * @param [out] mean the output mean vector of size D
 * @param [out] var the output variance vector of size D
 * @param [in] data the input matrix of size [N, D]
 * @param [in] D number of columns of data
 * @param [in] N number of rows of data
 * @param [in] sample whether to evaluate sample variance or not. In other words, whether to
 * normalize the variance using N-1 or N, for true or false respectively.
 * @param [in] rowMajor whether the input data is row- or col-major, for true or false respectively.
 * @param [in] stream
 */
template <typename Type, typename IdxType = int>
void meanvar(Type* mean,
             Type* var,
             const Type* data,
             IdxType D,
             IdxType N,
             bool sample,
             bool rowMajor,
             cudaStream_t stream)
{
  detail::meanvar(mean, var, data, D, N, sample, rowMajor, stream);
}

/**
 * @brief Compute mean and variance for each column of a given matrix.
 *
 * The operation is performed in a single sweep. Consider using it when you need to compute
 * both mean and variance, or when you need to compute variance but don't have the mean.
 * It's almost twice faster than running `mean` and `vars` sequentially, because all three
 * kernels are memory-bound.
 *
 * @tparam Type the data type
 * @tparam IdxType Integer type used for addressing
 * @tparam LayoutPolicy Layout type of the input matrix.
 * @tparam AccessorPolicy Accessor for the input and output, must be valid accessor on
 *                        device.
 * @param handle the raft handle
 * @param [out] mean the output mean vector of size D
 * @param [out] var the output variance vector of size D
 * @param [in] data the input matrix of size [N, D]
 * @param [in] sample whether to evaluate sample variance or not. In other words, whether to
 * normalize the variance using N-1 or N, for true or false respectively.
 */
template <typename Type, typename IdxType = int, typename LayoutPolicy = raft::row_major, typename AccessorPolicy>
void meanvar(const raft::handle_t& handle,
             raft::mdspan<Type, raft::vector_extent<IdxType>, LayoutPolicy, AccessorPolicy> mean,
             raft::mdspan<Type, raft::vector_extent<IdxType>, LayoutPolicy, AccessorPolicy> var,
             raft::mdspan<Type, raft::matrix_extent<IdxType>, LayoutPolicy, AccessorPolicy> data,
             bool sample)
{
  detail::meanvar(mean.data_handle(),
                  var.data_handle(),
                  data.data_handle(),
                  data.extent(1),
                  data.extent(0),
                  sample,
                  std::is_same_v<LayoutPolicy, raft::row_major>,
                  handle.get_stream());
}

};  // namespace raft::stats

#endif
