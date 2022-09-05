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

#ifndef __HISTOGRAM_H
#define __HISTOGRAM_H

#pragma once

#include <raft/core/mdarray.hpp>
#include <raft/stats/common.hpp>
#include <raft/stats/detail/histogram.cuh>

// This file is a shameless amalgamation of independent works done by
// Lars Nyland and Andy Adinets

///@todo: add cub's histogram as another option

namespace raft {
namespace stats {

/**
 * @brief Perform histogram on the input data. It chooses the right load size
 * based on the input data vector length. It also supports large-bin cases
 * using a specialized smem-based hashing technique.
 * @tparam DataT input data type
 * @tparam IdxT data type used to compute indices
 * @tparam BinnerOp takes the input data and computes its bin index
 * @param type histogram implementation type to choose
 * @param bins the output bins (length = ncols * nbins)
 * @param nbins number of bins
 * @param data input data (length = ncols * nrows)
 * @param nrows data array length in each column (or batch)
 * @param ncols number of columns (or batch size)
 * @param stream cuda stream
 * @param binner the operation that computes the bin index of the input data
 *
 * @note signature of BinnerOp is `int func(DataT, IdxT);`
 */
template <typename DataT, typename IdxT = int, typename BinnerOp = IdentityBinner<DataT, IdxT>>
void histogram(HistType type,
               int* bins,
               IdxT nbins,
               const DataT* data,
               IdxT nrows,
               IdxT ncols,
               cudaStream_t stream,
               BinnerOp binner = IdentityBinner<DataT, IdxT>())
{
  detail::histogram<DataT, IdxT, BinnerOp>(type, bins, nbins, data, nrows, ncols, stream, binner);
}

/**
 * @brief Perform histogram on the input data. It chooses the right load size
 * based on the input data vector length. It also supports large-bin cases
 * using a specialized smem-based hashing technique.
 * @tparam DataT input data type
 * @tparam IdxType data type used to compute indices
 * @tparam BinnerOp takes the input data and computes its bin index
 * @tparam LayoutPolicy Layout type of the input data.
 * @tparam AccessorPolicy Accessor for the input and output, must be valid accessor on
 *                        device.
 * @param handle the raft handle
 * @param type histogram implementation type to choose
 * @param bins the output bins (length = ncols * nbins)
 * @param data input data (length = ncols * nrows)
 * @param binner the operation that computes the bin index of the input data
 *
 * @note signature of BinnerOp is `int func(DataT, IdxT);`
 */
template <typename DataT,
          typename IdxType  = int,
          typename BinnerOp = IdentityBinner<DataT, IdxType>,
          typename LayoutPolicy,
          typename AccessorPolicy>
void histogram(const raft::handle_t& handle,
               HistType type,
               raft::mdspan<int, raft::matrix_extent<IdxType>, LayoutPolicy, AccessorPolicy> bins,
               raft::mdspan<DataT, raft::matrix_extent<IdxType>, LayoutPolicy, AccessorPolicy> data,
               BinnerOp binner = IdentityBinner<DataT, IdxType>())
{
  detail::histogram<DataT, IdxType, BinnerOp>(type,
                                              bins.data_handle(),
                                              bins.extent(1),
                                              data.data_handle(),
                                              data.extent(0),
                                              data.extent(1),
                                              handle.get_stream(),
                                              binner);
}
};  // end namespace stats
};  // end namespace raft

#endif