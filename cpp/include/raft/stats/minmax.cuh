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
#ifndef __MINMAX_H
#define __MINMAX_H

#pragma once

#include <optional>
#include <raft/core/mdarray.hpp>
#include <raft/stats/detail/minmax.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <limits>

namespace raft {
namespace stats {

/**
 * @brief Computes min/max across every column of the input matrix, as well as
 * optionally allow to subsample based on the given row/col ID mapping vectors
 *
 * @tparam T the data type
 * @tparam TPB number of threads per block
 * @param data input data
 * @param rowids actual row ID mappings. It is of length nrows. If you want to
 * skip this index lookup entirely, pass nullptr
 * @param colids actual col ID mappings. It is of length ncols. If you want to
 * skip this index lookup entirely, pass nullptr
 * @param nrows number of rows of data to be worked upon. The actual rows of the
 * input "data" can be bigger than this!
 * @param ncols number of cols of data to be worked upon. The actual cols of the
 * input "data" can be bigger than this!
 * @param row_stride stride (in number of elements) between 2 adjacent columns
 * @param globalmin final col-wise global minimum (size = ncols)
 * @param globalmax final col-wise global maximum (size = ncols)
 * @param sampledcols output sampled data. Pass nullptr if you don't need this
 * @param stream cuda stream
 * @note This method makes the following assumptions:
 * 1. input and output matrices are assumed to be col-major
 * 2. ncols is small enough to fit the whole of min/max values across all cols
 *    in shared memory
 */
template <typename T, int TPB = 512>
void minmax(const T* data,
            const unsigned* rowids,
            const unsigned* colids,
            int nrows,
            int ncols,
            int row_stride,
            T* globalmin,
            T* globalmax,
            T* sampledcols,
            cudaStream_t stream)
{
  detail::minmax<T, TPB>(
    data, rowids, colids, nrows, ncols, row_stride, globalmin, globalmax, sampledcols, stream);
}

/**
 * @brief Computes min/max across every column of the input matrix, as well as
 * optionally allow to subsample based on the given row/col ID mapping vectors
 *
 * @tparam DataT Data type of input matrix element.
 * @tparam IdxType Index type of matrix extent.
 * @tparam TPB number of threads per block
 * @param handle the raft handle
 * @param data input data col-major of size [nrows, ncols], unless rowids or
 * colids length is smaller
 * @param rowids actual row ID mappings. It is of length nrows. If you want to
 * skip this index lookup entirely, pass nullptr
 * @param colids actual col ID mappings. It is of length ncols. If you want to
 * skip this index lookup entirely, pass nullptr
 * @param globalmin final col-wise global minimum (size = ncols)
 * @param globalmax final col-wise global maximum (size = ncols)
 * @param sampledcols output sampled data. Pass nullptr if you don't need this
 * @param TPB threads_pre_block
 * @note This method makes the following assumptions:
 * 1. input and output matrices are assumed to be col-major
 * 2. ncols is small enough to fit the whole of min/max values across all cols
 *    in shared memory
 */
template <typename DataT, typename IdxType, int TPB = 512>
void minmax(const raft::handle_t& handle,
            raft::device_matrix_view<const DataT, IdxType, raft::col_major> data,
            std::optional<raft::device_vector_view<const unsigned, IdxType>> rowids,
            std::optional<raft::device_vector_view<const unsigned, IdxType>> colids,
            raft::device_vector_view<DataT, IdxType> globalmin,
            raft::device_vector_view<DataT, IdxType> globalmax,
            std::optional<raft::device_vector_view<DataT, IdxType>> sampledcols,
            std::integral_constant<int, TPB>)
{
  const unsigned* rowids_ptr = nullptr;
  const unsigned* colids_ptr = nullptr;
  DataT* sampledcols_ptr     = nullptr;
  auto nrows                 = data.extent(0);
  auto ncols                 = data.extent(1);
  auto row_stride            = data.stride(1);
  if (rowids.has_value()) {
    rowids_ptr = rowids.value().data_handle();
    nrows      = rowids.value().extent(0);
  }
  if (colids.has_value()) {
    colids_ptr = colids.value().data_handle();
    ncols      = colids.value().extent(0);
  }
  if (sampledcols.has_value()) { sampledcols_ptr = sampledcols.value().data_handle(); }
  detail::minmax<DataT, TPB>(data.data_handle(),
                             rowids_ptr,
                             colids_ptr,
                             nrows,
                             ncols,
                             row_stride,
                             globalmin.data_handle(),
                             globalmax.data_handle(),
                             sampledcols_ptr,
                             handle.get_stream());
}

};  // namespace stats
};  // namespace raft
#endif