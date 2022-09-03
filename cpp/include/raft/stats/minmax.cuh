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
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/stats/detail/minmax.cuh>

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
 * @tparam T Data type of input matrix element.
 * @tparam IndexType Index type of matrix extent.
 * @tparam LayoutPolicy Layout type of the input matrix. When layout is strided, it can
 *                      be a submatrix of a larger matrix. Arbitrary stride is not supported.
 * @tparam AccessorPolicy Accessor for the input and output, must be valid accessor on
 *                        device.
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
 * @note This method makes the following assumptions:
 * 1. input and output matrices are assumed to be col-major
 * 2. ncols is small enough to fit the whole of min/max values across all cols
 *    in shared memory
 */
template <typename T,
          typename IdxType,
          typename LayoutPolicy,
          typename AccessorPolicy,
          int TPB = 512>
void minmax(const raft::handle_t& handle,
            raft::mdspan<T, raft::matrix_extent<IdxType>, LayoutPolicy, AccessorPolicy> data,
            std::optional<raft::mdspan<unsigned, raft::vector_extent<IdxType>>> rowids,
            std::optional<raft::mdspan<unsigned, raft::vector_extent<IdxType>>> colids,
            raft::mdspan<T, raft::vector_extent<IdxType>> globalmin,
            raft::mdspan<T, raft::vector_extent<IdxType>> globalmax,
            std::optional<raft::mdspan<T, raft::vector_extent<IdxType>>> sampledcols)
{
  static_assert(std::is_same_v<LayoutPolicy, raft::col_major>, "Data should be col-major");
  const unsigned* rowids_ptr = nullptr;
  const unsigned* colids_ptr = nullptr;
  T* sampledcols_ptr         = nullptr;
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
  detail::minmax<T, TPB>(data.data_handle(),
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