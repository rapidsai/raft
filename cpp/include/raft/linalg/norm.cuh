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
#ifndef __NORM_H
#define __NORM_H

#pragma once

#include "linalg_types.hpp"
#include "detail/norm.cuh"

#include <raft/core/mdarray.hpp>

namespace raft {
namespace linalg {

/** different types of norms supported on the input buffers */
using detail::L1Norm;
using detail::L2Norm;
using detail::NormType;

/**
 * @brief Compute row-wise norm of the input matrix and perform fin_op lambda
 *
 * Row-wise norm is useful while computing pairwise distance matrix, for
 * example.
 * This is used in many clustering algos like knn, kmeans, dbscan, etc... The
 * current implementation is optimized only for bigger values of 'D'.
 *
 * @tparam Type the data type
 * @tparam Lambda device final lambda
 * @tparam IdxType Integer type used to for addressing
 * @param dots the output vector of row-wise dot products
 * @param data the input matrix (currently assumed to be row-major)
 * @param D number of columns of data
 * @param N number of rows of data
 * @param type the type of norm to be applied
 * @param rowMajor whether the input is row-major or not
 * @param stream cuda stream where to launch work
 * @param fin_op the final lambda op
 */
template <typename Type, typename IdxType = int, typename Lambda = raft::Nop<Type, IdxType>>
void rowNorm(Type* dots,
             const Type* data,
             IdxType D,
             IdxType N,
             NormType type,
             bool rowMajor,
             cudaStream_t stream,
             Lambda fin_op = raft::Nop<Type, IdxType>())
{
  detail::rowNormCaller(dots, data, D, N, type, rowMajor, stream, fin_op);
}

/**
 * @brief Compute column-wise norm of the input matrix and perform fin_op
 * @tparam Type the data type
 * @tparam Lambda device final lambda
 * @tparam IdxType Integer type used to for addressing
 * @param dots the output vector of column-wise dot products
 * @param data the input matrix (currently assumed to be row-major)
 * @param D number of columns of data
 * @param N number of rows of data
 * @param type the type of norm to be applied
 * @param rowMajor whether the input is row-major or not
 * @param stream cuda stream where to launch work
 * @param fin_op the final lambda op
 */
template <typename Type, typename IdxType = int, typename Lambda = raft::Nop<Type, IdxType>>
void colNorm(Type* dots,
             const Type* data,
             IdxType D,
             IdxType N,
             NormType type,
             bool rowMajor,
             cudaStream_t stream,
             Lambda fin_op = raft::Nop<Type, IdxType>())
{
  detail::colNormCaller(dots, data, D, N, type, rowMajor, stream, fin_op);
}

/**
 * @brief Compute norm of the input matrix and perform fin_op
 * @tparam ElementType Input/Output data type
 * @tparam LayoutPolicy the layout of input (raft::row_major or raft::col_major)
 * @tparam IdxType Integer type used to for addressing
 * @tparam Lambda device final lambda
 * @param[in] handle raft::handle_t
 * @param[in] in the input raft::device_matrix_view
 * @param[out] out the output raft::device_vector_view
 * @param[in] type the type of norm to be applied
 * @param[in] apply Whether to apply the norm along rows (raft::linalg::Apply::ALONG_ROWS)
                    or along columns (raft::linalg::Apply::ALONG_COLUMNS)
 * @param[in] fin_op the final lambda op
 */
template <typename ElementType,
          typename LayoutPolicy,
          typename IndexType,
          typename Lambda = raft::Nop<ElementType, IndexType>>
void norm(const raft::handle_t& handle,
          raft::device_matrix_view<const ElementType, IndexType, LayoutPolicy> in,
          raft::device_vector_view<ElementType, IndexType> out,
          NormType type,
          Apply apply,
          Lambda fin_op = raft::Nop<ElementType, IndexType>())
{
  RAFT_EXPECTS(out.is_exhaustive(), "Output must be contiguous");
  RAFT_EXPECTS(in.is_exhaustive(), "Input must be contiguous");

  auto constexpr row_major = std::is_same_v<typename decltype(out)::layout_type, raft::row_major>;
  auto along_rows          = apply == Apply::ALONG_ROWS;

  if (along_rows) {
    RAFT_EXPECTS(static_cast<IndexType>(out.size()) == in.extent(0),
                 "Output should be equal to number of rows in Input");
    rowNorm(out.data_handle(),
            in.data_handle(),
            in.extent(1),
            in.extent(0),
            type,
            row_major,
            handle.get_stream(),
            fin_op);
  } else {
    RAFT_EXPECTS(static_cast<IndexType>(out.size()) == in.extent(1),
                 "Output should be equal to number of columns in Input");
    colNorm(out.data_handle(),
            in.data_handle(),
            in.extent(1),
            in.extent(0),
            type,
            row_major,
            handle.get_stream(),
            fin_op);
  }
}

};  // end namespace linalg
};  // end namespace raft

#endif