/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#ifndef __REDUCE_H
#define __REDUCE_H

#pragma once

#include "detail/reduce.cuh"
#include "linalg_types.hpp"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/input_validation.hpp>

namespace raft {
namespace linalg {

/**
 * @brief Compute reduction of the input matrix along the requested dimension
 *        In case of an add-reduction, a compensated summation will be performed
 *        in order to reduce numerical error. Note that the compensation will not
 *        be equivalent to a sequential compensation to preserve parallel efficiency.
 *
 * @tparam InType the data type of the input
 * @tparam OutType the data type of the output (as well as the data type for
 *  which reduction is performed)
 * @tparam IdxType data type of the indices of the array
 * @tparam MainLambda Unary lambda applied while acculumation (eg: L1 or L2 norm)
 * It must be a 'callable' supporting the following input and output:
 * <pre>OutType (*MainLambda)(InType, IdxType);</pre>
 * @tparam ReduceLambda Binary lambda applied for reduction (eg: addition(+) for L2 norm)
 * It must be a 'callable' supporting the following input and output:
 * <pre>OutType (*ReduceLambda)(OutType);</pre>
 * @tparam FinalLambda the final lambda applied before STG (eg: Sqrt for L2 norm)
 * It must be a 'callable' supporting the following input and output:
 * <pre>OutType (*FinalLambda)(OutType);</pre>
 * @param dots the output reduction vector
 * @param data the input matrix
 * @param D number of columns
 * @param N number of rows
 * @param init initial value to use for the reduction
 * @param rowMajor input matrix is row-major or not
 * @param alongRows whether to reduce along rows or columns
 * @param stream cuda stream where to launch work
 * @param inplace reduction result added inplace or overwrites old values?
 * @param main_op elementwise operation to apply before reduction
 * @param reduce_op binary reduction operation
 * @param final_op elementwise operation to apply before storing results
 */
template <typename InType,
          typename OutType      = InType,
          typename IdxType      = int,
          typename MainLambda   = raft::identity_op,
          typename ReduceLambda = raft::add_op,
          typename FinalLambda  = raft::identity_op>
void reduce(OutType* dots,
            const InType* data,
            IdxType D,
            IdxType N,
            OutType init,
            bool rowMajor,
            bool alongRows,
            cudaStream_t stream,
            bool inplace           = false,
            MainLambda main_op     = raft::identity_op(),
            ReduceLambda reduce_op = raft::add_op(),
            FinalLambda final_op   = raft::identity_op())
{
  detail::reduce<InType, OutType, IdxType>(
    dots, data, D, N, init, rowMajor, alongRows, stream, inplace, main_op, reduce_op, final_op);
}

/**
 * @defgroup reduction Reduction Along Requested Dimension
 * @{
 */

/**
 * @brief Compute reduction of the input matrix along the requested dimension
 *        This API computes a reduction of a matrix whose underlying storage
 *        is either row-major or column-major, while allowing the choose the
 *        dimension for reduction. Depending upon the dimension chosen for
 *        reduction, the memory accesses may be coalesced or strided.
 *        In case of an add-reduction, a compensated summation will be performed
 *        in order to reduce numerical error. Note that the compensation will not
 *        be equivalent to a sequential compensation to preserve parallel efficiency.
 * @tparam InElementType the input data-type of underlying raft::matrix_view
 * @tparam LayoutPolicy The layout of Input/Output (row or col major)
 * @tparam OutElementType the output data-type of underlying raft::matrix_view and reduction
 * @tparam IndexType Integer type used to for addressing
 * @tparam MainLambda Unary lambda applied while acculumation (eg: L1 or L2 norm)
 * It must be a 'callable' supporting the following input and output:
 * <pre>OutType (*MainLambda)(InType, IdxType);</pre>
 * @tparam ReduceLambda Binary lambda applied for reduction (eg: addition(+) for L2 norm)
 * It must be a 'callable' supporting the following input and output:
 * <pre>OutType (*ReduceLambda)(OutType);</pre>
 * @tparam FinalLambda the final lambda applied before STG (eg: Sqrt for L2 norm)
 * It must be a 'callable' supporting the following input and output:
 * <pre>OutType (*FinalLambda)(OutType);</pre>
 * @param[in] handle raft::resources
 * @param[in] data Input of type raft::device_matrix_view
 * @param[out] dots Output of type raft::device_matrix_view
 * @param[in] init initial value to use for the reduction
 * @param[in] apply whether to reduce along rows or along columns (using raft::linalg::Apply)
 * @param[in] main_op fused elementwise operation to apply before reduction
 * @param[in] reduce_op fused binary reduction operation
 * @param[in] final_op fused elementwise operation to apply before storing results
 * @param[in] inplace reduction result added inplace or overwrites old values?
 */
template <typename InElementType,
          typename LayoutPolicy,
          typename OutElementType = InElementType,
          typename IdxType        = std::uint32_t,
          typename MainLambda     = raft::identity_op,
          typename ReduceLambda   = raft::add_op,
          typename FinalLambda    = raft::identity_op>
void reduce(raft::resources const& handle,
            raft::device_matrix_view<const InElementType, IdxType, LayoutPolicy> data,
            raft::device_vector_view<OutElementType, IdxType> dots,
            OutElementType init,
            Apply apply,
            bool inplace           = false,
            MainLambda main_op     = raft::identity_op(),
            ReduceLambda reduce_op = raft::add_op(),
            FinalLambda final_op   = raft::identity_op())
{
  RAFT_EXPECTS(raft::is_row_or_column_major(data), "Input must be contiguous");

  auto constexpr row_major = std::is_same_v<typename decltype(data)::layout_type, raft::row_major>;
  bool along_rows          = apply == Apply::ALONG_ROWS;

  if (along_rows) {
    RAFT_EXPECTS(static_cast<IdxType>(dots.size()) == data.extent(1),
                 "Output should be equal to number of columns in Input");
  } else {
    RAFT_EXPECTS(static_cast<IdxType>(dots.size()) == data.extent(0),
                 "Output should be equal to number of rows in Input");
  }

  reduce(dots.data_handle(),
         data.data_handle(),
         data.extent(1),
         data.extent(0),
         init,
         row_major,
         along_rows,
         resource::get_cuda_stream(handle),
         inplace,
         main_op,
         reduce_op,
         final_op);
}

/** @} */  // end of group reduction

};  // end namespace linalg
};  // end namespace raft

#endif