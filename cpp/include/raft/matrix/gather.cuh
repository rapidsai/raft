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

#pragma once
#include <raft/matrix/detail/gather.cuh>

namespace raft {
namespace matrix {

/**
 * @brief  gather copies rows from a source matrix into a destination matrix according to a map.
 *
 * @tparam MatrixIteratorT      Random-access iterator type, for reading input matrix (may be a
 * simple pointer type).
 * @tparam MapIteratorT         Random-access iterator type, for reading input map (may be a simple
 * pointer type).
 *
 * @param  in           Pointer to the input matrix (assumed to be row-major)
 * @param  D            Leading dimension of the input matrix 'in', which in-case of row-major
 * storage is the number of columns
 * @param  N            Second dimension
 * @param  map          Pointer to the input sequence of gather locations
 * @param  map_length   The length of 'map' and 'stencil'
 * @param  out          Pointer to the output matrix (assumed to be row-major)
 * @param  stream       CUDA stream to launch kernels within
 */
template <typename MatrixIteratorT, typename MapIteratorT>
void gather(const MatrixIteratorT in,
            int D,
            int N,
            MapIteratorT map,
            int map_length,
            MatrixIteratorT out,
            cudaStream_t stream)
{
  detail::gather(in, D, N, map, map_length, out, stream);
}

/**
 * @brief  gather copies rows from a source matrix into a destination matrix according to a
 * transformed map.
 *
 * @tparam MatrixIteratorT      Random-access iterator type, for reading input matrix (may be a
 * simple pointer type).
 * @tparam MapIteratorT         Random-access iterator type, for reading input map (may be a simple
 * pointer type).
 * @tparam MapTransformOp       Unary lambda expression or operator type, MapTransformOp's result
 * type must be convertible to IndexT (= int) type.
 *
 * @param  in           Pointer to the input matrix (assumed to be row-major)
 * @param  D            Leading dimension of the input matrix 'in', which in-case of row-major
 * storage is the number of columns
 * @param  N            Second dimension
 * @param  map          Pointer to the input sequence of gather locations
 * @param  map_length   The length of 'map' and 'stencil'
 * @param  out          Pointer to the output matrix (assumed to be row-major)
 * @param  transform_op The transformation operation, transforms the map values to IndexT
 * @param  stream       CUDA stream to launch kernels within
 */
template <typename MatrixIteratorT, typename MapIteratorT, typename MapTransformOp>
void gather(const MatrixIteratorT in,
            int D,
            int N,
            MapIteratorT map,
            int map_length,
            MatrixIteratorT out,
            MapTransformOp transform_op,
            cudaStream_t stream)
{
  detail::gather(in, D, N, map, map_length, out, transform_op, stream);
}

/**
 * @brief  gather_if conditionally copies rows from a source matrix into a destination matrix
 * according to a map.
 *
 * @tparam MatrixIteratorT      Random-access iterator type, for reading input matrix (may be a
 * simple pointer type).
 * @tparam MapIteratorT         Random-access iterator type, for reading input map (may be a simple
 * pointer type).
 * @tparam StencilIteratorT     Random-access iterator type, for reading input stencil (may be a
 * simple pointer type).
 * @tparam UnaryPredicateOp     Unary lambda expression or operator type, UnaryPredicateOp's result
 * type must be convertible to bool type.
 *
 * @param  in           Pointer to the input matrix (assumed to be row-major)
 * @param  D            Leading dimension of the input matrix 'in', which in-case of row-major
 * storage is the number of columns
 * @param  N            Second dimension
 * @param  map          Pointer to the input sequence of gather locations
 * @param  stencil      Pointer to the input sequence of stencil or predicate values
 * @param  map_length   The length of 'map' and 'stencil'
 * @param  out          Pointer to the output matrix (assumed to be row-major)
 * @param  pred_op      Predicate to apply to the stencil values
 * @param  stream       CUDA stream to launch kernels within
 */
template <typename MatrixIteratorT,
          typename MapIteratorT,
          typename StencilIteratorT,
          typename UnaryPredicateOp>
void gather_if(const MatrixIteratorT in,
               int D,
               int N,
               MapIteratorT map,
               StencilIteratorT stencil,
               int map_length,
               MatrixIteratorT out,
               UnaryPredicateOp pred_op,
               cudaStream_t stream)
{
  detail::gather_if(in, D, N, map, stencil, map_length, out, pred_op, stream);
}

/**
 * @brief  gather_if conditionally copies rows from a source matrix into a destination matrix
 * according to a transformed map.
 *
 * @tparam MatrixIteratorT      Random-access iterator type, for reading input matrix (may be a
 * simple pointer type).
 * @tparam MapIteratorT         Random-access iterator type, for reading input map (may be a simple
 * pointer type).
 * @tparam StencilIteratorT     Random-access iterator type, for reading input stencil (may be a
 * simple pointer type).
 * @tparam UnaryPredicateOp     Unary lambda expression or operator type, UnaryPredicateOp's result
 * type must be convertible to bool type.
 * @tparam MapTransformOp       Unary lambda expression or operator type, MapTransformOp's result
 * type must be convertible to IndexT (= int) type.
 *
 * @param  in           Pointer to the input matrix (assumed to be row-major)
 * @param  D            Leading dimension of the input matrix 'in', which in-case of row-major
 * storage is the number of columns
 * @param  N            Second dimension
 * @param  map          Pointer to the input sequence of gather locations
 * @param  stencil      Pointer to the input sequence of stencil or predicate values
 * @param  map_length   The length of 'map' and 'stencil'
 * @param  out          Pointer to the output matrix (assumed to be row-major)
 * @param  pred_op      Predicate to apply to the stencil values
 * @param  transform_op The transformation operation, transforms the map values to IndexT
 * @param  stream       CUDA stream to launch kernels within
 */
template <typename MatrixIteratorT,
          typename MapIteratorT,
          typename StencilIteratorT,
          typename UnaryPredicateOp,
          typename MapTransformOp>
void gather_if(const MatrixIteratorT in,
               int D,
               int N,
               MapIteratorT map,
               StencilIteratorT stencil,
               int map_length,
               MatrixIteratorT out,
               UnaryPredicateOp pred_op,
               MapTransformOp transform_op,
               cudaStream_t stream)
{
  detail::gather_if(in, D, N, map, stencil, map_length, out, pred_op, transform_op, stream);
}
}  // namespace matrix
}  // namespace raft
