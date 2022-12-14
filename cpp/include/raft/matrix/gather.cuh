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

#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/matrix/detail/gather.cuh>

namespace raft::matrix {

/**
 * @defgroup matrix_gather Matrix gather operations
 * @{
 */

/**
 * @brief  gather copies rows from a source matrix into a destination matrix according to a map.
 *
 * @tparam InputIteratorT       Random-access iterator type, for reading input matrix (may be a
 * simple pointer type).
 * @tparam MapIteratorT         Random-access iterator type, for reading input map (may be a simple
 * pointer type).
 * @tparam OutputIteratorT      Random-access iterator type, for writing output matrix (may be a
 * simple pointer type).
 * @tparam IndexT               Index type.
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
template <typename InputIteratorT, typename MapIteratorT, typename OutputIteratorT, typename IndexT>
void gather(const InputIteratorT in,
            IndexT D,
            IndexT N,
            const MapIteratorT map,
            IndexT map_length,
            OutputIteratorT out,
            cudaStream_t stream)
{
  detail::gather(in, D, N, map, map_length, out, stream);
}

/**
 * @brief  gather copies rows from a source matrix into a destination matrix according to a
 * transformed map.
 *
 * @tparam InputIteratorT       Random-access iterator type, for reading input matrix (may be a
 * simple pointer type).
 * @tparam MapIteratorT         Random-access iterator type, for reading input map (may be a simple
 * pointer type).
 * @tparam MapTransformOp       Unary lambda expression or operator type, MapTransformOp's result
 * type must be convertible to IndexT.
 * @tparam OutputIteratorT      Random-access iterator type, for writing output matrix (may be a
 * simple pointer type).
 * @tparam IndexT               Index type.
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
template <typename InputIteratorT,
          typename MapIteratorT,
          typename MapTransformOp,
          typename OutputIteratorT,
          typename IndexT>
void gather(const InputIteratorT in,
            IndexT D,
            IndexT N,
            const MapIteratorT map,
            IndexT map_length,
            OutputIteratorT out,
            MapTransformOp transform_op,
            cudaStream_t stream)
{
  detail::gather(in, D, N, map, map_length, out, transform_op, stream);
}

/**
 * @brief  gather_if conditionally copies rows from a source matrix into a destination matrix
 * according to a map.
 *
 * @tparam InputIteratorT      Random-access iterator type, for reading input matrix (may be a
 * simple pointer type).
 * @tparam MapIteratorT         Random-access iterator type, for reading input map (may be a simple
 * pointer type).
 * @tparam StencilIteratorT     Random-access iterator type, for reading input stencil (may be a
 * simple pointer type).
 * @tparam UnaryPredicateOp     Unary lambda expression or operator type, UnaryPredicateOp's result
 * type must be convertible to bool type.
 * @tparam OutputIteratorT      Random-access iterator type, for writing output matrix (may be a
 * simple pointer type).
 * @tparam IndexT               Index type.
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
template <typename InputIteratorT,
          typename MapIteratorT,
          typename StencilIteratorT,
          typename UnaryPredicateOp,
          typename OutputIteratorT,
          typename IndexT>
void gather_if(const InputIteratorT in,
               IndexT D,
               IndexT N,
               const MapIteratorT map,
               StencilIteratorT stencil,
               IndexT map_length,
               OutputIteratorT out,
               UnaryPredicateOp pred_op,
               cudaStream_t stream)
{
  detail::gather_if(in, D, N, map, stencil, map_length, out, pred_op, stream);
}

/**
 * @brief  gather_if conditionally copies rows from a source matrix into a destination matrix
 * according to a transformed map.
 *
 * @tparam InputIteratorT      Random-access iterator type, for reading input matrix (may be a
 * simple pointer type).
 * @tparam MapIteratorT         Random-access iterator type, for reading input map (may be a simple
 * pointer type).
 * @tparam StencilIteratorT     Random-access iterator type, for reading input stencil (may be a
 * simple pointer type).
 * @tparam UnaryPredicateOp     Unary lambda expression or operator type, UnaryPredicateOp's result
 * type must be convertible to bool type.
 * @tparam MapTransformOp       Unary lambda expression or operator type, MapTransformOp's result
 * type must be convertible to IndexT.
 * @tparam OutputIteratorT      Random-access iterator type, for writing output matrix (may be a
 * simple pointer type).
 * @tparam IndexT               Index type.
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
template <typename InputIteratorT,
          typename MapIteratorT,
          typename StencilIteratorT,
          typename UnaryPredicateOp,
          typename MapTransformOp,
          typename OutputIteratorT,
          typename IndexT>
void gather_if(const InputIteratorT in,
               IndexT D,
               IndexT N,
               const MapIteratorT map,
               StencilIteratorT stencil,
               IndexT map_length,
               OutputIteratorT out,
               UnaryPredicateOp pred_op,
               MapTransformOp transform_op,
               cudaStream_t stream)
{
  detail::gather_if(in, D, N, map, stencil, map_length, out, pred_op, transform_op, stream);
}

/**
 * @brief  gather copies rows from a source matrix into a destination matrix according to a map.
 *
 * @tparam matrix_t      Matrix element type
 * @tparam map_t         Map vector type
 * @tparam idx_t integer type used for indexing
 * @param[in] handle            raft handle for managing resources
 * @param[in]  in           Input matrix (assumed to be row-major)
 * @param[in]  map          Vector of gather locations
 * @param[out]  out         Output matrix (assumed to be row-major)
 */
template <typename matrix_t, typename map_t, typename idx_t>
void gather(const raft::handle_t& handle,
            raft::device_matrix_view<const matrix_t, idx_t, row_major> in,
            raft::device_vector_view<const map_t, idx_t> map,
            raft::device_matrix_view<matrix_t, idx_t, row_major> out)
{
  RAFT_EXPECTS(out.extent(0) == map.extent(0),
               "Number of rows in output matrix must equal the size of the map vector");
  RAFT_EXPECTS(out.extent(1) == in.extent(1),
               "Number of columns in input and output matrices must be equal.");

  raft::matrix::detail::gather(
    const_cast<matrix_t*>(in.data_handle()),  // TODO: There's a better way to handle this
    in.extent(1),
    in.extent(0),
    map.data_handle(),
    map.extent(0),
    out.data_handle(),
    handle.get_stream());
}

/**
 * @brief  gather copies rows from a source matrix into a destination matrix according to a
 * transformed map.
 *
 * @tparam matrix_t     Matrix type
 * @tparam map_t        Map vector type
 * @tparam map_xform_t       Unary lambda expression or operator type, MapTransformOp's result
 * type must be convertible to idx_t.
 * @tparam idx_t integer type for indexing
 * @param[in] handle        raft handle for managing resources
 * @param[in]  in           Input matrix (assumed to be row-major)
 * @param[in]  map          Input vector of gather locations
 * @param[out]  out         Output matrix (assumed to be row-major)
 * @param[in]  transform_op The transformation operation, transforms the map values to idx_t
 */
template <typename matrix_t, typename map_t, typename map_xform_t, typename idx_t>
void gather(const raft::handle_t& handle,
            raft::device_matrix_view<const matrix_t, idx_t, row_major> in,
            raft::device_vector_view<const map_t, idx_t> map,
            raft::device_matrix_view<const matrix_t, idx_t, row_major> out,
            map_xform_t transform_op)
{
  RAFT_EXPECTS(out.extent(0) == map.extent(0),
               "Number of rows in output matrix must equal the size of the map vector");
  RAFT_EXPECTS(out.extent(1) == in.extent(1),
               "Number of columns in input and output matrices must be equal.");

  detail::gather(
    const_cast<matrix_t*>(in.data_handle()),  // TODO: There's a better way to handle this
    in.extent(1),
    in.extent(0),
    map,
    map.extent(0),
    out.data_handle(),
    transform_op,
    handle.get_stream());
}

/**
 * @brief  gather_if conditionally copies rows from a source matrix into a destination matrix
 * according to a map.
 *
 * @tparam matrix_t      Matrix value type
 * @tparam map_t         Map vector type
 * @tparam stencil_t     Stencil vector type
 * @tparam unary_pred_t     Unary lambda expression or operator type, unary_pred_t's result
 * type must be convertible to bool type.
 * @tparam idx_t integer type for indexing
 * @param[in] handle        raft handle for managing resources
 * @param[in]  in           Input matrix (assumed to be row-major)
 * @param[in]  map          Input vector of gather locations
 * @param[in]  stencil      Input vector of stencil or predicate values
 * @param[out]  out         Output matrix (assumed to be row-major)
 * @param[in]  pred_op      Predicate to apply to the stencil values
 */
template <typename matrix_t,
          typename map_t,
          typename stencil_t,
          typename unary_pred_t,
          typename idx_t>
void gather_if(const raft::handle_t& handle,
               raft::device_matrix_view<const matrix_t, idx_t, row_major> in,
               raft::device_matrix_view<matrix_t, idx_t, row_major> out,
               raft::device_vector_view<const map_t, idx_t> map,
               raft::device_vector_view<const stencil_t, idx_t> stencil,
               unary_pred_t pred_op)
{
  RAFT_EXPECTS(out.extent(0) == map.extent(0),
               "Number of rows in output matrix must equal the size of the map vector");
  RAFT_EXPECTS(out.extent(1) == in.extent(1),
               "Number of columns in input and output matrices must be equal.");
  RAFT_EXPECTS(map.extent(0) == stencil.extent(0),
               "Number of elements in stencil must equal number of elements in map");

  detail::gather_if(const_cast<matrix_t*>(in.data_handle()),
                    out.extent(1),
                    out.extent(0),
                    map.data_handle(),
                    stencil.data_handle(),
                    map.extent(0),
                    out.data_handle(),
                    pred_op,
                    handle.get_stream());
}

/**
 * @brief  gather_if conditionally copies rows from a source matrix into a destination matrix
 * according to a transformed map.
 *
 * @tparam matrix_t      Matrix value type, for reading input matrix
 * @tparam map_t         Vector value type for map
 * @tparam stencil_t     Vector value type for stencil
 * @tparam unary_pred_t     Unary lambda expression or operator type, unary_pred_t's result
 * type must be convertible to bool type.
 * @tparam map_xform_t       Unary lambda expression or operator type, map_xform_t's result
 * type must be convertible to idx_t.
 * @tparam idx_t integer type for indexing
 * @param[in] handle        raft handle for managing resources
 * @param[in]  in           Input matrix (assumed to be row-major)
 * @param[in]  map          Vector of gather locations
 * @param[in]  stencil      Vector of stencil or predicate values
 * @param[out]  out          Output matrix (assumed to be row-major)
 * @param[in]  pred_op      Predicate to apply to the stencil values
 * @param[in]  transform_op The transformation operation, transforms the map values to idx_t
 */
template <typename matrix_t,
          typename map_t,
          typename stencil_t,
          typename unary_pred_t,
          typename map_xform_t,
          typename idx_t>
void gather_if(const raft::handle_t& handle,
               raft::device_matrix_view<const matrix_t, idx_t, row_major> in,
               raft::device_matrix_view<matrix_t, idx_t, row_major> out,
               raft::device_vector_view<const map_t> map,
               raft::device_vector_view<const stencil_t> stencil,
               unary_pred_t pred_op,
               map_xform_t transform_op)
{
  RAFT_EXPECTS(out.extent(0) == map.extent(0),
               "Number of rows in output matrix must equal the size of the map vector");
  RAFT_EXPECTS(out.extent(1) == in.extent(1),
               "Number of columns in input and output matrices must be equal.");
  RAFT_EXPECTS(map.extent(0) == stencil.extent(0),
               "Number of elements in stencil must equal number of elements in map");

  detail::gather_if(const_cast<matrix_t*>(in.data_handle()),
                    in.extent(1),
                    in.extent(0),
                    map.data_handle(),
                    stencil.data_handle(),
                    map.extent(0),
                    out.data_handle(),
                    pred_op,
                    transform_op,
                    handle.get_stream());
}

/** @} */  // end of group matrix_gather

}  // namespace raft::matrix
