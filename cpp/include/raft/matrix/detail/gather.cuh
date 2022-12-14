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

namespace raft {
namespace matrix {
namespace detail {

// gatherKernel conditionally copies rows from the source matrix 'in' into the destination matrix
// 'out' according to a map (or a transformed map)
template <int TPB,
          typename InputIteratorT,
          typename MapIteratorT,
          typename StencilIteratorT,
          typename PredicateOp,
          typename MapTransformOp,
          typename OutputIteratorT,
          typename IndexT>
__global__ void gatherKernel(const InputIteratorT in,
                             IndexT D,
                             IndexT N,
                             const MapIteratorT map,
                             StencilIteratorT stencil,
                             OutputIteratorT out,
                             PredicateOp pred_op,
                             MapTransformOp transform_op)
{
  typedef typename std::iterator_traits<MapIteratorT>::value_type MapValueT;
  typedef typename std::iterator_traits<StencilIteratorT>::value_type StencilValueT;

  IndexT outRowStart        = blockIdx.x * D;
  MapValueT map_val         = map[blockIdx.x];
  StencilValueT stencil_val = stencil[blockIdx.x];

  bool predicate = pred_op(stencil_val);
  if (predicate) {
    IndexT inRowStart = transform_op(map_val) * D;
    for (IndexT i = threadIdx.x; i < D; i += TPB) {
      out[outRowStart + i] = in[inRowStart + i];
    }
  }
}

/**
 * @brief  gather conditionally copies rows from a source matrix into a destination matrix according
 * to a transformed map.
 *
 * @tparam InputIteratorT       Random-access iterator type, for reading input matrix (may be a
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
void gatherImpl(const InputIteratorT in,
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
  // skip in case of 0 length input
  if (map_length <= 0 || N <= 0 || D <= 0) return;

  // map value type
  typedef typename std::iterator_traits<MapIteratorT>::value_type MapValueT;

  // stencil value type
  typedef typename std::iterator_traits<StencilIteratorT>::value_type StencilValueT;

  // return type of MapTransformOp, must be convertible to IndexT
  typedef typename std::result_of<decltype(transform_op)(MapValueT)>::type MapTransformOpReturnT;
  static_assert((std::is_convertible<MapTransformOpReturnT, IndexT>::value),
                "MapTransformOp's result type must be convertible to signed integer");

  // return type of UnaryPredicateOp, must be convertible to bool
  typedef typename std::result_of<decltype(pred_op)(StencilValueT)>::type PredicateOpReturnT;
  static_assert((std::is_convertible<PredicateOpReturnT, bool>::value),
                "UnaryPredicateOp's result type must be convertible to bool type");

  if (D <= 32) {
    gatherKernel<32>
      <<<map_length, 32, 0, stream>>>(in, D, N, map, stencil, out, pred_op, transform_op);
  } else if (D <= 64) {
    gatherKernel<64>
      <<<map_length, 64, 0, stream>>>(in, D, N, map, stencil, out, pred_op, transform_op);
  } else if (D <= 128) {
    gatherKernel<128>
      <<<map_length, 128, 0, stream>>>(in, D, N, map, stencil, out, pred_op, transform_op);
  } else {
    gatherKernel<256>
      <<<map_length, 256, 0, stream>>>(in, D, N, map, stencil, out, pred_op, transform_op);
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

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
  typedef typename std::iterator_traits<MapIteratorT>::value_type MapValueT;
  gatherImpl(
    in,
    D,
    N,
    map,
    map,
    map_length,
    out,
    [] __device__(MapValueT val) { return true; },
    [] __device__(MapValueT val) { return val; },
    stream);
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
  typedef typename std::iterator_traits<MapIteratorT>::value_type MapValueT;
  gatherImpl(
    in,
    D,
    N,
    map,
    map,
    map_length,
    out,
    [] __device__(MapValueT val) { return true; },
    transform_op,
    stream);
}

/**
 * @brief  gather_if conditionally copies rows from a source matrix into a destination matrix
 * according to a map.
 *
 * @tparam InputIteratorT       Random-access iterator type, for reading input matrix (may be a
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
  typedef typename std::iterator_traits<MapIteratorT>::value_type MapValueT;
  gatherImpl(
    in,
    D,
    N,
    map,
    stencil,
    map_length,
    out,
    pred_op,
    [] __device__(MapValueT val) { return val; },
    stream);
}

/**
 * @brief  gather_if conditionally copies rows from a source matrix into a destination matrix
 * according to a transformed map.
 *
 * @tparam InputIteratorT       Random-access iterator type, for reading input matrix (may be a
 * simple pointer type).
 * @tparam MapIteratorT         Random-access iterator type, for reading input map (may be a simple
 * pointer type).
 * @tparam StencilIteratorT     Random-access iterator type, for reading input stencil (may be a
 * simple pointer type).
 * @tparam UnaryPredicateOp     Unary lambda expression or operator type, UnaryPredicateOp's result
 * type must be convertible to bool type.
 * @tparam MapTransformOp       Unary lambda expression or operator type, MapTransformOp's result
 * type must be convertible to IndexT type.
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
  typedef typename std::iterator_traits<MapIteratorT>::value_type MapValueT;
  gatherImpl(in, D, N, map, stencil, map_length, out, pred_op, transform_op, stream);
}
}  // namespace detail
}  // namespace matrix
}  // namespace raft
