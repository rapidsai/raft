/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <functional>
#include <raft/core/operators.hpp>
#include <raft/util/cudart_utils.hpp>

namespace raft {
namespace matrix {
namespace detail {

/** Tiling policy for the gather kernel.
 *
 * The output matrix is considered as a flattened array, an approach that provides much better
 * performance than 1 row per block when D is small. Additionally, each thread works on multiple
 * output elements using an unrolled loop (approx. 30% faster than working on a single element)
 */
template <int tpb, int wpt>
struct gather_policy {
  static constexpr int n_threads       = tpb;
  static constexpr int work_per_thread = wpt;
  static constexpr int stride          = tpb * wpt;
};

/** Conditionally copies rows from the source matrix 'in' into the destination matrix
 * 'out' according to a map (or a transformed map) */
template <typename Policy,
          typename InputIteratorT,
          typename MapIteratorT,
          typename StencilIteratorT,
          typename PredicateOp,
          typename MapTransformOp,
          typename OutputIteratorT,
          typename IndexT>
RAFT_KERNEL gather_kernel(const InputIteratorT in,
                          IndexT D,
                          IndexT len,
                          const MapIteratorT map,
                          StencilIteratorT stencil,
                          OutputIteratorT out,
                          PredicateOp pred_op,
                          MapTransformOp transform_op)
{
  typedef typename std::iterator_traits<MapIteratorT>::value_type MapValueT;
  typedef typename std::iterator_traits<StencilIteratorT>::value_type StencilValueT;

#pragma unroll
  for (IndexT wid = 0; wid < Policy::work_per_thread; wid++) {
    IndexT tid = threadIdx.x + (Policy::work_per_thread * static_cast<IndexT>(blockIdx.x) + wid) *
                                 Policy::n_threads;
    if (tid < len) {
      IndexT i_dst = tid / D;
      IndexT j     = tid % D;

      MapValueT map_val         = map[i_dst];
      StencilValueT stencil_val = stencil[i_dst];

      bool predicate = pred_op(stencil_val);
      if (predicate) {
        IndexT i_src = transform_op(map_val);
        out[tid]     = in[i_src * D + j];
      }
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

  IndexT len        = map_length * D;
  constexpr int TPB = 128;
  const int n_sm    = raft::getMultiProcessorCount();
  // The following empirical heuristics enforce that we keep a good balance between having enough
  // blocks and enough work per thread.
  if (len < static_cast<IndexT>(32 * TPB * n_sm)) {
    using Policy    = gather_policy<TPB, 1>;
    IndexT n_blocks = raft::ceildiv(map_length * D, static_cast<IndexT>(Policy::stride));
    gather_kernel<Policy><<<n_blocks, Policy::n_threads, 0, stream>>>(
      in, D, len, map, stencil, out, pred_op, transform_op);
  } else if (len < static_cast<IndexT>(32 * 4 * TPB * n_sm)) {
    using Policy    = gather_policy<TPB, 4>;
    IndexT n_blocks = raft::ceildiv(map_length * D, static_cast<IndexT>(Policy::stride));
    gather_kernel<Policy><<<n_blocks, Policy::n_threads, 0, stream>>>(
      in, D, len, map, stencil, out, pred_op, transform_op);
  } else {
    using Policy    = gather_policy<TPB, 8>;
    IndexT n_blocks = raft::ceildiv(map_length * D, static_cast<IndexT>(Policy::stride));
    gather_kernel<Policy><<<n_blocks, Policy::n_threads, 0, stream>>>(
      in, D, len, map, stencil, out, pred_op, transform_op);
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
    in, D, N, map, map, map_length, out, raft::const_op(true), raft::identity_op(), stream);
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
  gatherImpl(in, D, N, map, map, map_length, out, raft::const_op(true), transform_op, stream);
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
  gatherImpl(in, D, N, map, stencil, map_length, out, pred_op, raft::identity_op(), stream);
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
