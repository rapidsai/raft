/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include <raft/linalg/distance_type.h>
#include <raft/distance/detail/distance.cuh>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

namespace raft {
namespace distance {

/**
* @brief Evaluate pairwise distances with the user epilogue lamba allowed
* @tparam DistanceType which distance to evaluate
* @tparam InType input argument type
* @tparam AccType accumulation type
* @tparam OutType output type
* @tparam FinalLambda user-defined epilogue lamba
* @tparam Index_ Index type
* @param x first set of points
* @param y second set of points
* @param dist output distance matrix
* @param m number of points in x
* @param n number of points in y
* @param k dimensionality
* @param workspace temporary workspace needed for computations
* @param worksize number of bytes of the workspace
* @param fin_op the final gemm epilogue lambda
* @param stream cuda stream
* @param isRowMajor whether the matrices are row-major or col-major
* @param metric_arg metric argument (used for Minkowski distance)
*
* @note fin_op: This is a device lambda which is supposed to operate upon the
* input which is AccType and returns the output in OutType. It's signature is
* as follows:  <pre>OutType fin_op(AccType in, int g_idx);</pre>. If one needs
* any other parameters, feel free to pass them via closure.
*/
template <raft::distance::DistanceType distanceType, typename InType,
          typename AccType, typename OutType, typename FinalLambda,
          typename Index_ = int>
void distance(const InType *x, const InType *y, OutType *dist, Index_ m,
              Index_ n, Index_ k, void *workspace, size_t worksize,
              FinalLambda fin_op, cudaStream_t stream, bool isRowMajor = true,
              InType metric_arg = 2.0f) {
  detail::distance<distanceType, InType, AccType, OutType, FinalLambda, Index_>(
    x, y, dist, m, n, k, workspace, worksize, fin_op, stream, isRowMajor,
    metric_arg);
}

/**
* @brief Evaluate pairwise distances for the simple use case
* @tparam DistanceType which distance to evaluate
* @tparam InType input argument type
* @tparam AccType accumulation type
* @tparam OutType output type
* @tparam Index_ Index type
* @param x first set of points
* @param y second set of points
* @param dist output distance matrix
* @param m number of points in x
* @param n number of points in y
* @param k dimensionality
* @param workspace temporary workspace needed for computations
* @param worksize number of bytes of the workspace
* @param stream cuda stream
* @param isRowMajor whether the matrices are row-major or col-major
* @param metric_arg metric argument (used for Minkowski distance)
*
* @note if workspace is passed as nullptr, this will return in
*  worksize, the number of bytes of workspace required
*/
template <raft::distance::DistanceType distanceType, typename InType,
          typename AccType, typename OutType, typename Index_ = int>
void distance(const InType *x, const InType *y, OutType *dist, Index_ m,
              Index_ n, Index_ k, void *workspace, size_t worksize,
              cudaStream_t stream, bool isRowMajor = true,
              InType metric_arg = 2.0f) {
  detail::distance<distanceType, InType, AccType, OutType, Index_>(
    x, y, dist, m, n, k, workspace, worksize, stream, isRowMajor, metric_arg);
}

/**
* @brief Return the exact workspace size to compute the distance
* @tparam DistanceType which distance to evaluate
* @tparam InType input argument type
* @tparam AccType accumulation type
* @tparam OutType output type
* @tparam Index_ Index type
* @param x first set of points
* @param y second set of points
* @param m number of points in x
* @param n number of points in y
* @param k dimensionality
*
* @note If the specified distanceType doesn't need the workspace at all, it
* returns 0.
*/
template <raft::distance::DistanceType distanceType, typename InType,
          typename AccType, typename OutType, typename Index_ = int>
size_t getWorkspaceSize(const InType *x, const InType *y, Index_ m, Index_ n,
                        Index_ k) {
  return detail::getWorkspaceSize<distanceType, InType, AccType, OutType,
                                  Index_>(x, y, m, n, k);
}

/**
* @brief Evaluate pairwise distances for the simple use case
* @tparam DistanceType which distance to evaluate
* @tparam InType input argument type
* @tparam AccType accumulation type
* @tparam OutType output type
* @tparam Index_ Index type
* @param x first set of points
* @param y second set of points
* @param dist output distance matrix
* @param m number of points in x
* @param n number of points in y
* @param k dimensionality
* @param stream cuda stream
* @param isRowMajor whether the matrices are row-major or col-major
* @param metric_arg metric argument (used for Minkowski distance)
*
* @note if workspace is passed as nullptr, this will return in
*  worksize, the number of bytes of workspace required
*/
template <raft::distance::DistanceType distanceType, typename InType,
          typename AccType, typename OutType, typename Index_ = int>
void distance(const InType *x, const InType *y, OutType *dist, Index_ m,
              Index_ n, Index_ k, cudaStream_t stream, bool isRowMajor = true,
              InType metric_arg = 2.0f) {
  rmm::device_uvector<char> workspace(0, stream);
  auto worksize =
    getWorkspaceSize<distanceType, InType, AccType, OutType, Index_>(x, y, m, n,
                                                                     k);
  workspace.resize(worksize, stream);
  detail::distance<distanceType, InType, AccType, OutType, Index_>(
    x, y, dist, m, n, k, workspace.data(), worksize, stream, isRowMajor,
    metric_arg);
}

/**
 * @defgroup pairwise_distance pairwise distance prims
 * @{
 * @brief Convenience wrapper around 'distance' prim to convert runtime metric
 * into compile time for the purpose of dispatch
 * @tparam Type input/accumulation/output data-type
 * @tparam Index_ indexing type
 * @param x first set of points
 * @param y second set of points
 * @param dist output distance matrix
 * @param m number of points in x
 * @param n number of points in y
 * @param k dimensionality
 * @param workspace temporary workspace buffer which can get resized as per the
 * needed workspace size
 * @param metric distance metric
 * @param stream cuda stream
 * @param isRowMajor whether the matrices are row-major or col-major
 */
template <typename Type, typename Index_ = int>
void pairwise_distance(const raft::handle_t &handle, const Type *x,
                       const Type *y, Type *dist, Index_ m, Index_ n, Index_ k,
                       rmm::device_uvector<char> &workspace,
                       raft::distance::DistanceType metric,
                       bool isRowMajor = true, Type metric_arg = 2.0f) {
  switch (metric) {
    case raft::distance::DistanceType::L2Expanded:
      detail::pairwise_distance_impl<Type, Index_,
                                     raft::distance::DistanceType::L2Expanded>(
        x, y, dist, m, n, k, workspace, handle.get_stream(), isRowMajor);
      break;
    case raft::distance::DistanceType::L2SqrtExpanded:
      detail::pairwise_distance_impl<
        Type, Index_, raft::distance::DistanceType::L2SqrtExpanded>(
        x, y, dist, m, n, k, workspace, handle.get_stream(), isRowMajor);
      break;
    case raft::distance::DistanceType::CosineExpanded:
      detail::pairwise_distance_impl<
        Type, Index_, raft::distance::DistanceType::CosineExpanded>(
        x, y, dist, m, n, k, workspace, handle.get_stream(), isRowMajor);
      break;
    case raft::distance::DistanceType::L1:
      detail::pairwise_distance_impl<Type, Index_,
                                     raft::distance::DistanceType::L1>(
        x, y, dist, m, n, k, workspace, handle.get_stream(), isRowMajor);
      break;
    case raft::distance::DistanceType::L2Unexpanded:
      detail::pairwise_distance_impl<
        Type, Index_, raft::distance::DistanceType::L2Unexpanded>(
        x, y, dist, m, n, k, workspace, handle.get_stream(), isRowMajor);
      break;
    case raft::distance::DistanceType::L2SqrtUnexpanded:
      detail::pairwise_distance_impl<
        Type, Index_, raft::distance::DistanceType::L2SqrtUnexpanded>(
        x, y, dist, m, n, k, workspace, handle.get_stream(), isRowMajor);
      break;
    case raft::distance::DistanceType::Linf:
      detail::pairwise_distance_impl<Type, Index_,
                                     raft::distance::DistanceType::Linf>(
        x, y, dist, m, n, k, workspace, handle.get_stream(), isRowMajor);
      break;
    case raft::distance::DistanceType::HellingerExpanded:
      detail::pairwise_distance_impl<
        Type, Index_, raft::distance::DistanceType::HellingerExpanded>(
        x, y, dist, m, n, k, workspace, handle.get_stream(), isRowMajor);
      break;
    case raft::distance::DistanceType::LpUnexpanded:
      detail::pairwise_distance_impl<
        Type, Index_, raft::distance::DistanceType::LpUnexpanded>(
        x, y, dist, m, n, k, workspace, handle.get_stream(), isRowMajor,
        metric_arg);
      break;
    case raft::distance::DistanceType::Canberra:
      detail::pairwise_distance_impl<Type, Index_,
                                     raft::distance::DistanceType::Canberra>(
        x, y, dist, m, n, k, workspace, handle.get_stream(), isRowMajor);
      break;
    case raft::distance::DistanceType::HammingUnexpanded:
      detail::pairwise_distance_impl<
        Type, Index_, raft::distance::DistanceType::HammingUnexpanded>(
        x, y, dist, m, n, k, workspace, handle.get_stream(), isRowMajor);
      break;
    case raft::distance::DistanceType::JensenShannon:
      detail::pairwise_distance_impl<
        Type, Index_, raft::distance::DistanceType::JensenShannon>(
        x, y, dist, m, n, k, workspace, handle.get_stream(), isRowMajor);
      break;
    case raft::distance::DistanceType::RusselRaoExpanded:
      detail::pairwise_distance_impl<
        Type, Index_, raft::distance::DistanceType::RusselRaoExpanded>(
        x, y, dist, m, n, k, workspace, handle.get_stream(), isRowMajor);
      break;
    case raft::distance::DistanceType::KLDivergence:
      detail::pairwise_distance_impl<
        Type, Index_, raft::distance::DistanceType::KLDivergence>(
        x, y, dist, m, n, k, workspace, handle.get_stream(), isRowMajor);
      break;
    case raft::distance::DistanceType::CorrelationExpanded:
      detail::pairwise_distance_impl<
        Type, Index_, raft::distance::DistanceType::CorrelationExpanded>(
        x, y, dist, m, n, k, workspace, handle.get_stream(), isRowMajor);
      break;
    default:
      THROW("Unknown or unsupported distance metric '%d'!", (int)metric);
  };
}
/** @} */

/**
     * @defgroup pairwise_distance pairwise distance prims
     * @{
     * @brief Convenience wrapper around 'distance' prim to convert runtime metric
     * into compile time for the purpose of dispatch
     * @tparam Type input/accumulation/output data-type
     * @tparam Index_ indexing type
     * @param x first set of points
     * @param y second set of points
     * @param dist output distance matrix
     * @param m number of points in x
     * @param n number of points in y
     * @param k dimensionality
     * @param metric distance metric
     * @param stream cuda stream
     * @param isRowMajor whether the matrices are row-major or col-major
     */
template <typename Type, typename Index_ = int>
void pairwise_distance(const raft::handle_t &handle, const Type *x,
                       const Type *y, Type *dist, Index_ m, Index_ n, Index_ k,
                       raft::distance::DistanceType metric,
                       bool isRowMajor = true, Type metric_arg = 2.0f) {
  rmm::device_uvector<char> workspace(0, handle.get_stream());
  pairwise_distance<Type, Index_>(handle, x, y, dist, m, n, k, workspace,
                                  metric, isRowMajor, metric_arg);
}

};  // namespace distance
};  // namespace raft
