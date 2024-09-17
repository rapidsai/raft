/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/detail/distance.cuh>
#include <raft/distance/distance_types.hpp>

#include <rmm/device_uvector.hpp>

#include <type_traits>

namespace raft {
namespace distance {

/**
 * @defgroup pairwise_distance pointer-based pairwise distance prims
 * @{
 */

/**
 * @brief Evaluate pairwise distances with the user epilogue lamba allowed
 * @tparam DistanceType which distance to evaluate
 * @tparam DataT input argument type
 * @tparam AccT accumulation type
 * @tparam OutT output type
 * @tparam FinalLambda user-defined epilogue lamba
 * @tparam IdxT Index type
 * @param handle raft handle for managing expensive resources
 * @param x first set of points
 * @param y second set of points
 * @param dist output distance matrix
 * @param m number of points in x
 * @param n number of points in y
 * @param k dimensionality
 * @param workspace temporary workspace needed for computations
 * @param worksize number of bytes of the workspace
 * @param fin_op the final gemm epilogue lambda
 * @param isRowMajor whether the matrices are row-major or col-major
 * @param metric_arg metric argument (used for Minkowski distance)
 *
 * @note fin_op: This is a device lambda which is supposed to operate upon the
 * input which is AccT and returns the output in OutT. It's signature is
 * as follows:  <pre>OutT fin_op(AccT in, int g_idx);</pre>. If one needs
 * any other parameters, feel free to pass them via closure.
 */
template <raft::distance::DistanceType DistT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename FinalLambda,
          typename IdxT = int>
void distance(raft::resources const& handle,
              const DataT* x,
              const DataT* y,
              OutT* dist,
              IdxT m,
              IdxT n,
              IdxT k,
              void* workspace,
              size_t worksize,
              FinalLambda fin_op,
              bool isRowMajor  = true,
              DataT metric_arg = 2.0f)
{
  detail::distance<DistT, DataT, AccT, OutT, FinalLambda, IdxT>(
    handle, x, y, dist, m, n, k, workspace, worksize, fin_op, isRowMajor, metric_arg);
}

/**
 * @brief Evaluate pairwise distances for the simple use case
 * @tparam DistanceType which distance to evaluate
 * @tparam DataT input argument type
 * @tparam AccT accumulation type
 * @tparam OutT output type
 * @tparam IdxT Index type
 * @param handle raft handle for managing expensive resources
 * @param x first set of points
 * @param y second set of points
 * @param dist output distance matrix
 * @param m number of points in x
 * @param n number of points in y
 * @param k dimensionality
 * @param workspace temporary workspace needed for computations
 * @param worksize number of bytes of the workspace
 * @param isRowMajor whether the matrices are row-major or col-major
 * @param metric_arg metric argument (used for Minkowski distance)
 */
template <raft::distance::DistanceType DistT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT = int>
void distance(raft::resources const& handle,
              const DataT* x,
              const DataT* y,
              OutT* dist,
              IdxT m,
              IdxT n,
              IdxT k,
              void* workspace,
              size_t worksize,
              bool isRowMajor  = true,
              DataT metric_arg = 2.0f)
{
  detail::distance<DistT, DataT, AccT, OutT, IdxT>(
    handle, x, y, dist, m, n, k, workspace, worksize, isRowMajor, metric_arg);
}

/**
 * @brief Return the exact workspace size to compute the distance
 * @tparam DistanceType which distance to evaluate
 * @tparam DataT input argument type
 * @tparam AccT accumulation type
 * @tparam OutT output type
 * @tparam IdxT Index type
 * @param x first set of points
 * @param y second set of points
 * @param m number of points in x
 * @param n number of points in y
 * @param k dimensionality
 *
 * @note If the specified DistT doesn't need the workspace at all, it
 * returns 0.
 */
template <raft::distance::DistanceType DistT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT = int>
size_t getWorkspaceSize(const DataT* x, const DataT* y, IdxT m, IdxT n, IdxT k)
{
  return detail::getWorkspaceSize<DistT, DataT, AccT, OutT, IdxT>(x, y, m, n, k);
}

/**
 * @brief Return the exact workspace size to compute the distance
 * @tparam DistanceType which distance to evaluate
 * @tparam DataT input argument type
 * @tparam AccT accumulation type
 * @tparam OutT output type
 * @tparam IdxT Index type
 * @param x first set of points (size m*k)
 * @param y second set of points (size n*k)
 * @return number of bytes needed in workspace
 *
 * @note If the specified DistT doesn't need the workspace at all, it
 * returns 0.
 */
template <raft::distance::DistanceType DistT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT = int,
          typename layout>
size_t getWorkspaceSize(raft::device_matrix_view<DataT, IdxT, layout> const& x,
                        raft::device_matrix_view<DataT, IdxT, layout> const& y)
{
  RAFT_EXPECTS(x.extent(1) == y.extent(1), "Number of columns must be equal.");

  return getWorkspaceSize<DistT, DataT, AccT, OutT, IdxT>(
    x.data_handle(), y.data_handle(), x.extent(0), y.extent(0), x.extent(1));
}

/**
 * @brief Evaluate pairwise distances for the simple use case
 * @tparam DistanceType which distance to evaluate
 * @tparam DataT input argument type
 * @tparam AccT accumulation type
 * @tparam OutT output type
 * @tparam IdxT Index type
 * @param handle raft handle for managing expensive resources
 * @param x first set of points
 * @param y second set of points
 * @param dist output distance matrix
 * @param m number of points in x
 * @param n number of points in y
 * @param k dimensionality
 * @param isRowMajor whether the matrices are row-major or col-major
 * @param metric_arg metric argument (used for Minkowski distance)
 */
template <raft::distance::DistanceType DistT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT = int>
void distance(raft::resources const& handle,
              const DataT* x,
              const DataT* y,
              OutT* dist,
              IdxT m,
              IdxT n,
              IdxT k,
              bool isRowMajor  = true,
              DataT metric_arg = 2.0f)
{
  auto stream = raft::resource::get_cuda_stream(handle);
  rmm::device_uvector<char> workspace(0, stream);
  auto worksize = getWorkspaceSize<DistT, DataT, AccT, OutT, IdxT>(x, y, m, n, k);
  workspace.resize(worksize, stream);
  detail::distance<DistT, DataT, AccT, OutT, IdxT>(
    handle, x, y, dist, m, n, k, workspace.data(), worksize, isRowMajor, metric_arg);
}

/**
 * @brief Convenience wrapper around 'distance' prim to convert runtime metric
 * into compile time for the purpose of dispatch
 * @tparam Type input/accumulation/output data-type
 * @tparam IdxT indexing type
 * @param handle raft handle for managing expensive resources
 * @param x first set of points
 * @param y second set of points
 * @param dist output distance matrix
 * @param m number of points in x
 * @param n number of points in y
 * @param k dimensionality
 * @param workspace temporary workspace buffer which can get resized as per the
 * needed workspace size
 * @param metric distance metric
 * @param isRowMajor whether the matrices are row-major or col-major
 * @param metric_arg metric argument (used for Minkowski distance)
 */
template <typename Type, typename IdxT = int>
void pairwise_distance(raft::resources const& handle,
                       const Type* x,
                       const Type* y,
                       Type* dist,
                       IdxT m,
                       IdxT n,
                       IdxT k,
                       rmm::device_uvector<char>& workspace,
                       raft::distance::DistanceType metric,
                       bool isRowMajor = true,
                       Type metric_arg = 2.0f)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  auto dispatch = [&](auto distance_type) {
    auto worksize = getWorkspaceSize<distance_type(), Type, Type, Type, IdxT>(x, y, m, n, k);
    workspace.resize(worksize, stream);
    detail::distance<distance_type(), Type, Type, Type, IdxT>(
      handle, x, y, dist, m, n, k, workspace.data(), worksize, isRowMajor, metric_arg);
  };

  switch (metric) {
    case DistanceType::Canberra:
      dispatch(std::integral_constant<DistanceType, DistanceType::Canberra>{});
      break;
    case DistanceType::CorrelationExpanded:
      dispatch(std::integral_constant<DistanceType, DistanceType::CorrelationExpanded>{});
      break;
    case DistanceType::CosineExpanded:
      dispatch(std::integral_constant<DistanceType, DistanceType::CosineExpanded>{});
      break;
    case DistanceType::HammingUnexpanded:
      dispatch(std::integral_constant<DistanceType, DistanceType::HammingUnexpanded>{});
      break;
    case DistanceType::HellingerExpanded:
      dispatch(std::integral_constant<DistanceType, DistanceType::HellingerExpanded>{});
      break;
    case raft::distance::DistanceType::InnerProduct:
      dispatch(std::integral_constant<DistanceType, DistanceType::InnerProduct>{});
      break;
    case DistanceType::JensenShannon:
      dispatch(std::integral_constant<DistanceType, DistanceType::JensenShannon>{});
      break;
    case DistanceType::KLDivergence:
      dispatch(std::integral_constant<DistanceType, DistanceType::KLDivergence>{});
      break;
    case DistanceType::L1:
      dispatch(std::integral_constant<DistanceType, DistanceType::L1>{});
      break;
    case DistanceType::L2Expanded:
      dispatch(std::integral_constant<DistanceType, DistanceType::L2Expanded>{});
      break;
    case DistanceType::L2SqrtExpanded:
      dispatch(std::integral_constant<DistanceType, DistanceType::L2SqrtExpanded>{});
      break;
    case DistanceType::L2SqrtUnexpanded:
      dispatch(std::integral_constant<DistanceType, DistanceType::L2SqrtUnexpanded>{});
      break;
    case DistanceType::L2Unexpanded:
      dispatch(std::integral_constant<DistanceType, DistanceType::L2Unexpanded>{});
      break;
    case DistanceType::Linf:
      dispatch(std::integral_constant<DistanceType, DistanceType::Linf>{});
      break;
    case DistanceType::LpUnexpanded:
      dispatch(std::integral_constant<DistanceType, DistanceType::LpUnexpanded>{});
      break;
    case DistanceType::RusselRaoExpanded:
      dispatch(std::integral_constant<DistanceType, DistanceType::RusselRaoExpanded>{});
      break;
    case DistanceType::DiceExpanded:
      dispatch(std::integral_constant<DistanceType, DistanceType::DiceExpanded>{});
      break;
    default: THROW("Unknown or unsupported distance metric '%d'!", (int)metric);
  };
}

/**
 * @brief Convenience wrapper around 'distance' prim to convert runtime metric
 * into compile time for the purpose of dispatch
 * @tparam Type input/accumulation/output data-type
 * @tparam IdxT indexing type
 * @param handle raft handle for managing expensive resources
 * @param x first set of points
 * @param y second set of points
 * @param dist output distance matrix
 * @param m number of points in x
 * @param n number of points in y
 * @param k dimensionality
 * @param metric distance metric
 * @param isRowMajor whether the matrices are row-major or col-major
 * @param metric_arg metric argument (used for Minkowski distance)
 */
template <typename Type, typename IdxT = int>
void pairwise_distance(raft::resources const& handle,
                       const Type* x,
                       const Type* y,
                       Type* dist,
                       IdxT m,
                       IdxT n,
                       IdxT k,
                       raft::distance::DistanceType metric,
                       bool isRowMajor = true,
                       Type metric_arg = 2.0f)
{
  auto stream = raft::resource::get_cuda_stream(handle);
  rmm::device_uvector<char> workspace(0, stream);
  pairwise_distance<Type, IdxT>(
    handle, x, y, dist, m, n, k, workspace, metric, isRowMajor, metric_arg);
}

/** @} */

/**
 * \defgroup distance_mdspan Pairwise distance functions
 * @{
 */

/**
 * @brief Evaluate pairwise distances for the simple use case.
 *
 * Note: Only contiguous row- or column-major layouts supported currently.
 *
 * Usage example:
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/device_mdarray.hpp>
 * #include <raft/random/make_blobs.cuh>
 * #include <raft/distance/distance.cuh>
 *
 * raft::raft::resources handle;
 * int n_samples = 5000;
 * int n_features = 50;
 *
 * auto input = raft::make_device_matrix<float>(handle, n_samples, n_features);
 * auto labels = raft::make_device_vector<int>(handle, n_samples);
 * auto output = raft::make_device_matrix<float>(handle, n_samples, n_samples);
 *
 * raft::random::make_blobs(handle, input.view(), labels.view());
 * auto metric = raft::distance::DistanceType::L2SqrtExpanded;
 * raft::distance::pairwise_distance(handle, input.view(), input.view(), output.view(), metric);
 * @endcode
 *
 * @tparam DistanceType which distance to evaluate
 * @tparam DataT input argument type
 * @tparam AccT accumulation type
 * @tparam OutT output type
 * @tparam IdxT Index type
 * @param handle raft handle for managing expensive resources
 * @param x first set of points (size n*k)
 * @param y second set of points (size m*k)
 * @param dist output distance matrix (size n*m)
 * @param metric_arg metric argument (used for Minkowski distance)
 */
template <raft::distance::DistanceType DistT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename layout = raft::layout_c_contiguous,
          typename IdxT   = int>
void distance(raft::resources const& handle,
              raft::device_matrix_view<DataT, IdxT, layout> const x,
              raft::device_matrix_view<DataT, IdxT, layout> const y,
              raft::device_matrix_view<OutT, IdxT, layout> dist,
              DataT metric_arg = 2.0f)
{
  RAFT_EXPECTS(x.extent(1) == y.extent(1), "Number of columns must be equal.");
  RAFT_EXPECTS(dist.extent(0) == x.extent(0),
               "Number of rows in output must be equal to "
               "number of rows in X");
  RAFT_EXPECTS(dist.extent(1) == y.extent(0),
               "Number of columns in output must be equal to "
               "number of rows in Y");

  RAFT_EXPECTS(x.is_exhaustive(), "Input x must be contiguous.");
  RAFT_EXPECTS(y.is_exhaustive(), "Input y must be contiguous.");

  constexpr auto is_rowmajor = std::is_same_v<layout, layout_c_contiguous>;

  distance<DistT, DataT, AccT, OutT, IdxT>(handle,
                                           x.data_handle(),
                                           y.data_handle(),
                                           dist.data_handle(),
                                           x.extent(0),
                                           y.extent(0),
                                           x.extent(1),
                                           is_rowmajor,
                                           metric_arg);
}

/**
 * @brief Convenience wrapper around 'distance' prim to convert runtime metric
 * into compile time for the purpose of dispatch
 * @tparam Type input/accumulation/output data-type
 * @tparam IdxT indexing type
 * @param handle raft handle for managing expensive resources
 * @param x first matrix of points (size mxk)
 * @param y second matrix of points (size nxk)
 * @param dist output distance matrix (size mxn)
 * @param metric distance metric
 * @param metric_arg metric argument (used for Minkowski distance)
 */
template <typename Type, typename layout = layout_c_contiguous, typename IdxT = int>
void pairwise_distance(raft::resources const& handle,
                       device_matrix_view<Type, IdxT, layout> const x,
                       device_matrix_view<Type, IdxT, layout> const y,
                       device_matrix_view<Type, IdxT, layout> dist,
                       raft::distance::DistanceType metric,
                       Type metric_arg = 2.0f)
{
  RAFT_EXPECTS(x.extent(1) == y.extent(1), "Number of columns must be equal.");
  RAFT_EXPECTS(dist.extent(0) == x.extent(0),
               "Number of rows in output must be equal to "
               "number of rows in X");
  RAFT_EXPECTS(dist.extent(1) == y.extent(0),
               "Number of columns in output must be equal to "
               "number of rows in Y");

  RAFT_EXPECTS(x.is_exhaustive(), "Input x must be contiguous.");
  RAFT_EXPECTS(y.is_exhaustive(), "Input y must be contiguous.");
  RAFT_EXPECTS(dist.is_exhaustive(), "Output must be contiguous.");

  constexpr auto rowmajor = std::is_same_v<layout, layout_c_contiguous>;

  auto stream = raft::resource::get_cuda_stream(handle);
  rmm::device_uvector<char> workspace(0, stream);

  pairwise_distance(handle,
                    x.data_handle(),
                    y.data_handle(),
                    dist.data_handle(),
                    x.extent(0),
                    y.extent(0),
                    x.extent(1),
                    metric,
                    rowmajor,
                    metric_arg);
}

/** @} */

};  // namespace distance
};  // namespace raft
