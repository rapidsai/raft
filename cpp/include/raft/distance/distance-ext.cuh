/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

#include <raft/core/device_mdspan.hpp>       // raft::device_matrix_view
#include <raft/core/operators.hpp>           // raft::identity_op
#include <raft/core/resources.hpp>           // raft::resources
#include <raft/distance/distance_types.hpp>  // raft::distance::DistanceType
#include <raft/util/raft_explicit.hpp>       // RAFT_EXPLICIT
#include <rmm/device_uvector.hpp>            // rmm::device_uvector

#ifdef RAFT_EXPLICIT_INSTANTIATE

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
              DataT metric_arg = 2.0f) RAFT_EXPLICIT;

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
              DataT metric_arg = 2.0f) RAFT_EXPLICIT;

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
size_t getWorkspaceSize(const DataT* x, const DataT* y, IdxT m, IdxT n, IdxT k) RAFT_EXPLICIT;

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
                        raft::device_matrix_view<DataT, IdxT, layout> const& y) RAFT_EXPLICIT;

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
              DataT metric_arg = 2.0f) RAFT_EXPLICIT;

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
                       Type metric_arg = 2.0f) RAFT_EXPLICIT;

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
                       Type metric_arg = 2.0f) RAFT_EXPLICIT;
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
 * #include <raft/core/device_resources.hpp>
 * #include <raft/core/device_mdarray.hpp>
 * #include <raft/random/make_blobs.cuh>
 * #include <raft/distance/distance.cuh>
 *
 * raft::raft::device_resources handle;
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
              DataT metric_arg = 2.0f) RAFT_EXPLICIT;

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
                       Type metric_arg = 2.0f) RAFT_EXPLICIT;

/** @} */

};  // namespace distance
};  // namespace raft

#endif  // RAFT_EXPLICIT_INSTANTIATE

#define instantiate_raft_distance_distance(DT, DataT, AccT, OutT, FinalLambda, IdxT)       \
  extern template void raft::distance::distance<DT, DataT, AccT, OutT, FinalLambda, IdxT>( \
    raft::resources const& handle,                                                         \
    const DataT* x,                                                                        \
    const DataT* y,                                                                        \
    OutT* dist,                                                                            \
    IdxT m,                                                                                \
    IdxT n,                                                                                \
    IdxT k,                                                                                \
    void* workspace,                                                                       \
    size_t worksize,                                                                       \
    FinalLambda fin_op,                                                                    \
    bool isRowMajor,                                                                       \
    DataT metric_arg)

instantiate_raft_distance_distance(
  raft::distance::DistanceType::Canberra, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::Canberra, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::CorrelationExpanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::CorrelationExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::identity_op,
                                   int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::CosineExpanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::CosineExpanded, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::HammingUnexpanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::HammingUnexpanded, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::HellingerExpanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::HellingerExpanded, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::InnerProduct, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::InnerProduct, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::JensenShannon, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::JensenShannon, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::KLDivergence, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::KLDivergence, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L1, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L1, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Expanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Expanded, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2SqrtExpanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2SqrtExpanded, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2SqrtUnexpanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2SqrtUnexpanded, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Unexpanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Unexpanded, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::Linf, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::Linf, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::LpUnexpanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::LpUnexpanded, double, double, double, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::RusselRaoExpanded, float, float, float, raft::identity_op, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::RusselRaoExpanded, double, double, double, raft::identity_op, int);

#undef instantiate_raft_distance_distance

// Same, but without raft::identity_op
#define instantiate_raft_distance_distance(DT, DataT, AccT, OutT, IdxT)       \
  extern template void raft::distance::distance<DT, DataT, AccT, OutT, IdxT>( \
    raft::resources const& handle,                                            \
    const DataT* x,                                                           \
    const DataT* y,                                                           \
    OutT* dist,                                                               \
    IdxT m,                                                                   \
    IdxT n,                                                                   \
    IdxT k,                                                                   \
    void* workspace,                                                          \
    size_t worksize,                                                          \
    bool isRowMajor,                                                          \
    DataT metric_arg)

instantiate_raft_distance_distance(
  raft::distance::DistanceType::Canberra, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::Canberra, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::CorrelationExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::CorrelationExpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::CosineExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::CosineExpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::HammingUnexpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::HammingUnexpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::HellingerExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::HellingerExpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::InnerProduct, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::InnerProduct, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::JensenShannon, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::JensenShannon, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::KLDivergence, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::KLDivergence, double, double, double, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::L1, float, float, float, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::L1, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Expanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Expanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2SqrtExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2SqrtExpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2SqrtUnexpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2SqrtUnexpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Unexpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Unexpanded, double, double, double, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::Linf, float, float, float, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::Linf, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::LpUnexpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::LpUnexpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::RusselRaoExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::RusselRaoExpanded, double, double, double, int);

#undef instantiate_raft_distance_distance

// Same, but without workspace
#define instantiate_raft_distance_distance(DT, DataT, AccT, OutT, IdxT)       \
  extern template void raft::distance::distance<DT, DataT, AccT, OutT, IdxT>( \
    raft::resources const& handle,                                            \
    const DataT* x,                                                           \
    const DataT* y,                                                           \
    OutT* dist,                                                               \
    IdxT m,                                                                   \
    IdxT n,                                                                   \
    IdxT k,                                                                   \
    bool isRowMajor,                                                          \
    DataT metric_arg)

instantiate_raft_distance_distance(
  raft::distance::DistanceType::Canberra, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::Canberra, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::CorrelationExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::CorrelationExpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::CosineExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::CosineExpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::HammingUnexpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::HammingUnexpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::HellingerExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::HellingerExpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::InnerProduct, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::InnerProduct, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::JensenShannon, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::JensenShannon, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::KLDivergence, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::KLDivergence, double, double, double, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::L1, float, float, float, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::L1, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Expanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Expanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2SqrtExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2SqrtExpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2SqrtUnexpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2SqrtUnexpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Unexpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Unexpanded, double, double, double, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::Linf, float, float, float, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::Linf, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::LpUnexpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::LpUnexpanded, double, double, double, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::RusselRaoExpanded, float, float, float, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::RusselRaoExpanded, double, double, double, int);

#undef instantiate_raft_distance_distance

#define instantiate_raft_distance_getWorkspaceSize(DistT, DataT, AccT, OutT, IdxT)         \
  extern template size_t raft::distance::getWorkspaceSize<DistT, DataT, AccT, OutT, IdxT>( \
    const DataT* x, const DataT* y, IdxT m, IdxT n, IdxT k)

instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::Canberra, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::Canberra, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::CorrelationExpanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::CorrelationExpanded, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::CosineExpanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::CosineExpanded, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::HammingUnexpanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::HammingUnexpanded, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::HellingerExpanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::HellingerExpanded, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::InnerProduct, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::InnerProduct, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::JensenShannon, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::JensenShannon, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::KLDivergence, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::KLDivergence, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L1, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L1, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2Expanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2Expanded, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2SqrtExpanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2SqrtExpanded, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2SqrtUnexpanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2SqrtUnexpanded, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2Unexpanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2Unexpanded, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::Linf, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::Linf, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::LpUnexpanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::LpUnexpanded, double, double, double, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::RusselRaoExpanded, float, float, float, int);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::RusselRaoExpanded, double, double, double, int);

#undef instantiate_raft_distance_getWorkspaceSize

#define instantiate_raft_distance_getWorkspaceSize(DistT, DataT, AccT, OutT, IdxT, layout)         \
  extern template size_t raft::distance::getWorkspaceSize<DistT, DataT, AccT, OutT, IdxT, layout>( \
    raft::device_matrix_view<DataT, IdxT, layout> const& x,                                        \
    raft::device_matrix_view<DataT, IdxT, layout> const& y)

// We could consider not taking template parameters for this function. The
// number of instantiations seems a bit excessive..
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::Canberra, float, float, float, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::Canberra, double, double, double, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::Canberra, float, float, float, int, raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::Canberra, double, double, double, int, raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::CorrelationExpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::CorrelationExpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::CorrelationExpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::CorrelationExpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::CosineExpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::CosineExpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::CosineExpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::CosineExpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::HammingUnexpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::HammingUnexpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::HammingUnexpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::HammingUnexpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::HellingerExpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::HellingerExpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::HellingerExpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::HellingerExpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::InnerProduct, float, float, float, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::InnerProduct,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::InnerProduct, float, float, float, int, raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::InnerProduct,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::JensenShannon, float, float, float, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::JensenShannon,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::JensenShannon, float, float, float, int, raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::JensenShannon,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::KLDivergence, float, float, float, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::KLDivergence,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::KLDivergence, float, float, float, int, raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::KLDivergence,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L1, float, float, float, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L1, double, double, double, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L1, float, float, float, int, raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L1, double, double, double, int, raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2Expanded, float, float, float, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2Expanded, double, double, double, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2Expanded, float, float, float, int, raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2Expanded, double, double, double, int, raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::L2SqrtExpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::L2SqrtExpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::L2SqrtExpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::L2SqrtExpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::L2SqrtUnexpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::L2SqrtUnexpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::L2SqrtUnexpanded,
                                           float,
                                           float,
                                           float,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::L2SqrtUnexpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_f_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2Unexpanded, float, float, float, int, raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(raft::distance::DistanceType::L2Unexpanded,
                                           double,
                                           double,
                                           double,
                                           int,
                                           raft::layout_c_contiguous);
instantiate_raft_distance_getWorkspaceSize(
  raft::distance::DistanceType::L2Unexpanded, float, float, float, int, raft::layout_f_contiguous);

#undef instantiate_raft_distance_getWorkspaceSize

#define instantiate_raft_distance_pairwise_distance(DataT, IdxT)                               \
  extern template void raft::distance::pairwise_distance(raft::resources const& handle,        \
                                                         const DataT* x,                       \
                                                         const DataT* y,                       \
                                                         DataT* dist,                          \
                                                         IdxT m,                               \
                                                         IdxT n,                               \
                                                         IdxT k,                               \
                                                         rmm::device_uvector<char>& workspace, \
                                                         raft::distance::DistanceType metric,  \
                                                         bool isRowMajor,                      \
                                                         DataT metric_arg)

instantiate_raft_distance_pairwise_distance(float, int);
instantiate_raft_distance_pairwise_distance(double, int);

#undef instantiate_raft_distance_pairwise_distance

// Same, but without workspace
#define instantiate_raft_distance_pairwise_distance(DataT, IdxT)                              \
  extern template void raft::distance::pairwise_distance(raft::resources const& handle,       \
                                                         const DataT* x,                      \
                                                         const DataT* y,                      \
                                                         DataT* dist,                         \
                                                         IdxT m,                              \
                                                         IdxT n,                              \
                                                         IdxT k,                              \
                                                         raft::distance::DistanceType metric, \
                                                         bool isRowMajor,                     \
                                                         DataT metric_arg)

instantiate_raft_distance_pairwise_distance(float, int);
instantiate_raft_distance_pairwise_distance(double, int);

#undef instantiate_raft_distance_pairwise_distance

// Version with mdspan
#define instantiate_raft_distance_distance(DistT, DataT, AccT, OutT, layout, IdxT)       \
  extern template void raft::distance::distance<DistT, DataT, AccT, OutT, layout, IdxT>( \
    raft::resources const& handle,                                                       \
    raft::device_matrix_view<DataT, IdxT, layout> const x,                               \
    raft::device_matrix_view<DataT, IdxT, layout> const y,                               \
    raft::device_matrix_view<OutT, IdxT, layout> dist,                                   \
    DataT metric_arg)

// Again, we might want to consider reigning in the number of instantiations...
instantiate_raft_distance_distance(
  raft::distance::DistanceType::Canberra, float, float, float, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::Canberra, double, double, double, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::Canberra, float, float, float, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::Canberra, double, double, double, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::CorrelationExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::CorrelationExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::CorrelationExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::CorrelationExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::CosineExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::CosineExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::CosineExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::CosineExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::HammingUnexpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::HammingUnexpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::HammingUnexpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::HammingUnexpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::HellingerExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::HellingerExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::HellingerExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::HellingerExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::InnerProduct, float, float, float, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::InnerProduct,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::InnerProduct, float, float, float, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::InnerProduct,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::JensenShannon, float, float, float, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::JensenShannon,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::JensenShannon, float, float, float, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::JensenShannon,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::KLDivergence, float, float, float, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::KLDivergence,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::KLDivergence, float, float, float, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::KLDivergence,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L1, float, float, float, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L1, double, double, double, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L1, float, float, float, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L1, double, double, double, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Expanded, float, float, float, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Expanded, double, double, double, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Expanded, float, float, float, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Expanded, double, double, double, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::L2SqrtExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::L2SqrtExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::L2SqrtExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::L2SqrtExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::L2SqrtUnexpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::L2SqrtUnexpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::L2SqrtUnexpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::L2SqrtUnexpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Unexpanded, float, float, float, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::L2Unexpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::L2Unexpanded, float, float, float, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::L2Unexpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::Linf, float, float, float, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::Linf, double, double, double, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::Linf, float, float, float, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::Linf, double, double, double, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::LpUnexpanded, float, float, float, raft::layout_c_contiguous, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::LpUnexpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(
  raft::distance::DistanceType::LpUnexpanded, float, float, float, raft::layout_f_contiguous, int);
instantiate_raft_distance_distance(raft::distance::DistanceType::LpUnexpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::RusselRaoExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::RusselRaoExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_c_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::RusselRaoExpanded,
                                   float,
                                   float,
                                   float,
                                   raft::layout_f_contiguous,
                                   int);
instantiate_raft_distance_distance(raft::distance::DistanceType::RusselRaoExpanded,
                                   double,
                                   double,
                                   double,
                                   raft::layout_f_contiguous,
                                   int);

#undef instantiate_raft_distance_distance

#define instantiate_raft_distance_pairwise_distance(DataT, layout, IdxT) \
  extern template void raft::distance::pairwise_distance(                \
    raft::resources const& handle,                                       \
    raft::device_matrix_view<DataT, IdxT, layout> const x,               \
    raft::device_matrix_view<DataT, IdxT, layout> const y,               \
    raft::device_matrix_view<DataT, IdxT, layout> dist,                  \
    raft::distance::DistanceType metric,                                 \
    DataT metric_arg)

instantiate_raft_distance_pairwise_distance(float, raft::layout_c_contiguous, int);
instantiate_raft_distance_pairwise_distance(float, raft::layout_f_contiguous, int);
instantiate_raft_distance_pairwise_distance(double, raft::layout_c_contiguous, int);
instantiate_raft_distance_pairwise_distance(double, raft::layout_f_contiguous, int);

#undef instantiate_raft_distance_pairwise_distance
