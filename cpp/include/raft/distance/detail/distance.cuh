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

#include <cuda_runtime_api.h>
#include <type_traits>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/linalg/unary_op.cuh>

#include <raft/core/operators.hpp>

#include <raft/distance/detail/distance_ops/canberra.cuh>
#include <raft/distance/detail/distance_ops/chebyshev.cuh>
#include <raft/distance/detail/distance_ops/correlation.cuh>
#include <raft/distance/detail/distance_ops/cosine.cuh>
#include <raft/distance/detail/distance_ops/hamming.cuh>
#include <raft/distance/detail/distance_ops/hellinger.cuh>
#include <raft/distance/detail/distance_ops/jensen_shannon.cuh>
#include <raft/distance/detail/distance_ops/kl_divergence.cuh>
#include <raft/distance/detail/distance_ops/l1.cuh>
#include <raft/distance/detail/distance_ops/l2_exp.cuh>
#include <raft/distance/detail/distance_ops/l2_unexp.cuh>
#include <raft/distance/detail/distance_ops/lp_unexp.cuh>
#include <raft/distance/detail/distance_ops/russel_rao.cuh>

#include <raft/distance/detail/pairwise_matrix/dispatch.cuh>

#include <raft/distance/distance_types.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/util/cuda_utils.cuh>
#include <rmm/device_uvector.hpp>

namespace raft {
namespace distance {
namespace detail {

/**
 * @brief: A tag type for overload resolution based on DistanceType
 *
 * It is not possible to partially specialize function templates on a single
 * parameter. Instead, it is often easier to use a combination of conventional
 * method overloading and a parameter with a specific tag type. The following
 * type is used to help method overloading based on the DistanceType enum.
 */
template <DistanceType d>
using distance_tag = std::integral_constant<DistanceType, d>;

/**
 * @brief Implement pairwise_matrix for specific distance
 *
 * There are multiple overloads for this function, one for each distance type.
 * They are implemented below. The documentation of this function serves as
 * documentation for all functions. The following overloads are defined:
 *
 * - DistanceType::Canberra:
 * - DistanceType::CorrelationExpanded:
 * - DistanceType::CosineExpanded:
 * - DistanceType::HammingUnexpanded:
 * - DistanceType::HellingerExpanded:
 * - DistanceType::JensenShannon:
 * - DistanceType::KLDivergence:
 * - DistanceType::L1:
 * - DistanceType::L2Expanded:
 * - DistanceType::L2SqrtExpanded:
 * - DistanceType::L2Unexpanded:
 * - DistanceType::L2SqrtUnexpanded:
 * - DistanceType::Linf:
 * - DistanceType::LpUnexpanded:
 * - DistanceType::RusselRaoExpanded:
 *
 * @tparam DataT   Input data type
 * @tparam AccT    Accumulation data type
 * @tparam OutT    Output data type
 * @tparam FinOpT  Type of final operation
 * @tparam IdxT    Index type
 *
 * @param handle        RAFT resources handle
 * @param distance_type A tag type to indicate which distance is calculated.
 * @param x             First set of points
 * @param y             Second set of points
 * @param out           Output distance matrix
 * @param m             Number of points in x
 * @param n             Number of points in y
 * @param k             Dimensionality of points in x, y
 * @param workspace     Temporary workspace needed for computations
 * @param worksize      Number of bytes of the workspace
 * @param is_row_major  Whether the matrices are row-major or col-major
 * @param metric_arg    The `p` argument for Lp.
 */
template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::Canberra> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT* workspace,  // unused
                   size_t worksize,  // unused
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT metric_arg)  // unused
{
  ops::canberra_distance_op distance_op{};

  const DataT* x_norm = nullptr;
  const DataT* y_norm = nullptr;

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  distance_matrix_dispatch<decltype(distance_op), DataT, AccT, OutT, FinOpT, IdxT>(
    distance_op, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::CorrelationExpanded> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT* workspace,
                   size_t worksize,
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // unused
{
  ASSERT(
    !(((x != y) && (worksize < 2 * (m + n) * sizeof(AccT))) || (worksize < 2 * m * sizeof(AccT))),
    "workspace size error");
  ASSERT(workspace != nullptr, "workspace is null");

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  AccT* norm_col_vec    = workspace;
  AccT* norm_row_vec    = workspace;
  AccT* sq_norm_col_vec = workspace;
  AccT* sq_norm_row_vec = workspace;
  if (x != y) {
    norm_row_vec += m;

    raft::linalg::reduce(norm_col_vec,
                         x,
                         k,
                         m,
                         (AccT)0,
                         is_row_major,
                         true,
                         stream,
                         false,
                         raft::identity_op(),
                         raft::add_op());
    raft::linalg::reduce(norm_row_vec,
                         y,
                         k,
                         n,
                         (AccT)0,
                         is_row_major,
                         true,
                         stream,
                         false,
                         raft::identity_op(),
                         raft::add_op());

    sq_norm_col_vec += (m + n);
    sq_norm_row_vec = sq_norm_col_vec + m;
    raft::linalg::rowNorm(sq_norm_col_vec, x, k, m, raft::linalg::L2Norm, is_row_major, stream);
    raft::linalg::rowNorm(sq_norm_row_vec, y, k, n, raft::linalg::L2Norm, is_row_major, stream);
  } else {
    raft::linalg::reduce(norm_col_vec,
                         x,
                         k,
                         m,
                         (AccT)0,
                         is_row_major,
                         true,
                         stream,
                         false,
                         raft::identity_op(),
                         raft::add_op());
    sq_norm_col_vec += m;
    sq_norm_row_vec = sq_norm_col_vec;
    raft::linalg::rowNorm(sq_norm_col_vec, x, k, m, raft::linalg::L2Norm, is_row_major, stream);
  }

  using CorrOp = ops::correlation_distance_op<DataT, IdxT>;
  CorrOp corr_op(is_row_major, sq_norm_col_vec, sq_norm_row_vec, m, n, k);
  distance_matrix_dispatch<decltype(corr_op), DataT, AccT, OutT, FinOpT, IdxT>(
    corr_op, m, n, k, x, y, norm_col_vec, norm_row_vec, out, fin_op, stream, is_row_major);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::CosineExpanded> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT* workspace,
                   size_t worksize,
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // unused
{
  // raft distance support inputs as float/double and output as uint8_t/float/double.
  static_assert(!((sizeof(OutT) > 1) && (sizeof(AccT) != sizeof(OutT))),
                "OutT can be uint8_t, float, double,"
                "if sizeof(OutT) > 1 then sizeof(AccT) == sizeof(OutT).");

  ASSERT(!(((x != y) && (worksize < (m + n) * sizeof(AccT))) || (worksize < m * sizeof(AccT))),
         "workspace size error");
  ASSERT(workspace != nullptr, "workspace is null");

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  DataT* norm_A = workspace;
  DataT* norm_B = workspace;
  if (x != y) {
    norm_B += m;
    raft::linalg::rowNorm(
      norm_A, x, k, m, raft::linalg::L2Norm, is_row_major, stream, raft::sqrt_op{});
    raft::linalg::rowNorm(
      norm_B, y, k, n, raft::linalg::L2Norm, is_row_major, stream, raft::sqrt_op{});
  } else {
    raft::linalg::rowNorm(
      norm_A, x, k, m, raft::linalg::L2Norm, is_row_major, stream, raft::sqrt_op{});
  }

  // On CUDA 12:
  // - always execute normal kernel
  //
  // On CUDA 11 and below:
  // - execute CUTLASS-based kernel on SM_80 and above
  // - execute normal kernel otherwise.

  if constexpr (__CUDACC_VER_MAJOR__ == 12) {
    // Always execute legacy kernels on CUDA 12
    ops::cosine_distance_op distance_op{};
    distance_matrix_dispatch<decltype(distance_op), DataT, AccT, OutT, FinOpT, IdxT>(
      distance_op, m, n, k, x, y, norm_A, norm_B, out, fin_op, stream, is_row_major);
  } else {
    const auto deviceVersion = getComputeCapability();
    if (deviceVersion.first >= 8) {
      // If device is SM_80 or later, use CUTLASS-based kernel.
      using Op = ops::cosine_cutlass_op<DataT, AccT>;
      Op distance_op{};

      distance_matrix_cutlass_dispatch<decltype(distance_op), DataT, AccT, OutT, FinOpT, IdxT>(
        distance_op, m, n, k, x, y, norm_A, norm_B, out, fin_op, stream, is_row_major);
    } else {
      // Else use "legacy" L2
      ops::cosine_distance_op distance_op{};
      distance_matrix_dispatch<decltype(distance_op), DataT, AccT, OutT, FinOpT, IdxT>(
        distance_op, m, n, k, x, y, norm_A, norm_B, out, fin_op, stream, is_row_major);
    }
  }
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::HammingUnexpanded> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT*,   // workspace unused
                   size_t,  // worksize unused
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // metric_arg unused
{
  ops::hamming_distance_op<IdxT> distance_op{k};

  const DataT* x_norm = nullptr;
  const DataT* y_norm = nullptr;

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  distance_matrix_dispatch<decltype(distance_op), DataT, AccT, OutT, FinOpT, IdxT>(
    distance_op, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::InnerProduct> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT*,   // workspace unused
                   size_t,  // worksize unused
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // metric_arg unused
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  raft::linalg::gemm(handle,
                     out,
                     const_cast<DataT*>(x),
                     const_cast<DataT*>(y),
                     m,
                     n,
                     k,
                     !is_row_major,
                     !is_row_major,
                     is_row_major,
                     stream);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::HellingerExpanded> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT*,   // workspace unused
                   size_t,  // worksize unused
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // metric_arg unused
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  // First sqrt x and y
  const auto raft_sqrt = raft::linalg::unaryOp<DataT, raft::sqrt_op, IdxT>;

  raft_sqrt((DataT*)x, x, m * k, raft::sqrt_op{}, stream);
  if (x != y) { raft_sqrt((DataT*)y, y, n * k, raft::sqrt_op{}, stream); }

  // Then calculate Hellinger distance
  ops::hellinger_distance_op distance_op{};

  const DataT* x_norm = nullptr;
  const DataT* y_norm = nullptr;

  distance_matrix_dispatch<decltype(distance_op), DataT, AccT, OutT, FinOpT, IdxT>(
    distance_op, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);

  // Finally revert sqrt of x and y
  raft_sqrt((DataT*)x, x, m * k, raft::sqrt_op{}, stream);
  if (x != y) { raft_sqrt((DataT*)y, y, n * k, raft::sqrt_op{}, stream); }

  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::JensenShannon> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT*,   // workspace unused
                   size_t,  // worksize unused
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // metric_arg unused
{
  ops::jensen_shannon_distance_op distance_op{};

  const DataT* x_norm = nullptr;
  const DataT* y_norm = nullptr;

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  distance_matrix_dispatch<decltype(distance_op), DataT, AccT, OutT, FinOpT, IdxT>(
    distance_op, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::KLDivergence> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT*,   // workspace unused
                   size_t,  // worksize unused
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // metric_arg unused
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  auto unaryOp_lambda = [] __device__(DataT input) {
    const bool x_zero = (input == 0);
    return (!x_zero) * raft::log(input + x_zero);
  };

  auto unaryOp_lambda_reverse = [] __device__(DataT input) {
    // reverse previous log (x) back to x using (e ^ log(x))
    const bool x_zero = (input == 0);
    return (!x_zero) * raft::exp(input);
  };

  // This op takes some shortcuts when x equals y. So its behavior changes based
  // on this.
  ops::kl_divergence_op kl_divergence{is_row_major, x == y};

  if (x != y) {
    raft::linalg::unaryOp<DataT, decltype(unaryOp_lambda), IdxT>(
      (DataT*)y, y, n * k, unaryOp_lambda, stream);
  }

  const DataT* x_norm = nullptr;
  const DataT* y_norm = nullptr;

  distance_matrix_dispatch<decltype(kl_divergence), DataT, AccT, OutT, FinOpT, IdxT>(
    kl_divergence, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);

  if (x != y) {
    // Now reverse previous log (x) back to x using (e ^ log(x))
    raft::linalg::unaryOp<DataT, decltype(unaryOp_lambda_reverse), IdxT>(
      (DataT*)y, y, n * k, unaryOp_lambda_reverse, stream);
  }
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::L1> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT*,   // workspace unused
                   size_t,  // worksize unused
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // metric_arg unused
{
  ops::l1_distance_op distance_op{};

  const DataT* x_norm = nullptr;
  const DataT* y_norm = nullptr;

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  distance_matrix_dispatch<ops::l1_distance_op, DataT, AccT, OutT, FinOpT, IdxT>(
    distance_op, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);
}

template <typename DataT,
          typename AccT,
          typename OutT,
          typename FinOpT,
          typename IdxT = int>
void distance_impl_l2_expanded(  // NOTE: different name
  bool perform_sqrt,             // dispatch on sqrt
  const DataT* x,
  const DataT* y,
  OutT* out,
  IdxT m,
  IdxT n,
  IdxT k,
  AccT* workspace,
  size_t worksize,
  FinOpT fin_op,
  cudaStream_t stream,
  bool is_row_major)
{
  // raft distance support inputs as float/double and output as uint8_t/float/double.
  static_assert(!((sizeof(OutT) > 1) && (sizeof(AccT) != sizeof(OutT))),
                "OutT can be uint8_t, float, double,"
                "if sizeof(OutT) > 1 then sizeof(AccT) == sizeof(OutT).");

  ASSERT(!(((x != y) && (worksize < (m + n) * sizeof(AccT))) || (worksize < m * sizeof(AccT))),
         "workspace size error");
  ASSERT(workspace != nullptr, "workspace is null");

  DataT* norm_A = workspace;
  DataT* norm_B = workspace;
  if (x != y) {
    norm_B += m;
    raft::linalg::rowNorm(
      norm_A, x, k, m, raft::linalg::L2Norm, is_row_major, stream, raft::identity_op{});
    raft::linalg::rowNorm(
      norm_B, y, k, n, raft::linalg::L2Norm, is_row_major, stream, raft::identity_op{});
  } else {
    raft::linalg::rowNorm(
      norm_A, x, k, m, raft::linalg::L2Norm, is_row_major, stream, raft::identity_op{});
  }

  // On CUDA 12:
  // - always execute normal kernel
  //
  // On CUDA 11 and below:
  // - execute CUTLASS-based kernel on SM_80 and above
  // - execute normal kernel otherwise.

  if constexpr (__CUDACC_VER_MAJOR__ == 12) {
    // Always execute legacy kernels on CUDA 12
    ops::l2_exp_distance_op l2_op(perform_sqrt);
    distance_matrix_dispatch<decltype(l2_op), DataT, AccT, OutT, FinOpT, IdxT>(
      l2_op, m, n, k, x, y, norm_A, norm_B, out, fin_op, stream, is_row_major);
  } else {
    const auto deviceVersion = getComputeCapability();
    if (deviceVersion.first >= 8) {
      // If device is SM_80 or later, use CUTLASS-based kernel.
      using L2Op = ops::l2_exp_cutlass_op<DataT, AccT>;
      L2Op l2_op(perform_sqrt);

      distance_matrix_cutlass_dispatch<decltype(l2_op), DataT, AccT, OutT, FinOpT, IdxT>(
        l2_op, m, n, k, x, y, norm_A, norm_B, out, fin_op, stream, is_row_major);
    } else {
      // Else use "legacy" L2
      ops::l2_exp_distance_op l2_op(perform_sqrt);
      distance_matrix_dispatch<decltype(l2_op), DataT, AccT, OutT, FinOpT, IdxT>(
        l2_op, m, n, k, x, y, norm_A, norm_B, out, fin_op, stream, is_row_major);
    }
  }
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::L2Expanded> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT* workspace,
                   size_t worksize,
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // metric_arg unused
{
  bool perform_sqrt   = false;
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  distance_impl_l2_expanded(
    perform_sqrt, x, y, out, m, n, k, workspace, worksize, fin_op, stream, is_row_major);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::L2SqrtExpanded> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT* workspace,
                   size_t worksize,
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // metric_arg unused
{
  bool perform_sqrt   = true;
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  distance_impl_l2_expanded(
    perform_sqrt, x, y, out, m, n, k, workspace, worksize, fin_op, stream, is_row_major);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::L2Unexpanded> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT*,   // workspace unused
                   size_t,  // worksize unused
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // metric_arg unused
{
  bool perform_sqrt = false;
  ops::l2_unexp_distance_op l2_op(perform_sqrt);

  // The unexpanded L2 does not require the norms of a and b to be calculated.
  const DataT* norm_A = nullptr;
  const DataT* norm_B = nullptr;

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  distance_matrix_dispatch<decltype(l2_op), DataT, AccT, OutT, FinOpT, IdxT>(
    l2_op, m, n, k, x, y, norm_A, norm_B, out, fin_op, stream, is_row_major);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::L2SqrtUnexpanded> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT*,   // workspace unused
                   size_t,  // worksize unused
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // metric_arg unused
{
  bool perform_sqrt = true;
  ops::l2_unexp_distance_op l2_op(perform_sqrt);

  // The unexpanded L2 does not require the norms of a and b to be calculated.
  const DataT* norm_A = nullptr;
  const DataT* norm_B = nullptr;

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  distance_matrix_dispatch<decltype(l2_op), DataT, AccT, OutT, FinOpT, IdxT>(
    l2_op, m, n, k, x, y, norm_A, norm_B, out, fin_op, stream, is_row_major);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::Linf> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT*,   // workspace unused
                   size_t,  // worksize unused
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // metric_arg unused
{
  ops::chebyshev_distance_op distance_op{};

  const DataT* x_norm = nullptr;
  const DataT* y_norm = nullptr;

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  distance_matrix_dispatch<decltype(distance_op), DataT, AccT, OutT, FinOpT, IdxT>(
    distance_op, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::LpUnexpanded> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT*,   // workspace unused
                   size_t,  // worksize unused
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT metric_arg)
{
  ops::lp_unexp_distance_op<DataT> distance_op{metric_arg};

  const DataT* x_norm = nullptr;
  const DataT* y_norm = nullptr;

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  distance_matrix_dispatch<decltype(distance_op), DataT, AccT, OutT, FinOpT, IdxT>(
    distance_op, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::RusselRaoExpanded> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT*,   // workspace unused
                   size_t,  // worksize unused
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // metric_arg unused
{
  ops::russel_rao_distance_op<IdxT> distance_op{k};

  const DataT* x_norm = nullptr;
  const DataT* y_norm = nullptr;

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  distance_matrix_dispatch<decltype(distance_op), DataT, AccT, OutT, FinOpT, IdxT>(
    distance_op, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);
}

/**
 * @brief Evaluate pairwise distances with the user epilogue lamba allowed
 * @tparam DistanceType which distance to evaluate
 * @tparam InType input argument type
 * @tparam AccType accumulation type
 * @tparam OutType output type
 * @tparam FinalLambda user-defined epilogue lamba
 * @tparam Index_ Index type
 *
 * @param x first set of points
 * @param y second set of points
 * @param out output distance matrix
 * @param m number of points in x
 * @param n number of points in y
 * @param k dimensionality
 * @param workspace temporary workspace needed for computations
 * @param worksize number of bytes of the workspace
 * @param fin_op the final gemm epilogue lambda
 * @param stream cuda stream
 * @param isRowMajor whether the matrices are row-major or col-major
 *
 * @note fin_op: This is a device lambda which is supposed to operate upon the
 * input which is AccType and returns the output in OutType. It's signature is
 * as follows:  <pre>OutType fin_op(AccType in, int g_idx);</pre>. If one needs
 * any other parameters, feel free to pass them via closure.
 */
template <raft::distance::DistanceType distanceType,
          typename InType,
          typename AccType,
          typename OutType,
          typename FinalLambda,
          typename Index_ = int>
void distance(raft::resources const& handle,
              const InType* x,
              const InType* y,
              OutType* out,
              Index_ m,
              Index_ n,
              Index_ k,
              void* workspace,
              size_t worksize,
              FinalLambda fin_op,
              bool isRowMajor   = true,
              InType metric_arg = 2.0f)
{
  // raft distance support inputs as float/double and output as uint8_t/float/double.
  static_assert(!((sizeof(OutType) > 1) && (sizeof(AccType) != sizeof(OutType))),
                "OutType can be uint8_t, float, double,"
                "if sizeof(OutType) > 1 then sizeof(AccType) == sizeof(OutType).");

  distance_impl<InType, AccType, OutType, FinalLambda, Index_>(
    handle,
    distance_tag<distanceType>{},
    x,
    y,
    out,
    m,
    n,
    k,
    reinterpret_cast<AccType*>(workspace),
    worksize,
    fin_op,
    isRowMajor,
    metric_arg);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
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
 *
 * @note if workspace is passed as nullptr, this will return in
 *  worksize, the number of bytes of workspace required
 */
template <raft::distance::DistanceType distanceType,
          typename InType,
          typename AccType,
          typename OutType,
          typename Index_ = int>
void distance(raft::resources const& handle,
              const InType* x,
              const InType* y,
              OutType* out,
              Index_ m,
              Index_ n,
              Index_ k,
              void* workspace,
              size_t worksize,
              bool isRowMajor   = true,
              InType metric_arg = 2.0f)
{
  auto fin_op = raft::identity_op();

  distance<distanceType, InType, AccType, OutType, decltype(fin_op), Index_>(
    handle, x, y, out, m, n, k, workspace, worksize, fin_op, isRowMajor, metric_arg);
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
template <raft::distance::DistanceType distanceType,
          typename InType,
          typename AccType,
          typename OutType,
          typename Index_ = int>
size_t getWorkspaceSize(const InType* x, const InType* y, Index_ m, Index_ n, Index_ k)
{
  size_t worksize             = 0;
  constexpr bool is_allocated = (distanceType <= raft::distance::DistanceType::CosineExpanded) ||
                                (distanceType == raft::distance::DistanceType::CorrelationExpanded);
  constexpr int numOfBuffers =
    (distanceType == raft::distance::DistanceType::CorrelationExpanded) ? 2 : 1;

  if (is_allocated) {
    worksize += numOfBuffers * m * sizeof(AccType);
    if (x != y) worksize += numOfBuffers * n * sizeof(AccType);
  }

  return worksize;
}

};  // namespace detail
};  // namespace distance
};  // namespace raft
