/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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
#include <raft/distance/detail/canberra.cuh>
#include <raft/distance/detail/chebyshev.cuh>
#include <raft/distance/detail/correlation.cuh>
#include <raft/distance/detail/cosine.cuh>
#include <raft/distance/detail/euclidean.cuh>
#include <raft/distance/detail/hamming.cuh>
#include <raft/distance/detail/hellinger.cuh>
#include <raft/distance/detail/jensen_shannon.cuh>
#include <raft/distance/detail/kl_divergence.cuh>
#include <raft/distance/detail/l1.cuh>
#include <raft/distance/detail/minkowski.cuh>
#include <raft/distance/detail/russell_rao.cuh>
#include <raft/distance/distance_types.hpp>
#include <raft/util/cuda_utils.cuh>
#include <rmm/device_uvector.hpp>

namespace raft {
namespace distance {
namespace detail {

namespace {
template <raft::distance::DistanceType distanceType,
          typename InType,
          typename AccType,
          typename OutType,
          typename FinalLambda,
          typename Index_>
struct DistanceImpl {
  void run(const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void* workspace,
           size_t worksize,
           FinalLambda fin_op,
           cudaStream_t stream,
           bool isRowMajor,
           InType metric_arg = 2.0f)
  {
  }
};

template <typename InType,
          typename AccType,
          typename OutType,
          typename FinalLambda,
          typename Index_>
struct DistanceImpl<raft::distance::DistanceType::L2Expanded,
                    InType,
                    AccType,
                    OutType,
                    FinalLambda,
                    Index_> {
  void run(const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void* workspace,
           size_t worksize,
           FinalLambda fin_op,
           cudaStream_t stream,
           bool isRowMajor,
           InType)
  {
    raft::distance::detail::euclideanAlgo1<InType, AccType, OutType, FinalLambda, Index_>(
      m, n, k, x, y, dist, false, (AccType*)workspace, worksize, fin_op, stream, isRowMajor);
  }
};

template <typename InType,
          typename AccType,
          typename OutType,
          typename FinalLambda,
          typename Index_>
struct DistanceImpl<raft::distance::DistanceType::L2SqrtExpanded,
                    InType,
                    AccType,
                    OutType,
                    FinalLambda,
                    Index_> {
  void run(const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void* workspace,
           size_t worksize,
           FinalLambda fin_op,
           cudaStream_t stream,
           bool isRowMajor,
           InType)
  {
    raft::distance::detail::euclideanAlgo1<InType, AccType, OutType, FinalLambda, Index_>(
      m, n, k, x, y, dist, true, (AccType*)workspace, worksize, fin_op, stream, isRowMajor);
  }
};

template <typename InType,
          typename AccType,
          typename OutType,
          typename FinalLambda,
          typename Index_>
struct DistanceImpl<raft::distance::DistanceType::CosineExpanded,
                    InType,
                    AccType,
                    OutType,
                    FinalLambda,
                    Index_> {
  void run(const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void* workspace,
           size_t worksize,
           FinalLambda fin_op,
           cudaStream_t stream,
           bool isRowMajor,
           InType)
  {
    raft::distance::detail::cosineAlgo1<InType, AccType, OutType, FinalLambda, Index_>(
      m, n, k, x, y, dist, (AccType*)workspace, worksize, fin_op, stream, isRowMajor);
  }
};

template <typename InType,
          typename AccType,
          typename OutType,
          typename FinalLambda,
          typename Index_>
struct DistanceImpl<raft::distance::DistanceType::L2Unexpanded,
                    InType,
                    AccType,
                    OutType,
                    FinalLambda,
                    Index_> {
  void run(const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void*,
           size_t,
           FinalLambda fin_op,
           cudaStream_t stream,
           bool isRowMajor,
           InType)
  {
    raft::distance::detail::euclideanAlgo2<InType, AccType, OutType, FinalLambda, Index_>(
      m, n, k, x, y, dist, false, fin_op, stream, isRowMajor);
  }
};

template <typename InType,
          typename AccType,
          typename OutType,
          typename FinalLambda,
          typename Index_>
struct DistanceImpl<raft::distance::DistanceType::L2SqrtUnexpanded,
                    InType,
                    AccType,
                    OutType,
                    FinalLambda,
                    Index_> {
  void run(const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void*,
           size_t,
           FinalLambda fin_op,
           cudaStream_t stream,
           bool isRowMajor,
           InType)
  {
    raft::distance::detail::euclideanAlgo2<InType, AccType, OutType, FinalLambda, Index_>(
      m, n, k, x, y, dist, true, fin_op, stream, isRowMajor);
  }
};

template <typename InType,
          typename AccType,
          typename OutType,
          typename FinalLambda,
          typename Index_>
struct DistanceImpl<raft::distance::DistanceType::L1,
                    InType,
                    AccType,
                    OutType,
                    FinalLambda,
                    Index_> {
  void run(const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void*,
           size_t,
           FinalLambda fin_op,
           cudaStream_t stream,
           bool isRowMajor,
           InType)
  {
    raft::distance::detail::l1Impl<InType, AccType, OutType, FinalLambda, Index_>(
      m, n, k, x, y, dist, fin_op, stream, isRowMajor);
  }
};

template <typename InType,
          typename AccType,
          typename OutType,
          typename FinalLambda,
          typename Index_>
struct DistanceImpl<raft::distance::DistanceType::Linf,
                    InType,
                    AccType,
                    OutType,
                    FinalLambda,
                    Index_> {
  void run(const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void*,
           size_t,
           FinalLambda fin_op,
           cudaStream_t stream,
           bool isRowMajor,
           InType)
  {
    raft::distance::detail::chebyshevImpl<InType, AccType, OutType, FinalLambda, Index_>(
      m, n, k, x, y, dist, fin_op, stream, isRowMajor);
  }
};

template <typename InType,
          typename AccType,
          typename OutType,
          typename FinalLambda,
          typename Index_>
struct DistanceImpl<raft::distance::DistanceType::HellingerExpanded,
                    InType,
                    AccType,
                    OutType,
                    FinalLambda,
                    Index_> {
  void run(const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void*,
           size_t,
           FinalLambda fin_op,
           cudaStream_t stream,
           bool isRowMajor,
           InType)
  {
    raft::distance::detail::hellingerImpl<InType, AccType, OutType, FinalLambda, Index_>(
      m, n, k, x, y, dist, fin_op, stream, isRowMajor);
  }
};

template <typename InType,
          typename AccType,
          typename OutType,
          typename FinalLambda,
          typename Index_>
struct DistanceImpl<raft::distance::DistanceType::LpUnexpanded,
                    InType,
                    AccType,
                    OutType,
                    FinalLambda,
                    Index_> {
  void run(const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void*,
           size_t,
           FinalLambda fin_op,
           cudaStream_t stream,
           bool isRowMajor,
           InType metric_arg)
  {
    raft::distance::detail::minkowskiImpl<InType, AccType, OutType, FinalLambda, Index_>(
      m, n, k, x, y, dist, fin_op, stream, isRowMajor, metric_arg);
  }
};

template <typename InType,
          typename AccType,
          typename OutType,
          typename FinalLambda,
          typename Index_>
struct DistanceImpl<raft::distance::DistanceType::Canberra,
                    InType,
                    AccType,
                    OutType,
                    FinalLambda,
                    Index_> {
  void run(const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void*,
           size_t,
           FinalLambda fin_op,
           cudaStream_t stream,
           bool isRowMajor,
           InType)
  {
    raft::distance::detail::canberraImpl<InType, AccType, OutType, FinalLambda, Index_>(
      m, n, k, x, y, dist, fin_op, stream, isRowMajor);
  }
};

template <typename InType,
          typename AccType,
          typename OutType,
          typename FinalLambda,
          typename Index_>
struct DistanceImpl<raft::distance::DistanceType::HammingUnexpanded,
                    InType,
                    AccType,
                    OutType,
                    FinalLambda,
                    Index_> {
  void run(const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void*,
           size_t,
           FinalLambda fin_op,
           cudaStream_t stream,
           bool isRowMajor,
           InType)
  {
    raft::distance::detail::hammingUnexpandedImpl<InType, AccType, OutType, FinalLambda, Index_>(
      m, n, k, x, y, dist, fin_op, stream, isRowMajor);
  }
};

template <typename InType,
          typename AccType,
          typename OutType,
          typename FinalLambda,
          typename Index_>
struct DistanceImpl<raft::distance::DistanceType::JensenShannon,
                    InType,
                    AccType,
                    OutType,
                    FinalLambda,
                    Index_> {
  void run(const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void*,
           size_t,
           FinalLambda fin_op,
           cudaStream_t stream,
           bool isRowMajor,
           InType)
  {
    raft::distance::detail::jensenShannonImpl<InType, AccType, OutType, FinalLambda, Index_>(
      m, n, k, x, y, dist, fin_op, stream, isRowMajor);
  }
};

template <typename InType,
          typename AccType,
          typename OutType,
          typename FinalLambda,
          typename Index_>
struct DistanceImpl<raft::distance::DistanceType::RusselRaoExpanded,
                    InType,
                    AccType,
                    OutType,
                    FinalLambda,
                    Index_> {
  void run(const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void*,
           size_t,
           FinalLambda fin_op,
           cudaStream_t stream,
           bool isRowMajor,
           InType)
  {
    raft::distance::detail::russellRaoImpl<InType, AccType, OutType, FinalLambda, Index_>(
      m, n, k, x, y, dist, fin_op, stream, isRowMajor);
  }
};

template <typename InType,
          typename AccType,
          typename OutType,
          typename FinalLambda,
          typename Index_>
struct DistanceImpl<raft::distance::DistanceType::KLDivergence,
                    InType,
                    AccType,
                    OutType,
                    FinalLambda,
                    Index_> {
  void run(const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void*,
           size_t,
           FinalLambda fin_op,
           cudaStream_t stream,
           bool isRowMajor,
           InType)
  {
    raft::distance::detail::klDivergenceImpl<InType, AccType, OutType, FinalLambda, Index_>(
      m, n, k, x, y, dist, fin_op, stream, isRowMajor);
  }
};

template <typename InType,
          typename AccType,
          typename OutType,
          typename FinalLambda,
          typename Index_>
struct DistanceImpl<raft::distance::DistanceType::CorrelationExpanded,
                    InType,
                    AccType,
                    OutType,
                    FinalLambda,
                    Index_> {
  void run(const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void* workspace,
           size_t worksize,
           FinalLambda fin_op,
           cudaStream_t stream,
           bool isRowMajor,
           InType)
  {
    raft::distance::detail::correlationImpl<InType, AccType, OutType, FinalLambda, Index_>(
      m, n, k, x, y, dist, (AccType*)workspace, worksize, fin_op, stream, isRowMajor);
  }
};

}  // anonymous namespace

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
void distance(const InType* x,
              const InType* y,
              OutType* dist,
              Index_ m,
              Index_ n,
              Index_ k,
              void* workspace,
              size_t worksize,
              FinalLambda fin_op,
              cudaStream_t stream,
              bool isRowMajor   = true,
              InType metric_arg = 2.0f)
{
  DistanceImpl<distanceType, InType, AccType, OutType, FinalLambda, Index_> distImpl;
  distImpl.run(x, y, dist, m, n, k, workspace, worksize, fin_op, stream, isRowMajor, metric_arg);
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

// Default final op functor which facilitates elementwise operation on
// final distance value if any.
template <typename AccType, typename OutType, typename Index>
struct default_fin_op {
  __host__ __device__ default_fin_op() noexcept {};
  // functor signature.
  __host__ __device__ OutType operator()(AccType d_val, Index g_d_idx) const noexcept
  {
    return d_val;
  }
};

template <raft::distance::DistanceType distanceType,
          typename InType,
          typename AccType,
          typename OutType,
          typename Index_ = int>
void distance(const InType* x,
              const InType* y,
              OutType* dist,
              Index_ m,
              Index_ n,
              Index_ k,
              void* workspace,
              size_t worksize,
              cudaStream_t stream,
              bool isRowMajor   = true,
              InType metric_arg = 2.0f)
{
  using final_op_type = default_fin_op<AccType, OutType, Index_>;
  final_op_type fin_op;

  // raft distance support inputs as float/double and output as uint8_t/float/double.
  static_assert(!((sizeof(OutType) > 1) && (sizeof(AccType) != sizeof(OutType))),
                "OutType can be uint8_t, float, double,"
                "if sizeof(OutType) > 1 then sizeof(AccType) == sizeof(OutType).");
  distance<distanceType, InType, AccType, OutType, final_op_type, Index_>(
    x, y, dist, m, n, k, workspace, worksize, fin_op, stream, isRowMajor, metric_arg);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
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
