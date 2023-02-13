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
#include <raft/core/resource/cuda_stream.hpp>
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
#include <raft/linalg/gemm.cuh>
#include <raft/util/cuda_utils.cuh>
#include <rmm/device_uvector.hpp>

namespace raft {
namespace distance {
namespace detail {

/** enum to tell how to compute distance */
enum DistanceType : unsigned short {

  /** evaluate as dist_ij = sum(x_ik^2) + sum(y_ij)^2 - 2*sum(x_ik * y_jk) */
  L2Expanded = 0,
  /** same as above, but inside the epilogue, perform square root operation */
  L2SqrtExpanded = 1,
  /** cosine distance */
  CosineExpanded = 2,
  /** L1 distance */
  L1 = 3,
  /** evaluate as dist_ij += (x_ik - y-jk)^2 */
  L2Unexpanded = 4,
  /** same as above, but inside the epilogue, perform square root operation */
  L2SqrtUnexpanded = 5,
  /** basic inner product **/
  InnerProduct = 6,
  /** Chebyshev (Linf) distance **/
  Linf = 7,
  /** Canberra distance **/
  Canberra = 8,
  /** Generalized Minkowski distance **/
  LpUnexpanded = 9,
  /** Correlation distance **/
  CorrelationExpanded = 10,
  /** Jaccard distance **/
  JaccardExpanded = 11,
  /** Hellinger distance **/
  HellingerExpanded = 12,
  /** Haversine distance **/
  Haversine = 13,
  /** Bray-Curtis distance **/
  BrayCurtis = 14,
  /** Jensen-Shannon distance**/
  JensenShannon = 15,
  /** Hamming distance **/
  HammingUnexpanded = 16,
  /** KLDivergence **/
  KLDivergence = 17,
  /** RusselRao **/
  RusselRaoExpanded = 18,
  /** Dice-Sorensen distance **/
  DiceExpanded = 19,
  /** Precomputed (special value) **/
  Precomputed = 100
};

namespace {
template <raft::distance::DistanceType distanceType,
          typename InType,
          typename AccType,
          typename OutType,
          typename FinalLambda,
          typename Index_>
struct DistanceImpl {
  void run(raft::resources const& handle,
           const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void* workspace,
           size_t worksize,
           FinalLambda fin_op,
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
  void run(raft::resources const& handle,
           const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void* workspace,
           size_t worksize,
           FinalLambda fin_op,
           bool isRowMajor,
           InType)
  {
    raft::distance::detail::euclideanAlgo1<InType, AccType, OutType, FinalLambda, Index_>(
      m,
      n,
      k,
      x,
      y,
      dist,
      false,
      (AccType*)workspace,
      worksize,
      fin_op,
      raft::resource::get_cuda_stream(handle),
      isRowMajor);
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
  void run(raft::resources const& handle,
           const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void* workspace,
           size_t worksize,
           FinalLambda fin_op,
           bool isRowMajor,
           InType)
  {
    raft::distance::detail::euclideanAlgo1<InType, AccType, OutType, FinalLambda, Index_>(
      m,
      n,
      k,
      x,
      y,
      dist,
      true,
      (AccType*)workspace,
      worksize,
      fin_op,
      raft::resource::get_cuda_stream(handle),
      isRowMajor);
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
  void run(raft::resources const& handle,
           const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void* workspace,
           size_t worksize,
           FinalLambda fin_op,
           bool isRowMajor,
           InType)
  {
    raft::distance::detail::cosineAlgo1<InType, AccType, OutType, FinalLambda, Index_>(
      m,
      n,
      k,
      x,
      y,
      dist,
      (AccType*)workspace,
      worksize,
      fin_op,
      raft::resource::get_cuda_stream(handle),
      isRowMajor);
  }
};

template <typename InType,
          typename AccType,
          typename OutType,
          typename FinalLambda,
          typename Index_>
struct DistanceImpl<raft::distance::DistanceType::InnerProduct,
                    InType,
                    AccType,
                    OutType,
                    FinalLambda,
                    Index_> {
  void run(raft::resources const& handle,
           const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void*,
           size_t,
           FinalLambda,
           bool isRowMajor,
           InType)
  {
    cudaStream_t stream = raft::resource::get_cuda_stream(handle);
    raft::linalg::gemm(handle,
                       dist,
                       const_cast<InType*>(x),
                       const_cast<InType*>(y),
                       m,
                       n,
                       k,
                       !isRowMajor,
                       !isRowMajor,
                       isRowMajor,
                       stream);
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
  void run(raft::resources const& handle,
           const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void*,
           size_t,
           FinalLambda fin_op,
           bool isRowMajor,
           InType)
  {
    raft::distance::detail::euclideanAlgo2<InType, AccType, OutType, FinalLambda, Index_>(
      m, n, k, x, y, dist, false, fin_op, raft::resource::get_cuda_stream(handle), isRowMajor);
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
  void run(raft::resources const& handle,
           const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void*,
           size_t,
           FinalLambda fin_op,
           bool isRowMajor,
           InType)
  {
    raft::distance::detail::euclideanAlgo2<InType, AccType, OutType, FinalLambda, Index_>(
      m, n, k, x, y, dist, true, fin_op, raft::resource::get_cuda_stream(handle), isRowMajor);
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
  void run(raft::resources const& handle,
           const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void*,
           size_t,
           FinalLambda fin_op,
           bool isRowMajor,
           InType)
  {
    raft::distance::detail::l1Impl<InType, AccType, OutType, FinalLambda, Index_>(
      m, n, k, x, y, dist, fin_op, raft::resource::get_cuda_stream(handle), isRowMajor);
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
  void run(raft::resources const& handle,
           const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void*,
           size_t,
           FinalLambda fin_op,
           bool isRowMajor,
           InType)
  {
    raft::distance::detail::chebyshevImpl<InType, AccType, OutType, FinalLambda, Index_>(
      m, n, k, x, y, dist, fin_op, raft::resource::get_cuda_stream(handle), isRowMajor);
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
  void run(raft::resources const& handle,
           const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void*,
           size_t,
           FinalLambda fin_op,
           bool isRowMajor,
           InType)
  {
    raft::distance::detail::hellingerImpl<InType, AccType, OutType, FinalLambda, Index_>(
      m, n, k, x, y, dist, fin_op, raft::resource::get_cuda_stream(handle), isRowMajor);
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
  void run(raft::resources const& handle,
           const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void*,
           size_t,
           FinalLambda fin_op,
           bool isRowMajor,
           InType metric_arg)
  {
    raft::distance::detail::minkowskiImpl<InType, AccType, OutType, FinalLambda, Index_>(
      m, n, k, x, y, dist, fin_op, raft::resource::get_cuda_stream(handle), isRowMajor, metric_arg);
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
  void run(raft::resources const& handle,
           const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void*,
           size_t,
           FinalLambda fin_op,
           bool isRowMajor,
           InType)
  {
    raft::distance::detail::canberraImpl<InType, AccType, OutType, FinalLambda, Index_>(
      m, n, k, x, y, dist, fin_op, raft::resource::get_cuda_stream(handle), isRowMajor);
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
  void run(raft::resources const& handle,
           const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void*,
           size_t,
           FinalLambda fin_op,
           bool isRowMajor,
           InType)
  {
    raft::distance::detail::hammingUnexpandedImpl<InType, AccType, OutType, FinalLambda, Index_>(
      m, n, k, x, y, dist, fin_op, raft::resource::get_cuda_stream(handle), isRowMajor);
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
  void run(raft::resources const& handle,
           const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void*,
           size_t,
           FinalLambda fin_op,
           bool isRowMajor,
           InType)
  {
    raft::distance::detail::jensenShannonImpl<InType, AccType, OutType, FinalLambda, Index_>(
      m, n, k, x, y, dist, fin_op, raft::resource::get_cuda_stream(handle), isRowMajor);
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
  void run(raft::resources const& handle,
           const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void*,
           size_t,
           FinalLambda fin_op,
           bool isRowMajor,
           InType)
  {
    raft::distance::detail::russellRaoImpl<InType, AccType, OutType, FinalLambda, Index_>(
      m, n, k, x, y, dist, fin_op, raft::resource::get_cuda_stream(handle), isRowMajor);
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
  void run(raft::resources const& handle,
           const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void*,
           size_t,
           FinalLambda fin_op,
           bool isRowMajor,
           InType)
  {
    raft::distance::detail::klDivergenceImpl<InType, AccType, OutType, FinalLambda, Index_>(
      m, n, k, x, y, dist, fin_op, raft::resource::get_cuda_stream(handle), isRowMajor);
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
  void run(raft::resources const& handle,
           const InType* x,
           const InType* y,
           OutType* dist,
           Index_ m,
           Index_ n,
           Index_ k,
           void* workspace,
           size_t worksize,
           FinalLambda fin_op,
           bool isRowMajor,
           InType)
  {
    raft::distance::detail::correlationImpl<InType, AccType, OutType, FinalLambda, Index_>(
      m,
      n,
      k,
      x,
      y,
      dist,
      (AccType*)workspace,
      worksize,
      fin_op,
      raft::resource::get_cuda_stream(handle),
      isRowMajor);
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
              OutType* dist,
              Index_ m,
              Index_ n,
              Index_ k,
              void* workspace,
              size_t worksize,
              FinalLambda fin_op,
              bool isRowMajor   = true,
              InType metric_arg = 2.0f)
{
  DistanceImpl<distanceType, InType, AccType, OutType, FinalLambda, Index_> distImpl;
  distImpl.run(handle, x, y, dist, m, n, k, workspace, worksize, fin_op, isRowMajor, metric_arg);
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
void distance(raft::resources const& handle,
              const InType* x,
              const InType* y,
              OutType* dist,
              Index_ m,
              Index_ n,
              Index_ k,
              void* workspace,
              size_t worksize,
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
    handle, x, y, dist, m, n, k, workspace, worksize, fin_op, isRowMajor, metric_arg);
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
 * @param isRowMajor whether the matrices are row-major or col-major
 */
template <typename Type, typename Index_, raft::distance::DistanceType DistType>
void pairwise_distance_impl(raft::resources const& handle,
                            const Type* x,
                            const Type* y,
                            Type* dist,
                            Index_ m,
                            Index_ n,
                            Index_ k,
                            rmm::device_uvector<char>& workspace,
                            bool isRowMajor,
                            Type metric_arg = 2.0f)
{
  auto worksize = getWorkspaceSize<DistType, Type, Type, Type, Index_>(x, y, m, n, k);
  workspace.resize(worksize, raft::resource::get_cuda_stream(handle));
  distance<DistType, Type, Type, Type, Index_>(
    handle, x, y, dist, m, n, k, workspace.data(), worksize, isRowMajor, metric_arg);
}
/** @} */
};  // namespace detail
};  // namespace distance
};  // namespace raft
