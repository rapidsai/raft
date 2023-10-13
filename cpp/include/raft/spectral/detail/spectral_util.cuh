/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/spectral/matrix_wrappers.hpp>
#include <raft/util/cudart_utils.hpp>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <algorithm>

namespace raft {
namespace spectral {

template <typename index_type_t, typename value_type_t>
RAFT_KERNEL scale_obs_kernel(index_type_t m, index_type_t n, value_type_t* obs)
{
  index_type_t i, j, k, index, mm;
  value_type_t alpha, v, last;
  bool valid;
  // ASSUMPTION: kernel is launched with either 2, 4, 8, 16 or 32 threads in x-dimension

  // compute alpha
  mm    = (((m + blockDim.x - 1) / blockDim.x) * blockDim.x);  // m in multiple of blockDim.x
  alpha = 0.0;

  for (j = threadIdx.y + blockIdx.y * blockDim.y; j < n; j += blockDim.y * gridDim.y) {
    for (i = threadIdx.x; i < mm; i += blockDim.x) {
      // check if the thread is valid
      valid = i < m;

      // get the value of the last thread
      last = __shfl_sync(warp_full_mask(), alpha, blockDim.x - 1, blockDim.x);

      // if you are valid read the value from memory, otherwise set your value to 0
      alpha = (valid) ? obs[i + j * m] : 0.0;
      alpha = alpha * alpha;

      // do prefix sum (of size warpSize=blockDim.x =< 32)
      for (k = 1; k < blockDim.x; k *= 2) {
        v = __shfl_up_sync(warp_full_mask(), alpha, k, blockDim.x);
        if (threadIdx.x >= k) alpha += v;
      }
      // shift by last
      alpha += last;
    }
  }

  // scale by alpha
  alpha = __shfl_sync(warp_full_mask(), alpha, blockDim.x - 1, blockDim.x);
  alpha = raft::sqrt(alpha);
  for (j = threadIdx.y + blockIdx.y * blockDim.y; j < n; j += blockDim.y * gridDim.y) {
    for (i = threadIdx.x; i < m; i += blockDim.x) {  // blockDim.x=32
      index      = i + j * m;
      obs[index] = obs[index] / alpha;
    }
  }
}

template <typename index_type_t>
index_type_t next_pow2(index_type_t n)
{
  index_type_t v;
  // Reference:
  // http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2Float
  v = n - 1;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}

template <typename index_type_t, typename value_type_t>
cudaError_t scale_obs(index_type_t m, index_type_t n, value_type_t* obs)
{
  index_type_t p2m;

  // find next power of 2
  p2m = next_pow2<index_type_t>(m);
  // setup launch configuration
  unsigned int xsize = std::max(2, std::min(p2m, 32));
  dim3 nthreads{xsize, 256 / xsize, 1};

  dim3 nblocks{1, (n + nthreads.y - 1) / nthreads.y, 1};

  // launch scaling kernel (scale each column of obs by its norm)
  scale_obs_kernel<index_type_t, value_type_t><<<nblocks, nthreads>>>(m, n, obs);

  return cudaSuccess;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void transform_eigen_matrix(raft::resources const& handle,
                            edge_t n,
                            vertex_t nEigVecs,
                            weight_t* eigVecs)
{
  auto stream             = resource::get_cuda_stream(handle);
  auto cublas_h           = resource::get_cublas_handle(handle);
  auto thrust_exec_policy = resource::get_thrust_policy(handle);

  const weight_t zero{0.0};
  const weight_t one{1.0};

  // Whiten eigenvector matrix
  for (auto i = 0; i < nEigVecs; ++i) {
    weight_t mean, std;

    mean = thrust::reduce(thrust_exec_policy,
                          thrust::device_pointer_cast(eigVecs + IDX(0, i, n)),
                          thrust::device_pointer_cast(eigVecs + IDX(0, i + 1, n)));
    RAFT_CHECK_CUDA(stream);
    mean /= n;
    thrust::transform(thrust_exec_policy,
                      thrust::device_pointer_cast(eigVecs + IDX(0, i, n)),
                      thrust::device_pointer_cast(eigVecs + IDX(0, i + 1, n)),
                      thrust::make_constant_iterator(mean),
                      thrust::device_pointer_cast(eigVecs + IDX(0, i, n)),
                      thrust::minus<weight_t>());
    RAFT_CHECK_CUDA(stream);

    // TODO: Call from public API when ready
    RAFT_CUBLAS_TRY(
      raft::linalg::detail::cublasnrm2(cublas_h, n, eigVecs + IDX(0, i, n), 1, &std, stream));

    std /= std::sqrt(static_cast<weight_t>(n));

    thrust::transform(thrust_exec_policy,
                      thrust::device_pointer_cast(eigVecs + IDX(0, i, n)),
                      thrust::device_pointer_cast(eigVecs + IDX(0, i + 1, n)),
                      thrust::make_constant_iterator(std),
                      thrust::device_pointer_cast(eigVecs + IDX(0, i, n)),
                      thrust::divides<weight_t>());
    RAFT_CHECK_CUDA(stream);
  }

  // Transpose eigenvector matrix
  //   TODO: in-place transpose
  {
    raft::spectral::matrix::vector_t<weight_t> work(handle, nEigVecs * n);
    // TODO: Call from public API when ready
    RAFT_CUBLAS_TRY(
      raft::linalg::detail::cublassetpointermode(cublas_h, CUBLAS_POINTER_MODE_HOST, stream));

    // TODO: Call from public API when ready
    RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgeam(cublas_h,
                                                     CUBLAS_OP_T,
                                                     CUBLAS_OP_N,
                                                     nEigVecs,
                                                     n,
                                                     &one,
                                                     eigVecs,
                                                     n,
                                                     &zero,
                                                     (weight_t*)NULL,
                                                     nEigVecs,
                                                     work.raw(),
                                                     nEigVecs,
                                                     stream));

    RAFT_CUDA_TRY(cudaMemcpyAsync(
      eigVecs, work.raw(), nEigVecs * n * sizeof(weight_t), cudaMemcpyDeviceToDevice, stream));
  }
}

namespace {
/// Functor to generate indicator vectors
/** For use in Thrust transform
 */
template <typename index_type_t, typename value_type_t>
struct equal_to_i_op {
  const index_type_t i;

 public:
  equal_to_i_op(index_type_t _i) : i(_i) {}
  template <typename Tuple_>
  __host__ __device__ void operator()(Tuple_ t)
  {
    thrust::get<1>(t) = (thrust::get<0>(t) == i) ? (value_type_t)1.0 : (value_type_t)0.0;
  }
};
}  // namespace

// Construct indicator vector for ith partition
//
template <typename vertex_t, typename edge_t, typename weight_t>
bool construct_indicator(raft::resources const& handle,
                         edge_t index,
                         edge_t n,
                         weight_t& clustersize,
                         weight_t& partStats,
                         vertex_t const* __restrict__ clusters,
                         raft::spectral::matrix::vector_t<weight_t>& part_i,
                         raft::spectral::matrix::vector_t<weight_t>& Bx,
                         raft::spectral::matrix::laplacian_matrix_t<vertex_t, weight_t> const& B)
{
  auto stream             = resource::get_cuda_stream(handle);
  auto cublas_h           = resource::get_cublas_handle(handle);
  auto thrust_exec_policy = resource::get_thrust_policy(handle);

  thrust::for_each(
    thrust_exec_policy,
    thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(clusters),
                                                 thrust::device_pointer_cast(part_i.raw()))),
    thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(clusters + n),
                                                 thrust::device_pointer_cast(part_i.raw() + n))),
    equal_to_i_op<vertex_t, weight_t>(index));
  RAFT_CHECK_CUDA(stream);

  // Compute size of ith partition
  // TODO: Call from public API when ready
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(
    cublas_h, n, part_i.raw(), 1, part_i.raw(), 1, &clustersize, stream));

  clustersize = round(clustersize);
  if (clustersize < 0.5) { return false; }

  // Compute part stats
  B.mv(1, part_i.raw(), 0, Bx.raw());
  // TODO: Call from public API when ready
  RAFT_CUBLAS_TRY(
    raft::linalg::detail::cublasdot(cublas_h, n, Bx.raw(), 1, part_i.raw(), 1, &partStats, stream));

  return true;
}

}  // namespace spectral
}  // namespace raft
