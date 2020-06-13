/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <raft/handle.hpp>
#include <raft/spectral/error_temp.hpp>
#include <raft/spectral/sm_utils.hpp>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

namespace raft {
namespace spectral {

template <typename IndexType_, typename ValueType_>
static __global__ void scale_obs_kernel(IndexType_ m, IndexType_ n,
                                        ValueType_* obs) {
  IndexType_ i, j, k, index, mm;
  ValueType_ alpha, v, last;
  bool valid;
  // ASSUMPTION: kernel is launched with either 2, 4, 8, 16 or 32 threads in x-dimension

  // compute alpha
  mm = (((m + blockDim.x - 1) / blockDim.x) *
        blockDim.x);  // m in multiple of blockDim.x
  alpha = 0.0;
  // printf("[%d,%d,%d,%d] n=%d, li=%d, mn=%d \n",threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y, n,
  // li, mn);
  for (j = threadIdx.y + blockIdx.y * blockDim.y; j < n;
       j += blockDim.y * gridDim.y) {
    for (i = threadIdx.x; i < mm; i += blockDim.x) {
      // check if the thread is valid
      valid = i < m;

      // get the value of the last thread
      last = utils::shfl(alpha, blockDim.x - 1, blockDim.x);

      // if you are valid read the value from memory, otherwise set your value to 0
      alpha = (valid) ? obs[i + j * m] : 0.0;
      alpha = alpha * alpha;

      // do prefix sum (of size warpSize=blockDim.x =< 32)
      for (k = 1; k < blockDim.x; k *= 2) {
        v = utils::shfl_up(alpha, k, blockDim.x);
        if (threadIdx.x >= k) alpha += v;
      }
      // shift by last
      alpha += last;
    }
  }

  // scale by alpha
  alpha = utils::shfl(alpha, blockDim.x - 1, blockDim.x);
  alpha = std::sqrt(alpha);
  for (j = threadIdx.y + blockIdx.y * blockDim.y; j < n;
       j += blockDim.y * gridDim.y) {
    for (i = threadIdx.x; i < m; i += blockDim.x) {  // blockDim.x=32
      index = i + j * m;
      obs[index] = obs[index] / alpha;
    }
  }
}

template <typename IndexType_>
IndexType_ next_pow2(IndexType_ n) {
  IndexType_ v;
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

template <typename IndexType_, typename ValueType_>
cudaError_t scale_obs(IndexType_ m, IndexType_ n, ValueType_* obs) {
  IndexType_ p2m;
  dim3 nthreads, nblocks;

  // find next power of 2
  p2m = next_pow2<IndexType_>(m);
  // setup launch configuration
  nthreads.x = max(2, min(p2m, 32));
  nthreads.y = 256 / nthreads.x;
  nthreads.z = 1;
  nblocks.x = 1;
  nblocks.y = (n + nthreads.y - 1) / nthreads.y;
  nblocks.z = 1;
  // printf("m=%d(%d),n=%d,obs=%p,
  // nthreads=(%d,%d,%d),nblocks=(%d,%d,%d)\n",m,p2m,n,obs,nthreads.x,nthreads.y,nthreads.z,nblocks.x,nblocks.y,nblocks.z);

  // launch scaling kernel (scale each column of obs by its norm)
  scale_obs_kernel<IndexType_, ValueType_><<<nblocks, nthreads>>>(m, n, obs);
  CUDA_CHECK_LAST();

  return cudaSuccess;
}

template <typename vertex_t, typename edge_t, typename weight_t,
          typename ThrustExePolicy>
void transform_eigen_matrix(handle_t handle, ThrustExePolicy thrust_exec_policy,
                            edge_t n, vertex_t nEigVecs, weight_t* eigVecs) {
  auto cublas_h = handle.get_cublas_handle();
  auto stream = handle.get_stream();

  // Whiten eigenvector matrix
  for (auto i = 0; i < nEigVecs; ++i) {
    weight_t mean, std;

    mean = thrust::reduce(
      thrust_exec_policy, thrust::device_pointer_cast(eigVecs + IDX(0, i, n)),
      thrust::device_pointer_cast(eigVecs + IDX(0, i + 1, n)));
    CUDA_CHECK_LAST();
    mean /= n;
    thrust::transform(thrust_exec_policy,
                      thrust::device_pointer_cast(eigVecs + IDX(0, i, n)),
                      thrust::device_pointer_cast(eigVecs + IDX(0, i + 1, n)),
                      thrust::make_constant_iterator(mean),
                      thrust::device_pointer_cast(eigVecs + IDX(0, i, n)),
                      thrust::minus<weight_t>());
    CUDA_CHECK_LAST();

    CUBLAS_CHECK(
      cublasnrm2(cublas_h, n, eigVecs + IDX(0, i, n), 1, &std, stream));

    std /= std::sqrt(static_cast<weight_t>(n));

    thrust::transform(thrust_exec_policy,
                      thrust::device_pointer_cast(eigVecs + IDX(0, i, n)),
                      thrust::device_pointer_cast(eigVecs + IDX(0, i + 1, n)),
                      thrust::make_constant_iterator(std),
                      thrust::device_pointer_cast(eigVecs + IDX(0, i, n)),
                      thrust::divides<weight_t>());
    CUDA_CHECK_LAST();
  }

  // Transpose eigenvector matrix
  //   TODO: in-place transpose
  {
    vector_t<weight_t> work(handle, nEigVecs * n);
    CUBLAS_CHECK(
      cublassetpointermode(cublas_h, CUBLAS_POINTER_MODE_HOST, stream));

    CUBLAS_CHECK(cublasgeam(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N, nEigVecs, n,
                            &one, eigVecs, n, &zero, (weight_t*)NULL, nEigVecs,
                            work.raw(), nEigVecs, stream));

    CUDA_TRY(cudaMemcpyAsync(eigVecs, work.raw(),
                             nEigVecs * n * sizeof(weight_t),
                             cudaMemcpyDeviceToDevice, stream));
  }
}

namespace {
/// Functor to generate indicator vectors
/** For use in Thrust transform
 */
template <typename IndexType_, typename ValueType_>
struct equal_to_i_op {
  const IndexType_ i;

 public:
  equal_to_i_op(IndexType_ _i) : i(_i) {}
  template <typename Tuple_>
  __host__ __device__ void operator()(Tuple_ t) {
    thrust::get<1>(t) =
      (thrust::get<0>(t) == i) ? (ValueType_)1.0 : (ValueType_)0.0;
  }
};
}  // namespace

// Construct indicator vector for ith partition
//
template <typename vertex_t, typename edge_t, typename weight_t>
bool construct_indicator(handle_t handle, ThrustExePolicy thrust_exec_policy,
                         edge_t n, weight_t& clustersize, weight_t& partStats,
                         vertex_t const* __restrict__ clusters,
                         vector_t<weight_t>& part_i, vector_t<weight_t>& Bx,
                         laplacian_matrix_t<vertex_t, weight_t> const& B) {
  auto cublas_h = handle.get_cublas_handle();
  auto stream = handle.get_stream();

  thrust::for_each(thrust_exec_policy,
                   thrust::make_zip_iterator(thrust::make_tuple(
                     thrust::device_pointer_cast(clusters),
                     thrust::device_pointer_cast(part_i.raw()))),
                   thrust::make_zip_iterator(thrust::make_tuple(
                     thrust::device_pointer_cast(clusters + n),
                     thrust::device_pointer_cast(part_i.raw() + n))),
                   equal_to_i_op<vertex_t, weight_t>(i));
  CUDA_CHECK_LAST();

  // Compute size of ith partition
  CUBLAS_CHECK(cublasdot(cublas_h, n, part_i.raw(), 1, part_i.raw(), 1,
                         &clustersize, stream));

  clustersize = round(clustersize);
  if (clustersize < 0.5) {
    return false;
  }

  // Compute part stats
  B.mv(1, part_i.raw(), 0, Bx.raw());
  CUBLAS_CHECK(
    cublasdot(cublas_h, n, Bx.raw(), 1, part_i.raw(), 1, &partStats, stream));

  return true;
}

}  // namespace spectral
}  // namespace raft
