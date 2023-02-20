/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cstddef>
#include <raft/core/operators.hpp>
#include <raft/distance/detail/pairwise_distance_base.cuh>

namespace raft::distance::detail {

template <typename Policy,
          bool row_major,
          typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT,
          typename opT,
          typename FinOpT>
__global__ __launch_bounds__(Policy::Nthreads, 2) void pairwise_matrix_kernel(const DataT* x,
                                                                              const DataT* y,
                                                                              const DataT* _xn,
                                                                              const DataT* _yn,
                                                                              IdxT m,
                                                                              IdxT n,
                                                                              IdxT k,
                                                                              IdxT lda,
                                                                              IdxT ldb,
                                                                              IdxT ldd,
                                                                              OutT* dOutput,
                                                                              opT distance_op,
                                                                              FinOpT fin_op)
{
  extern __shared__ char smem[];

  // Wrap operator back into lambdas. This is temporary and should be removed. (TODO)
  auto core_op = [distance_op] __device__(AccT & acc, DataT & x, DataT & y) {
    distance_op.core(acc, x, y);
  };
  auto epilog_op = [distance_op] __device__(AccT acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                                            DataT * regxn,
                                            DataT * regyn,
                                            IdxT gridStrideX,
                                            IdxT gridStrideY) {
    // Use .template to disambiguate (See:
    // https://en.cppreference.com/w/cpp/language/dependent_name)
    distance_op.template epilog<Policy, AccT, DataT, IdxT>(
      acc, regxn, regyn, gridStrideX, gridStrideY);
  };

  // No support for row_epilog_op.
  auto row_epilog_op = raft::void_op();

  // Always write output
  constexpr bool write_out = true;
  constexpr bool use_norms = distance_op.use_norms;
  PairwiseDistances<use_norms,
                    DataT,
                    AccT,
                    OutT,
                    IdxT,
                    Policy,
                    decltype(core_op),
                    decltype(epilog_op),
                    decltype(fin_op),
                    decltype(row_epilog_op),
                    row_major,
                    write_out>
    obj(x,
        y,
        m,
        n,
        k,
        lda,
        ldb,
        ldd,
        _xn,
        _yn,
        dOutput,
        smem,
        core_op,
        epilog_op,
        fin_op,
        row_epilog_op);
  obj.run();
}

template <typename Policy,
          bool row_major,
          typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT,
          typename OpT,
          typename FinOpT>
void pairwise_matrix(OpT distance_op,
                     FinOpT fin_op,
                     const DataT* x,
                     const DataT* y,
                     const DataT* _xn,
                     const DataT* _yn,
                     IdxT m,
                     IdxT n,
                     IdxT k,
                     IdxT lda,
                     IdxT ldb,
                     IdxT ldd,
                     OutT* dOutput,
                     cudaStream_t stream)
{
  dim3 blk(Policy::Nthreads);
  // Use .template to disambiguate (See:
  // https://en.cppreference.com/w/cpp/language/dependent_name)
  size_t smem_size = distance_op.template shared_mem_size<Policy, DataT>();
  // Obtain function pointer to kernel
  auto kernel = pairwise_matrix_kernel<Policy, row_major, DataT, AccT, OutT, IdxT, OpT, FinOpT>;
  dim3 grid   = launchConfigGenerator<Policy>(m, n, smem_size, kernel);

  kernel<<<grid, blk, smem_size, stream>>>(
    x, y, _xn, _yn, m, n, k, lda, ldb, ldd, dOutput, distance_op, fin_op);
  RAFT_CUDA_TRY(cudaGetLastError());
}

};  // namespace raft::distance::detail
