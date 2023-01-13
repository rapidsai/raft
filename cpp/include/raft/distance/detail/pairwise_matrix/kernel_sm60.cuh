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
#include <raft/util/cudart_utils.hpp> // TODO: remove

#include <raft/distance/detail/pairwise_distance_base.cuh>

namespace raft::distance::detail {

template <typename data_type,
          typename accumulate_type,
          typename out_type,
          typename index_type,

          typename policy,
          // Op (L2, L1, etc...)
          typename op_type,
          typename final_op_type,
          bool row_major>
struct kernel_params_T {
  using DataT                        = data_type;
  using AccT                         = accumulate_type;
  using OutT                         = out_type;
  using IdxT                         = index_type;
  using PolicyT                      = policy;
  using opT                          = op_type;
  using FinOpT                       = final_op_type;
  static constexpr bool is_row_major = row_major;
};

template <typename KP_T>
__global__ __launch_bounds__(KP_T::PolicyT::Nthreads, 2)

  void pairwise_matrix_kernel(const typename KP_T::DataT* x,
                              const typename KP_T::DataT* y,
                              const typename KP_T::DataT* _xn,
                              const typename KP_T::DataT* _yn,
                              typename KP_T::IdxT m,
                              typename KP_T::IdxT n,
                              typename KP_T::IdxT k,
                              typename KP_T::IdxT lda,
                              typename KP_T::IdxT ldb,
                              typename KP_T::IdxT ldd,
                              typename KP_T::OutT* dOutput,
                              typename KP_T::opT distance_op,
                              typename KP_T::FinOpT fin_op)
{
  using AccT  = typename KP_T::AccT;
  using DataT = typename KP_T::DataT;
  using OutT  = typename KP_T::OutT;
  using IdxT  = typename KP_T::IdxT;

  using Policy = typename KP_T::PolicyT;

  // Instantiate compile time parameters to access constexpr members.
  KP_T compile_time_params{};

  extern __shared__ char smem[];

  // Wrap operator back into lambdas. This is temporary and should be removed. (TODO)
  auto core_op = [distance_op] __device__(AccT & acc, DataT & x, DataT & y) {
    // use .template to disambiguate (See:
    // https://en.cppreference.com/w/cpp/language/dependent_name)
    distance_op.template core<AccT, DataT>(acc, x, y);
  };
  auto epilog_op = [distance_op] __device__(AccT acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                                            DataT * regxn,
                                            DataT * regyn,
                                            IdxT gridStrideX,
                                            IdxT gridStrideY) {
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
                    compile_time_params.is_row_major,
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

template <typename KP_T>
static void pairwise_matrix(typename KP_T::opT distance_op,
                            typename KP_T::FinOpT fin_op,
                            const typename KP_T::DataT* x,
                            const typename KP_T::DataT* y,
                            const typename KP_T::DataT* _xn,
                            const typename KP_T::DataT* _yn,
                            typename KP_T::IdxT m,
                            typename KP_T::IdxT n,
                            typename KP_T::IdxT k,
                            typename KP_T::IdxT lda,
                            typename KP_T::IdxT ldb,
                            typename KP_T::IdxT ldd,
                            typename KP_T::OutT* dOutput,
                            cudaStream_t stream)
{
  using Policy = typename KP_T::PolicyT;

  dim3 blk(Policy::Nthreads);
  size_t smem_size = distance_op.template shared_mem_size<Policy>();
  dim3 grid        = launchConfigGenerator<Policy>(m, n, smem_size, pairwise_matrix_kernel<KP_T>);

  pairwise_matrix_kernel<KP_T><<<grid, blk, smem_size, stream>>>(
    x, y, _xn, _yn, m, n, k, lda, ldb, ldd, dOutput, distance_op, fin_op);

  RAFT_CUDA_TRY(cudaGetLastError());
}

};  // namespace raft::distance::detail
