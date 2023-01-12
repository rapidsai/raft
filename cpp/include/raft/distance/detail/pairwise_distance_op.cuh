/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <raft/core/operators.hpp>
#include <raft/linalg/contractions.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/vectorized.cuh>

#include <cstddef>

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
struct params_CT {
  using DataT = data_type;
  using AccT = accumulate_type;
  using OutT = out_type;
  using IdxT = index_type;

  using PolicyT = policy;

  using opT = op_type;
  using FinOpT = final_op_type;
  static constexpr bool is_row_major = row_major;
};

template <typename PCT>
__global__ __launch_bounds__(PCT::PolicyT::Nthreads, 2)

  void pairwiseDistanceOpKernel(
    const typename PCT::DataT* x,
    const typename PCT::DataT* y,
    const typename PCT::DataT* _xn,
    const typename PCT::DataT* _yn,
    typename PCT::IdxT m,
    typename PCT::IdxT n,
    typename PCT::IdxT k,
    typename PCT::IdxT lda,
    typename PCT::IdxT ldb,
    typename PCT::IdxT ldd,
    typename PCT::OutT* dOutput,
    typename PCT::opT distance_op,
    typename PCT::FinOpT fin_op)
{
  using AccT = typename PCT::AccT;
  using DataT = typename PCT::DataT;
  using OutT = typename PCT::OutT;
  using IdxT = typename PCT::IdxT;

  using Policy = typename PCT::PolicyT;

  // Instantiate PCT to access constexpr members.
  PCT compile_time_params{};

  extern __shared__ char smem[];

  // Wrap operator back into lambdas. This is temporary and should be removed. (TODO)
  auto core_op = [distance_op] __device__(AccT & acc, DataT & x, DataT & y) {
    // use .template to disambiguate (See: https://en.cppreference.com/w/cpp/language/dependent_name)
    distance_op.template core<AccT, DataT>(acc, x, y);
  };
  auto epilog_op = [distance_op] __device__(AccT acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                                     DataT * regxn,
                                     DataT * regyn,
                                     IdxT gridStrideX,
                                     IdxT gridStrideY) {
    distance_op.template epilog<Policy, AccT, DataT, IdxT>(acc, regxn, regyn, gridStrideX, gridStrideY);
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
    obj(
      x, y, m, n, k, lda, ldb, ldd, _xn, _yn, dOutput, smem, core_op, epilog_op, fin_op, row_epilog_op);
  obj.run();

}
};  // namespace detail
