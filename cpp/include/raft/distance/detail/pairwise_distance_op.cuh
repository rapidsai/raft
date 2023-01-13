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
  using AccT  = accumulate_type;
  using OutT  = out_type;
  using IdxT  = index_type;
  using PolicyT = policy;
  using opT                          = op_type;
  using FinOpT                       = final_op_type;
  static constexpr bool is_row_major = row_major;
};

template <typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT,
          typename opT,
          typename FinOpT>
struct params_RT {
  int vectorized_load_num_elem = 1;
  bool row_major               = true;

  // Turn run-time parameters into compile-time parameters.
  // Call the provided function f with these compile-time parameters.
  // Returns false if dispatch fails, i.e., if there is no implementation
  // for the given runtime parameters.
  template <typename F>
  bool dispatch_with_compile_time_params(F&& f) const
  {
    return convert_vectorized_load_num_elem(f);
  }

  // Step 1: convert alignment into a compile time constant
  template <typename F>
  bool convert_vectorized_load_num_elem(F&& f) const
  {
    bool fail = false;
    switch (vectorized_load_num_elem) {
      case 1: return layout<1>(f);
      case 2: return layout<2>(f);
      case 4:
        // We need "if constexpr" here, to prevent the if else to be delegated
        // to run time (in which case a kernel that loads 4 doubles is
        // generated). This is especially important, because that leads to
        // compilation errors (which we want to avoid).
        if constexpr (sizeof(DataT) < 8) {
          return layout<4>(f);
        } else {
          // For doubles, load at most 2 elements in one instruction.
          return layout<2>(f);
        }
      default: return fail;
    };
  }

  // Step 2: convert layout into a compile time constant
  template <int vec_len, typename F>
  bool layout(F&& f) const
  {
    if (row_major) {
      return to_compile_time_params<vec_len, true>(f);
    } else {
      return to_compile_time_params<vec_len, false>(f);
    }
  }

  // Step 3: convert compile-time constant into compile-time parameter struct and invoke
  // function f with these compile time parameters.
  template <int vec_len, bool is_row_major, typename F>
  bool to_compile_time_params(F&& f) const
  {
    // Determine kernel policy using vec_len and layout
    typedef typename raft::linalg::Policy4x4<DataT, vec_len>::Policy RowPolicy;
    typedef typename raft::linalg::Policy4x4<DataT, vec_len>::ColPolicy ColPolicy;
    typedef typename std::conditional<is_row_major, RowPolicy, ColPolicy>::type Policy;

    // Create compile-time parameter type and instantiate a struct;
    using PCT = params_CT<DataT, AccT, OutT, IdxT, Policy, opT, FinOpT, is_row_major>;
    PCT compile_time_params{};

    // Dispatch to f
    f(compile_time_params);

    bool dispatch_success = true;
    return dispatch_success;
  }
};

template <typename PCT>
__global__ __launch_bounds__(PCT::PolicyT::Nthreads, 2)

  void pairwiseDistanceOpKernel(const typename PCT::DataT* x,
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
  using AccT  = typename PCT::AccT;
  using DataT = typename PCT::DataT;
  using OutT  = typename PCT::OutT;
  using IdxT  = typename PCT::IdxT;

  using Policy = typename PCT::PolicyT;

  // Instantiate compile time parameters to access constexpr members.
  PCT compile_time_params{};

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

};  // namespace raft::distance::detail
