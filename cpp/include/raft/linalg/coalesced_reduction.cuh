/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include <cub/cub.cuh>
#include <raft/cuda_utils.cuh>

namespace raft {
namespace linalg {

// Kernel (based on norm.cuh) to perform reductions along the coalesced dimension
// of the matrix, i.e. reduce along rows for row major or reduce along columns
// for column major layout. Kernel does an inplace reduction adding to original
// values of dots.
template <typename InType, typename OutType, typename IdxType, int TPB,
          typename MainLambda, typename ReduceLambda, typename FinalLambda>
__global__ void coalesced_reduction_kernel(OutType *dots, const InType *data,
                                         int D, int N, OutType init,
                                         MainLambda main_op,
                                         ReduceLambda reduce_op,
                                         FinalLambda final_op,
                                         bool inplace = false) {
  using block_reduce_t = cub::BlockReduce<OutType, TPB>;
  __shared__ typename block_reduce_t::TempStorage temp_storage;  // NOLINT
  auto thread_data = init;
  IdxType row_start = blockIdx.x * D;
  for (IdxType i = threadIdx.x; i < D; i += TPB) {
    auto idx = row_start + i;
    thread_data = reduce_op(thread_data, main_op(data[idx], i));
  }
  OutType acc = block_reduce_t(temp_storage).Reduce(thread_data, reduce_op);
  if (threadIdx.x == 0) {
    if (inplace) {
      dots[blockIdx.x] = final_op(reduce_op(dots[blockIdx.x], acc));
    } else {
      dots[blockIdx.x] = final_op(acc);
    }
  }
}

/**
 * @brief Compute reduction of the input matrix along the leading dimension
 *
 * @tparam InType the data type of the input
 * @tparam OutType the data type of the output (as well as the data type for
 *  which reduction is performed)
 * @tparam IdxType data type of the indices of the array
 * @tparam MainLambda Unary lambda applied while acculumation (eg: L1 or L2 norm)
 * It must be a 'callable' supporting the following input and output:
 * <pre>OutType (*MainLambda)(InType, IdxType);</pre>
 * @tparam ReduceLambda Binary lambda applied for reduction (eg: addition(+) for L2 norm)
 * It must be a 'callable' supporting the following input and output:
 * <pre>OutType (*ReduceLambda)(OutType);</pre>
 * @tparam FinalLambda the final lambda applied before STG (eg: Sqrt for L2 norm)
 * It must be a 'callable' supporting the following input and output:
 * <pre>OutType (*FinalLambda)(OutType);</pre>
 * @param dots the output reduction vector
 * @param data the input matrix
 * @param D leading dimension of data
 * @param N second dimension data
 * @param init initial value to use for the reduction
 * @param main_op elementwise operation to apply before reduction
 * @param reduce_op binary reduction operation
 * @param final_op elementwise operation to apply before storing results
 * @param inplace reduction result added inplace or overwrites old values?
 * @param stream cuda stream where to launch work
 */
template <typename InType, typename OutType = InType, typename IdxType = int,
          typename MainLambda = raft::Nop<InType, IdxType>,
          typename ReduceLambda = raft::Sum<OutType>,
          typename FinalLambda = raft::Nop<OutType>>
void coalescedReduction(OutType *dots, const InType *data, int D,  // NOLINT
                         int N, OutType init, cudaStream_t stream, bool inplace = false,
                        MainLambda main_op = raft::Nop<InType, IdxType>(),
                        ReduceLambda reduce_op = raft::Sum<OutType>(),
                        FinalLambda final_op = raft::Nop<OutType>()) {
  // One block per reduction
  // Efficient only for large leading dimensions
  if (D <= 32) {
    coalesced_reduction_kernel<InType, OutType, IdxType, 32>
      <<<N, 32, 0, stream>>>(dots, data, D, N, init, main_op, reduce_op,
                             final_op, inplace);
  } else if (D <= 64) {
    coalesced_reduction_kernel<InType, OutType, IdxType, 64>
      <<<N, 64, 0, stream>>>(dots, data, D, N, init, main_op, reduce_op,
                             final_op, inplace);
  } else if (D <= 128) {
    coalesced_reduction_kernel<InType, OutType, IdxType, 128>
      <<<N, 128, 0, stream>>>(dots, data, D, N, init, main_op, reduce_op,
                              final_op, inplace);
  } else {
    coalesced_reduction_kernel<InType, OutType, IdxType, 256>
      <<<N, 256, 0, stream>>>(dots, data, D, N, init, main_op, reduce_op,
                              final_op, inplace);
  }
  CUDA_CHECK(cudaPeekAtLastError());
}

};  // end namespace linalg
};  // end namespace raft
