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

#include <cub/cub.cuh>
#include <raft/util/cuda_utils.cuh>

namespace raft {
namespace linalg {
namespace detail {

// Kernel (based on norm.cuh) to perform reductions along the coalesced dimension
// of the matrix, i.e. reduce along rows for row major or reduce along columns
// for column major layout. Kernel does an inplace reduction adding to original
// values of dots.
template <typename InType,
          typename OutType,
          typename IdxType,
          int TPB,
          typename MainLambda,
          typename ReduceLambda,
          typename FinalLambda>
__global__ void coalescedReductionKernel(OutType* dots,
                                         const InType* data,
                                         int D,
                                         int N,
                                         OutType init,
                                         MainLambda main_op,
                                         ReduceLambda reduce_op,
                                         FinalLambda final_op,
                                         bool inplace = false)
{
  typedef cub::BlockReduce<OutType, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  OutType thread_data = init;
  IdxType rowStart    = blockIdx.x * D;
  for (IdxType i = threadIdx.x; i < D; i += TPB) {
    IdxType idx = rowStart + i;
    thread_data = reduce_op(thread_data, main_op(data[idx], i));
  }
  OutType acc = BlockReduce(temp_storage).Reduce(thread_data, reduce_op);
  if (threadIdx.x == 0) {
    if (inplace) {
      dots[blockIdx.x] = final_op(reduce_op(dots[blockIdx.x], acc));
    } else {
      dots[blockIdx.x] = final_op(acc);
    }
  }
}

template <typename InType,
          typename OutType      = InType,
          typename IdxType      = int,
          typename MainLambda   = raft::Nop<InType, IdxType>,
          typename ReduceLambda = raft::Sum<OutType>,
          typename FinalLambda  = raft::Nop<OutType>>
void coalescedReduction(OutType* dots,
                        const InType* data,
                        int D,
                        int N,
                        OutType init,
                        cudaStream_t stream,
                        bool inplace           = false,
                        MainLambda main_op     = raft::Nop<InType, IdxType>(),
                        ReduceLambda reduce_op = raft::Sum<OutType>(),
                        FinalLambda final_op   = raft::Nop<OutType>())
{
  // One block per reduction
  // Efficient only for large leading dimensions
  if (D <= 32) {
    coalescedReductionKernel<InType, OutType, IdxType, 32>
      <<<N, 32, 0, stream>>>(dots, data, D, N, init, main_op, reduce_op, final_op, inplace);
  } else if (D <= 64) {
    coalescedReductionKernel<InType, OutType, IdxType, 64>
      <<<N, 64, 0, stream>>>(dots, data, D, N, init, main_op, reduce_op, final_op, inplace);
  } else if (D <= 128) {
    coalescedReductionKernel<InType, OutType, IdxType, 128>
      <<<N, 128, 0, stream>>>(dots, data, D, N, init, main_op, reduce_op, final_op, inplace);
  } else {
    coalescedReductionKernel<InType, OutType, IdxType, 256>
      <<<N, 256, 0, stream>>>(dots, data, D, N, init, main_op, reduce_op, final_op, inplace);
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace detail
}  // namespace linalg
}  // namespace raft