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

#include <raft/cuda_utils.cuh>
#include <raft/linalg/unary_op.cuh>
#include "../test_utils.h"

namespace raft {
namespace linalg {

template <typename InType, typename OutType, typename IdxType>
__global__ void naive_scale_kernel(OutType *out, const InType *in,
                                   InType scalar, IdxType len) {
  IdxType idx = threadIdx.x + (static_cast<IdxType>(blockIdx.x) *
                               static_cast<IdxType>(blockDim.x));
  if (idx < len) {
    if (in == nullptr) {
      // used for testing writeOnlyUnaryOp
      out[idx] = static_cast<OutType>(scalar * idx);
    } else {
      out[idx] = static_cast<OutType>(scalar * in[idx]);
    }
  }
}

template <typename InType, typename IdxType = int, typename OutType = InType>
void naive_scale(OutType *out, const InType *in, InType scalar, int len,
                 cudaStream_t stream) {
  static const int kTpb = 64;
  int nblks = raft::ceildiv(len, kTpb);
  naive_scale_kernel<InType, OutType, IdxType>
    <<<nblks, kTpb, 0, stream>>>(out, in, scalar, len);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename InType, typename IdxType = int, typename OutType = InType>
struct unary_op_inputs {
  OutType tolerance;
  IdxType len;
  InType scalar;
  uint64_t seed;
};

template <typename InType, typename IdxType = int, typename OutType = InType>
::std::ostream &operator<<(::std::ostream &os,
                           const unary_op_inputs<InType, IdxType, OutType> &d) {
  return os;
}

}  // end namespace linalg
}  // end namespace raft
