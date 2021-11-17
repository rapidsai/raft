/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
 #include <raft/linalg/binary_op.hpp>
 #include <raft/linalg/unary_op.hpp>
 
 namespace raft {
 namespace linalg {
 namespace detail {
 
 template <class math_t, typename IdxType>
 __global__ void subtract_dev_scalar_kernel(math_t *outDev, const math_t *inDev,
                                            const math_t *singleScalarDev,
                                            IdxType len) {
   //TODO: kernel do not use shared memory in current implementation
   int i = ((IdxType)blockIdx.x * (IdxType)blockDim.x) + threadIdx.x;
   if (i < len) {
     outDev[i] = inDev[i] - *singleScalarDev;
   }
 }
 
 template <typename math_t, typename IdxType = int, int TPB = 256>
 void subtractDevScalar(math_t *outDev, const math_t *inDev,
                        const math_t *singleScalarDev, IdxType len,
                        cudaStream_t stream) {
   // Just for the note - there is no way to express such operation with cuBLAS in effective way
   // https://stackoverflow.com/questions/14051064/add-scalar-to-vector-in-blas-cublas-cuda
   const IdxType nblks = raft::ceildiv(len, (IdxType)TPB);
   subtract_dev_scalar_kernel<math_t>
     <<<nblks, TPB, 0, stream>>>(outDev, inDev, singleScalarDev, len);
   CUDA_CHECK(cudaPeekAtLastError());
 }
 
 };  // end namespace detail
 };  // end namespace linalg
 };  // end namespace raft
 