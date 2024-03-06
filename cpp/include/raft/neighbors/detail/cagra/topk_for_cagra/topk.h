/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cuda_fp16.h>

#include <stdint.h>

namespace raft::neighbors::cagra::detail {

//
size_t _cuann_find_topk_bufferSize(uint32_t topK,
                                   uint32_t sizeBatch,
                                   uint32_t numElements,
                                   cudaDataType_t sampleDtype = CUDA_R_32F);

//
template <class ValT>
void _cuann_find_topk(uint32_t topK,
                      uint32_t sizeBatch,
                      uint32_t numElements,
                      const float* inputKeys,  // [sizeBatch, ldIK,]
                      uint32_t ldIK,           // (*) ldIK >= numElements
                      const ValT* inputVals,   // [sizeBatch, ldIV,]
                      uint32_t ldIV,           // (*) ldIV >= numElements
                      float* outputKeys,       // [sizeBatch, ldOK,]
                      uint32_t ldOK,           // (*) ldOK >= topK
                      ValT* outputVals,        // [sizeBatch, ldOV,]
                      uint32_t ldOV,           // (*) ldOV >= topK
                      void* workspace,
                      bool sort           = false,
                      uint32_t* hint      = NULL,
                      cudaStream_t stream = 0);

#ifdef __CUDA_ARCH__
#define CUDA_DEVICE_HOST_FUNC __device__
#else
#define CUDA_DEVICE_HOST_FUNC
#endif
//
CUDA_DEVICE_HOST_FUNC inline size_t _cuann_aligned(size_t size, size_t unit = 128)
{
  if (size % unit) { size += unit - (size % unit); }
  return size;
}
}  // namespace raft::neighbors::cagra::detail
