/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
