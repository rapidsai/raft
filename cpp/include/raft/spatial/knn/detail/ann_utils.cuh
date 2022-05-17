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

#include "../ann_common.h"
#include "knn_brute_force_faiss.cuh"

#include "common_faiss.h"
#include "processing.hpp"

#include "processing.hpp"
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>

#include <raft/distance/distance.hpp>
#include <raft/spatial/knn/faiss_mr.hpp>

#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/Select.cuh>
#include <faiss/gpu/utils/Tensor.cuh>
#include <faiss/utils/Heap.h>

#include <thrust/iterator/transform_iterator.h>

#include <raft/distance/distance_type.hpp>

#include <iostream>
#include <set>

namespace raft {
namespace spatial {
namespace knn {
namespace detail {
namespace utils {

// bool check(cudaError_t e, int iLine, const char *szFile) {
//         if (e != cudaSuccess) {
//             std::cout << "CUDA runtime API error " << cudaGetErrorName(e) << " at line " << iLine
//                     << " in file " << szFile << std::endl;
//             exit(0);
//             return false;
//         }
//         return true;
// }

// const char *cublasGetErrorString(cublasStatus_t status) {
//     switch (status) {
//     case CUBLAS_STATUS_SUCCESS:
//         return "CUBLAS_STATUS_SUCCESS";
//     case CUBLAS_STATUS_NOT_INITIALIZED:
//         return "CUBLAS_STATUS_NOT_INITIALIZED";
//     case CUBLAS_STATUS_ALLOC_FAILED:
//         return "CUBLAS_STATUS_ALLOC_FAILED";
//     case CUBLAS_STATUS_INVALID_VALUE:
//         return "CUBLAS_STATUS_INVALID_VALUE";
//     case CUBLAS_STATUS_ARCH_MISMATCH:
//         return "CUBLAS_STATUS_ARCH_MISMATCH";
//     case CUBLAS_STATUS_MAPPING_ERROR:
//         return "CUBLAS_STATUS_MAPPING_ERROR";
//     case CUBLAS_STATUS_EXECUTION_FAILED:
//         return "CUBLAS_STATUS_EXECUTION_FAILED";
//     case CUBLAS_STATUS_INTERNAL_ERROR:
//         return "CUBLAS_STATUS_INTERNAL_ERROR";
//     }
//     return "unknown error";
// }

// bool check(cublasStatus_t e, int iLine, const char *szFile) {
//     if (e != CUBLAS_STATUS_SUCCESS) {
//         std::cout << "CUDA runtime API error " << cublasGetErrorString(e) << " at line "
//                 << iLine << " in file " << szFile << std::endl;
//         exit(0);
//         return false;
//     }
//     return true;
// }
constexpr int kThreadPerBlock = 128;
constexpr int kNumWarps       = kThreadPerBlock / WarpSize;

namespace numeric {

// a new type should specialize get_lower_bound() & get_upper_bound()
// rather than get_dummy()
template <typename T>
constexpr T get_lower_bound()
{
  if (std::numeric_limits<T>::has_infinity && std::numeric_limits<T>::is_signed) {
    return -std::numeric_limits<T>::infinity();
  } else {
    return std::numeric_limits<T>::lowest();
  }
}

template <typename T>
constexpr T get_upper_bound()
{
  if (std::numeric_limits<T>::has_infinity) {
    return std::numeric_limits<T>::infinity();
  } else {
    return std::numeric_limits<T>::max();
  }
}

template <typename T>
constexpr T get_dummy(bool greater)
{
  // TODO: for unsigned and greater=true, dummy will be 0
  //       find better way to warn about this
  assert(!(std::is_unsigned<T>::value && greater));
  return greater ? get_lower_bound<T>() : get_upper_bound<T>();
}

template <bool greater, typename T>
__device__ inline bool is_better_than(T val, T baseline)
{
  return (val > baseline && greater) || (val < baseline && !greater);
}

}  // namespace numeric

/*******************************************************/
/*                   Debug Function                    */
/*******************************************************/

template <typename T>
void printDevPtr(const T* d_cache, int len, const char* name)
{
  T* res = (T*)malloc(sizeof(T) * len);
  RAFT_CUDA_TRY(cudaMemcpy(res, d_cache, sizeof(T) * len, cudaMemcpyDeviceToHost));
  printf("%s ", name);
  for (int i = 0; i < len; i++) {
    printf("%d(%f) ", i, (float)res[i]);
    if (i % 10 == 9) { printf("\n"); }
  }
  printf("\n");
  free(res);
}

inline size_t calc_aligned_size(const std::vector<size_t>& sizes)
{
  const size_t ALIGN_BYTES = 256;
  const size_t ALIGN_MASK  = ~(ALIGN_BYTES - 1);
  size_t total             = 0;
  for (auto sz : sizes) {
    total += (sz + ALIGN_BYTES - 1) & ALIGN_MASK;
  }
  return total + ALIGN_BYTES - 1;
}

inline std::vector<void*> calc_aligned_pointers(const void* p, const std::vector<size_t>& sizes)
{
  const size_t ALIGN_BYTES = 256;
  const size_t ALIGN_MASK  = ~(ALIGN_BYTES - 1);

  char* ptr = reinterpret_cast<char*>((reinterpret_cast<size_t>(p) + ALIGN_BYTES - 1) & ALIGN_MASK);

  std::vector<void*> aligned_pointers;
  aligned_pointers.reserve(sizes.size());
  for (auto sz : sizes) {
    aligned_pointers.push_back(ptr);
    ptr += (sz + ALIGN_BYTES - 1) & ALIGN_MASK;
  }

  return aligned_pointers;
}

//
size_t _cuann_aligned(size_t size, size_t unit = 128)
{
  if (size % unit) { size += unit - (size % unit); }
  return size;
}

// memset
void _cuann_memset(void* ptr, int value, size_t count)
{
  cudaPointerAttributes attr;
  cudaPointerGetAttributes(&attr, ptr);
  if (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged) {
    RAFT_CUDA_TRY(cudaMemset(ptr, value, count));
    // if (ret != cudaSuccess) {
    //     fprintf(stderr, "(%s) cudaMemset() failed\n", __func__);
    //     exit(-1);
    // }
  } else {
    memset(ptr, value, count);
  }
}

// argmin along column
__global__ void kern_argmin(uint32_t nRows,
                            uint32_t nCols,
                            const float* a,  // [nRows, nCols]
                            uint32_t* out    // [nRows]
)
{
  __shared__ uint32_t smCol[1024];
  __shared__ float smVal[1024];
  uint32_t iRow = blockIdx.x;
  if (iRow >= nRows) return;
  uint32_t iCol   = threadIdx.x;
  uint32_t minCol = nCols;
  float minVal    = FLT_MAX;
  for (iCol = threadIdx.x; iCol < nCols; iCol += blockDim.x) {
    if (minVal > a[iCol + (nCols * iRow)]) {
      minVal = a[iCol + (nCols * iRow)];
      minCol = iCol;
    }
  }
  smVal[threadIdx.x] = minVal;
  smCol[threadIdx.x] = minCol;
  __syncthreads();
  for (uint32_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      if (smVal[threadIdx.x] < smVal[threadIdx.x + offset]) {
      } else if (smVal[threadIdx.x] > smVal[threadIdx.x + offset]) {
        smVal[threadIdx.x] = smVal[threadIdx.x + offset];
        smCol[threadIdx.x] = smCol[threadIdx.x + offset];
      } else if (smCol[threadIdx.x] > smCol[threadIdx.x + offset]) {
        smCol[threadIdx.x] = smCol[threadIdx.x + offset];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) { out[iRow] = smCol[0]; }
}

// argmin along column
void _cuann_argmin(uint32_t nRows,
                   uint32_t nCols,
                   const float* a,  // [nRows, nCols]
                   uint32_t* out    // [nRows]
)
{
  uint32_t nThreads = 1024;
  while (nThreads > nCols) {
    nThreads /= 2;
  }
  nThreads = max(nThreads, 128);
  kern_argmin<<<nRows, nThreads>>>(nRows, nCols, a, out);
}

// square sum along column
__global__ void kern_sqsum(uint32_t nRows,
                           uint32_t nCols,
                           const float* a,  // [nRows, nCols]
                           float* out       // [nRows]
)
{
  uint64_t iRow = threadIdx.y + (blockDim.y * blockIdx.x);
  if (iRow >= nRows) return;

  float sqsum = 0.0;
  for (uint64_t iCol = threadIdx.x; iCol < nCols; iCol += blockDim.x) {
    float val = a[iCol + (nCols * iRow)];
    sqsum += val * val;
  }
  sqsum += __shfl_xor_sync(0xffffffff, sqsum, 1);
  sqsum += __shfl_xor_sync(0xffffffff, sqsum, 2);
  sqsum += __shfl_xor_sync(0xffffffff, sqsum, 4);
  sqsum += __shfl_xor_sync(0xffffffff, sqsum, 8);
  sqsum += __shfl_xor_sync(0xffffffff, sqsum, 16);
  if (threadIdx.x == 0) { out[iRow] = sqsum; }
}

// square sum along column
void _cuann_sqsum(uint32_t nRows,
                  uint32_t nCols,
                  const float* a,  // [numDataset, dimDataset]
                  float* out       // [numDataset,]
)
{
  dim3 threads(32, 4, 1);  // DO NOT CHANGE
  dim3 blocks((nRows + threads.y - 1) / threads.y, 1, 1);
  kern_sqsum<<<blocks, threads>>>(nRows, nCols, a, out);
}

// copy

template <typename S, typename D>
__global__ void kern_copy(uint32_t nRows,
                          uint32_t nCols,
                          const S* src,  // [nRows, ldSrc]
                          uint32_t ldSrc,
                          D* dst,  // [nRows, ldDst]
                          uint32_t ldDst)
{
  uint32_t gid  = threadIdx.x + (blockDim.x * blockIdx.x);
  uint32_t iCol = gid % nCols;
  uint32_t iRow = gid / nCols;
  if (iRow >= nRows) return;
  dst[iCol + (ldDst * iRow)] = src[iCol + (ldSrc * iRow)];
}

// copy
template <typename S, typename D>
__global__ void kern_copy(uint32_t nRows,
                          uint32_t nCols,
                          const S* src,  // [nRows, ldSrc]
                          uint32_t ldSrc,
                          D* dst,  // [nRows, ldDst]
                          uint32_t ldDst,
                          D divisor)
{
  uint32_t gid  = threadIdx.x + (blockDim.x * blockIdx.x);
  uint32_t iCol = gid % nCols;
  uint32_t iRow = gid / nCols;
  if (iRow >= nRows) return;
  dst[iCol + (ldDst * iRow)] = src[iCol + (ldSrc * iRow)] / divisor;
}

template <typename S, typename D>
void _cuann_copy(uint32_t nRows,
                 uint32_t nCols,
                 const S* src,  // [nRows, ldSrc]
                 uint32_t ldSrc,
                 D* dst,  // [nRows, ldDst]
                 uint32_t ldDst,
                 D divisor)
{
  uint32_t nThreads = 128;
  uint32_t nBlocks  = ((nRows * nCols) + nThreads - 1) / nThreads;
  kern_copy<S, D><<<nBlocks, nThreads>>>(nRows, nCols, src, ldSrc, dst, ldDst, divisor);
}

template <typename S, typename D>
void _cuann_copy(uint32_t nRows,
                 uint32_t nCols,
                 const S* src,  // [nRows, ldSrc]
                 uint32_t ldSrc,
                 D* dst,  // [nRows, ldDst]
                 uint32_t ldDst,
                 cudaStream_t stream,
                 D divisor)
{
  uint32_t nThreads = 128;
  uint32_t nBlocks  = ((nRows * nCols) + nThreads - 1) / nThreads;
  kern_copy<S, D><<<nBlocks, nThreads, 0, stream>>>(nRows, nCols, src, ldSrc, dst, ldDst, divisor);
}

// accumulate
template <typename T>
__global__ void kern_accumulate_with_label(uint32_t nRowsOutput,
                                           uint32_t nCols,
                                           float* output,    // [nRowsOutput, nCols,]
                                           uint32_t* count,  // [nRowsOutput,]
                                           uint32_t nRowsInput,
                                           const T* input,         // [nRowsInput, nCols,]
                                           const uint32_t* label,  // [nRowsInput,]
                                           float divisor)
{
  uint64_t gid       = threadIdx.x + (blockDim.x * blockIdx.x);
  uint64_t iCol      = gid % nCols;
  uint64_t iRowInput = gid / nCols;
  if (iRowInput >= nRowsInput) return;
  uint64_t iRowOutput = label[iRowInput];
  if (iCol == 0) { atomicAdd(&(count[iRowOutput]), 1); }
  atomicAdd(&(output[iCol + (nCols * iRowOutput)]), input[gid] / divisor);
}

// accumulate
template <typename T>
void _cuann_accumulate_with_label(uint32_t nRowsOutput,
                                  uint32_t nCols,
                                  float* output,    // [nRowsOutput, nCols,]
                                  uint32_t* count,  // [nRowsOutput,]
                                  uint32_t nRowsInput,
                                  const T* input,         // [nRowsInput, nCols,]
                                  const uint32_t* label,  // [nRowsInput,]
                                  float divisor = 1.0)
{
  bool useGPU = 1;
  cudaPointerAttributes attr;
  cudaPointerGetAttributes(&attr, output);
  if (attr.type == cudaMemoryTypeUnregistered || attr.type == cudaMemoryTypeHost) { useGPU = 0; }
  cudaPointerGetAttributes(&attr, count);
  if (attr.type == cudaMemoryTypeUnregistered || attr.type == cudaMemoryTypeHost) { useGPU = 0; }
  cudaPointerGetAttributes(&attr, input);
  if (attr.type == cudaMemoryTypeUnregistered || attr.type == cudaMemoryTypeHost) { useGPU = 0; }
  // _cuann_memset(output, 0, sizeof(float) * nRowsOutput * nCols);
  // _cuann_memset(count, 0, sizeof(uint32_t) * nRowsOutput);

  if (useGPU) {
    // GPU
    uint32_t nThreads = 128;
    uint64_t nBlocks  = (((uint64_t)nRowsInput * nCols) + nThreads - 1) / nThreads;
    kern_accumulate_with_label<T>
      <<<nBlocks, nThreads>>>(nRowsOutput, nCols, output, count, nRowsInput, input, label, divisor);
  } else {
    // CPU
    cudaDeviceSynchronize();
    for (uint64_t i = 0; i < nRowsInput; i++) {
      uint64_t l = label[i];
      count[l] += 1;
      for (uint64_t j = 0; j < nCols; j++) {
        output[j + (nCols * l)] += input[j + (nCols * i)] / divisor;
      }
    }
  }
}

// normalize
__global__ void kern_normalize(uint32_t nRows,
                               uint32_t nCols,
                               float* a,                   // [nRows, nCols]
                               const uint32_t* numSamples  // [nRows,]
)
{
  uint64_t iRow = threadIdx.y + (blockDim.y * blockIdx.x);
  if (iRow >= nRows) return;
  if (numSamples != NULL and numSamples[iRow] < 1) return;

  float sqsum = 0.0;
  for (uint32_t iCol = threadIdx.x; iCol < nCols; iCol += blockDim.x) {
    float val = a[iCol + (nCols * iRow)];
    sqsum += val * val;
  }
  sqsum += __shfl_xor_sync(0xffffffff, sqsum, 1);
  sqsum += __shfl_xor_sync(0xffffffff, sqsum, 2);
  sqsum += __shfl_xor_sync(0xffffffff, sqsum, 4);
  sqsum += __shfl_xor_sync(0xffffffff, sqsum, 8);
  sqsum += __shfl_xor_sync(0xffffffff, sqsum, 16);
  sqsum = sqrt(sqsum);
  for (uint32_t iCol = threadIdx.x; iCol < nCols; iCol += blockDim.x) {
    a[iCol + (nCols * iRow)] /= sqsum;
  }
}

// normalize
void _cuann_normalize(uint32_t nRows,
                      uint32_t nCols,
                      float* a,                   // [nRows, nCols]
                      const uint32_t* numSamples  // [nRows,]
)
{
  dim3 threads(32, 4, 1);  // DO NOT CHANGE
  dim3 blocks((nRows + threads.y - 1) / threads.y, 1, 1);
  kern_normalize<<<blocks, threads>>>(nRows, nCols, a, numSamples);
}

// divide
__global__ void kern_divide(uint32_t nRows,
                            uint32_t nCols,
                            float* a,                   // [nRows, nCols]
                            const uint32_t* numSamples  // [nRows,]
)
{
  uint64_t gid  = threadIdx.x + (blockDim.x * blockIdx.x);
  uint64_t iRow = gid / nCols;
  if (iRow >= nRows) return;
  if (numSamples[iRow] == 0) return;
  a[gid] /= numSamples[iRow];
}

// divide
void _cuann_divide(uint32_t nRows,
                   uint32_t nCols,
                   float* a,                   // [nRows, nCols]
                   const uint32_t* numSamples  // [nRows,]
)
{
  dim3 threads(128, 1, 1);
  dim3 blocks(((uint64_t)nRows * nCols + threads.x - 1) / threads.x, 1, 1);
  kern_divide<<<blocks, threads>>>(nRows, nCols, a, numSamples);
}

// outer add
__global__ void kern_outer_add(const float* a,
                               uint32_t numA,
                               const float* b,
                               uint32_t numB,
                               float* c  // [numA, numB]
)
{
  uint64_t gid = threadIdx.x + (blockDim.x * blockIdx.x);
  uint64_t iA  = gid / numB;
  uint64_t iB  = gid % numB;
  if (iA >= numA) return;
  float valA = (a == NULL) ? 0.0 : a[iA];
  float valB = (b == NULL) ? 0.0 : b[iB];
  c[gid]     = valA + valB;
}

// outer add
void _cuann_outer_add(const float* a,
                      uint32_t numA,
                      const float* b,
                      uint32_t numB,
                      float* c  // [numA, numB]
)
{
  dim3 threads(128, 1, 1);
  dim3 blocks(((uint64_t)numA * numB + threads.x - 1) / threads.x, 1, 1);
  kern_outer_add<<<blocks, threads>>>(a, numA, b, numB, c);
}

// copy with row list
template <typename T>
__global__ void kern_copy_with_list(uint32_t nRows,
                                    uint32_t nCols,
                                    const T* src,             // [..., ldSrc]
                                    const uint32_t* rowList,  // [nRows,]
                                    uint32_t ldSrc,
                                    float* dst,  // [nRows, ldDst]
                                    uint32_t ldDst)
{
  uint64_t gid  = threadIdx.x + (blockDim.x * blockIdx.x);
  uint64_t iCol = gid % nCols;
  uint64_t iRow = gid / nCols;
  if (iRow >= nRows) return;
  uint64_t iaRow             = rowList[iRow];
  dst[iCol + (ldDst * iRow)] = src[iCol + (ldSrc * iaRow)];
}

// copy with row list
template <typename T>
__global__ void kern_copy_with_list(uint32_t nRows,
                                    uint32_t nCols,
                                    const T* src,             // [..., ldSrc]
                                    const uint32_t* rowList,  // [nRows,]
                                    uint32_t ldSrc,
                                    float* dst,  // [nRows, ldDst]
                                    uint32_t ldDst,
                                    float divisor)
{
  uint64_t gid  = threadIdx.x + (blockDim.x * blockIdx.x);
  uint64_t iCol = gid % nCols;
  uint64_t iRow = gid / nCols;
  if (iRow >= nRows) return;
  uint64_t iaRow             = rowList[iRow];
  dst[iCol + (ldDst * iRow)] = src[iCol + (ldSrc * iaRow)] / divisor;
}

// copy with row list
template <typename T>
void _cuann_copy_with_list(uint32_t nRows,
                           uint32_t nCols,
                           const T* src,             // [..., ldSrc]
                           const uint32_t* rowList,  // [nRows,]
                           uint32_t ldSrc,
                           float* dst,  // [nRows, ldDst]
                           uint32_t ldDst,
                           float divisor = 1.0)
{
  cudaPointerAttributes attr;
  cudaPointerGetAttributes(&attr, src);
  if (attr.type == cudaMemoryTypeUnregistered || attr.type == cudaMemoryTypeHost) {
    for (uint64_t iRow = 0; iRow < nRows; iRow++) {
      uint64_t iaRow = rowList[iRow];
      for (uint64_t iCol = 0; iCol < nCols; iCol++) {
        dst[iCol + (ldDst * iRow)] = src[iCol + (ldSrc * iaRow)] / divisor;
      }
    }
  } else {
    uint32_t nThreads = 128;
    uint32_t nBlocks  = ((nRows * nCols) + nThreads - 1) / nThreads;
    kern_copy_with_list<T>
      <<<nBlocks, nThreads>>>(nRows, nCols, src, rowList, ldSrc, dst, ldDst, divisor);
  }
}
}  // namespace utils
}  // namespace detail
}  // namespace knn
}  // namespace spatial
}  // namespace raft
