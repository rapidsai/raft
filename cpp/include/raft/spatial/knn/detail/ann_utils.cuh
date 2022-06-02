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

#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/distance/distance.hpp>
#include <raft/distance/distance_type.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace raft::spatial ::knn::detail::utils {

constexpr int kThreadPerBlock = 128;
constexpr int kNumWarps       = kThreadPerBlock / WarpSize;

size_t _cuann_aligned(size_t size, size_t unit = 128)
{
  if (size % unit) { size += unit - (size % unit); }
  return size;
}

/**
 * @brief Sets the first num bytes of the block of memory pointed by ptr to the specified value.
 *
 * @param[out] ptr host or device pointer
 * @param[in] value
 * @param[in] count
 */
void _cuann_memset(void* ptr, int value, size_t count, rmm::cuda_stream_view stream)
{
  cudaPointerAttributes attr;
  cudaPointerGetAttributes(&attr, ptr);
  if (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged) {
    RAFT_CUDA_TRY(cudaMemsetAsync(ptr, value, count, stream));
  } else {
    stream.synchronize();
    memset(ptr, value, count);
    stream.synchronize();
  }
}

template <typename T>
struct config {
};

template <>
struct config<float> {
  using value_t                   = float;
  static constexpr float kDivisor = 1.0;
};
template <>
struct config<uint8_t> {
  using value_t                   = uint32_t;
  static constexpr float kDivisor = 256.0;
};
template <>
struct config<int8_t> {
  using value_t                   = int32_t;
  static constexpr float kDivisor = 128.0;
};

template <typename T, typename S>
struct mapping {
  HDI auto operator()(T x) -> S;
};

template <typename T>
struct mapping<T, T> {
  HDI auto operator()(T x) -> T { return x; }
};

template <typename T>
struct mapping<T, float> {
  HDI auto operator()(T x) -> float { return float(x) * kMult; }

 private:
  static constexpr float kMult = 1 / config<T>::kDivisor;
};

template <typename S>
struct mapping<float, S> {
  HDI auto operator()(float x) -> S { return S(x * kMult); }

 private:
  static constexpr float kMult = config<S>::kDivisor;
};

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

/**
 * argmin along column
 *
 * NB: device-only
 */
void _cuann_argmin(uint32_t nRows,
                   uint32_t nCols,
                   const float* a,  // [nRows, nCols]
                   uint32_t* out,   // [nRows]
                   rmm::cuda_stream_view stream)
{
  uint32_t nThreads = 1024;
  while (nThreads > nCols) {
    nThreads /= 2;
  }
  nThreads = max(nThreads, 128);
  kern_argmin<<<nRows, nThreads, 0, stream>>>(nRows, nCols, a, out);
}

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

/**
 * @brief Square sum along rows (row-major).
 *
 * NB: device-only
 *
 * @param nRows
 * @param nCols
 * @param[in] a device pointer to the row-major matrix [nRows, nCols]
 * @param[out] out device pointer to the vector of dot-products [nRows]
 * @param stream
 */
void _cuann_sqsum(uint32_t nRows,
                  uint32_t nCols,
                  const float* a,  // [numDataset, dimDataset]
                  float* out,      // [numDataset,]
                  rmm::cuda_stream_view stream)
{
  dim3 threads(32, 4, 1);
  dim3 blocks(ceildiv(nRows, threads.y), 1, 1);
  kern_sqsum<<<blocks, threads, 0, stream>>>(nRows, nCols, a, out);
  /**
   * TODO: this can be replaced with the rowNorm helper as shown below.
   * However, the rowNorm helper seems to incur a significant performance penalty
   * (example case ann-search slowed down from 150ms to 186ms).
   *
   * raft::linalg::rowNorm(out, a, nCols, nRows, raft::linalg::L2Norm, true, stream);
   */
}

template <typename T>
__global__ void kern_accumulate_with_label(uint32_t nRowsOutput,
                                           uint32_t nCols,
                                           float* output,    // [nRowsOutput, nCols,]
                                           uint32_t* count,  // [nRowsOutput,]
                                           uint32_t nRowsInput,
                                           const T* input,        // [nRowsInput, nCols,]
                                           const uint32_t* label  // [nRowsInput,]
)
{
  uint64_t gid       = threadIdx.x + (blockDim.x * blockIdx.x);
  uint64_t iCol      = gid % nCols;
  uint64_t iRowInput = gid / nCols;
  if (iRowInput >= nRowsInput) return;
  uint64_t iRowOutput = label[iRowInput];
  if (iCol == 0) { atomicAdd(&(count[iRowOutput]), 1); }
  atomicAdd(&(output[iCol + (nCols * iRowOutput)]), float(input[gid]) / config<T>::kDivisor);
}

/**
 * @brief Accumulate
 *
 * Pointer residency: altogether available either on GPU or on CPU
 *
 * @tparam T
 *
 * @param nRowsOutput
 * @param nCols
 * @param output device/host pointer
 * @param count device/host pointer
 * @param nRowsInput
 * @param input device/host pointer
 * @param label device/host pointer
 */
template <typename T>
void _cuann_accumulate_with_label(uint32_t nRowsOutput,
                                  uint32_t nCols,
                                  float* output,    // [nRowsOutput, nCols,]
                                  uint32_t* count,  // [nRowsOutput,]
                                  uint32_t nRowsInput,
                                  const T* input,         // [nRowsInput, nCols,]
                                  const uint32_t* label,  // [nRowsInput,]
                                  rmm::cuda_stream_view stream)
{
  bool useGPU = 1;
  cudaPointerAttributes attr;
  cudaPointerGetAttributes(&attr, output);
  if (attr.type == cudaMemoryTypeUnregistered || attr.type == cudaMemoryTypeHost) { useGPU = 0; }
  cudaPointerGetAttributes(&attr, count);
  if (attr.type == cudaMemoryTypeUnregistered || attr.type == cudaMemoryTypeHost) { useGPU = 0; }
  cudaPointerGetAttributes(&attr, input);
  if (attr.type == cudaMemoryTypeUnregistered || attr.type == cudaMemoryTypeHost) { useGPU = 0; }
  cudaPointerGetAttributes(&attr, label);
  if (attr.type == cudaMemoryTypeUnregistered || attr.type == cudaMemoryTypeHost) { useGPU = 0; }

  if (useGPU) {
    // GPU
    uint32_t nThreads = 128;
    uint64_t nBlocks  = ceildiv((uint64_t)nRowsInput * nCols, (uint64_t)nThreads);
    kern_accumulate_with_label<T><<<nBlocks, nThreads, 0, stream>>>(
      nRowsOutput, nCols, output, count, nRowsInput, input, label);
  } else {
    // CPU
    stream.synchronize();
    for (uint64_t i = 0; i < nRowsInput; i++) {
      uint64_t l = label[i];
      count[l] += 1;
      for (uint64_t j = 0; j < nCols; j++) {
        output[j + (nCols * l)] += float(input[j + (nCols * i)]) / config<T>::kDivisor;
      }
    }
    stream.synchronize();
  }
}

__global__ void kern_normalize(uint32_t nRows,
                               uint32_t nCols,
                               float* a,                   // [nRows, nCols]
                               const uint32_t* numSamples  // [nRows,]
)
{
  uint64_t iRow = threadIdx.y + (blockDim.y * blockIdx.x);
  if (iRow >= nRows) return;
  if (numSamples != nullptr and numSamples[iRow] < 1) return;

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

/**
 * @brief Normalize
 *
 * NB: device-only
 *
 * @param[in] nRows
 * @param[in] nCols
 * @param[inout] a device pointer
 * @param[in] numSamples device pointer
 */
void _cuann_normalize(uint32_t nRows,
                      uint32_t nCols,
                      float* a,                    // [nRows, nCols]
                      const uint32_t* numSamples,  // [nRows,]
                      rmm::cuda_stream_view stream)
{
  dim3 threads(32, 4, 1);  // DO NOT CHANGE
  dim3 blocks(ceildiv(nRows, threads.y), 1, 1);
  kern_normalize<<<blocks, threads, 0, stream>>>(nRows, nCols, a, numSamples);
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

/**
 * @brief Divide
 *
 * NB: device-only
 *
 * @param[in] nRows
 * @param[in] nCols
 * @param[inout] a device pointer
 * @param[in] numSamples device pointer
 */
void _cuann_divide(uint32_t nRows,
                   uint32_t nCols,
                   float* a,                    // [nRows, nCols]
                   const uint32_t* numSamples,  // [nRows,]
                   rmm::cuda_stream_view stream)
{
  dim3 threads(128, 1, 1);
  dim3 blocks(ceildiv<uint64_t>((uint64_t)nRows * (uint64_t)nCols, threads.x), 1, 1);
  kern_divide<<<blocks, threads, 0, stream>>>(nRows, nCols, a, numSamples);
}
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
  float valA = (a == nullptr) ? 0.0 : a[iA];
  float valB = (b == nullptr) ? 0.0 : b[iB];
  c[gid]     = valA + valB;
}

/**
 * outer add
 *
 * NB: device-only
 */
void _cuann_outer_add(const float* a,
                      uint32_t numA,
                      const float* b,
                      uint32_t numB,
                      float* c,  // [numA, numB]
                      rmm::cuda_stream_view stream)
{
  dim3 threads(128, 1, 1);
  dim3 blocks(ceildiv<uint64_t>((uint64_t)numA * (uint64_t)numB, threads.x), 1, 1);
  kern_outer_add<<<blocks, threads, 0, stream>>>(a, numA, b, numB, c);
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
  dst[iCol + (ldDst * iRow)] = float(src[iCol + (ldSrc * iaRow)]) / config<T>::kDivisor;
}

/**
 * copy with row list
 *
 * NB: host or device
 */
template <typename T>
void _cuann_copy_with_list(uint32_t nRows,
                           uint32_t nCols,
                           const T* src,             // [..., ldSrc]
                           const uint32_t* rowList,  // [nRows,]
                           uint32_t ldSrc,
                           float* dst,  // [nRows, ldDst]
                           uint32_t ldDst,
                           rmm::cuda_stream_view stream)
{
  cudaPointerAttributes attr;
  cudaPointerGetAttributes(&attr, src);
  if (attr.type == cudaMemoryTypeUnregistered || attr.type == cudaMemoryTypeHost) {
    stream.synchronize();
    for (uint64_t iRow = 0; iRow < nRows; iRow++) {
      uint64_t iaRow = rowList[iRow];
      for (uint64_t iCol = 0; iCol < nCols; iCol++) {
        dst[iCol + (ldDst * iRow)] = float(src[iCol + (ldSrc * iaRow)]) / config<T>::kDivisor;
      }
    }
    stream.synchronize();
  } else {
    uint32_t nThreads = 128;
    uint32_t nBlocks  = ceildiv(nRows * nCols, nThreads);
    kern_copy_with_list<T>
      <<<nBlocks, nThreads, 0, stream>>>(nRows, nCols, src, rowList, ldSrc, dst, ldDst);
  }
}
}  // namespace raft::spatial::knn::detail::utils
