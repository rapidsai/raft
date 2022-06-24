/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <raft/device_atomics.cuh>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

///////////////////
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <omp.h>

//////////////////

#define CUANN_DEBUG

namespace raft::spatial::knn::ivf_pq {

/**
 *
 *
 *
 *
 *
 * fp_8bit
 */

template <unsigned expBitLen>
struct fp_8bit;

template <unsigned expBitLen>
__device__ __host__ fp_8bit<expBitLen> __float2fp_8bit(const float v);
template <unsigned expBitLen>
__device__ __host__ float __fp_8bit2float(const fp_8bit<expBitLen>& v);

template <unsigned expBitLen>
struct fp_8bit {
  uint8_t bitstring;

  __device__ __host__ fp_8bit(const uint8_t bs) { bitstring = bs; }
  __device__ __host__ fp_8bit(const float fp)
  {
    bitstring = __float2fp_8bit<expBitLen>(fp).bitstring;
  }
  __device__ __host__ fp_8bit<expBitLen>& operator=(const float fp)
  {
    bitstring = __float2fp_8bit<expBitLen>(fp).bitstring;
    return *this;
  }

  __device__ __host__ operator float() const { return __fp_8bit2float(*this); }
};

// Since __float_as_uint etc can not be used in host codes,
// these converters are needed for test.
union cvt_fp_32bit {
  float fp;
  uint32_t bs;
};
union cvt_fp_16bit {
  half fp;
  uint16_t bs;
};

// Type converters
template <unsigned expBitLen>
__device__ __host__ fp_8bit<expBitLen> __float2fp_8bit(const float v)
{
  if (v < 1. / (1u << ((1u << (expBitLen - 1)) - 1)))
    return fp_8bit<expBitLen>{static_cast<uint8_t>(0)};
  return fp_8bit<expBitLen>{static_cast<uint8_t>(
    (cvt_fp_32bit{.fp = v}.bs + (((1u << (expBitLen - 1)) - 1) << 23) - 0x3f800000u) >>
    (15 + expBitLen))};
}

template <unsigned expBitLen>
__device__ __host__ float __fp_8bit2float(const fp_8bit<expBitLen>& v)
{
  return cvt_fp_32bit{.bs = ((v.bitstring << (15 + expBitLen)) +
                             (0x3f800000u | (0x00400000u >> (8 - expBitLen))) -
                             (((1u << (expBitLen - 1)) - 1) << 23))}
    .fp;
}

/**
 *
 * end of fp8bit
 *
 */

using namespace cub;

//
extern __shared__ float smemArray[];

#define FP16_MAX 65504.0

/* CUANN status type */
typedef enum {
  CUANN_STATUS_SUCCESS           = 0,
  CUANN_STATUS_ALLOC_FAILED      = 1,
  CUANN_STATUS_NOT_INITIALIZED   = 2,
  CUANN_STATUS_INVALID_VALUE     = 3,
  CUANN_STATUS_INTERNAL_ERROR    = 4,
  CUANN_STATUS_FILEIO_ERROR      = 5,
  CUANN_STATUS_CUDA_ERROR        = 6,
  CUANN_STATUS_CUBLAS_ERROR      = 7,
  CUANN_STATUS_INVALID_POINTER   = 8,
  CUANN_STATUS_VERSION_ERROR     = 9,
  CUANN_STATUS_UNSUPPORTED_DTYPE = 10,
} cuannStatus_t;

/* CUANN similarity type */
typedef enum {
  CUANN_SIMILARITY_INNER = 0,
  CUANN_SIMILARITY_L2    = 1,
} cuannSimilarity_t;

/* CUANN PQ center type */
typedef enum {
  CUANN_PQ_CENTER_PER_SUBSPACE = 0,
  CUANN_PQ_CENTER_PER_CLUSTER  = 1,
} cuannPqCenter_t;

/* Context */
struct cuannContext {
  int devId;
  cudaStream_t stream;
  cudaDeviceProp deviceProp;
  cublasHandle_t cublasHandle;

  int numDevices;
  cudaStream_t* streams;
  cudaDeviceProp* deviceProps;
  cublasHandle_t* cublasHandles;
};
typedef struct cuannContext* cuannHandle_t;

/* IvfPq */
struct cuannIvfPqDescriptor {
  uint32_t numClusters;
  uint32_t numDataset;
  uint32_t dimDataset;
  uint32_t dimDatasetExt;
  uint32_t dimRotDataset;
  uint32_t dimPq;
  uint32_t bitPq;
  cuannSimilarity_t similarity;
  cuannPqCenter_t typePqCenter;
  cudaDataType_t dtypeDataset;
  cudaDataType_t internalDistanceDtype;
  cudaDataType_t smemLutDtype;
  uint32_t indexVersion;
  uint32_t maxClusterSize;
  uint32_t lenPq;  // dimRotDataset / dimPq
  uint32_t numProbes;
  uint32_t topK;
  uint32_t maxQueries;
  uint32_t maxBatchSize;
  uint32_t maxSamples;
  uint32_t* inclusiveSumSortedClusterSize;  // [numClusters,]
  float* sqsumClusters;                     // [numClusters,]
  size_t sizeCubWorkspace;
  uint32_t _numClustersSize0;  // (*) urgent WA, need to be fixed
  uint32_t preferredThreadBlockSize;
};
typedef struct cuannIvfPqDescriptor* cuannIvfPqDescriptor_t;

// header of index
struct cuannIvfPqIndexHeader {
  // (*) DO NOT CHANGE ORDER
  size_t indexSize;
  uint32_t version;
  uint32_t numClusters;
  uint32_t numDataset;
  uint32_t dimDataset;
  uint32_t dimPq;
  uint32_t similarity;
  uint32_t maxClusterSize;
  uint32_t dimRotDataset;
  uint32_t bitPq;
  uint32_t typePqCenter;
  uint32_t dtypeDataset;
  uint32_t dimDatasetExt;
  uint32_t numDatasetAdded;
  uint32_t _dummy[256 - 15];
};

//
char* _cuann_get_dtype_string(cudaDataType_t dtype, char* string)
{
  if (dtype == CUDA_R_32F)
    sprintf(string, "float (CUDA_R_32F)");
  else if (dtype == CUDA_R_16F)
    sprintf(string, "half (CUDA_R_16F)");
  else if (dtype == CUDA_R_8U)
    sprintf(string, "uint8 (CUDA_R_8U)");
  else if (dtype == CUDA_R_8I)
    sprintf(string, "int8 (CUDA_R_8I)");
  else
    sprintf(string, "unknown");
  return string;
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
    cudaError_t ret = cudaMemset(ptr, value, count);
    if (ret != cudaSuccess) {
      fprintf(stderr, "(%s) cudaMemset() failed\n", __func__);
      exit(-1);
    }
  } else {
    memset(ptr, value, count);
  }
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

// copy
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

template void _cuann_copy<float, float>(uint32_t nRows,
                                        uint32_t nCols,
                                        const float* src,
                                        uint32_t ldSrc,
                                        float* dst,
                                        uint32_t ldDst,
                                        float divisor);
template void _cuann_copy<uint32_t, uint8_t>(uint32_t nRows,
                                             uint32_t nCols,
                                             const uint32_t* src,
                                             uint32_t ldSrc,
                                             uint8_t* dst,
                                             uint32_t ldDst,
                                             uint8_t divisor);
template void _cuann_copy<uint8_t, float>(uint32_t nRows,
                                          uint32_t nCols,
                                          const uint8_t* src,
                                          uint32_t ldSrc,
                                          float* dst,
                                          uint32_t ldDst,
                                          float divisor);
template void _cuann_copy<int8_t, float>(uint32_t nRows,
                                         uint32_t nCols,
                                         const int8_t* src,
                                         uint32_t ldSrc,
                                         float* dst,
                                         uint32_t ldDst,
                                         float divisor);

// copy_CPU
template <typename S, typename D>
void _cuann_copy_CPU(uint32_t nRows,
                     uint32_t nCols,
                     const S* src,  // [nRows, ldSrc]
                     uint32_t ldSrc,
                     D* dst,  // [nRows, ldDst]
                     uint32_t ldDst)
{
  for (uint32_t ir = 0; ir < nRows; ir++) {
    for (uint32_t ic = 0; ic < nCols; ic++) {
      dst[ic + (ldDst * ir)] = src[ic + (ldSrc * ir)];
    }
  }
}

template void _cuann_copy_CPU<float, float>(
  uint32_t nRows, uint32_t nCols, const float* src, uint32_t ldSrc, float* dst, uint32_t ldDst);

// copy_fill
template <typename S, typename D>
__global__ void kern_copy_fill(uint32_t nRows,
                               uint32_t nCols,
                               const S* src,  // [nRows, ldSrc]
                               uint32_t ldSrc,
                               D* dst,  // [nRows, ldDst]
                               uint32_t ldDst,
                               D fillValue,
                               D divisor)
{
  uint32_t gid  = threadIdx.x + (blockDim.x * blockIdx.x);
  uint32_t iCol = gid % ldDst;
  uint32_t iRow = gid / ldDst;
  if (iRow >= nRows) return;
  if (iCol < nCols) {
    dst[iCol + (ldDst * iRow)] = src[iCol + (ldSrc * iRow)] / divisor;
  } else {
    dst[iCol + (ldDst * iRow)] = fillValue;
  }
}

// copy_fill
template <typename S, typename D>
void _cuann_copy_fill(uint32_t nRows,
                      uint32_t nCols,
                      const S* src,  // [nRows, ldSrc]
                      uint32_t ldSrc,
                      D* dst,  // [nRows, ldDst]
                      uint32_t ldDst,
                      D fillValue,
                      D divisor,
                      cudaStream_t stream)
{
  assert(ldSrc >= nCols);
  assert(ldDst >= nCols);
  uint32_t nThreads = 128;
  uint32_t nBlocks  = ((nRows * ldDst) + nThreads - 1) / nThreads;
  kern_copy_fill<S, D>
    <<<nBlocks, nThreads, 0, stream>>>(nRows, nCols, src, ldSrc, dst, ldDst, fillValue, divisor);
}

template void _cuann_copy_fill<float, float>(uint32_t nRows,
                                             uint32_t nCols,
                                             const float* src,
                                             uint32_t ldSrc,
                                             float* dst,
                                             uint32_t ldDst,
                                             float fillValue,
                                             float divisor,
                                             cudaStream_t stream);
template void _cuann_copy_fill<uint8_t, float>(uint32_t nRows,
                                               uint32_t nCols,
                                               const uint8_t* src,
                                               uint32_t ldSrc,
                                               float* dst,
                                               uint32_t ldDst,
                                               float fillValue,
                                               float divisor,
                                               cudaStream_t stream);
template void _cuann_copy_fill<int8_t, float>(uint32_t nRows,
                                              uint32_t nCols,
                                              const int8_t* src,
                                              uint32_t ldSrc,
                                              float* dst,
                                              uint32_t ldDst,
                                              float fillValue,
                                              float divisor,
                                              cudaStream_t stream);

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
                           float divisor = 1.0f)
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

template void _cuann_copy_with_list<float>(uint32_t nRows,
                                           uint32_t nCols,
                                           const float* src,
                                           const uint32_t* rowList,
                                           uint32_t ldSrc,
                                           float* dst,
                                           uint32_t ldDst,
                                           float divisor);
template void _cuann_copy_with_list<uint8_t>(uint32_t nRows,
                                             uint32_t nCols,
                                             const uint8_t* src,
                                             const uint32_t* rowList,
                                             uint32_t ldSrc,
                                             float* dst,
                                             uint32_t ldDst,
                                             float divisor);
template void _cuann_copy_with_list<int8_t>(uint32_t nRows,
                                            uint32_t nCols,
                                            const int8_t* src,
                                            const uint32_t* rowList,
                                            uint32_t ldSrc,
                                            float* dst,
                                            uint32_t ldDst,
                                            float divisor);

// a -= b
__global__ void kern_a_me_b(uint32_t nRows,
                            uint32_t nCols,
                            float* a,  // [nRows, nCols]
                            uint32_t ldA,
                            float* b  // [nCols]
)
{
  uint64_t gid  = threadIdx.x + (blockDim.x * blockIdx.x);
  uint64_t iCol = gid % nCols;
  uint64_t iRow = gid / nCols;
  if (iRow >= nRows) return;
  a[iCol + (ldA * iRow)] -= b[iCol];
}

// a -= b
void _cuann_a_me_b(uint32_t nRows,
                   uint32_t nCols,
                   float* a,  // [nRows, nCols]
                   uint32_t ldA,
                   float* b  // [nCols]
)
{
  uint32_t nThreads = 128;
  uint32_t nBlocks  = ((nRows * nCols) + nThreads - 1) / nThreads;
  kern_a_me_b<<<nBlocks, nThreads>>>(nRows, nCols, a, ldA, b);
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
                      float* a,                             // [nRows, nCols]
                      const uint32_t* numSamples = nullptr  // [nRows,]
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

//
template <typename D, typename S>
__global__ void kern_transpose_copy_3d(uint32_t num0,
                                       uint32_t num1,
                                       uint32_t num2,
                                       D* dst,  // [num2, ld1, ld0]
                                       uint32_t ld0,
                                       uint32_t ld1,
                                       const S* src,  // [...]
                                       uint32_t stride0,
                                       uint32_t stride1,
                                       uint32_t stride2)
{
  uint32_t tid = threadIdx.x + (blockDim.x * blockIdx.x);
  if (tid >= num0 * num1 * num2) return;
  uint32_t i0 = tid % num0;
  uint32_t i1 = (tid / num0) % num1;
  uint32_t i2 = (tid / num0) / num1;

  dst[i0 + (ld0 * i1) + (ld0 * ld1 * i2)] = src[(stride0 * i0) + (stride1 * i1) + (stride2 * i2)];
}

// transpose_copy_3d
template <typename D, typename S>
void _cuann_transpose_copy_3d(uint32_t num0,
                              uint32_t num1,
                              uint32_t num2,
                              D* dst,  // [num2, ld1, ld0]
                              uint32_t ld0,
                              uint32_t ld1,
                              const S* src,  // [...]
                              uint32_t stride0,
                              uint32_t stride1,
                              uint32_t stride2)
{
  uint32_t nThreads = 128;
  uint32_t nBlocks  = ((num0 * num1 * num2) + nThreads - 1) / nThreads;
  kern_transpose_copy_3d<D, S>
    <<<nBlocks, nThreads>>>(num0, num1, num2, dst, ld0, ld1, src, stride0, stride1, stride2);
}

template void _cuann_transpose_copy_3d<float, float>(uint32_t num0,
                                                     uint32_t num1,
                                                     uint32_t num2,
                                                     float* dst,
                                                     uint32_t ld0,
                                                     uint32_t ld1,
                                                     const float* src,
                                                     uint32_t stride0,
                                                     uint32_t stride1,
                                                     uint32_t stride2);

//
template <typename T>
__global__ void kern_axpy(int num, T alpha, const T* x, T* y)
{
  uint32_t tid = threadIdx.x + (blockDim.x * blockIdx.x);
  if (tid >= num) return;
  y[tid] += alpha * x[tid];
}

//
template <typename T>
void _cuann_axpy(int num, T alpha, const T* x, T* y)
{
  uint32_t nThreads = 128;
  uint32_t nBlocks  = (num + nThreads - 1) / nThreads;
  kern_axpy<T><<<nBlocks, nThreads>>>(num, alpha, x, y);
}

template void _cuann_axpy<float>(int num, float alpha, const float* x, float* y);
template void _cuann_axpy<uint32_t>(int num, uint32_t alpha, const uint32_t* x, uint32_t* y);

//
template <typename T>
T** _cuann_multi_device_malloc(int numDevices,
                               size_t numArrayElements,
                               const char* arrayName,
                               bool useCudaMalloc = false  // If true, cudaMalloc() used,
                                                           // otherwise, cudaMallocManaged() used.
)
{
  cudaError_t cudaError;
  int orgDevId;
  cudaError = cudaGetDevice(&orgDevId);
  if (cudaError != cudaSuccess) {
    fprintf(
      stderr, "(%s, %d) cudaGetDevice() failed (arrayName: %s).\n", __func__, __LINE__, arrayName);
    exit(-1);
  }
  T** arrays = (T**)malloc(sizeof(T*) * numDevices);
  for (int devId = 0; devId < numDevices; devId++) {
    cudaError = cudaSetDevice(devId);
    if (cudaError != cudaSuccess) {
      fprintf(stderr,
              "(%s, %d) cudaSetDevice() failed (arrayName: %s).\n",
              __func__,
              __LINE__,
              arrayName);
      exit(-1);
    }
    if (useCudaMalloc) {
      cudaError = cudaMalloc(&(arrays[devId]), sizeof(T) * numArrayElements);
      if (cudaError != cudaSuccess) {
        fprintf(
          stderr, "(%s, %d) cudaMalloc() failed (arrayName: %s).\n", __func__, __LINE__, arrayName);
        exit(-1);
      }
    } else {
      cudaError = cudaMallocManaged(&(arrays[devId]), sizeof(T) * numArrayElements);
      if (cudaError != cudaSuccess) {
        fprintf(stderr,
                "(%s, %d) cudaMallocManaged() failed (arrayName: %s).\n",
                __func__,
                __LINE__,
                arrayName);
        exit(-1);
      }
    }
  }
  cudaError = cudaSetDevice(orgDevId);
  if (cudaError != cudaSuccess) {
    fprintf(
      stderr, "(%s, %d) cudaSetDevice() failed (arrayName: %s)\n", __func__, __LINE__, arrayName);
    exit(-1);
  }
  return arrays;
}

// multi_device_free
template <typename T>
void _cuann_multi_device_free(T** arrays, int numDevices)
{
  for (int devId = 0; devId < numDevices; devId++) {
    cudaFree(arrays[devId]);
  }
  free(arrays);
}

template void _cuann_multi_device_free<float>(float** arrays, int numDevices);
template void _cuann_multi_device_free<uint32_t>(uint32_t** arrays, int numDevices);
template void _cuann_multi_device_free<uint8_t>(uint8_t** arrays, int numDevices);

/**
 * End of utils
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 * start of kmeans
 */

// update kmeans centers
void _cuann_kmeans_update_centers(float* centers,  // [numCenters, dimCenters]
                                  uint32_t numCenters,
                                  uint32_t dimCenters,
                                  const void* dataset,  // [numDataset, dimCenters]
                                  cudaDataType_t dtype,
                                  uint32_t numDataset,
                                  uint32_t* labels,  // [numDataset]
                                  cuannSimilarity_t similarity,
                                  uint32_t* clusterSize,  // [numCenters]
                                  float* accumulatedCenters)
{
  if (accumulatedCenters == NULL) {
    // accumulate
    _cuann_memset(centers, 0, sizeof(float) * numCenters * dimCenters);
    _cuann_memset(clusterSize, 0, sizeof(uint32_t) * numCenters);
    if (dtype == CUDA_R_32F) {
      _cuann_accumulate_with_label<float>(
        numCenters, dimCenters, centers, clusterSize, numDataset, (const float*)dataset, labels);
    } else if (dtype == CUDA_R_8U) {
      float divisor = 256.0;
      _cuann_accumulate_with_label<uint8_t>(numCenters,
                                            dimCenters,
                                            centers,
                                            clusterSize,
                                            numDataset,
                                            (const uint8_t*)dataset,
                                            labels,
                                            divisor);
    } else if (dtype == CUDA_R_8I) {
      float divisor = 128.0;
      _cuann_accumulate_with_label<int8_t>(numCenters,
                                           dimCenters,
                                           centers,
                                           clusterSize,
                                           numDataset,
                                           (const int8_t*)dataset,
                                           labels,
                                           divisor);
    }
  } else {
    cudaMemcpy(
      centers, accumulatedCenters, sizeof(float) * numCenters * dimCenters, cudaMemcpyDefault);
  }

  if (similarity == CUANN_SIMILARITY_INNER) {
    // normalize
    _cuann_normalize(numCenters, dimCenters, centers, clusterSize);
  } else {
    // average
    _cuann_divide(numCenters, dimCenters, centers, clusterSize);
  }
}

//
static cudaStream_t _cuann_set_cublas_stream(cublasHandle_t cublasHandle, cudaStream_t stream)
{
  cublasStatus_t cublasError;
  cudaStream_t cublasStream;
  cublasError = cublasGetStream(cublasHandle, &cublasStream);
  if (cublasError != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "(%s, %d) cublasGetStream() failed.\n", __func__, __LINE__);
    exit(-1);
  }
  cublasError = cublasSetStream(cublasHandle, stream);
  if (cublasError != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "(%s, %d) cublasSetStream() failed.\n", __func__, __LINE__);
    exit(-1);
  }
  return cublasStream;
}

// predict label of dataset
void _cuann_kmeans_predict_core(cublasHandle_t cublasHandle,
                                const float* centers,  // [numCenters, dimCenters]
                                uint32_t numCenters,
                                uint32_t dimCenters,
                                const float* dataset,  // [numDataset, dimCenters]
                                uint32_t numDataset,
                                uint32_t* labels,  // [numDataset]
                                cuannSimilarity_t similarity,
                                float* workspace)
{
  cublasStatus_t cublasError;
  const uint32_t dimDataset = dimCenters;
  float* sqsumCenters;  // [numCenters]
  float* sqsumDataset;  // [numDataset]
  float* distances;     // [numDataset, numCenters]

  sqsumCenters = workspace;
  sqsumDataset = sqsumCenters + numCenters;
  distances    = sqsumDataset + numDataset;

  float alpha;
  float beta;
  if (similarity == CUANN_SIMILARITY_INNER) {
    alpha = -1.0;
    beta  = 0.0;
  } else {
    _cuann_sqsum(numCenters, dimCenters, centers, sqsumCenters);
    _cuann_sqsum(numDataset, dimDataset, dataset, sqsumDataset);
    _cuann_outer_add(sqsumDataset, numDataset, sqsumCenters, numCenters, distances);
    alpha = -2.0;
    beta  = 1.0;
  }
  cudaStream_t cublasStream = _cuann_set_cublas_stream(cublasHandle, NULL);
  cublasError               = cublasGemmEx(cublasHandle,
                             CUBLAS_OP_T,
                             CUBLAS_OP_N,
                             numCenters,
                             numDataset,
                             dimCenters,
                             &alpha,
                             centers,
                             CUDA_R_32F,
                             dimCenters,
                             dataset,
                             CUDA_R_32F,
                             dimDataset,
                             &beta,
                             distances,
                             CUDA_R_32F,
                             numCenters,
                             CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (cublasError != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "(%s, %d) cublasGemmEx() failed.\n", __func__, __LINE__);
    exit(-1);
  }
  _cuann_set_cublas_stream(cublasHandle, cublasStream);
  _cuann_argmin(numDataset, numCenters, distances, labels);
}

//
uint32_t _cuann_kmeans_predict_chunkSize(uint32_t numCenters, uint32_t numDataset)
{
  uint32_t chunk = (1 << 20);
  if (chunk > (1 << 28) / numCenters) {
    chunk = (1 << 28) / numCenters;
    if (chunk > 31) {
      chunk += 32;
      chunk -= chunk % 64;
    } else {
      chunk = 64;
    }
  }
  chunk = min(chunk, numDataset);
  return chunk;
}

//
size_t _cuann_kmeans_predict_bufferSize(uint32_t numCenters,
                                        uint32_t dimCenters,
                                        uint32_t numDataset)
{
  uint32_t chunk = _cuann_kmeans_predict_chunkSize(numCenters, numDataset);
  size_t size    = 0;
  // float *curDataset;  // [chunk, dimCenters]
  size += _cuann_aligned(sizeof(float) * chunk * dimCenters);
  // void *bufDataset;  // [chunk, dimCenters]
  size += _cuann_aligned(sizeof(float) * chunk * dimCenters);
  // float *workspace;
  size += _cuann_aligned(sizeof(float) * (numCenters + chunk + (numCenters * chunk)));
  return size;
}

// predict label of dataset
void _cuann_kmeans_predict(cublasHandle_t cublasHandle,
                           float* centers,  // [numCenters, dimCenters]
                           uint32_t numCenters,
                           uint32_t dimCenters,
                           const void* dataset,  // [numDataset, dimCenters]
                           cudaDataType_t dtype,
                           uint32_t numDataset,
                           uint32_t* labels,  // [numDataset]
                           cuannSimilarity_t similarity,
                           bool isCenterSet,
                           void* _workspace,
                           float* tempCenters,     // [numCenters, dimCenters]
                           uint32_t* clusterSize,  // [numCenters,]
                           bool updateCenter)
{
  if (!isCenterSet) {
    // If centers are not set, the labels will be determined randomly.
    for (uint32_t i = 0; i < numDataset; i++) {
      labels[i] = i % numCenters;
    }
    if (tempCenters != NULL && clusterSize != NULL) {
      // update centers
      _cuann_kmeans_update_centers(centers,
                                   numCenters,
                                   dimCenters,
                                   dataset,
                                   dtype,
                                   numDataset,
                                   labels,
                                   similarity,
                                   clusterSize,
                                   nullptr);
    }
    return;
  }

  cudaError_t cudaError;
  uint32_t chunk  = _cuann_kmeans_predict_chunkSize(numCenters, numDataset);
  void* workspace = _workspace;
  if (_workspace == NULL) {
    size_t sizeWorkspace = _cuann_kmeans_predict_bufferSize(numCenters, dimCenters, numDataset);
    cudaError            = cudaMallocManaged(&workspace, sizeWorkspace);
    if (cudaError != cudaSuccess) {
      fprintf(stderr, "(%s, %d) cudaMallocManaged() failed.\n", __func__, __LINE__);
      exit(-1);
    }
  }
  float* curDataset;  // [chunk, dimCenters]
  void* bufDataset;   // [chunk, dimCenters]
  float* workspace_core;
  curDataset = (float*)workspace;
  bufDataset = (void*)((uint8_t*)curDataset + _cuann_aligned(sizeof(float) * chunk * dimCenters));
  workspace_core =
    (float*)((uint8_t*)bufDataset + _cuann_aligned(sizeof(float) * chunk * dimCenters));

  if (tempCenters != NULL && clusterSize != NULL) {
    _cuann_memset(tempCenters, 0, sizeof(float) * numCenters * dimCenters);
    _cuann_memset(clusterSize, 0, sizeof(uint32_t) * numCenters);
  }

  cudaMemcpyKind kind;
  cudaPointerAttributes attr;
  cudaPointerGetAttributes(&attr, dataset);
  if (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged) {
    kind = cudaMemcpyDeviceToDevice;
  } else {
    kind = cudaMemcpyHostToDevice;
  }

  for (uint64_t is = 0; is < numDataset; is += chunk) {
    uint64_t ie       = min(is + chunk, (uint64_t)numDataset);
    uint32_t nDataset = ie - is;

    if (dtype == CUDA_R_32F) {
      cudaError = cudaMemcpyAsync(bufDataset,
                                  (float*)dataset + (is * dimCenters),
                                  sizeof(float) * nDataset * dimCenters,
                                  kind,
                                  NULL);
    } else if (dtype == CUDA_R_8U) {
      cudaError = cudaMemcpyAsync(bufDataset,
                                  (uint8_t*)dataset + (is * dimCenters),
                                  sizeof(uint8_t) * nDataset * dimCenters,
                                  kind,
                                  NULL);
    } else if (dtype == CUDA_R_8I) {
      cudaError = cudaMemcpyAsync(bufDataset,
                                  (int8_t*)dataset + (is * dimCenters),
                                  sizeof(int8_t) * nDataset * dimCenters,
                                  kind,
                                  NULL);
    }
    if (cudaError != cudaSuccess) {
      fprintf(stderr, "(%s, %d) cudaMemcpy() failed.\n", __func__, __LINE__);
      exit(-1);
    }

    if (dtype == CUDA_R_32F) {
#if 0
            _cuann_copy<float, float>(nDataset, dimCenters,
                                      (const float*)bufDataset, dimCenters,
                                      curDataset, dimCenters);
#else
      // No need to copy when dtype is CUDA_R_32F
      curDataset = (float*)bufDataset;
#endif
    } else if (dtype == CUDA_R_8U) {
      float divisor = 256.0;
      _cuann_copy<uint8_t, float>(nDataset,
                                  dimCenters,
                                  (const uint8_t*)bufDataset,
                                  dimCenters,
                                  curDataset,
                                  dimCenters,
                                  divisor);
    } else if (dtype == CUDA_R_8I) {
      float divisor = 128.0;
      _cuann_copy<int8_t, float>(nDataset,
                                 dimCenters,
                                 (const int8_t*)bufDataset,
                                 dimCenters,
                                 curDataset,
                                 dimCenters,
                                 divisor);
    }

    // predict
    _cuann_kmeans_predict_core(cublasHandle,
                               centers,
                               numCenters,
                               dimCenters,
                               curDataset,
                               nDataset,
                               labels + is,
                               similarity,
                               workspace_core);

    if ((tempCenters != NULL) && (clusterSize != NULL)) {
      // accumulate
      _cuann_accumulate_with_label<float>(
        numCenters, dimCenters, tempCenters, clusterSize, nDataset, curDataset, labels + is);
    }
#if 0
        // debug
        cudaError = cudaDeviceSynchronize();
        if (cudaError != cudaSuccess) {
            fprintf(stderr, "(%s, %d) cudaDeviceSynchronize() failed.\n",
                    __func__, __LINE__);
            exit(-1);
        }
#endif
  }

  if ((tempCenters != NULL) && (clusterSize != NULL) && updateCenter) {
    _cuann_kmeans_update_centers(centers,
                                 numCenters,
                                 dimCenters,
                                 dataset,
                                 dtype,
                                 numDataset,
                                 labels,
                                 similarity,
                                 clusterSize,
                                 tempCenters);
  }

  if (_workspace == NULL) { cudaFree(workspace); }
}

//
// predict label of dataset with multiple devices
//
void _cuann_kmeans_predict_MP(int numDevices,
                              cublasHandle_t* cublasHandles,  // [numDevices]
                              float* clusterCenters,          // [numCenters, dimCenters]
                              uint32_t numCenters,
                              uint32_t dimCenters,
                              const void* dataset,  // [numDataset, dimCenters]
                              cudaDataType_t dtype,
                              uint32_t numDataset,
                              uint32_t* labels,  // [numDataset]
                              cuannSimilarity_t similarity,
                              bool isCenterSet,
                              uint32_t* clusterSize,  // [numCenters]
                              bool updateCenter       // If true, cluster Centers will be updated.
)
{
  // [numDevices][numCenters, dimCenters]
  float** clusterCentersCopy = _cuann_multi_device_malloc<float>(
    numDevices, numCenters * dimCenters, "clusterCentersCopy", true /* use cudaMalloc() */);

  // [numDevices][numCenters, dimCenters]
  float** clusterCentersMP =
    _cuann_multi_device_malloc<float>(numDevices, numCenters * dimCenters, "clusterCentersMP");

  // [numDevices][numCenters]
  uint32_t** clusterSizeMP =
    _cuann_multi_device_malloc<uint32_t>(numDevices, numCenters, "clusterSizeMP");

  // [numDevices][...]
  size_t sizePredictWorkspace =
    _cuann_kmeans_predict_bufferSize(numCenters, dimCenters, numDataset);
  void** predictWorkspaceMP = (void**)_cuann_multi_device_malloc<uint8_t>(
    numDevices, sizePredictWorkspace, "predictWorkspaceMP");

  int orgDevId;
  cudaGetDevice(&orgDevId);
#pragma omp parallel num_threads(numDevices)
  {
    int devId = omp_get_thread_num();
    cudaSetDevice(devId);
    cudaMemcpy(clusterCentersCopy[devId],
               clusterCenters,
               sizeof(float) * numCenters * dimCenters,
               cudaMemcpyDefault);
    uint64_t d0       = (uint64_t)numDataset * (devId) / numDevices;
    uint64_t d1       = (uint64_t)numDataset * (devId + 1) / numDevices;
    uint64_t nDataset = d1 - d0;
    void* ptrDataset;
    if (dtype == CUDA_R_32F) {
      ptrDataset = (void*)((float*)dataset + (uint64_t)dimCenters * d0);
    } else if (dtype == CUDA_R_8U) {
      ptrDataset = (void*)((uint8_t*)dataset + (uint64_t)dimCenters * d0);
    } else if (dtype == CUDA_R_8I) {
      ptrDataset = (void*)((int8_t*)dataset + (uint64_t)dimCenters * d0);
    }
    _cuann_kmeans_predict(cublasHandles[devId],
                          clusterCentersCopy[devId],
                          numCenters,
                          dimCenters,
                          ptrDataset,
                          dtype,
                          nDataset,
                          labels + d0,
                          similarity,
                          isCenterSet,
                          predictWorkspaceMP[devId],
                          clusterCentersMP[devId],
                          clusterSizeMP[devId],
                          false /* do not update centers */);
  }
  for (int devId = 0; devId < numDevices; devId++) {
    // Barrier
    cudaSetDevice(devId);
    cudaDeviceSynchronize();
  }
  cudaSetDevice(orgDevId);
  if (clusterSize != NULL) {
    // Reduce results to main thread
    _cuann_memset(clusterSize, 0, sizeof(uint32_t) * numCenters);
    for (int devId = 0; devId < numDevices; devId++) {
      _cuann_axpy<uint32_t>(numCenters, 1, clusterSizeMP[devId], clusterSize);
      if (devId != orgDevId) {
        _cuann_axpy<float>(
          numCenters * dimCenters, 1, clusterCentersMP[devId], clusterCentersMP[orgDevId]);
      }
    }
    if (updateCenter) {
      _cuann_kmeans_update_centers(clusterCenters,
                                   numCenters,
                                   dimCenters,
                                   dataset,
                                   dtype,
                                   numDataset,
                                   labels,
                                   similarity,
                                   clusterSize,
                                   clusterCentersMP[orgDevId]);
    }
  }

  _cuann_multi_device_free<float>(clusterCentersCopy, numDevices);
  _cuann_multi_device_free<float>(clusterCentersMP, numDevices);
  _cuann_multi_device_free<uint32_t>(clusterSizeMP, numDevices);
  _cuann_multi_device_free<uint8_t>((uint8_t**)predictWorkspaceMP, numDevices);
}

// predict labe of dataset (naive CPU version).
// (*) available only for prediction, but not for training.
void _cuann_kmeans_predict_CPU(float* centers,  // [numCenters, dimCenters]
                               uint32_t numCenters,
                               uint32_t dimCenters,
                               const void* dataset,  // [numDataset, dimCenters]
                               cudaDataType_t dtype,
                               uint32_t numDataset,
                               uint32_t* labels,  // [numDataset]
                               cuannSimilarity_t similarity)
{
  float multiplier = 1.0;
  if (dtype == CUDA_R_8U) {
    multiplier = 1.0 / 256.0;
  } else if (dtype == CUDA_R_8I) {
    multiplier = 1.0 / 128.0;
  }
  for (uint32_t i = 0; i < numDataset; i++) {
    float* vector = (float*)malloc(sizeof(float) * dimCenters);
    for (uint32_t j = 0; j < dimCenters; j++) {
      if (dtype == CUDA_R_32F) {
        vector[j] = ((float*)dataset)[j + (dimCenters * i)];
      } else if (dtype == CUDA_R_8U) {
        vector[j] = ((uint8_t*)dataset)[j + (dimCenters * i)];
        vector[j] *= multiplier;
      } else if (dtype == CUDA_R_8I) {
        vector[j] = ((int8_t*)dataset)[j + (dimCenters * i)];
        vector[j] *= multiplier;
      }
    }
    float best_score;
    for (uint32_t l = 0; l < numCenters; l++) {
      float score = 0.0;
      for (uint32_t j = 0; j < dimCenters; j++) {
        if (similarity == CUANN_SIMILARITY_INNER) {
          score -= vector[j] * centers[j + (dimCenters * l)];
        } else {
          float diff = vector[j] - centers[j + (dimCenters * l)];
          score += diff * diff;
        }
      }
      if ((l == 0) || (score < best_score)) {
        labels[i]  = l;
        best_score = score;
      }
    }
    free(vector);
  }
}

#define R_FACTOR 8

//
template <typename T, int _divisor>
__global__ void kern_adjust_centers(float* centers,  // [numCenters, dimCenters]
                                    uint32_t numCenters,
                                    uint32_t dimCenters,
                                    const void* _dataset,  // [numDataet, dimCenters]
                                    uint32_t numDataset,
                                    const uint32_t* labels,  // [numDataset]
                                    cuannSimilarity_t similarity,
                                    const uint32_t* clusterSize,  // [numCenters]
                                    float threshold,
                                    uint32_t average,
                                    uint32_t ofst,
                                    uint32_t* count)
{
  const T* dataset = (const T*)_dataset;
  float divisor    = (float)_divisor;
  uint32_t l       = threadIdx.y + blockDim.y * blockIdx.y;
  if (l >= numCenters) return;
  if (clusterSize[l] > (int)(average * threshold)) return;

  uint32_t laneId = threadIdx.x % 32;
  uint32_t i;
  if (laneId == 0) {
    do {
      uint32_t old = atomicAdd(count, 1);
      i            = (ofst * (old + 1)) % numDataset;
    } while (clusterSize[labels[i]] < average);
  }
  i           = __shfl_sync(0xffffffff, i, 0);
  uint32_t li = labels[i];
  float sqsum = 0.0;
  for (uint32_t j = laneId; j < dimCenters; j += 32) {
    float val = centers[j + (uint64_t)dimCenters * li] * (R_FACTOR - 1);
    val += (float)(dataset[j + (uint64_t)dimCenters * i]) / divisor;
    val /= R_FACTOR;
    sqsum += val * val;
    centers[j + (uint64_t)dimCenters * l] = val;
  }
  if (similarity == CUANN_SIMILARITY_INNER) {
    sqsum += __shfl_xor_sync(0xffffffff, sqsum, 1);
    sqsum += __shfl_xor_sync(0xffffffff, sqsum, 2);
    sqsum += __shfl_xor_sync(0xffffffff, sqsum, 4);
    sqsum += __shfl_xor_sync(0xffffffff, sqsum, 8);
    sqsum += __shfl_xor_sync(0xffffffff, sqsum, 16);
    sqsum = sqrt(sqsum);
    for (uint32_t j = laneId; j < dimCenters; j += 32) {
      centers[j + ((uint64_t)dimCenters * l)] /= sqsum;
    }
  }
}

// adjust centers which have small number of entries
bool _cuann_kmeans_adjust_centers(float* centers,  // [numCenters, dimCenters]
                                  uint32_t numCenters,
                                  uint32_t dimCenters,
                                  const void* dataset,  // [numDataset, dimCenters]
                                  cudaDataType_t dtype,
                                  uint32_t numDataset,
                                  const uint32_t* labels,  // [numDataset]
                                  cuannSimilarity_t similarity,
                                  const uint32_t* clusterSize,  // [numCenters]
                                  float threshold,
                                  void* ws)
{
  if (dtype != CUDA_R_32F && dtype != CUDA_R_8U && dtype != CUDA_R_8I) {
    fprintf(stderr, "(%s, %d) Unsupported dtype (%d)\n", __func__, __LINE__, dtype);
    exit(-1);
  }
  bool adjusted                = false;
  static uint32_t iPrimes      = 0;
  constexpr uint32_t numPrimes = 40;
  uint32_t primes[numPrimes]   = {29,   71,   113,  173,  229,  281,  349,  409,  463,  541,
                                601,  659,  733,  809,  863,  941,  1013, 1069, 1151, 1223,
                                1291, 1373, 1451, 1511, 1583, 1657, 1733, 1811, 1889, 1987,
                                2053, 2129, 2213, 2287, 2357, 2423, 2531, 2617, 2687, 2741};
  uint32_t average             = (numDataset + numCenters - 1) / numCenters;
  uint32_t ofst;
  do {
    iPrimes = (iPrimes + 1) % numPrimes;
    ofst    = primes[iPrimes];
  } while (numDataset % ofst == 0);

  cudaDeviceSynchronize();
  cudaPointerAttributes attr;
  cudaPointerGetAttributes(&attr, dataset);
  if (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged) {
    // GPU
    uint32_t* count;
    if (ws == NULL) {
      cudaMallocManaged(&count, sizeof(uint32_t));
    } else {
      count = (uint32_t*)ws;
    }
    count[0] = 0;
    void (*kernel)(float*,
                   uint32_t,
                   uint32_t,
                   const void*,
                   uint32_t,
                   const uint32_t*,
                   cuannSimilarity_t,
                   const uint32_t*,
                   float,
                   uint32_t,
                   uint32_t,
                   uint32_t*);
    if (dtype == CUDA_R_32F) {
      kernel = kern_adjust_centers<float, 1>;
    } else if (dtype == CUDA_R_8U) {
      kernel = kern_adjust_centers<uint8_t, 256>;
    } else if (dtype == CUDA_R_8I) {
      kernel = kern_adjust_centers<int8_t, 128>;
    }
    dim3 threads(32, 4, 1);
    dim3 blocks(1, (numCenters + threads.y - 1) / threads.y, 1);
    kernel<<<blocks, threads>>>(centers,
                                numCenters,
                                dimCenters,
                                dataset,
                                numDataset,
                                labels,
                                similarity,
                                clusterSize,
                                threshold,
                                average,
                                ofst,
                                count);
    cudaDeviceSynchronize();
    if (count[0] > 0) { adjusted = true; }
    if (ws == NULL) { cudaFree(count); }
  } else {
    // CPU
    uint32_t i     = 0;
    uint32_t count = 0;
    for (uint32_t l = 0; l < numCenters; l++) {
      if (clusterSize[l] > (int)(average * threshold)) continue;
      do {
        i = (i + ofst) % numDataset;
      } while (clusterSize[labels[i]] < average);
      uint32_t li = labels[i];
      float sqsum = 0.0;
      for (uint32_t j = 0; j < dimCenters; j++) {
        float val = centers[j + ((uint64_t)dimCenters * li)] * (R_FACTOR - 1);
        if (dtype == CUDA_R_32F) {
          val += ((float*)dataset)[j + ((uint64_t)dimCenters * i)];
        } else if (dtype == CUDA_R_8U) {
          float divisor = 256.0;
          val += ((uint8_t*)dataset)[j + ((uint64_t)dimCenters * i)] / divisor;
        } else if (dtype == CUDA_R_8I) {
          float divisor = 128.0;
          val += ((int8_t*)dataset)[j + ((uint64_t)dimCenters * i)] / divisor;
        }
        val /= R_FACTOR;
        sqsum += val * val;
        centers[j + ((uint64_t)dimCenters * l)] = val;
      }
      if (similarity == CUANN_SIMILARITY_INNER) {
        sqsum = sqrt(sqsum);
        for (uint32_t j = 0; j < dimCenters; j++) {
          centers[j + ((uint64_t)dimCenters * l)] /= sqsum;
        }
      }
      count += 1;
    }
    if (count > 0) {
      adjusted = true;
#ifdef CUANN_DEBUG
      fprintf(stderr,
              "(%s) num adjusted: %u / %u, threshold: %d \n",
              __func__,
              count,
              numCenters,
              (int)(average * threshold));
#endif
    }
  }
  return adjusted;
}

/**
 * end of kmeans
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 * Start of topk
 */

//
#define NUM_THREADS      1024  // DO NOT CHANGE
#define STATE_BIT_LENGTH 8     // 0: state not used,  8: state used
#define MAX_VEC_LENGTH   8     // 1, 2, 4 or 8
// #define CUANN_DEBUG

//
__device__ inline uint32_t convert(uint32_t x)
{
  if (x & 0x80000000) {
    return x ^ 0xffffffff;
  } else {
    return x ^ 0x80000000;
  }
}

//
struct u32_vector {
  uint1 x1;
  uint2 x2;
  uint4 x4;
  ulonglong4 x8;
};

//
template <int vecLen>
__device__ inline void load_u32_vector(struct u32_vector& vec, const uint32_t* x, int i)
{
  if (vecLen == 1) {
    vec.x1 = ((uint1*)(x + i))[0];
  } else if (vecLen == 2) {
    vec.x2 = ((uint2*)(x + i))[0];
  } else if (vecLen == 4) {
    vec.x4 = ((uint4*)(x + i))[0];
  } else if (vecLen == 8) {
    vec.x8 = ((ulonglong4*)(x + i))[0];
  }
}

//
template <int vecLen>
__device__ inline uint32_t get_element_from_u32_vector(struct u32_vector& vec, int i)
{
  uint32_t xi;
  if (vecLen == 1) {
    xi = convert(vec.x1.x);
  } else if (vecLen == 2) {
    if (i == 0)
      xi = convert(vec.x2.x);
    else
      xi = convert(vec.x2.y);
  } else if (vecLen == 4) {
    if (i == 0)
      xi = convert(vec.x4.x);
    else if (i == 1)
      xi = convert(vec.x4.y);
    else if (i == 2)
      xi = convert(vec.x4.z);
    else
      xi = convert(vec.x4.w);
  } else if (vecLen == 8) {
    if (i == 0)
      xi = convert((uint32_t)(vec.x8.x & 0xffffffff));
    else if (i == 1)
      xi = convert((uint32_t)(vec.x8.x >> 32));
    else if (i == 2)
      xi = convert((uint32_t)(vec.x8.y & 0xffffffff));
    else if (i == 3)
      xi = convert((uint32_t)(vec.x8.y >> 32));
    else if (i == 4)
      xi = convert((uint32_t)(vec.x8.z & 0xffffffff));
    else if (i == 5)
      xi = convert((uint32_t)(vec.x8.z >> 32));
    else if (i == 6)
      xi = convert((uint32_t)(vec.x8.w & 0xffffffff));
    else
      xi = convert((uint32_t)(vec.x8.w >> 32));
  }
  return xi;
}

//
template <int blockDim_x, int stateBitLen, int vecLen>
__launch_bounds__(1024, 2) __global__
  void kern_topk_cg_11(uint32_t topk,
                       uint32_t size_batch,
                       uint32_t max_len_x,
                       uint32_t* len_x,     // [size_batch,]
                       const uint32_t* _x,  // [size_batch, max_len_x,]
                       uint8_t* _state,     // [size_batch, max_len_x / 8,]
                       uint32_t* _labels,   // [size_batch, topk,]
                       uint32_t* _count     // [size_batch, 5 * 1024,]
  )
{
  __shared__ uint32_t smem[2048 + 6];
  uint32_t* best_index = &(smem[2048]);
  uint32_t* best_csum  = &(smem[2048 + 3]);
  typedef BlockScan<uint32_t, blockDim_x> BlockScanT;
  __shared__ typename BlockScanT::TempStorage temp_storage;
  namespace cg        = cooperative_groups;
  cg::grid_group grid = cg::this_grid();
  uint32_t i_batch    = blockIdx.y;
  if (i_batch >= size_batch) return;

  uint32_t nx;
  if (len_x == NULL) {
    nx = max_len_x;
  } else {
    nx = len_x[i_batch];
  }

  uint32_t num_threads = blockDim_x * gridDim.x;
  uint32_t thread_id   = threadIdx.x + (blockDim_x * blockIdx.x);

  const uint32_t* x = _x + (max_len_x * i_batch);
  uint8_t* state    = NULL;
  if (stateBitLen == 8) {
    uint32_t numSample_perThread = (max_len_x + num_threads - 1) / num_threads;
    uint32_t numState_perThread  = (numSample_perThread + stateBitLen - 1) / stateBitLen;
    state                        = _state + (numState_perThread * num_threads * i_batch);
  }
  uint32_t* labels = _labels + (topk * i_batch);
  if (threadIdx.x < 6) { smem[2048 + threadIdx.x] = 0; }

  uint32_t* count = _count + (5 * 1024 * i_batch);
  for (int i = thread_id; i < 5 * 1024; i += num_threads) {
    count[i] = 0;
  }
  cg::sync(grid);

  uint32_t count_below = 0;
  uint32_t threshold   = 0;

  //
  // Search for the maximum threshold that satisfies "(x < threshold).sum() < topk".
  //
  for (int j = 0; j < 2; j += 1) {
    uint32_t shift = (21 - 11 * j);
    for (int i = threadIdx.x; i < 2048; i += blockDim_x) {
      smem[i] = 0;
    }
    __syncthreads();

    int ii = 0;
    for (int i = thread_id * vecLen; i < nx; i += num_threads * max(vecLen, stateBitLen), ii++) {
      uint8_t iState = 0;
      if (stateBitLen == 8 && j > 0) { iState = state[thread_id + (num_threads * ii)]; }
#pragma unroll
      for (int v = 0; v < max(vecLen, stateBitLen); v += vecLen) {
        int iv = i + (num_threads * v);
        if (iv >= nx) break;

        struct u32_vector x_vec;
        load_u32_vector<vecLen>(x_vec, x, iv);
#pragma unroll
        for (int u = 0; u < vecLen; u++) {
          int ivu = iv + u;
          if (ivu >= nx) break;

          uint8_t mask = (uint8_t)0x1 << (v + u);
          uint32_t xi  = get_element_from_u32_vector<vecLen>(x_vec, u);
          if (xi < threshold) {
            if (stateBitLen == 8) {
              labels[atomicAdd(&count[0], 1)] = ivu;
              iState |= mask;
            }
          } else {
            uint32_t k = (xi - threshold) >> shift;  // 0 <= k
            if (k >= 2048) {
              if (stateBitLen == 8) { iState |= mask; }
            } else if (k + 1 < 2048) {
              atomicAdd(&(smem[k + 1]), 1);
            }
          }
        }
      }
      if (stateBitLen == 8) { state[thread_id + (num_threads * ii)] = iState; }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < 2048; i += blockDim_x) {
      if (smem[i] > 0) { atomicAdd(&(count[i + (2048 * j)]), smem[i]); }
    }
    cg::sync(grid);

    constexpr int n_data = 2048 / blockDim_x;
    uint32_t csum[n_data];
#pragma unroll
    for (int i = 0; i < n_data; i++) {
      csum[i] = count[i + (n_data * threadIdx.x) + (2048 * j)];
    }
    BlockScanT(temp_storage).InclusiveSum(csum, csum);

#pragma unroll
    for (int i = n_data - 1; i >= 0; i--) {
      if (count_below + csum[i] >= topk) continue;
      uint32_t index = i + (n_data * threadIdx.x);
      atomicMax(&(best_index[j]), index);
      atomicMax(&(best_csum[j]), csum[i]);
      break;
    }
    __syncthreads();

    count_below += best_csum[j];
    threshold += (best_index[j] << shift);
  }

  {
    uint32_t j = 2;
    for (int i = threadIdx.x; i < 1024; i += blockDim_x) {
      smem[i] = 0;
    }
    __syncthreads();

    int ii = 0;
    for (int i = thread_id * vecLen; i < nx; i += num_threads * max(vecLen, stateBitLen), ii++) {
      uint8_t iState = 0;
      if (stateBitLen == 8) {
        iState = state[thread_id + (num_threads * ii)];
        if (iState == (uint8_t)0xff) continue;
      }
#pragma unroll
      for (int v = 0; v < max(vecLen, stateBitLen); v += vecLen) {
        int iv = i + (num_threads * v);
        if (iv >= nx) break;

        struct u32_vector x_vec;
        load_u32_vector<vecLen>(x_vec, x, iv);
#pragma unroll
        for (int u = 0; u < vecLen; u++) {
          int ivu = iv + u;
          if (ivu >= nx) break;

          uint8_t mask = (uint8_t)0x1 << (v + u);
          if ((stateBitLen == 8) && (iState & mask)) continue;
          uint32_t xi = get_element_from_u32_vector<vecLen>(x_vec, u);
          if (xi < threshold) {
            if (stateBitLen == 8) {
              labels[atomicAdd(&count[0], 1)] = ivu;
              iState |= mask;
            }
          } else {
            uint32_t k = (xi - threshold);  // 0 <= k
            if (k >= 1024) {
              if (stateBitLen == 8) { iState |= mask; }
            } else if (k + 1 < 1024) {
              atomicAdd(&(smem[k + 1]), 1);
            }
          }
        }
      }
      if (stateBitLen == 8) { state[thread_id + (num_threads * ii)] = iState; }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < 1024; i += blockDim_x) {
      if (smem[i] > 0) { atomicAdd(&(count[i + (2048 * j)]), smem[i]); }
    }
    cg::sync(grid);

    constexpr int n_data = 1024 / blockDim_x;
    uint32_t csum[n_data];
#pragma unroll
    for (int i = 0; i < n_data; i++) {
      csum[i] = count[i + (n_data * threadIdx.x) + (2048 * j)];
    }
    BlockScanT(temp_storage).InclusiveSum(csum, csum);

#pragma unroll
    for (int i = n_data - 1; i >= 0; i--) {
      if (count_below + csum[i] >= topk) continue;
      uint32_t index = i + (n_data * threadIdx.x);
      atomicMax(&(best_index[j]), index);
      atomicMax(&(best_csum[j]), csum[i]);
      break;
    }
    __syncthreads();

    count_below += best_csum[j];
    threshold += best_index[j];
  }

  //
  // Get labels that satifies "x[i] < threshold".
  //
  int ii = 0;
  for (int i = thread_id * vecLen; i < nx; i += num_threads * max(vecLen, stateBitLen), ii++) {
    uint8_t iState = 0;
    if (stateBitLen == 8) {
      iState = state[thread_id + (num_threads * ii)];
      if (iState == (uint8_t)0xff) continue;
    }
#pragma unroll
    for (int v = 0; v < max(vecLen, stateBitLen); v += vecLen) {
      int iv = i + (num_threads * v);
      if (iv >= nx) break;

      struct u32_vector vec;
      load_u32_vector<vecLen>(vec, x, iv);
#pragma unroll
      for (int u = 0; u < vecLen; u++) {
        int ivu = iv + u;
        if (ivu >= nx) break;

        uint8_t mask = (uint8_t)0x1 << (v + u);
        if ((stateBitLen == 8) && (iState & mask)) continue;
        uint32_t xi = get_element_from_u32_vector<vecLen>(vec, u);
        if (xi < threshold) {
          labels[atomicAdd(&count[0], 1)] = ivu;
        } else if ((xi == threshold) && (count_below + count[2048] < topk)) {
          if (count_below + atomicAdd(&count[2048], 1) < topk) {
            labels[atomicAdd(&count[0], 1)] = ivu;
          }
        }
      }
    }
  }

#ifdef CUANN_DEBUG
  cg::sync(grid);
  if (thread_id == 0 && count[0] < topk) {
    printf("# i_batch:%d, topk:%d, count[0]:%d, count_below:%d, threshold:%08x\n",
           i_batch,
           topk,
           count[0],
           count_below,
           threshold);
  }
#endif
}

//
template <int blockDim_x, int stateBitLen, int vecLen>
__launch_bounds__(1024, 2) __global__
  void kern_topk_cta_11(uint32_t topk,
                        uint32_t size_batch,
                        uint32_t max_len_x,
                        uint32_t* len_x,     // [size_batch, max_len_x,]
                        const uint32_t* _x,  // [size_batch, max_len_x,]
                        uint8_t* _state,     // [size_batch, max_len_x / 8,]
                        uint32_t* _labels    // [size_batch, topk,]
  )
{
  __shared__ uint32_t smem[2048 + 3 + 3 + 2];
  uint32_t* best_index = &(smem[2048]);
  uint32_t* best_csum  = &(smem[2048 + 3]);
  uint32_t* count      = &(smem[2048 + 6]);
  typedef BlockScan<uint32_t, blockDim_x> BlockScanT;
  __shared__ typename BlockScanT::TempStorage temp_storage;
  uint32_t i_batch = blockIdx.y;
  if (i_batch >= size_batch) return;

  uint32_t nx;
  if (len_x == NULL) {
    nx = max_len_x;
  } else {
    nx = len_x[i_batch];
  }

  uint32_t num_threads = blockDim_x;
  uint32_t thread_id   = threadIdx.x;

  const uint32_t* x = _x + (max_len_x * i_batch);
  uint8_t* state    = NULL;
  if (stateBitLen == 8) {
    uint32_t numSample_perThread = (max_len_x + num_threads - 1) / num_threads;
    uint32_t numState_perThread  = (numSample_perThread + stateBitLen - 1) / stateBitLen;
    state                        = _state + (numState_perThread * num_threads * i_batch);
  }
  uint32_t* labels = _labels + (topk * i_batch);
  if (threadIdx.x < 8) { smem[2048 + threadIdx.x] = 0; }

  uint32_t count_below = 0;
  uint32_t threshold   = 0;

  //
  // Search for the maximum threshold that satisfies "(x < threshold).sum() < topk".
  //
  for (int j = 0; j < 2; j += 1) {
    uint32_t shift = (21 - 11 * j);
    for (int i = threadIdx.x; i < 2048; i += blockDim_x) {
      smem[i] = 0;
    }
    __syncthreads();

    int ii = 0;
    for (int i = thread_id * vecLen; i < nx; i += num_threads * max(vecLen, stateBitLen), ii++) {
      uint8_t iState = 0;
      if (stateBitLen == 8 && j > 0) { iState = state[thread_id + (num_threads * ii)]; }
#pragma unroll
      for (int v = 0; v < max(vecLen, stateBitLen); v += vecLen) {
        int iv = i + (num_threads * v);
        if (iv >= nx) break;

        struct u32_vector x_vec;
        load_u32_vector<vecLen>(x_vec, x, iv);
#pragma unroll
        for (int u = 0; u < vecLen; u++) {
          int ivu = iv + u;
          if (ivu >= nx) break;

          uint8_t mask = (uint8_t)0x1 << (v + u);
          uint32_t xi  = get_element_from_u32_vector<vecLen>(x_vec, u);
          if (xi < threshold) {
            if (stateBitLen == 8) {
              labels[atomicAdd(&count[0], 1)] = ivu;
              iState |= mask;
            }
          } else {
            uint32_t k = (xi - threshold) >> shift;  // 0 <= k
            if (k >= 2048) {
              if (stateBitLen == 8) { iState |= mask; }
            } else if (k + 1 < 2048) {
              atomicAdd(&(smem[k + 1]), 1);
            }
          }
        }
      }
      if (stateBitLen == 8) { state[thread_id + (num_threads * ii)] = iState; }
    }
    __syncthreads();

    constexpr int n_data = 2048 / blockDim_x;
    uint32_t csum[n_data];
#pragma unroll
    for (int i = 0; i < n_data; i++) {
      csum[i] = smem[i + (n_data * threadIdx.x)];
    }
    BlockScanT(temp_storage).InclusiveSum(csum, csum);

#pragma unroll
    for (int i = n_data - 1; i >= 0; i--) {
      if (count_below + csum[i] > topk) continue;
      uint32_t index = i + (n_data * threadIdx.x);
      atomicMax(&(best_index[j]), index);
      atomicMax(&(best_csum[j]), csum[i]);
      break;
    }
    __syncthreads();

    count_below += best_csum[j];
    threshold += (best_index[j] << shift);
    if (count_below == topk) break;
  }

  {
    uint32_t j = 2;
    for (int i = threadIdx.x; i < 1024; i += blockDim_x) {
      smem[i] = 0;
    }
    __syncthreads();

    int ii = 0;
    for (int i = thread_id * vecLen; i < nx; i += num_threads * max(vecLen, stateBitLen), ii++) {
      uint8_t iState = 0;
      if (stateBitLen == 8) {
        iState = state[thread_id + (num_threads * ii)];
        if (iState == (uint8_t)0xff) continue;
      }
#pragma unroll
      for (int v = 0; v < max(vecLen, stateBitLen); v += vecLen) {
        int iv = i + (num_threads * v);
        if (iv >= nx) break;

        struct u32_vector x_vec;
        load_u32_vector<vecLen>(x_vec, x, iv);
#pragma unroll
        for (int u = 0; u < vecLen; u++) {
          int ivu = iv + u;
          if (ivu >= nx) break;

          uint8_t mask = (uint8_t)0x1 << (v + u);
          if ((stateBitLen == 8) && (iState & mask)) continue;
          uint32_t xi = get_element_from_u32_vector<vecLen>(x_vec, u);
          if (xi < threshold) {
            if (stateBitLen == 8) {
              labels[atomicAdd(&count[0], 1)] = ivu;
              iState |= mask;
            }
          } else {
            uint32_t k = (xi - threshold);  // 0 <= k
            if (k >= 1024) {
              if (stateBitLen == 8) { iState |= mask; }
            } else if (k + 1 < 1024) {
              atomicAdd(&(smem[k + 1]), 1);
            }
          }
        }
      }
      if (stateBitLen == 8) { state[thread_id + (num_threads * ii)] = iState; }
    }
    __syncthreads();

    constexpr int n_data = 1024 / blockDim_x;
    uint32_t csum[n_data];
#pragma unroll
    for (int i = 0; i < n_data; i++) {
      csum[i] = smem[i + (n_data * threadIdx.x)];
    }
    BlockScanT(temp_storage).InclusiveSum(csum, csum);

#pragma unroll
    for (int i = n_data - 1; i >= 0; i--) {
      if (count_below + csum[i] > topk) continue;
      uint32_t index = i + (n_data * threadIdx.x);
      atomicMax(&(best_index[j]), index);
      atomicMax(&(best_csum[j]), csum[i]);
      break;
    }
    __syncthreads();

    count_below += best_csum[j];
    threshold += best_index[j];
  }

  //
  // Get labels that satifies "x[i] < threshold".
  //
  int ii = 0;
  for (int i = thread_id * vecLen; i < nx; i += num_threads * max(vecLen, stateBitLen), ii++) {
    uint8_t iState = 0;
    if (stateBitLen == 8) {
      iState = state[thread_id + (num_threads * ii)];
      if (iState == (uint8_t)0xff) continue;
    }
#pragma unroll
    for (int v = 0; v < max(vecLen, stateBitLen); v += vecLen) {
      int iv = i + (num_threads * v);
      if (iv >= nx) break;

      struct u32_vector vec;
      load_u32_vector<vecLen>(vec, x, iv);
#pragma unroll
      for (int u = 0; u < vecLen; u++) {
        int ivu = iv + u;
        if (ivu >= nx) break;

        uint8_t mask = (uint8_t)0x1 << (v + u);
        if ((stateBitLen == 8) && (iState & mask)) continue;
        uint32_t xi = get_element_from_u32_vector<vecLen>(vec, u);
        if (xi < threshold) {
          labels[atomicAdd(&count[0], 1)] = ivu;
        } else if ((xi == threshold) && (count_below + count[1] < topk)) {
          if (count_below + atomicAdd(&count[1], 1) < topk) {
            labels[atomicAdd(&count[0], 1)] = ivu;
          }
        }
      }
    }
  }

#ifdef CUANN_DEBUG
  __syncthreads();
  if (thread_id == 0 && count[0] < topk) {
    printf("# i_batch:%d, topk:%d, count[0]:%d, count_below:%d, threshold:%08x\n",
           i_batch,
           topk,
           count[0],
           count_below,
           threshold);
  }
#endif
}

//
__device__ inline uint16_t convert(uint16_t x)
{
  if (x & 0x8000) {
    return x ^ 0xffff;
  } else {
    return x ^ 0x8000;
  }
}

//
struct u16_vector {
  ushort1 x1;
  ushort2 x2;
  ushort4 x4;
  uint4 x8;
};

//
template <int vecLen>
__device__ inline void load_u16_vector(struct u16_vector& vec, const uint16_t* x, int i)
{
  if (vecLen == 1) {
    vec.x1 = ((ushort1*)(x + i))[0];
  } else if (vecLen == 2) {
    vec.x2 = ((ushort2*)(x + i))[0];
  } else if (vecLen == 4) {
    vec.x4 = ((ushort4*)(x + i))[0];
  } else if (vecLen == 8) {
    vec.x8 = ((uint4*)(x + i))[0];
  }
}

//
template <int vecLen>
__device__ inline uint16_t get_element_from_u16_vector(struct u16_vector& vec, int i)
{
  uint16_t xi;
  if (vecLen == 1) {
    xi = convert(vec.x1.x);
  } else if (vecLen == 2) {
    if (i == 0)
      xi = convert(vec.x2.x);
    else
      xi = convert(vec.x2.y);
  } else if (vecLen == 4) {
    if (i == 0)
      xi = convert(vec.x4.x);
    else if (i == 1)
      xi = convert(vec.x4.y);
    else if (i == 2)
      xi = convert(vec.x4.z);
    else
      xi = convert(vec.x4.w);
  } else if (vecLen == 8) {
    if (i == 0)
      xi = convert((uint16_t)(vec.x8.x & 0xffff));
    else if (i == 1)
      xi = convert((uint16_t)(vec.x8.x >> 16));
    else if (i == 2)
      xi = convert((uint16_t)(vec.x8.y & 0xffff));
    else if (i == 3)
      xi = convert((uint16_t)(vec.x8.y >> 16));
    else if (i == 4)
      xi = convert((uint16_t)(vec.x8.z & 0xffff));
    else if (i == 5)
      xi = convert((uint16_t)(vec.x8.z >> 16));
    else if (i == 6)
      xi = convert((uint16_t)(vec.x8.w & 0xffff));
    else
      xi = convert((uint16_t)(vec.x8.w >> 16));
  }
  return xi;
}

//
template <int blockDim_x, int stateBitLen, int vecLen>
__launch_bounds__(1024, 2) __global__
  void kern_topk_cg_8(uint32_t topk,
                      uint32_t size_batch,
                      uint32_t max_len_x,
                      uint32_t* len_x,     // [size_batch,]
                      const uint16_t* _x,  // [size_batch, max_len_x,]
                      uint8_t* _state,     // [size_batch, max_len_x / 8,]
                      uint32_t* _labels,   // [size_batch, topk,]
                      uint32_t* _count     // [size_batch, 5 * 1024,]
  )
{
  __shared__ uint32_t smem[256 + 4];
  uint32_t* best_index = &(smem[256]);
  uint32_t* best_csum  = &(smem[256 + 2]);
  typedef BlockScan<uint32_t, blockDim_x> BlockScanT;
  __shared__ typename BlockScanT::TempStorage temp_storage;
  namespace cg        = cooperative_groups;
  cg::grid_group grid = cg::this_grid();
  uint32_t i_batch    = blockIdx.y;
  if (i_batch >= size_batch) return;

  uint32_t nx;
  if (len_x == NULL) {
    nx = max_len_x;
  } else {
    nx = len_x[i_batch];
  }

  uint32_t num_threads = blockDim_x * gridDim.x;
  uint32_t thread_id   = threadIdx.x + (blockDim_x * blockIdx.x);

  const uint16_t* x = _x + (max_len_x * i_batch);
  uint8_t* state    = NULL;
  if (stateBitLen == 8) {
    uint32_t numSample_perThread = (max_len_x + num_threads - 1) / num_threads;
    uint32_t numState_perThread  = (numSample_perThread + stateBitLen - 1) / stateBitLen;
    state                        = _state + (numState_perThread * num_threads * i_batch);
  }
  uint32_t* labels = _labels + (topk * i_batch);
  if (threadIdx.x < 4) { smem[256 + threadIdx.x] = 0; }

  uint32_t* count = _count + (2 * 256 * i_batch);
  for (int i = thread_id; i < 2 * 256; i += num_threads) {
    count[i] = 0;
  }
  cg::sync(grid);

  uint32_t count_below = 0;
  uint32_t threshold   = 0;

  //
  // Search for the maximum threshold that satisfies "(x < threshold).sum() < topk".
  //
  for (int j = 0; j < 2; j += 1) {
    uint32_t shift = (8 - 8 * j);
    for (int i = threadIdx.x; i < 256; i += blockDim_x) {
      smem[i] = 0;
    }
    __syncthreads();

    int ii = 0;
    for (int i = thread_id * vecLen; i < nx; i += num_threads * max(vecLen, stateBitLen), ii++) {
      uint8_t iState = 0;
      if (stateBitLen == 8 && j > 0) { iState = state[thread_id + (num_threads * ii)]; }
#pragma unroll
      for (int v = 0; v < max(vecLen, stateBitLen); v += vecLen) {
        int iv = i + (num_threads * v);
        if (iv >= nx) break;

        struct u16_vector x_vec;
        load_u16_vector<vecLen>(x_vec, x, iv);
#pragma unroll
        for (int u = 0; u < vecLen; u++) {
          int ivu = iv + u;
          if (ivu >= nx) break;

          uint8_t mask = (uint8_t)0x1 << (v + u);
          uint32_t xi  = get_element_from_u16_vector<vecLen>(x_vec, u);
          if (xi < threshold) {
            if (stateBitLen == 8) {
              labels[atomicAdd(&count[0], 1)] = ivu;
              iState |= mask;
            }
          } else {
            uint32_t k = (xi - threshold) >> shift;  // 0 <= k
            if (k >= 256) {
              if (stateBitLen == 8) { iState |= mask; }
            } else if (k + 1 < 256) {
              atomicAdd(&(smem[k + 1]), 1);
            }
          }
        }
      }
      if (stateBitLen == 8) { state[thread_id + (num_threads * ii)] = iState; }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < 256; i += blockDim_x) {
      if (smem[i] > 0) { atomicAdd(&(count[i + (256 * j)]), smem[i]); }
    }
    cg::sync(grid);

    uint32_t csum[1];
    csum[0] = 0;
    if (threadIdx.x < 256) { csum[0] = count[threadIdx.x + (256 * j)]; }
    BlockScanT(temp_storage).InclusiveSum(csum, csum);

    if (threadIdx.x < 256) {
      if (count_below + csum[0] < topk) {
        uint32_t index = threadIdx.x;
        atomicMax(&(best_index[j]), index);
        atomicMax(&(best_csum[j]), csum[0]);
      }
    }
    __syncthreads();

    count_below += best_csum[j];
    threshold += (best_index[j] << shift);
  }

  //
  // Get labels that satifies "x[i] < threshold".
  //
  int ii = 0;
  for (int i = thread_id * vecLen; i < nx; i += num_threads * max(vecLen, stateBitLen), ii++) {
    uint8_t iState = 0;
    if (stateBitLen == 8) {
      iState = state[thread_id + (num_threads * ii)];
      if (iState == (uint8_t)0xff) continue;
    }
#pragma unroll
    for (int v = 0; v < max(vecLen, stateBitLen); v += vecLen) {
      int iv = i + (num_threads * v);
      if (iv >= nx) break;

      struct u16_vector vec;
      load_u16_vector<vecLen>(vec, x, iv);
#pragma unroll
      for (int u = 0; u < vecLen; u++) {
        int ivu = iv + u;
        if (ivu >= nx) break;

        uint8_t mask = (uint8_t)0x1 << (v + u);
        if ((stateBitLen == 8) && (iState & mask)) continue;
        uint32_t xi = get_element_from_u16_vector<vecLen>(vec, u);
        if (xi < threshold) {
          labels[atomicAdd(&count[0], 1)] = ivu;
        } else if ((xi == threshold) && (count_below + count[256] < topk)) {
          if (count_below + atomicAdd(&count[256], 1) < topk) {
            labels[atomicAdd(&count[0], 1)] = ivu;
          }
        }
      }
    }
  }

#ifdef CUANN_DEBUG
  cg::sync(grid);
  if (thread_id == 0 && count[0] < topk) {
    printf("# i_batch:%d, topk:%d, count[0]:%d, count_below:%d, threshold:%08x\n",
           i_batch,
           topk,
           count[0],
           count_below,
           threshold);
  }
#endif
}

//
template <int blockDim_x, int stateBitLen, int vecLen>
__launch_bounds__(1024, 2) __global__
  void kern_topk_cta_8(uint32_t topk,
                       uint32_t size_batch,
                       uint32_t max_len_x,
                       uint32_t* len_x,     // [size_batch, max_len_x,]
                       const uint16_t* _x,  // [size_batch, max_len_x,]
                       uint8_t* _state,     // [size_batch, max_len_x / 8,]
                       uint32_t* _labels    // [size_batch, topk,]
  )
{
  __shared__ uint32_t smem[256 + 6];
  uint32_t* best_index = &(smem[256]);
  uint32_t* best_csum  = &(smem[256 + 2]);
  uint32_t* count      = &(smem[256 + 4]);
  typedef BlockScan<uint32_t, blockDim_x> BlockScanT;
  __shared__ typename BlockScanT::TempStorage temp_storage;
  uint32_t i_batch = blockIdx.y;
  if (i_batch >= size_batch) return;

  uint32_t nx;
  if (len_x == NULL) {
    nx = max_len_x;
  } else {
    nx = len_x[i_batch];
  }

  uint32_t num_threads = blockDim_x;
  uint32_t thread_id   = threadIdx.x;

  const uint16_t* x = _x + (max_len_x * i_batch);
  uint8_t* state    = NULL;
  if (stateBitLen == 8) {
    uint32_t numSample_perThread = (max_len_x + num_threads - 1) / num_threads;
    uint32_t numState_perThread  = (numSample_perThread + stateBitLen - 1) / stateBitLen;
    state                        = _state + (numState_perThread * num_threads * i_batch);
  }
  uint32_t* labels = _labels + (topk * i_batch);
  if (threadIdx.x < 6) { smem[256 + threadIdx.x] = 0; }

  uint32_t count_below = 0;
  uint32_t threshold   = 0;

  //
  // Search for the maximum threshold that satisfies "(x < threshold).sum() < topk".
  //
  for (int j = 0; j < 2; j += 1) {
    uint32_t shift = (8 - 8 * j);
    for (int i = threadIdx.x; i < 256; i += blockDim_x) {
      smem[i] = 0;
    }
    __syncthreads();

    int ii = 0;
    for (int i = thread_id * vecLen; i < nx; i += num_threads * max(vecLen, stateBitLen), ii++) {
      uint8_t iState = 0;
      if (stateBitLen == 8 && j > 0) { iState = state[thread_id + (num_threads * ii)]; }
#pragma unroll
      for (int v = 0; v < max(vecLen, stateBitLen); v += vecLen) {
        int iv = i + (num_threads * v);
        if (iv >= nx) break;

        struct u16_vector x_vec;
        load_u16_vector<vecLen>(x_vec, x, iv);
#pragma unroll
        for (int u = 0; u < vecLen; u++) {
          int ivu = iv + u;
          if (ivu >= nx) break;

          uint8_t mask = (uint8_t)0x1 << (v + u);
          uint32_t xi  = get_element_from_u16_vector<vecLen>(x_vec, u);
          if (xi < threshold) {
            if (stateBitLen == 8) {
              labels[atomicAdd(&count[0], 1)] = ivu;
              iState |= mask;
            }
          } else {
            uint32_t k = (xi - threshold) >> shift;  // 0 <= k
            if (k >= 256) {
              if (stateBitLen == 8) { iState |= mask; }
            } else if (k + 1 < 256) {
              atomicAdd(&(smem[k + 1]), 1);
            }
          }
        }
      }
      if (stateBitLen == 8) { state[thread_id + (num_threads * ii)] = iState; }
    }
    __syncthreads();

    uint32_t csum[1];
    if (threadIdx.x < 256) { csum[0] = smem[threadIdx.x]; }
    BlockScanT(temp_storage).InclusiveSum(csum, csum);

    if (threadIdx.x < 256) {
      if (count_below + csum[0] < topk) {
        uint32_t index = threadIdx.x;
        atomicMax(&(best_index[j]), index);
        atomicMax(&(best_csum[j]), csum[0]);
      }
    }
    __syncthreads();

    count_below += best_csum[j];
    threshold += (best_index[j] << shift);
    if (count_below == topk) break;
  }

  //
  // Get labels that satifies "x[i] < threshold".
  //
  int ii = 0;
  for (int i = thread_id * vecLen; i < nx; i += num_threads * max(vecLen, stateBitLen), ii++) {
    uint8_t iState = 0;
    if (stateBitLen == 8) {
      iState = state[thread_id + (num_threads * ii)];
      if (iState == (uint8_t)0xff) continue;
    }
#pragma unroll
    for (int v = 0; v < max(vecLen, stateBitLen); v += vecLen) {
      int iv = i + (num_threads * v);
      if (iv >= nx) break;

      struct u16_vector vec;
      load_u16_vector<vecLen>(vec, x, iv);
#pragma unroll
      for (int u = 0; u < vecLen; u++) {
        int ivu = iv + u;
        if (ivu >= nx) break;

        uint8_t mask = (uint8_t)0x1 << (v + u);
        if ((stateBitLen == 8) && (iState & mask)) continue;
        uint32_t xi = get_element_from_u16_vector<vecLen>(vec, u);
        if (xi < threshold) {
          labels[atomicAdd(&count[0], 1)] = ivu;
        } else if ((xi == threshold) && (count_below + count[1] < topk)) {
          if (count_below + atomicAdd(&count[1], 1) < topk) {
            labels[atomicAdd(&count[0], 1)] = ivu;
          }
        }
      }
    }
  }

#ifdef CUANN_DEBUG
  __syncthreads();
  if (thread_id == 0 && count[0] < topk) {
    printf("# i_batch:%d, topk:%d, count[0]:%d, count_below:%d, threshold:%08x\n",
           i_batch,
           topk,
           count[0],
           count_below,
           threshold);
  }
#endif
}

//
__global__ void _sort_topk_prep(uint32_t sizeBatch,
                                uint32_t topK,
                                uint32_t maxSamples,
                                const uint32_t* labels,  // [sizeBatch, topK]
                                const float* samples,    // [sizeBatch, maxSamples]
                                int* offsets,            // [sizeBatch + 1]
                                float* outputs           // [sizeBatch, topK]
)
{
  uint32_t tid = threadIdx.x + (blockDim.x * blockIdx.x);
  if (tid < sizeBatch + 1) { offsets[tid] = tid * topK; }
  if (tid < sizeBatch * topK) {
    uint32_t label  = labels[tid];
    uint32_t iBatch = tid / topK;
    float value     = samples[label + (maxSamples * iBatch)];
    outputs[tid]    = value;
  }
}

//
size_t _cuann_find_topk_bufferSize(cuannHandle_t handle,
                                   uint32_t topK,
                                   uint32_t sizeBatch,
                                   uint32_t maxSamples,
                                   cudaDataType_t sampleDtype = CUDA_R_32F)
{
  constexpr int numThreads  = NUM_THREADS;
  constexpr int stateBitLen = STATE_BIT_LENGTH;
  assert(stateBitLen == 0 || stateBitLen == 8);

  size_t workspaceSize = 0;
  // count
  if (sampleDtype == CUDA_R_16F) {
    workspaceSize += _cuann_aligned(sizeof(uint32_t) * sizeBatch * 2 * 256);
  } else {
    workspaceSize += _cuann_aligned(sizeof(uint32_t) * sizeBatch * 5 * 1024);
  }
  // state
  if (stateBitLen == 8) {
    // (*) Each thread has at least one array element for state
    uint32_t numBlocks_perBatch =
      ((handle->deviceProp).multiProcessorCount * 2 + sizeBatch) / sizeBatch;
    uint32_t numThreads_perBatch = numThreads * numBlocks_perBatch;
    uint32_t numSample_perThread = (maxSamples + numThreads_perBatch - 1) / numThreads_perBatch;
    uint32_t numState_perThread  = (numSample_perThread + stateBitLen - 1) / stateBitLen;
    workspaceSize +=
      _cuann_aligned(sizeof(uint8_t) * numState_perThread * numThreads_perBatch * sizeBatch);
  }

  size_t workspaceSize2 = 0;
  // offsets
  workspaceSize2 += _cuann_aligned(sizeof(int) * (sizeBatch + 1));
  // keys_in, keys_out, values_out
  workspaceSize2 += _cuann_aligned(sizeof(float) * sizeBatch * topK);
  workspaceSize2 += _cuann_aligned(sizeof(float) * sizeBatch * topK);
  workspaceSize2 += _cuann_aligned(sizeof(uint32_t) * sizeBatch * topK);
  // cub_ws
  size_t cub_ws_size = 0;
  cub::DeviceSegmentedRadixSort::SortPairs(NULL,
                                           cub_ws_size,
                                           (float*)NULL,
                                           (float*)NULL,
                                           (uint32_t*)NULL,
                                           (uint32_t*)NULL,
                                           sizeBatch * topK,
                                           sizeBatch,
                                           (int*)NULL,
                                           (int*)NULL);
  workspaceSize2 += _cuann_aligned(cub_ws_size);
  workspaceSize = max(workspaceSize, workspaceSize2);

  return workspaceSize;
}

//
int _get_vecLen(uint32_t maxSamples, int maxVecLen = MAX_VEC_LENGTH)
{
  int vecLen = min(maxVecLen, MAX_VEC_LENGTH);
  while ((maxSamples % vecLen) != 0) {
    vecLen /= 2;
  }
  return vecLen;
}

//
void _cuann_find_topk(cuannHandle_t handle,
                      uint32_t topK,
                      uint32_t sizeBatch,
                      uint32_t maxSamples,
                      uint32_t* numSamples,  // [sizeBatch,]
                      const float* samples,  // [sizeBatch, maxSamples,]
                      uint32_t* labels,      // [sizeBatch, topK,]
                      void* workspace,
                      bool sort = false)
{
  constexpr int numThreads  = NUM_THREADS;
  constexpr int stateBitLen = STATE_BIT_LENGTH;
  assert(stateBitLen == 0 || stateBitLen == 8);
#ifdef CUANN_DEBUG
  cudaMemsetAsync(labels, 0xff, sizeof(uint32_t) * sizeBatch * topK, handle->stream);
#endif

  // Limit the maximum value of vecLen to 4. In the case of FP32,
  // setting vecLen = 8 in cg_kernel causes too much register usage.
  int vecLen = _get_vecLen(maxSamples, 4);
  void* cg_kernel;
  if (vecLen == 4) {
    cg_kernel = (void*)kern_topk_cg_11<numThreads, stateBitLen, 4>;
  } else if (vecLen == 2) {
    cg_kernel = (void*)kern_topk_cg_11<numThreads, stateBitLen, 2>;
  } else if (vecLen == 1) {
    cg_kernel = (void*)kern_topk_cg_11<numThreads, stateBitLen, 1>;
  }

  int numBlocksPerSm_topk;
  size_t dynamicSMemSize = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &numBlocksPerSm_topk, cg_kernel, numThreads, dynamicSMemSize);
  int numBlocks_perBatch = (maxSamples + (numThreads * vecLen) - 1) / (numThreads * vecLen);
  int numBlocks          = min(numBlocks_perBatch * sizeBatch,
                      (handle->deviceProp).multiProcessorCount * numBlocksPerSm_topk);
  numBlocks_perBatch     = max(numBlocks / sizeBatch, 1);
  if (maxSamples <= numThreads * 10) {
    // When number of sample is small, using multiple thread-blocks does not
    // improve performance, in which case cta_kernel is used. Tentatively,
    // "numThreads * 10" is used as the threshold, but this may be better
    // determined by auto-tuning, etc.
    numBlocks_perBatch = 1;
  }
  uint32_t* count = (uint32_t*)workspace;
  uint8_t* state  = NULL;
  if (stateBitLen == 8) {
    state = (uint8_t*)count + _cuann_aligned(sizeof(uint32_t) * sizeBatch * 5 * 1024);
  }

  dim3 threads(numThreads, 1, 1);
  dim3 blocks(numBlocks_perBatch, sizeBatch, 1);
  if (numBlocks_perBatch <= 1) {
    void (*cta_kernel)(
      uint32_t, uint32_t, uint32_t, uint32_t*, const uint32_t*, uint8_t*, uint32_t*);
    int vecLen = _get_vecLen(maxSamples);
    if (vecLen == 8) {
      cta_kernel = kern_topk_cta_11<numThreads, stateBitLen, 8>;
    } else if (vecLen == 4) {
      cta_kernel = kern_topk_cta_11<numThreads, stateBitLen, 4>;
    } else if (vecLen == 2) {
      cta_kernel = kern_topk_cta_11<numThreads, stateBitLen, 2>;
    } else if (vecLen == 1) {
      cta_kernel = kern_topk_cta_11<numThreads, stateBitLen, 1>;
    }
    cta_kernel<<<blocks, threads, 0, handle->stream>>>(
      topK, sizeBatch, maxSamples, numSamples, (const uint32_t*)samples, state, labels);
  } else {
    void* args[9];
    args[0] = {&(topK)};
    args[1] = {&(sizeBatch)};
    args[2] = {&(maxSamples)};
    args[3] = {&(numSamples)};
    args[4] = {&(samples)};
    args[5] = {&(state)};
    args[6] = {&(labels)};
    args[7] = {&(count)};
    args[8] = {nullptr};
    cudaLaunchCooperativeKernel((void*)cg_kernel, blocks, threads, args, 0, handle->stream);
  }
  if (!sort) { return; }

  // offsets: [sizeBatch + 1]
  // keys_in, keys_out, values_out: [sizeBatch, topK]
  int* offsets    = (int*)workspace;
  float* keys_in  = (float*)((uint8_t*)offsets + _cuann_aligned(sizeof(int) * (sizeBatch + 1)));
  float* keys_out = (float*)((uint8_t*)keys_in + _cuann_aligned(sizeof(float) * sizeBatch * topK));
  uint32_t* values_out =
    (uint32_t*)((uint8_t*)keys_out + _cuann_aligned(sizeof(float) * sizeBatch * topK));
  void* cub_ws =
    (void*)((uint8_t*)values_out + _cuann_aligned(sizeof(uint32_t) * sizeBatch * topK));

  dim3 stpThreads(128, 1, 1);
  dim3 stpBlocks((max(sizeBatch + 1, sizeBatch * topK) + stpThreads.x - 1) / stpThreads.x, 1, 1);
  _sort_topk_prep<<<stpBlocks, stpThreads, 0, handle->stream>>>(
    sizeBatch, topK, maxSamples, labels, samples, offsets, keys_in);

  size_t cub_ws_size = 0;
  cub::DeviceSegmentedRadixSort::SortPairs(NULL,
                                           cub_ws_size,
                                           keys_in,
                                           keys_out,
                                           labels,
                                           values_out,
                                           sizeBatch * topK,
                                           sizeBatch,
                                           offsets,
                                           offsets + 1);

  cub::DeviceSegmentedRadixSort::SortPairs(cub_ws,
                                           cub_ws_size,
                                           keys_in,
                                           keys_out,
                                           labels,
                                           values_out,
                                           sizeBatch * topK,
                                           sizeBatch,
                                           offsets,
                                           offsets + 1,
                                           (int)0,
                                           (int)(sizeof(float) * 8),
                                           handle->stream);

  cudaMemcpyAsync(labels,
                  values_out,
                  sizeof(uint32_t) * sizeBatch * topK,
                  cudaMemcpyDeviceToDevice,
                  handle->stream);
}

//
void _cuann_find_topk(cuannHandle_t handle,
                      uint32_t topK,
                      uint32_t sizeBatch,
                      uint32_t maxSamples,
                      uint32_t* numSamples,  // [sizeBatch,]
                      const half* samples,   // [sizeBatch, maxSamples,]
                      uint32_t* labels,      // [sizeBatch, topK,]
                      void* workspace,
                      bool sort = false)
{
  constexpr int numThreads  = NUM_THREADS;
  constexpr int stateBitLen = STATE_BIT_LENGTH;
  assert(stateBitLen == 0 || stateBitLen == 8);
#ifdef CUANN_DEBUG
  cudaMemsetAsync(labels, 0xff, sizeof(uint32_t) * sizeBatch * topK, handle->stream);
#endif

  int vecLen = _get_vecLen(maxSamples);
  void* cg_kernel;
  if (vecLen == 8) {
    cg_kernel = (void*)kern_topk_cg_8<numThreads, stateBitLen, 8>;
  } else if (vecLen == 4) {
    cg_kernel = (void*)kern_topk_cg_8<numThreads, stateBitLen, 4>;
  } else if (vecLen == 2) {
    cg_kernel = (void*)kern_topk_cg_8<numThreads, stateBitLen, 2>;
  } else if (vecLen == 1) {
    cg_kernel = (void*)kern_topk_cg_8<numThreads, stateBitLen, 1>;
  }

  int numBlocksPerSm_topk;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm_topk, cg_kernel, numThreads, 0);
  int numBlocks_perBatch = (maxSamples + (numThreads * vecLen) - 1) / (numThreads * vecLen);
  int numBlocks          = min(numBlocks_perBatch * sizeBatch,
                      (handle->deviceProp).multiProcessorCount * numBlocksPerSm_topk);
  numBlocks_perBatch     = max(numBlocks / sizeBatch, 1);
  if (maxSamples <= numThreads * 10) {
    // When number of sample is small, using multiple thread-blocks does not
    // improve performance, in which case cta_kernel is used. Tentatively,
    // "numThreads * 10" is used as the threshold, but this may be better
    // determined by auto-tuning, etc.
    numBlocks_perBatch = 1;
  }
  uint32_t* count = (uint32_t*)workspace;
  uint8_t* state  = NULL;
  if (stateBitLen == 8) {
    state = (uint8_t*)count + _cuann_aligned(sizeof(uint32_t) * sizeBatch * 2 * 256);
  }

  dim3 threads(numThreads, 1, 1);
  dim3 blocks(numBlocks_perBatch, sizeBatch, 1);
  if (numBlocks_perBatch <= 1) {
    void (*cta_kernel)(
      uint32_t, uint32_t, uint32_t, uint32_t*, const uint16_t*, uint8_t*, uint32_t*);
    int vecLen = _get_vecLen(maxSamples);
    if (vecLen == 8) {
      cta_kernel = kern_topk_cta_8<numThreads, stateBitLen, 8>;
    } else if (vecLen == 4) {
      cta_kernel = kern_topk_cta_8<numThreads, stateBitLen, 4>;
    } else if (vecLen == 2) {
      cta_kernel = kern_topk_cta_8<numThreads, stateBitLen, 2>;
    } else if (vecLen == 1) {
      cta_kernel = kern_topk_cta_8<numThreads, stateBitLen, 1>;
    }
    cta_kernel<<<blocks, threads, 0, handle->stream>>>(
      topK, sizeBatch, maxSamples, numSamples, (const uint16_t*)samples, state, labels);
  } else {
    void* args[9];
    args[0] = {&(topK)};
    args[1] = {&(sizeBatch)};
    args[2] = {&(maxSamples)};
    args[3] = {&(numSamples)};
    args[4] = {&(samples)};
    args[5] = {&(state)};
    args[6] = {&(labels)};
    args[7] = {&(count)};
    args[8] = {nullptr};
    cudaLaunchCooperativeKernel((void*)cg_kernel, blocks, threads, args, 0, handle->stream);
  }
}

/**
 *
 * End of topk
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 * Start of ivfpq
 */

//
size_t ivfpq_search_bufferSize(cuannHandle_t handle, cuannIvfPqDescriptor_t desc);

// search
template <typename scoreDtype, typename smemLutDtype>
void ivfpq_search(cuannHandle_t handle,
                  cuannIvfPqDescriptor_t desc,
                  uint32_t numQueries,
                  const float* clusterCenters,           // [numDataset, dimDataset]
                  const float* pqCenters,                // [dimPq, 256, lenPq]
                  const uint8_t* pqDataset,              // [numDataset, dimPq]
                  const uint32_t* originalNumbers,       // [numDataset]
                  const uint32_t* indexPtr,              // [numClusters + 1]
                  const uint32_t* clusterLabelsToProbe,  // [numQueries, numProbes]
                  const float* query,                    // [dimDataset]
                  uint64_t* topKNeighbors,               // [topK]
                  float* topKDistances,                  // [topK]
                  void* workspace);

void ivfpq_encode(uint32_t numDataset,
                  uint32_t ldDataset,  // (*) ldDataset >= numDataset
                  uint32_t dimPq,
                  uint32_t bitPq,         // 4 <= bitPq <= 8
                  const uint32_t* label,  // [dimPq, ldDataset]
                  uint8_t* output         // [numDataset, dimPq]
);

//
bool manage_local_topk(cuannIvfPqDescriptor_t desc);
size_t get_sizeSmemForLocalTopk(cuannIvfPqDescriptor_t desc, int numThreads);

//
__global__ void ivfpq_init_topkScores(float* topkScores,  // [num,]
                                      float initValue,
                                      uint32_t num);

//
__global__ void ivfpq_prep_sort(uint32_t numElement, uint32_t* indexList);

//
__global__ void ivfpq_make_chunk_index_ptr(
  uint32_t numProbes,
  uint32_t sizeBatch,
  const uint32_t* indexPtr,               // [numClusters + 1,]
  const uint32_t* _clusterLabelsToProbe,  // [sizeBatch, numProbes,]
  uint32_t* _chunkIndexPtr,               // [sizeBetch, numProbes,]
  uint32_t* numSamples                    // [sizeBatch,]
);

//
template <typename scoreDtype>
__global__ void ivfpq_make_outputs(uint32_t numProbes,
                                   uint32_t topk,
                                   uint32_t maxSamples,
                                   uint32_t sizeBatch,
                                   const uint32_t* clusterIndexPtr,  // [numClusters + 1]
                                   const uint32_t* originalNumbers,  // [numDataset]
                                   const uint32_t* clusterLabels,    // [sizeBatch, numProbes]
                                   const uint32_t* chunkIndexPtr,    // [sizeBatch, numProbes]
                                   const scoreDtype* scores,         // [sizeBatch, maxSamples] or
                                                                     // [sizeBatch, numProbes, topk]
                                   const uint32_t* scoreTopkIndex,   // [sizeBatch, numProbes, topk]
                                   const uint32_t* topkSampleIds,    // [sizeBatch, topk]
                                   uint64_t* topkNeighbors,          // [sizeBatch, topk]
                                   float* topkScores                 // [sizeBatch, topk]
);

//
__device__ inline uint32_t warp_scan(uint32_t x)
{
  uint32_t y;
  y = __shfl_up_sync(0xffffffff, x, 1);
  if (threadIdx.x % 32 >= 1) x += y;
  y = __shfl_up_sync(0xffffffff, x, 2);
  if (threadIdx.x % 32 >= 2) x += y;
  y = __shfl_up_sync(0xffffffff, x, 4);
  if (threadIdx.x % 32 >= 4) x += y;
  y = __shfl_up_sync(0xffffffff, x, 8);
  if (threadIdx.x % 32 >= 8) x += y;
  y = __shfl_up_sync(0xffffffff, x, 16);
  if (threadIdx.x % 32 >= 16) x += y;
  return x;
}

//
__device__ inline uint32_t thread_block_scan(uint32_t x, uint32_t* smem)
{
  x = warp_scan(x);
  __syncthreads();
  if (threadIdx.x % 32 == 31) { smem[threadIdx.x / 32] = x; }
  __syncthreads();
  if (threadIdx.x < 32) { smem[threadIdx.x] = warp_scan(smem[threadIdx.x]); }
  __syncthreads();
  if (threadIdx.x / 32 > 0) { x += smem[threadIdx.x / 32 - 1]; }
  __syncthreads();
  return x;
}

//
__global__ void ivfpq_make_chunk_index_ptr(
  uint32_t numProbes,
  uint32_t sizeBatch,
  const uint32_t* indexPtr,               // [numClusters + 1,]
  const uint32_t* _clusterLabelsToProbe,  // [sizeBatch, numProbes,]
  uint32_t* _chunkIndexPtr,               // [sizeBetch, numProbes,]
  uint32_t* numSamples                    // [sizeBatch,]
)
{
  __shared__ uint32_t smem_temp[32];
  __shared__ uint32_t smem_base[2];

  uint32_t iBatch = blockIdx.x;
  if (iBatch >= sizeBatch) return;
  const uint32_t* clusterLabelsToProbe = _clusterLabelsToProbe + (numProbes * iBatch);
  uint32_t* chunkIndexPtr              = _chunkIndexPtr + (numProbes * iBatch);

  //
  uint32_t j_end = (numProbes + 1024 - 1) / 1024;
  for (uint32_t j = 0; j < j_end; j++) {
    uint32_t i   = threadIdx.x + (1024 * j);
    uint32_t val = 0;
    if (i < numProbes) {
      uint32_t l = clusterLabelsToProbe[i];
      val        = indexPtr[l + 1] - indexPtr[l];
    }
    val = thread_block_scan(val, smem_temp);

    if (i < numProbes) {
      if (j > 0) { val += smem_base[(j - 1) & 0x1]; }
      chunkIndexPtr[i] = val;
      if (i == numProbes - 1) { numSamples[iBatch] = val; }
    }

    if ((j < j_end - 1) && (threadIdx.x == 1023)) { smem_base[j & 0x1] = val; }
  }
}

//
__global__ void ivfpq_init_topkScores(float* topkScores,  // [num,]
                                      float initValue,
                                      uint32_t num)
{
  uint32_t i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i >= num) return;
  topkScores[i] = initValue;
}

//
__global__ void ivfpq_prep_sort(uint32_t numElement, uint32_t* indexList)
{
  uint32_t i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i >= numElement) return;
  indexList[i] = i;
}

//
__device__ inline void ivfpq_get_id_dataset(uint32_t iSample,
                                            uint32_t numProbes,
                                            const uint32_t* clusterIndexPtr,  // [numClusters + 1,]
                                            const uint32_t* clusterLabels,    // [numProbes,]
                                            const uint32_t* chunkIndexPtr,    // [numProbes,]
                                            uint32_t& iChunk,
                                            uint32_t& label,
                                            uint32_t& iDataset)
{
  uint32_t minChunk = 0;
  uint32_t maxChunk = numProbes - 1;
  iChunk            = (minChunk + maxChunk) / 2;
  while (minChunk < maxChunk) {
    if (iSample >= chunkIndexPtr[iChunk]) {
      minChunk = iChunk + 1;
    } else {
      maxChunk = iChunk;
    }
    iChunk = (minChunk + maxChunk) / 2;
  }

  label                   = clusterLabels[iChunk];
  uint32_t iSampleInChunk = iSample;
  if (iChunk > 0) { iSampleInChunk -= chunkIndexPtr[iChunk - 1]; }
  iDataset = iSampleInChunk + clusterIndexPtr[label];
}

//
template <typename scoreDtype>
__global__ void ivfpq_make_outputs(uint32_t numProbes,
                                   uint32_t topk,
                                   uint32_t maxSamples,
                                   uint32_t sizeBatch,
                                   const uint32_t* clusterIndexPtr,  // [numClusters + 1]
                                   const uint32_t* originalNumbers,  // [numDataset]
                                   const uint32_t* clusterLabels,    // [sizeBatch, numProbes]
                                   const uint32_t* chunkIndexPtr,    // [sizeBatch, numProbes]
                                   const scoreDtype* scores,         // [sizeBatch, maxSamples] or
                                                                     // [sizeBatch, numProbes, topk]
                                   const uint32_t* scoreTopkIndex,   // [sizeBatch, numProbes, topk]
                                   const uint32_t* topkSampleIds,    // [sizeBatch, topk]
                                   uint64_t* topkNeighbors,          // [sizeBatch, topk]
                                   float* topkScores                 // [sizeBatch, topk]
)
{
  uint32_t i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i >= topk) return;
  uint32_t iBatch = blockIdx.y;
  if (iBatch >= sizeBatch) return;

  uint32_t iSample = topkSampleIds[i + (topk * iBatch)];
  if (scoreTopkIndex == NULL) {
    // 0 <= iSample < maxSamples
    topkScores[i + (topk * iBatch)] = scores[iSample + (maxSamples * iBatch)];
    uint32_t iChunk;
    uint32_t label;
    uint32_t iDataset;
    ivfpq_get_id_dataset(iSample,
                         numProbes,
                         clusterIndexPtr,
                         clusterLabels + (numProbes * iBatch),
                         chunkIndexPtr + (numProbes * iBatch),
                         iChunk,
                         label,
                         iDataset);
    topkNeighbors[i + (topk * iBatch)] = originalNumbers[iDataset];
  } else {
    // 0 <= iSample < (numProbes * topk)
    topkScores[i + (topk * iBatch)]    = scores[iSample + ((numProbes * topk) * iBatch)];
    uint32_t iDataset                  = scoreTopkIndex[iSample + ((numProbes * topk) * iBatch)];
    topkNeighbors[i + (topk * iBatch)] = originalNumbers[iDataset];
  }
}

//
bool manage_local_topk(cuannIvfPqDescriptor_t desc)
{
  int depth = (desc->topK + 31) / 32;
  if (depth > 4) { return false; }
  if (desc->numProbes < 16) { return false; }
  if (desc->maxBatchSize * desc->numProbes < 256) { return false; }
  return true;
}

//
size_t get_sizeSmemForLocalTopk(cuannIvfPqDescriptor_t desc, int numThreads)
{
  if (manage_local_topk(desc)) {
    int topk_32 = (desc->topK + 31) / 32;
    return (sizeof(float) + sizeof(uint32_t)) * (numThreads / 2) * topk_32;
  }
  return 0;
}

// return workspace size
size_t ivfpq_search_bufferSize(cuannHandle_t handle, cuannIvfPqDescriptor_t desc)
{
  size_t size = 0;
  // clusterLabelsOut  [maxBatchSize, numProbes]
  size += _cuann_aligned(sizeof(uint32_t) * desc->maxBatchSize * desc->numProbes);
  // indexList  [maxBatchSize * numProbes]
  size += _cuann_aligned(sizeof(uint32_t) * desc->maxBatchSize * desc->numProbes);
  // indexListSorted  [maxBatchSize * numProbes]
  size += _cuann_aligned(sizeof(uint32_t) * desc->maxBatchSize * desc->numProbes);
  // numSamples  [maxBatchSize,]
  size += _cuann_aligned(sizeof(uint32_t) * desc->maxBatchSize);
  // cubWorkspace
  void* d_temp_storage      = NULL;
  size_t temp_storage_bytes = 0;
  uint32_t* d_keys_in       = NULL;
  uint32_t* d_keys_out      = NULL;
  uint32_t* d_values_in     = NULL;
  uint32_t* d_values_out    = NULL;
  cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                  temp_storage_bytes,
                                  d_keys_in,
                                  d_keys_out,
                                  d_values_in,
                                  d_values_out,
                                  desc->maxBatchSize * desc->numProbes);
  desc->sizeCubWorkspace = _cuann_aligned(temp_storage_bytes);
  size += desc->sizeCubWorkspace;
  // chunkIndexPtr  [maxBatchSize, numProbes]
  size += _cuann_aligned(sizeof(uint32_t) * desc->maxBatchSize * desc->numProbes);
  // topkSids  [maxBatchSize, topk]
  size += _cuann_aligned(sizeof(uint32_t) * desc->maxBatchSize * desc->topK);
  // similarity
  size_t unit_size = sizeof(float);
  if (desc->internalDistanceDtype == CUDA_R_16F) { unit_size = sizeof(half); }
  if (manage_local_topk(desc)) {
    // [matBatchSize, numProbes, topK]
    size += _cuann_aligned(unit_size * desc->maxBatchSize * desc->numProbes * desc->topK);
  } else {
    // [matBatchSize, maxSamples]
    size += _cuann_aligned(unit_size * desc->maxBatchSize * desc->maxSamples);
  }
  // simTopkIndex
  if (manage_local_topk(desc)) {
    // [matBatchSize, numProbes, topk]
    size += _cuann_aligned(sizeof(uint32_t) * desc->maxBatchSize * desc->numProbes * desc->topK);
  }
  // topkScores
  if (manage_local_topk(desc)) {
    // [maxBatchSize, topk]
    size += _cuann_aligned(sizeof(float) * desc->maxBatchSize * desc->topK);
  }
  // preCompScores  [multiProcessorCount, dimPq, 1 << bitPq,]
  size += _cuann_aligned(sizeof(float) * (handle->deviceProp).multiProcessorCount * desc->dimPq *
                         (1 << desc->bitPq));
  // topkWorkspace
  if (manage_local_topk(desc)) {
    size += _cuann_find_topk_bufferSize(handle,
                                        desc->topK,
                                        desc->maxBatchSize,
                                        desc->numProbes * desc->topK,
                                        desc->internalDistanceDtype);
  } else {
    size += _cuann_find_topk_bufferSize(
      handle, desc->topK, desc->maxBatchSize, desc->maxSamples, desc->internalDistanceDtype);
  }
  return size;
}

//
__device__ __host__ inline void ivfpq_encode_core(
  uint32_t ldDataset, uint32_t dimPq, uint32_t bitPq, const uint32_t* label, uint8_t* output)
{
  for (uint32_t j = 0; j < dimPq; j++) {
    uint8_t code = label[(ldDataset * j)];
    if (bitPq == 8) {
      uint8_t* ptrOutput = output + j;
      ptrOutput[0]       = code;
    } else if (bitPq == 7) {
      uint8_t* ptrOutput = output + 7 * (j / 8);
      if (j % 8 == 0) {
        ptrOutput[0] |= code;
      } else if (j % 8 == 1) {
        ptrOutput[0] |= code << 7;
        ptrOutput[1] |= code >> 1;
      } else if (j % 8 == 2) {
        ptrOutput[1] |= code << 6;
        ptrOutput[2] |= code >> 2;
      } else if (j % 8 == 3) {
        ptrOutput[2] |= code << 5;
        ptrOutput[3] |= code >> 3;
      } else if (j % 8 == 4) {
        ptrOutput[3] |= code << 4;
        ptrOutput[4] |= code >> 4;
      } else if (j % 8 == 5) {
        ptrOutput[4] |= code << 3;
        ptrOutput[5] |= code >> 5;
      } else if (j % 8 == 6) {
        ptrOutput[5] |= code << 2;
        ptrOutput[6] |= code >> 6;
      } else if (j % 8 == 7) {
        ptrOutput[6] |= code << 1;
      }
    } else if (bitPq == 6) {
      uint8_t* ptrOutput = output + 3 * (j / 4);
      if (j % 4 == 0) {
        ptrOutput[0] |= code;
      } else if (j % 4 == 1) {
        ptrOutput[0] |= code << 6;
        ptrOutput[1] |= code >> 2;
      } else if (j % 4 == 2) {
        ptrOutput[1] |= code << 4;
        ptrOutput[2] |= code >> 4;
      } else if (j % 4 == 3) {
        ptrOutput[2] |= code << 2;
      }
    } else if (bitPq == 5) {
      uint8_t* ptrOutput = output + 5 * (j / 8);
      if (j % 8 == 0) {
        ptrOutput[0] |= code;
      } else if (j % 8 == 1) {
        ptrOutput[0] |= code << 5;
        ptrOutput[1] |= code >> 3;
      } else if (j % 8 == 2) {
        ptrOutput[1] |= code << 2;
      } else if (j % 8 == 3) {
        ptrOutput[1] |= code << 7;
        ptrOutput[2] |= code >> 1;
      } else if (j % 8 == 4) {
        ptrOutput[2] |= code << 4;
        ptrOutput[3] |= code >> 4;
      } else if (j % 8 == 5) {
        ptrOutput[3] |= code << 1;
      } else if (j % 8 == 6) {
        ptrOutput[3] |= code << 6;
        ptrOutput[4] |= code >> 2;
      } else if (j % 8 == 7) {
        ptrOutput[4] |= code << 3;
      }
    } else if (bitPq == 4) {
      uint8_t* ptrOutput = output + (j / 2);
      if (j % 2 == 0) {
        ptrOutput[0] |= code;
      } else {
        ptrOutput[0] |= code << 4;
      }
    }
  }
}

//
__global__ void ivfpq_encode_kernel(uint32_t numDataset,
                                    uint32_t ldDataset,  // (*) ldDataset >= numDataset
                                    uint32_t dimPq,
                                    uint32_t bitPq,         // 4 <= bitPq <= 8
                                    const uint32_t* label,  // [dimPq, ldDataset]
                                    uint8_t* output         // [numDataset, dimPq]
)
{
  uint32_t i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i >= numDataset) return;
  ivfpq_encode_core(ldDataset, dimPq, bitPq, label + i, output + (dimPq * bitPq / 8) * i);
}

//
void ivfpq_encode(uint32_t numDataset,
                  uint32_t ldDataset,  // (*) ldDataset >= numDataset
                  uint32_t dimPq,
                  uint32_t bitPq,         // 4 <= bitPq <= 8
                  const uint32_t* label,  // [dimPq, ldDataset]
                  uint8_t* output         // [numDataset, dimPq]
)
{
#if 1
  // GPU
  dim3 iekThreads(128, 1, 1);
  dim3 iekBlocks((numDataset + iekThreads.x - 1) / iekThreads.x, 1, 1);
  ivfpq_encode_kernel<<<iekBlocks, iekThreads>>>(
    numDataset, ldDataset, dimPq, bitPq, label, output);
#else
  // CPU
  cudaDeviceSynchronize();
  for (uint32_t i = 0; i < numDataset; i++) {
    ivfpq_encode_core(ldDataset, dimPq, bitPq, label + i, output + (dimPq * bitPq / 8) * i);
  }
#endif
}

//
template __global__ void ivfpq_make_outputs<float>(
  uint32_t numProbes,
  uint32_t topk,
  uint32_t maxSamples,
  uint32_t sizeBatch,
  const uint32_t* clusterIndexPtr,  // [numClusters + 1]
  const uint32_t* originalNumbers,  // [numDataset]
  const uint32_t* clusterLabels,    // [sizeBatch, numProbes]
  const uint32_t* chunkIndexPtr,    // [sizeBatch, numProbes]
  const float* scores,              // [sizeBatch, maxSamples] or
                                    // [sizeBatch, numProbes, topk]
  const uint32_t* scoreTopkIndex,   // [sizeBatch, numProbes, topk]
  const uint32_t* topkSampleIds,    // [sizeBatch, topk]
  uint64_t* topkNeighbors,          // [sizeBatch, topk]
  float* topkScores                 // [sizeBatch, topk]
);

//
template __global__ void ivfpq_make_outputs<half>(
  uint32_t numProbes,
  uint32_t topk,
  uint32_t maxSamples,
  uint32_t sizeBatch,
  const uint32_t* clusterIndexPtr,  // [numClusters + 1]
  const uint32_t* originalNumbers,  // [numDataset]
  const uint32_t* clusterLabels,    // [sizeBatch, numProbes]
  const uint32_t* chunkIndexPtr,    // [sizeBatch, numProbes]
  const half* scores,               // [sizeBatch, maxSamples] or
                                    // [sizeBatch, numProbes, topk]
  const uint32_t* scoreTopkIndex,   // [sizeBatch, numProbes, topk]
  const uint32_t* topkSampleIds,    // [sizeBatch, topk]
  uint64_t* topkNeighbors,          // [sizeBatch, topk]
  float* topkScores                 // [sizeBatch, topk]
);

/**
 * End of ivfpq
 *
 *
 *
 *
 */

cuannStatus_t cuannCreate(cuannHandle_t* handle);
cuannStatus_t cuannDestroy(cuannHandle_t handle);
cuannStatus_t cuannSetStream(cuannHandle_t handle, cudaStream_t stream);
cuannStatus_t cuannSetDevice(cuannHandle_t handle, int devId);

cuannStatus_t cuannIvfPqCreateDescriptor(cuannIvfPqDescriptor_t* desc);
cuannStatus_t cuannIvfPqDestroyDescriptor(cuannIvfPqDescriptor_t desc);

cuannStatus_t cuannIvfPqSetIndexParameters(
  cuannIvfPqDescriptor_t desc,
  const uint32_t numClusters, /* Number of clusters */
  const uint32_t numDataset,  /* Number of dataset entries */
  const uint32_t dimDataset,  /* Dimension of each entry */
  const uint32_t dimPq,       /* Dimension of each entry after product quantization */
  const uint32_t bitPq,       /* Bit length of PQ */
  const cuannSimilarity_t similarity,
  const cuannPqCenter_t typePqCenter);

cuannStatus_t cuannIvfPqGetIndexParameters(cuannIvfPqDescriptor_t desc,
                                           uint32_t* numClusters,
                                           uint32_t* numDataset,
                                           uint32_t* dimDataset,
                                           uint32_t* dimPq,
                                           uint32_t* bitPq,
                                           cuannSimilarity_t* similarity,
                                           cuannPqCenter_t* typePqCenter);

cuannStatus_t cuannIvfPqGetIndexSize(cuannIvfPqDescriptor_t desc,
                                     size_t* size /* bytes of dataset index */);

cuannStatus_t cuannIvfPqBuildIndex(
  cuannHandle_t handle,
  cuannIvfPqDescriptor_t desc,
  const void* dataset,  /* [numDataset, dimDataset] */
  const void* trainset, /* [numTrainset, dimDataset] */
  cudaDataType_t dtype,
  uint32_t numTrainset,        /* Number of train-set entries */
  uint32_t numIterations,      /* Number of iterations to train kmeans */
  bool randomRotation,         /* If true, rotate vectors with randamly created rotation matrix */
  bool hierarchicalClustering, /* If true, do kmeans training hierarchically */
  void* index /* database index to build */);

cuannStatus_t cuannIvfPqSaveIndex(cuannHandle_t handle,
                                  cuannIvfPqDescriptor_t desc,
                                  const void* index,
                                  const char* fileName);

cuannStatus_t cuannIvfPqLoadIndex(cuannHandle_t handle,
                                  cuannIvfPqDescriptor_t desc,
                                  void** index,
                                  const char* fileName);

cuannStatus_t cuannIvfPqCreateNewIndexByAddingVectorsToOldIndex(
  cuannHandle_t handle,
  const char* oldIndexFileName,
  const char* newIndexFileName,
  const void* newVectors, /* [numVectorsToAdd, dimDataset] */
  uint32_t numNewVectors);

cuannStatus_t cuannIvfPqSetSearchParameters(
  cuannIvfPqDescriptor_t desc,
  const uint32_t numProbes, /* Number of clusters to probe */
  const uint32_t topK);     /* Number of search results */

cuannStatus_t cuannIvfPqSetSearchTuningParameters(cuannIvfPqDescriptor_t desc,
                                                  cudaDataType_t internalDistanceDtype,
                                                  cudaDataType_t smemLutDtype,
                                                  const uint32_t preferredThreadBlockSize);

cuannStatus_t cuannIvfPqGetSearchParameters(cuannIvfPqDescriptor_t desc,
                                            uint32_t* numProbes,
                                            uint32_t* topK);

cuannStatus_t cuannIvfPqGetSearchTuningParameters(cuannIvfPqDescriptor_t desc,
                                                  cudaDataType_t* internalDistanceDtype,
                                                  cudaDataType_t* smemLutDtype,
                                                  uint32_t* preferredThreadBlockSize);

cuannStatus_t cuannIvfPqSearch_bufferSize(cuannHandle_t handle,
                                          cuannIvfPqDescriptor_t desc,
                                          const void* index,
                                          uint32_t numQueries,
                                          size_t maxWorkspaceSize,
                                          size_t* workspaceSize);

cuannStatus_t cuannIvfPqSearch(cuannHandle_t handle,
                               cuannIvfPqDescriptor_t desc,
                               const void* index,
                               const void* queries, /* [numQueries, dimDataset] */
                               cudaDataType_t dtype,
                               uint32_t numQueries,
                               uint64_t* neighbors, /* [numQueries, topK] */
                               float* distances,    /* [numQueries, topK] */
                               void* workspace);

cuannStatus_t cuannPostprocessingRefine(uint32_t numDataset,
                                        uint32_t numQueries,
                                        uint32_t dimDataset,
                                        const void* dataset, /* [numDataset, dimDataset] */
                                        const void* queries, /* [numQueries, dimDataset] */
                                        cudaDataType_t dtype,
                                        cuannSimilarity_t similarity,
                                        uint32_t topK,
                                        const uint64_t* neighbors, /* [numQueries, topK] */
                                        uint32_t refinedTopK,
                                        uint64_t* refinedNeighbors, /* [numQueries, refinedTopK] */
                                        float* refinedDistances     /* [numQueries, refinedTopK] */
);

cuannStatus_t cuannPostprocessingMerge(
  uint32_t numSplit,
  uint32_t numQueries,
  uint32_t topK,
  const uint32_t* eachNumDataset, /* [numSplit] */
  const uint64_t* eachNeighbors,  /* [numSplit, numQueries, topK] */
  const float* eachDistances,     /* [numSplit, numQueries, topK] */
  uint64_t* neighbors,            /* [numQueries, topK] */
  float* distances                /* [numQueries, topK] */
);

size_t _cuann_getIndexSize_clusterCenters(cuannIvfPqDescriptor_t desc)
{
  // [numClusters, dimDatasetExt]
  return _cuann_aligned(sizeof(float) * desc->numClusters * desc->dimDatasetExt);
}

size_t _cuann_getIndexSize_pqCenters(cuannIvfPqDescriptor_t desc)
{
  size_t size_base = sizeof(float) * (1 << desc->bitPq) * desc->lenPq;
  if (desc->typePqCenter == CUANN_PQ_CENTER_PER_SUBSPACE) {
    // [dimPq, 1 << bitPq, lenPq]
    return _cuann_aligned(desc->dimPq * size_base);
  } else {
    // [numClusters, 1 << bitPq, lenPq]
    return _cuann_aligned(desc->numClusters * size_base);
  }
}

size_t _cuann_getIndexSize_pqDataset(cuannIvfPqDescriptor_t desc)
{
  // [numDataset, dimPq * bitPq / 8]
  return _cuann_aligned(sizeof(uint8_t) * desc->numDataset * desc->dimPq * desc->bitPq / 8);
}

size_t _cuann_getIndexSize_originalNumbers(cuannIvfPqDescriptor_t desc)
{
  // [numDataset,]
  return _cuann_aligned(sizeof(uint32_t) * desc->numDataset);
}

size_t _cuann_getIndexSize_indexPtr(cuannIvfPqDescriptor_t desc)
{
  // [numClusters + 1,]
  return _cuann_aligned(sizeof(uint32_t) * (desc->numClusters + 1));
}

size_t _cuann_getIndexSize_rotationMatrix(cuannIvfPqDescriptor_t desc)
{
  // [dimDataset, dimRotDataset]
  return _cuann_aligned(sizeof(float) * desc->dimDataset * desc->dimRotDataset);
}

size_t _cuann_getIndexSize_clusterRotCenters(cuannIvfPqDescriptor_t desc)
{
  // [numClusters, dimRotDataset]
  return _cuann_aligned(sizeof(float) * desc->numClusters * desc->dimRotDataset);
}

void _cuann_get_index_pointers(cuannIvfPqDescriptor_t desc,
                               const void* index,
                               struct cuannIvfPqIndexHeader** header,
                               float** clusterCenters,      // [numClusters, dimDatasetExt]
                               float** pqCenters,           // [dimPq, 1 << bitPq, lenPq], or
                                                            // [numClusters, 1 << bitPq, lenPq]
                               uint8_t** pqDataset,         // [numDataset, dimPq * bitPq / 8]
                               uint32_t** originalNumbers,  // [numDataset]
                               uint32_t** indexPtr,         // [numClusters + 1]
                               float** rotationMatrix,      // [dimDataset, dimRotDataset]
                               float** clusterRotCenters    // [numClusters, dimRotDataset]
)
{
  *header         = (struct cuannIvfPqIndexHeader*)index;
  *clusterCenters = (float*)((uint8_t*)(*header) + sizeof(struct cuannIvfPqIndexHeader));
  *pqCenters = (float*)((uint8_t*)(*clusterCenters) + _cuann_getIndexSize_clusterCenters(desc));
  *pqDataset = (uint8_t*)((uint8_t*)(*pqCenters) + _cuann_getIndexSize_pqCenters(desc));
  *originalNumbers = (uint32_t*)((uint8_t*)(*pqDataset) + _cuann_getIndexSize_pqDataset(desc));
  *indexPtr = (uint32_t*)((uint8_t*)(*originalNumbers) + _cuann_getIndexSize_originalNumbers(desc));
  *rotationMatrix = (float*)((uint8_t*)(*indexPtr) + _cuann_getIndexSize_indexPtr(desc));
  *clusterRotCenters =
    (float*)((uint8_t*)(*rotationMatrix) + _cuann_getIndexSize_rotationMatrix(desc));
}

__global__ void kern_get_cluster_size(uint32_t numClusters,
                                      const uint32_t* indexPtr,  // [numClusters + 1,]
                                      uint32_t* clusterSize      // [numClusters,]
)
{
  uint32_t i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i >= numClusters) return;
  clusterSize[i] = indexPtr[i + 1] - indexPtr[i];
}

template <typename T>
int descending(const void* a, const void* b)
{
  T valA = ((T*)a)[0];
  T valB = ((T*)b)[0];
  if (valA > valB) return -1;
  if (valA < valB) return 1;
  return 0;
}

// (*) This is temporal. Need to be removed in future.
void _cuann_get_random_norm_vector(int len, float* vector)
{
  float sqsum = 0.0;
  for (int i = 0; i < len; i++) {
    vector[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0;
    sqsum += vector[i] * vector[i];
  }
  float norm = sqrt(sqsum);
  for (int i = 0; i < len; i++) {
    vector[i] /= norm;
  }
}

void _cuann_get_inclusiveSumSortedClusterSize(
  cuannIvfPqDescriptor_t desc,
  const uint32_t* indexPtr,  // [numClusters + 1]
  float* clusterCenters,     // [numClusters, dimDatasetExt]
  uint32_t** output          // [numClusters]
)
{
  // [CPU]
  *output                 = (uint32_t*)malloc(sizeof(uint32_t) * desc->numClusters);
  desc->_numClustersSize0 = 0;
  for (int i = 0; i < desc->numClusters; i++) {
    (*output)[i] = indexPtr[i + 1] - indexPtr[i];
    if ((*output)[i] > 0) continue;

    desc->_numClustersSize0 += 1;
    // Work-around for clusters of size 0
#if 0
        printf("# i:%d, %u ... ", i, (*output)[i]);
        for (int j = 0; j < desc->dimDatasetExt; j++) {
            printf( "%.3f, ", clusterCenters[ j + (desc->dimDatasetExt * i) ] );
        }
        printf( "\n" );
#endif
    _cuann_get_random_norm_vector(desc->dimDatasetExt, clusterCenters + (desc->dimDatasetExt * i));
#if 0
        printf("# i:%d, %u ... ", i, (*output)[i]);
        for (int j = 0; j < desc->dimDatasetExt; j++) {
            printf( "%.3f, ", clusterCenters[ j + (desc->dimDatasetExt * i) ] );
        }
        printf( "\n" );
#endif
  }
  if (1 || desc->_numClustersSize0 > 0) {
    fprintf(stderr, "# num clusters of size 0: %d\n", desc->_numClustersSize0);
  }
  // sort
  qsort(*output, desc->numClusters, sizeof(uint32_t), descending<uint32_t>);
  // scan
  for (int i = 1; i < desc->numClusters; i++) {
    (*output)[i] += (*output)[i - 1];
  }
  assert((*output)[desc->numClusters - 1] == desc->numDataset);
}

void _cuann_get_sqsumClusters(cuannIvfPqDescriptor_t desc,
                              const float* clusterCenters,  // [numClusters, dimDataset,]
                              float** output                // [numClusters,]
)
{
  cudaError_t cudaError;
  if (*output != NULL) { cudaFree(*output); }
  cudaError = cudaMallocManaged(output, sizeof(float) * desc->numClusters);
  if (cudaError != cudaSuccess) {
    fprintf(stderr, "(%s, %d) cudaMallocManaged() failed.\n", __func__, __LINE__);
    exit(-1);
  }
  _cuann_sqsum(desc->numClusters, desc->dimDataset, clusterCenters, *output);
}

//
template <typename T>
T _cuann_dot(int n, const T* x, int incX, const T* y, int incY)
{
  T val = 0;
  for (int i = 0; i < n; i++) {
    val += x[incX * i] * y[incY * i];
  }
  return val;
}

//
template <typename T, typename X, typename Y>
T _cuann_dot(int n, const X* x, int incX, const Y* y, int incY, T divisor = 1)
{
  T val = 0;
  for (int i = 0; i < n; i++) {
    val += (T)(x[incX * i]) * (T)(y[incY * i]) / divisor;
  }
  return val;
}

//
template <typename T>
T _cuann_rand()
{
  return (T)rand() / RAND_MAX;
}

// make rotation matrix
void _cuann_make_rotation_matrix(uint32_t nRows,
                                 uint32_t nCols,
                                 uint32_t lenPq,
                                 bool randomRotation,
                                 float* rotationMatrix  // [nRows, nCols]
)
{
  assert(nRows >= nCols);
  assert(nRows % lenPq == 0);

  if (randomRotation) {
    fprintf(stderr, "# create rotation matrix randomly.\n");
    double dot, norm;
    double* matrix = (double*)malloc(sizeof(double) * nRows * nCols);
    memset(matrix, 0, sizeof(double) * nRows * nCols);
    for (int i = 0; i < nRows * nCols; i++) {
      matrix[i] = _cuann_rand<double>() - 0.5;
    }
    for (int j = 0; j < nCols; j++) {
      // normalize the j-th col vector
      norm = sqrt(_cuann_dot<double>(nRows, matrix + j, nCols, matrix + j, nCols));
      for (int i = 0; i < nRows; i++) {
        matrix[j + (nCols * i)] /= norm;
      }
      // orthogonalize the j-th col vector with the previous col vectors
      for (int k = 0; k < j; k++) {
        dot = _cuann_dot<double>(nRows, matrix + j, nCols, matrix + k, nCols);
        for (int i = 0; i < nRows; i++) {
          matrix[j + (nCols * i)] -= dot * matrix[k + (nCols * i)];
        }
      }
      // normalize the j-th col vector again
      norm = sqrt(_cuann_dot<double>(nRows, matrix + j, nCols, matrix + j, nCols));
      for (int i = 0; i < nRows; i++) {
        matrix[j + (nCols * i)] /= norm;
      }
    }
    for (int i = 0; i < nRows * nCols; i++) {
      rotationMatrix[i] = (float)matrix[i];
    }
    free(matrix);
  } else {
    if (nRows == nCols) {
      memset(rotationMatrix, 0, sizeof(float) * nRows * nCols);
      for (int i = 0; i < nCols; i++) {
        rotationMatrix[i + (nCols * i)] = 1.0;
      }
    } else {
      memset(rotationMatrix, 0, sizeof(float) * nRows * nCols);
      int i = 0;
      for (int j = 0; j < nCols; j++) {
        rotationMatrix[j + (nCols * i)] = 1.0;
        i += lenPq;
        if (i >= nRows) { i = (i % nRows) + 1; }
      }
    }
  }
}

// show centers (for debuging)
void _cuann_kmeans_show_centers(const float* centers,  // [numCenters, dimCenters]
                                uint32_t numCenters,
                                uint32_t dimCenters,
                                const uint32_t* centerSize,
                                const int numShow = 5)
{
  for (uint64_t k = 0; k < numCenters; k++) {
    if ((numShow <= k) && (k < numCenters - numShow)) {
      if (k == numShow) fprintf(stderr, "...\n");
      continue;
    }
    fprintf(stderr, "# centers[%lu]:", k);
    for (uint64_t j = 0; j < dimCenters; j++) {
      if ((numShow <= j) && (j < dimCenters - numShow)) {
        if (j == numShow) fprintf(stderr, " ... ");
        continue;
      }
      fprintf(stderr, " %f,", centers[j + (dimCenters * k)]);
    }
    fprintf(stderr, " %d\n", centerSize[k]);
  }
}

// show dataset (for debugging)
void _cuann_show_dataset(const float* dataset,  // [numDataset, dimDataset]
                         uint32_t numDataset,
                         uint32_t dimDataset,
                         const int numShow = 5)
{
  for (uint64_t i = 0; i < numDataset; i++) {
    if ((numShow <= i) && (i < numDataset - numShow)) {
      if (i == numShow) fprintf(stderr, "...\n");
      continue;
    }
    fprintf(stderr, "# dataset[%lu]:", i);
    for (uint64_t j = 0; j < dimDataset; j++) {
      if ((numShow <= j) && (j < dimDataset - numShow)) {
        if (j == numShow) fprintf(stderr, " ... ");
        continue;
      }
      fprintf(stderr, " %.3f,", dataset[j + (dimDataset * i)]);
    }
    fprintf(stderr, "\n");
  }
}

// show pq code (for debuging)
void _cuann_show_pq_code(const uint8_t* pqDataset,  // [numDataset, dimPq]
                         uint32_t numDataset,
                         uint32_t dimPq,
                         const int numShow = 5)
{
  for (uint64_t i = 0; i < numDataset; i++) {
    if ((numShow <= i) && (i < numDataset - numShow)) {
      if (i == numShow) fprintf(stderr, "...\n");
      continue;
    }
    fprintf(stderr, "# dataset[%lu]:", i);
    for (uint64_t j = 0; j < dimPq; j++) {
      if ((numShow <= j) && (j < dimPq - numShow)) {
        if (j == numShow) fprintf(stderr, " ... ");
        continue;
      }
      fprintf(stderr, " %u,", pqDataset[j + (dimPq * i)]);
    }
    fprintf(stderr, "\n");
  }
}

//
int _cuann_set_device(int devId)
{
  int orgDevId;
  cudaError_t cudaError = cudaGetDevice(&orgDevId);
  if (cudaError != cudaSuccess) {
    fprintf(stderr, "(%s, %d) cudaGetDevice() failed (%d)\n", __func__, __LINE__, cudaError);
    exit(-1);
  }
  cudaError = cudaSetDevice(devId);
  if (cudaError != cudaSuccess) {
    fprintf(stderr, "(%s, %d) cudaSetDevice() failed (%d)\n", __func__, __LINE__, cudaError);
    exit(-1);
  }
  return orgDevId;
}

//
uint32_t _get_num_trainset(uint32_t clusterSize, uint32_t dimPq, uint32_t bitPq)
{
  return min(clusterSize * dimPq, 256 * max(1 << bitPq, dimPq));
}

//
void _cuann_compute_PQ_code(cuannHandle_t handle,
                            uint32_t numDataset,
                            uint32_t dimDataset,
                            uint32_t dimRotDataset,
                            uint32_t dimPq,
                            uint32_t lenPq,
                            uint32_t bitPq,
                            uint32_t numClusters,
                            cudaDataType_t dtype,
                            cuannPqCenter_t typePqCenter,
                            uint32_t maxClusterSize,
                            float* clusterCenters,            // [numClusters, dimDataset]
                            const float* rotationMatrix,      // [dimRotDataset, dimDataset]
                            const void* dataset,              // [numDataset]
                            const uint32_t* originalNumbers,  // [numDataset]
                            const uint32_t* clusterSize,      // [numClusters]
                            const uint32_t* indexPtr,         // [numClusters + 1]
                            float* pqCenters,                 // [...]
                            uint32_t numIterations,
                            uint8_t* pqDataset  // [numDataset, dimPq * bitPq / 8]
)
{
  //
  // Compute PQ code
  //
  memset(pqDataset, 0, sizeof(uint8_t) * numDataset * dimPq * bitPq / 8);
  float** resVectors;          // [numDevices][maxClusterSize, dimDataset]
  float** rotVectors;          // [numDevices][maxClusterSize, dimRotDataset]
  float** subVectors;          // [numDevices][dimPq, maxClusterSize, lenPq]
  uint32_t** subVectorLabels;  // [numDevices][dimPq, maxClusterSize]
  uint8_t** myPqDataset;       // [numDevices][maxCluserSize, dimPq * bitPq / 8]
  resVectors = _cuann_multi_device_malloc<float>(
    handle->numDevices, maxClusterSize * dimDataset, "resVectors");
  rotVectors = _cuann_multi_device_malloc<float>(
    handle->numDevices, maxClusterSize * dimRotDataset, "rotVectors");
  subVectors = _cuann_multi_device_malloc<float>(
    handle->numDevices, dimPq * maxClusterSize * lenPq, "subVectors");
  subVectorLabels = _cuann_multi_device_malloc<uint32_t>(
    handle->numDevices, dimPq * maxClusterSize, "subVectorLabels");
  myPqDataset = _cuann_multi_device_malloc<uint8_t>(
    handle->numDevices, maxClusterSize * dimPq * bitPq / 8, "myPqDataset");

  uint32_t maxTrainset = 0;
  if ((numIterations > 0) && (typePqCenter == CUANN_PQ_CENTER_PER_CLUSTER)) {
    maxTrainset = _get_num_trainset(maxClusterSize, dimPq, bitPq);
  }
  void** pqPredictWorkspace = (void**)_cuann_multi_device_malloc<uint8_t>(
    handle->numDevices,
    _cuann_kmeans_predict_bufferSize((1 << bitPq), lenPq, max(maxClusterSize, maxTrainset)),
    "pqPredictWorkspace");

  uint32_t** rotVectorLabels;  // [numDevices][maxClusterSize, dimPq,]
  uint32_t** pqClusterSize;    // [numDevices][1 << bitPq,]
  uint32_t** wsKAC;            // [numDevices][1]
  float** myPqCenters;         // [numDevices][1 << bitPq, lenPq]
  float** myPqCentersTemp;     // [numDevices][1 << bitPq, lenPq]
  if ((numIterations > 0) && (typePqCenter == CUANN_PQ_CENTER_PER_CLUSTER)) {
    memset(pqCenters, 0, sizeof(float) * numClusters * (1 << bitPq) * lenPq);
    rotVectorLabels = _cuann_multi_device_malloc<uint32_t>(
      handle->numDevices, maxClusterSize * dimPq, "rotVectorLabels");
    pqClusterSize =
      _cuann_multi_device_malloc<uint32_t>(handle->numDevices, (1 << bitPq), "pqClusterSize");
    wsKAC = _cuann_multi_device_malloc<uint32_t>(handle->numDevices, 1, "wsKAC");
    myPqCenters =
      _cuann_multi_device_malloc<float>(handle->numDevices, (1 << bitPq) * lenPq, "myPqCenters");
    myPqCentersTemp = _cuann_multi_device_malloc<float>(
      handle->numDevices, (1 << bitPq) * lenPq, "myPqCentersTemp");
  }

#pragma omp parallel for schedule(dynamic) num_threads(handle->numDevices)
  for (uint32_t l = 0; l < numClusters; l++) {
    int devId = omp_get_thread_num();
    cudaSetDevice(devId);
    if (devId == 0) {
      fprintf(stderr, "(%s) Making PQ dataset: %u / %u    \r", __func__, l, numClusters);
    }
    if (clusterSize[l] == 0) continue;

    //
    // Compute the residual vector of the new vector with its cluster
    // centroids.
    //   resVectors[..] = newVectors[..] - clusterCenters[..]
    //
    if (dtype == CUDA_R_32F) {
      _cuann_copy_with_list<float>(clusterSize[l],
                                   dimDataset,
                                   (float*)dataset,
                                   originalNumbers + indexPtr[l],
                                   dimDataset,
                                   resVectors[devId],
                                   dimDataset);
    } else if (dtype == CUDA_R_8U) {
      const float divisor = 256.0;
      _cuann_copy_with_list<uint8_t>(clusterSize[l],
                                     dimDataset,
                                     (uint8_t*)dataset,
                                     originalNumbers + indexPtr[l],
                                     dimDataset,
                                     resVectors[devId],
                                     dimDataset,
                                     divisor);
    } else if (dtype == CUDA_R_8I) {
      const float divisor = 128.0;
      _cuann_copy_with_list<int8_t>(clusterSize[l],
                                    dimDataset,
                                    (int8_t*)dataset,
                                    originalNumbers + indexPtr[l],
                                    dimDataset,
                                    resVectors[devId],
                                    dimDataset,
                                    divisor);
    }
    _cuann_a_me_b(clusterSize[l],
                  dimDataset,
                  resVectors[devId],
                  dimDataset,
                  clusterCenters + (uint64_t)l * dimDataset);

    //
    // Rotate the residual vectors using a rotation matrix
    //
    cudaStream_t cublasStream  = _cuann_set_cublas_stream(handle->cublasHandles[devId], NULL);
    float alpha                = 1.0;
    float beta                 = 0.0;
    cublasStatus_t cublasError = cublasGemmEx(handle->cublasHandles[devId],
                                              CUBLAS_OP_T,
                                              CUBLAS_OP_N,
                                              dimRotDataset,
                                              clusterSize[l],
                                              dimDataset,
                                              &alpha,
                                              rotationMatrix,
                                              CUDA_R_32F,
                                              dimDataset,
                                              resVectors[devId],
                                              CUDA_R_32F,
                                              dimDataset,
                                              &beta,
                                              rotVectors[devId],
                                              CUDA_R_32F,
                                              dimRotDataset,
                                              CUBLAS_COMPUTE_32F,
                                              CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (cublasError != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "(%s, %d) cublasGemmEx() failed.\n", __func__, __LINE__);
      // return CUANN_STATUS_CUBLAS_ERROR;
      exit(-1);
    }
    _cuann_set_cublas_stream(handle->cublasHandles[devId], cublasStream);

    //
    // Training PQ codebook if CUANN_PQ_CENTER_PER_CLUSTER
    // (*) PQ codebooks are trained for each cluster.
    //
    if ((numIterations > 0) && (typePqCenter == CUANN_PQ_CENTER_PER_CLUSTER)) {
      uint32_t numTrainset = _get_num_trainset(clusterSize[l], dimPq, bitPq);
      int numIterations_2  = numIterations * 2;
      for (int iter = 0; iter < numIterations_2; iter += 2) {
        if (devId == 0) {
          fprintf(stderr,
                  "(%s) Making PQ dataset: %u / %u, "
                  "Training PQ codebook (%u): %.1f / %u    \r",
                  __func__,
                  l,
                  numClusters,
                  numTrainset,
                  (float)iter / 2,
                  numIterations);
        }
        _cuann_kmeans_predict(handle->cublasHandles[devId],
                              myPqCenters[devId],
                              (1 << bitPq),
                              lenPq,
                              rotVectors[devId],
                              CUDA_R_32F,
                              numTrainset,
                              rotVectorLabels[devId],
                              CUANN_SIMILARITY_L2,
                              (iter != 0),
                              pqPredictWorkspace[devId],
                              myPqCentersTemp[devId],
                              pqClusterSize[devId],
                              true);
        if ((iter + 1 < numIterations_2) && _cuann_kmeans_adjust_centers(myPqCenters[devId],
                                                                         (1 << bitPq),
                                                                         lenPq,
                                                                         rotVectors[devId],
                                                                         CUDA_R_32F,
                                                                         numTrainset,
                                                                         rotVectorLabels[devId],
                                                                         CUANN_SIMILARITY_L2,
                                                                         pqClusterSize[devId],
                                                                         (float)1.0 / 4,
                                                                         wsKAC[devId])) {
          iter -= 1;
        }
      }
      cudaMemcpy(pqCenters + ((1 << bitPq) * lenPq) * l,
                 myPqCenters[devId],
                 sizeof(float) * (1 << bitPq) * lenPq,
                 cudaMemcpyDeviceToHost);
    }

    //
    // Change the order of the vector data to facilitate processing in
    // each vector subspace.
    //   input:  rotVectors[clusterSize, dimRotDataset]
    //   output: subVectors[dimPq, clusterSize, lenPq]
    //
    _cuann_transpose_copy_3d<float, float>(lenPq,
                                           clusterSize[l],
                                           dimPq,
                                           subVectors[devId],
                                           lenPq,
                                           clusterSize[l],
                                           rotVectors[devId],
                                           1,
                                           dimRotDataset,
                                           lenPq);

    //
    // Find a label (cluster ID) for each vector subspace.
    //
    for (uint32_t j = 0; j < dimPq; j++) {
      float* curPqCenters = NULL;
      if (typePqCenter == CUANN_PQ_CENTER_PER_SUBSPACE) {
        curPqCenters = pqCenters + ((1 << bitPq) * lenPq) * j;
      } else if (typePqCenter == CUANN_PQ_CENTER_PER_CLUSTER) {
        curPqCenters = pqCenters + ((1 << bitPq) * lenPq) * l;
        if (numIterations > 0) { curPqCenters = myPqCenters[devId]; }
      }
      _cuann_kmeans_predict(handle->cublasHandles[devId],
                            curPqCenters,
                            (1 << bitPq),
                            lenPq,
                            subVectors[devId] + j * (clusterSize[l] * lenPq),
                            CUDA_R_32F,
                            clusterSize[l],
                            subVectorLabels[devId] + j * clusterSize[l],
                            CUANN_SIMILARITY_L2,
                            true,
                            pqPredictWorkspace[devId],
                            nullptr,
                            nullptr,
                            true);
    }

    //
    // PQ encoding
    //
    ivfpq_encode(
      clusterSize[l], clusterSize[l], dimPq, bitPq, subVectorLabels[devId], myPqDataset[devId]);
    cudaMemcpy(pqDataset + ((uint64_t)indexPtr[l] * dimPq * bitPq / 8),
               myPqDataset[devId],
               sizeof(uint8_t) * clusterSize[l] * dimPq * bitPq / 8,
               cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();
  }
  cudaDeviceSynchronize();
  fprintf(stderr, "\n");

  //
  _cuann_multi_device_free<uint8_t>((uint8_t**)pqPredictWorkspace, handle->numDevices);
  _cuann_multi_device_free<uint8_t>(myPqDataset, handle->numDevices);
  _cuann_multi_device_free<uint32_t>(subVectorLabels, handle->numDevices);
  _cuann_multi_device_free<float>(subVectors, handle->numDevices);
  _cuann_multi_device_free<float>(rotVectors, handle->numDevices);
  _cuann_multi_device_free<float>(resVectors, handle->numDevices);
  if ((numIterations > 0) && (typePqCenter == CUANN_PQ_CENTER_PER_CLUSTER)) {
    _cuann_multi_device_free<uint32_t>(wsKAC, handle->numDevices);
    _cuann_multi_device_free<uint32_t>(rotVectorLabels, handle->numDevices);
    _cuann_multi_device_free<uint32_t>(pqClusterSize, handle->numDevices);
    _cuann_multi_device_free<float>(myPqCenters, handle->numDevices);
    _cuann_multi_device_free<float>(myPqCentersTemp, handle->numDevices);
  }
}

// cuannCreate
cuannStatus_t cuannCreate(cuannHandle_t* handle)
{
  cudaError_t cudaError;
  cublasStatus_t cublasError;

  *handle = (cuannHandle_t)malloc(sizeof(struct cuannContext));
  if (*handle == NULL) { return CUANN_STATUS_ALLOC_FAILED; }

  // Keep the current device ID.
  int devId;
  cudaError = cudaGetDevice(&devId);
  if (cudaError != cudaSuccess) {
    fprintf(stderr, "(%s, %d) cudaGetDevice() failed.\n", __func__, __LINE__);
    return CUANN_STATUS_CUDA_ERROR;
  }

  // numDevices
  cudaGetDeviceCount(&((*handle)->numDevices));
  if (cudaError != cudaSuccess) {
    fprintf(stderr, "(%s, %d) cudaGetDeviceCount() failed.\n", __func__, __LINE__);
    return CUANN_STATUS_CUDA_ERROR;
  }

  (*handle)->streams     = (cudaStream_t*)malloc(sizeof(cudaStream_t) * (*handle)->numDevices);
  (*handle)->deviceProps = (cudaDeviceProp*)malloc(sizeof(cudaDeviceProp) * (*handle)->numDevices);
  (*handle)->cublasHandles =
    (cublasHandle_t*)malloc(sizeof(cublasHandle_t) * (*handle)->numDevices);

  for (int i = 0; i < (*handle)->numDevices; i++) {
    cudaError = cudaSetDevice(i);
    if (cudaError != cudaSuccess) {
      fprintf(stderr, "(%s, %d) cudaSetDevice() failed.\n", __func__, __LINE__);
      return CUANN_STATUS_CUDA_ERROR;
    }

    // stream
    (*handle)->streams[i] = NULL;

    // deviceProp
    cudaError = cudaGetDeviceProperties(&((*handle)->deviceProps[i]), i);
    if (cudaError != cudaSuccess) {
      fprintf(stderr, "(%s, %d) cudaGetDeviceProperties() failed.\n", __func__, __LINE__);
      return CUANN_STATUS_CUDA_ERROR;
    }

    // cublasHandle
    cublasError = cublasCreate(&((*handle)->cublasHandles[i]));
    if (cublasError != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "(%s, %d) cublasCreate() failed.\n", __func__, __LINE__);
      return CUANN_STATUS_CUBLAS_ERROR;
    }
  }

  return cuannSetDevice(*handle, devId);
}

// cuannDestroy
cuannStatus_t cuannDestroy(cuannHandle_t handle)
{
  if (handle == NULL) { return CUANN_STATUS_NOT_INITIALIZED; }
  cublasStatus_t cublasError;
  for (int i = 0; i < handle->numDevices; i++) {
    cublasError = cublasDestroy(handle->cublasHandles[i]);
    if (cublasError != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "(%s, %d) cublasDestroy() failed.\n", __func__, __LINE__);
      return CUANN_STATUS_CUBLAS_ERROR;
    }
  }
  free(handle->streams);
  free(handle->deviceProps);
  free(handle->cublasHandles);
  free(handle);
  return CUANN_STATUS_SUCCESS;
}

// cuannSetStream
cuannStatus_t cuannSetStream(cuannHandle_t handle, cudaStream_t stream)
{
  if (handle == NULL) { return CUANN_STATUS_NOT_INITIALIZED; }
  int devId = handle->devId;
  cublasSetStream(handle->cublasHandles[devId], stream);
  handle->streams[devId] = stream;

  return cuannSetDevice(handle, devId);
}

// cuannSetDevice
cuannStatus_t cuannSetDevice(cuannHandle_t handle, int devId)
{
  if (handle == NULL) { return CUANN_STATUS_NOT_INITIALIZED; }
  if (devId < 0 || devId >= handle->numDevices) {
    fprintf(
      stderr, "(%s, %d) devId is out of range (devId:%d) failed.\n", __func__, __LINE__, devId);
    return CUANN_STATUS_INVALID_VALUE;
  }

  // (*) Need to re-consider whether it is good to call cudaSetDevice() here.
  cudaError_t cudaError = cudaSetDevice(devId);
  if (cudaError != cudaSuccess) {
    fprintf(stderr, "(%s, %d) cudaSetDevice() failed.\n", __func__, __LINE__);
    return CUANN_STATUS_CUDA_ERROR;
  }

  handle->devId        = devId;
  handle->stream       = handle->streams[devId];
  handle->deviceProp   = handle->deviceProps[devId];
  handle->cublasHandle = handle->cublasHandles[devId];
  return CUANN_STATUS_SUCCESS;
}

// cuannIvfPqCreateDescriptor
cuannStatus_t cuannIvfPqCreateDescriptor(cuannIvfPqDescriptor_t* desc)
{
  *desc = (cuannIvfPqDescriptor_t)malloc(sizeof(struct cuannIvfPqDescriptor));
  if (*desc == NULL) { return CUANN_STATUS_ALLOC_FAILED; }
  (*desc)->numClusters                   = 0;
  (*desc)->numDataset                    = 0;
  (*desc)->dimDataset                    = 0;
  (*desc)->dimDatasetExt                 = 0;
  (*desc)->dimRotDataset                 = 0;
  (*desc)->dimPq                         = 0;
  (*desc)->bitPq                         = 0;
  (*desc)->numProbes                     = 0;
  (*desc)->topK                          = 0;
  (*desc)->maxQueries                    = 0;
  (*desc)->maxBatchSize                  = 0;
  (*desc)->maxSamples                    = 0;
  (*desc)->inclusiveSumSortedClusterSize = NULL;
  (*desc)->sqsumClusters                 = NULL;
  return CUANN_STATUS_SUCCESS;
}

// cuannIvfPqDestroyDescriptor
cuannStatus_t cuannIvfPqDestroyDescriptor(cuannIvfPqDescriptor_t desc)
{
  if (desc == NULL) { return CUANN_STATUS_NOT_INITIALIZED; }
  if (desc->sqsumClusters != NULL) { cudaFree(desc->sqsumClusters); }
  free(desc);
  return CUANN_STATUS_SUCCESS;
}

// cuannIvfPqSetIndexParameters
cuannStatus_t cuannIvfPqSetIndexParameters(cuannIvfPqDescriptor_t desc,
                                           const uint32_t numClusters,
                                           const uint32_t numDataset,
                                           const uint32_t dimDataset,
                                           const uint32_t dimPq,
                                           const uint32_t bitPq,
                                           const cuannSimilarity_t similarity,
                                           const cuannPqCenter_t typePqCenter)
{
  if (desc == NULL) { return CUANN_STATUS_NOT_INITIALIZED; }
  if (numClusters == 0) {
    fprintf(
      stderr, "(%s) numClusters must be larger than zero (dimDataset:%u).\n", __func__, dimDataset);
    return CUANN_STATUS_INVALID_VALUE;
  }
  if (numDataset == 0) {
    fprintf(
      stderr, "(%s) numDataset must be larger than zero (numDataset:%u).\n", __func__, numDataset);
    return CUANN_STATUS_INVALID_VALUE;
  }
  if (dimDataset == 0) {
    fprintf(
      stderr, "(%s) dimDataset must be larger than zero (dimDataset:%u).\n", __func__, dimDataset);
    return CUANN_STATUS_INVALID_VALUE;
  }
  if (dimPq == 0) {
    fprintf(stderr, "(%s) dimPq must be larger than zero (dimPq:%u).\n", __func__, dimPq);
    return CUANN_STATUS_INVALID_VALUE;
  }
  if (numClusters > numDataset) {
    fprintf(stderr,
            "(%s) numClusters must be smaller than numDataset (numClusters:%u, numDataset:%u).\n",
            __func__,
            numClusters,
            numDataset);
    return CUANN_STATUS_INVALID_VALUE;
  }
  if (bitPq < 4 || bitPq > 8) {
    fprintf(stderr, "(%s) bitPq must be 4, 5, 6, 7 or 8 (bitPq:%u)\n", __func__, bitPq);
    return CUANN_STATUS_INVALID_VALUE;
  }
  if (bitPq == 4 && dimPq % 2 != 0) {
    fprintf(stderr,
            "(%s) dimPq must be multiple of 2 when bitPq is 4 (dimPq:%u, bitPq:%u)\n",
            __func__,
            dimPq,
            bitPq);
    return CUANN_STATUS_INVALID_VALUE;
  }
  if (bitPq == 5 && dimPq % 8 != 0) {
    fprintf(stderr,
            "(%s) dimPq must be multiple of 8 when bitPq is 5 (dimPq:%u, bitPq:%u)\n",
            __func__,
            dimPq,
            bitPq);
    return CUANN_STATUS_INVALID_VALUE;
  }
  if (bitPq == 6 && dimPq % 4 != 0) {
    fprintf(stderr,
            "(%s) dimPq must be multiple of 4 when bitPq is 6 (dimPq:%u, bitPq:%u)\n",
            __func__,
            dimPq,
            bitPq);
    return CUANN_STATUS_INVALID_VALUE;
  }
  if (bitPq == 7 && dimPq % 8 != 0) {
    fprintf(stderr,
            "(%s) dimPq must be multiple of 8 when bitPq is 7 (dimPq:%u, bitPq:%u)\n",
            __func__,
            dimPq,
            bitPq);
    return CUANN_STATUS_INVALID_VALUE;
  }
  desc->numClusters   = numClusters;
  desc->numDataset    = numDataset;
  desc->dimDataset    = dimDataset;
  desc->dimDatasetExt = dimDataset + 1;
  if (desc->dimDatasetExt % 8) { desc->dimDatasetExt += 8 - (desc->dimDatasetExt % 8); }
  assert(desc->dimDatasetExt >= dimDataset + 1);
  assert(desc->dimDatasetExt % 8 == 0);
  desc->dimPq        = dimPq;
  desc->bitPq        = bitPq;
  desc->similarity   = similarity;
  desc->typePqCenter = typePqCenter;

  desc->dimRotDataset = dimDataset;
  if (dimDataset % dimPq) { desc->dimRotDataset = ((dimDataset / dimPq) + 1) * dimPq; }
  desc->lenPq = desc->dimRotDataset / dimPq;
  return CUANN_STATUS_SUCCESS;
}

// cuannIvfPqGetIndexParameters
cuannStatus_t cuannIvfPqGetIndexParameters(cuannIvfPqDescriptor_t desc,
                                           uint32_t* numClusters,
                                           uint32_t* numDataset,
                                           uint32_t* dimDataset,
                                           uint32_t* dimPq,
                                           uint32_t* bitPq,
                                           cuannSimilarity_t* similarity,
                                           cuannPqCenter_t* typePqCenter)
{
  if (desc == NULL) { return CUANN_STATUS_NOT_INITIALIZED; }

  *numClusters  = desc->numClusters;
  *numDataset   = desc->numDataset;
  *dimDataset   = desc->dimDataset;
  *dimPq        = desc->dimPq;
  *bitPq        = desc->bitPq;
  *similarity   = desc->similarity;
  *typePqCenter = desc->typePqCenter;
  return CUANN_STATUS_SUCCESS;
}

// cuannIvfPqGetIndexSize
cuannStatus_t cuannIvfPqGetIndexSize(cuannIvfPqDescriptor_t desc, size_t* size)
{
  if (desc == NULL) { return CUANN_STATUS_NOT_INITIALIZED; }

  *size = sizeof(struct cuannIvfPqIndexHeader);
  if (*size != 1024) {
    fprintf(stderr, "(%s, %d) Unexpected Error!\n", __func__, __LINE__);
    exit(-1);
  }
  *size += _cuann_getIndexSize_clusterCenters(desc);
  *size += _cuann_getIndexSize_pqCenters(desc);
  *size += _cuann_getIndexSize_pqDataset(desc);
  *size += _cuann_getIndexSize_originalNumbers(desc);
  *size += _cuann_getIndexSize_indexPtr(desc);
  *size += _cuann_getIndexSize_rotationMatrix(desc);
  *size += _cuann_getIndexSize_clusterRotCenters(desc);
  return CUANN_STATUS_SUCCESS;
}

// cuannIvfPqBuildIndex
cuannStatus_t cuannIvfPqBuildIndex(cuannHandle_t handle,
                                   cuannIvfPqDescriptor_t desc,
                                   const void* dataset,
                                   const void* trainset,
                                   cudaDataType_t dtype,
                                   uint32_t numTrainset,
                                   uint32_t numIterations,
                                   bool randomRotation,
                                   bool hierarchicalClustering,
                                   void* index)
{
  if (handle == NULL || desc == NULL) { return CUANN_STATUS_NOT_INITIALIZED; }
  int cuannDevId  = handle->devId;
  int callerDevId = _cuann_set_device(cuannDevId);

  if (dtype != CUDA_R_32F && dtype != CUDA_R_8U && dtype != CUDA_R_8I) {
    return CUANN_STATUS_UNSUPPORTED_DTYPE;
  }
  if (desc->similarity == CUANN_SIMILARITY_INNER && dtype != CUDA_R_32F) {
    fprintf(
      stderr, "(%s, %d) CUANN_SIMILARITY_INNER supports float dtype only.\n", __func__, __LINE__);
    return CUANN_STATUS_UNSUPPORTED_DTYPE;
  }
  desc->dtypeDataset = dtype;
  char dtypeString[64];
  fprintf(stderr, "# dtypeDataset: %s\n", _cuann_get_dtype_string(desc->dtypeDataset, dtypeString));

  cudaError_t cudaError;
  cudaPointerAttributes attr;
  cudaPointerGetAttributes(&attr, dataset);
  if (attr.type == cudaMemoryTypeDevice) {
    fprintf(stderr, "(%s) dataset must be accessible from the host.\n", __func__);
    return CUANN_STATUS_INVALID_POINTER;
  }
  cudaPointerGetAttributes(&attr, trainset);
  if (attr.type == cudaMemoryTypeDevice) {
    fprintf(stderr, "(%s) trainset must be accessible from the host.\n", __func__);
    return CUANN_STATUS_INVALID_POINTER;
  }

  struct cuannIvfPqIndexHeader* header;
  float* clusterCenters;      // [numClusters, dimDataset]
  float* pqCenters;           // [dimPq, 1 << bitPq, lenPq], or
                              // [numClusters, 1 << bitPq, lenPq]
  uint8_t* pqDataset;         // [numDataset, dimPq * bitPq / 8]
  uint32_t* originalNumbers;  // [numDataset]
  uint32_t* indexPtr;         // [numClusters + 1]
  float* rotationMatrix;      // [dimDataset, dimRotDataset]
  float* clusterRotCenters;   // [numClusters, dimRotDataset]
  _cuann_get_index_pointers(desc,
                            index,
                            &header,
                            &clusterCenters,
                            &pqCenters,
                            &pqDataset,
                            &originalNumbers,
                            &indexPtr,
                            &rotationMatrix,
                            &clusterRotCenters);

  uint32_t* trainsetLabels;  // [numTrainset]
  cudaError = cudaMallocManaged(&trainsetLabels, sizeof(uint32_t) * numTrainset);
  if (cudaError != cudaSuccess) {
    fprintf(stderr, "(%s, %d) cudaMallocManaged() failed.\n", __func__, __LINE__);
    return CUANN_STATUS_ALLOC_FAILED;
  }

  uint32_t* clusterSize;  // [numClusters]
  cudaError = cudaMallocManaged(&clusterSize, sizeof(uint32_t) * desc->numClusters);
  if (cudaError != cudaSuccess) {
    fprintf(stderr, "(%s, %d) cudaMallocManaged() failed.\n", __func__, __LINE__);
    return CUANN_STATUS_ALLOC_FAILED;
  }

  float* clusterCentersTemp;  // [numClusters, dimDataset]
  cudaError =
    cudaMallocManaged(&clusterCentersTemp, sizeof(float) * desc->numClusters * desc->dimDataset);
  if (cudaError != cudaSuccess) {
    fprintf(stderr, "(%s, %d) cudaMallocManaged() failed.\n", __func__, __LINE__);
    return CUANN_STATUS_ALLOC_FAILED;
  }

  uint32_t** wsKAC = _cuann_multi_device_malloc<uint32_t>(handle->numDevices, 1, "wsKAC");

  //
  // Training kmeans
  //
  fprintf(stderr, "# hierarchicalClustering: %u\n", hierarchicalClustering);
  if (hierarchicalClustering) {
    // Hierarchical kmeans
    uint32_t numMesoClusters = pow((double)(desc->numClusters), (double)1.0 / 2.0) + 0.5;
    fprintf(stderr, "# numMesoClusters: %u\n", numMesoClusters);

    float* mesoClusterCenters;  // [numMesoClusters, dimDataset]
    cudaError =
      cudaMallocManaged(&mesoClusterCenters, sizeof(float) * numMesoClusters * desc->dimDataset);
    if (cudaError != cudaSuccess) {
      fprintf(stderr, "(%s, %d) cudaMallocManaged() failed.\n", __func__, __LINE__);
      return CUANN_STATUS_ALLOC_FAILED;
    }
    float* mesoClusterCentersTemp;  // [numMesoClusters, dimDataset]
    cudaError = cudaMallocManaged(&mesoClusterCentersTemp,
                                  sizeof(float) * numMesoClusters * desc->dimDataset);
    if (cudaError != cudaSuccess) {
      fprintf(stderr, "(%s, %d) cudaMallocManaged() failed.\n", __func__, __LINE__);
      return CUANN_STATUS_ALLOC_FAILED;
    }

    uint32_t* mesoClusterLabels;  // [numTrainset,]
    cudaError = cudaMallocManaged(&mesoClusterLabels, sizeof(uint32_t) * numTrainset);
    if (cudaError != cudaSuccess) {
      fprintf(stderr, "(%s, %d) cudaMallocManaged() failed.\n", __func__, __LINE__);
      return CUANN_STATUS_ALLOC_FAILED;
    }

    uint32_t* mesoClusterSize;  // [numMesoClusters,]
    cudaError = cudaMallocManaged(&mesoClusterSize, sizeof(uint32_t) * numMesoClusters);
    if (cudaError != cudaSuccess) {
      fprintf(stderr, "(%s, %d) cudaMallocManaged() failed.\n", __func__, __LINE__);
      return CUANN_STATUS_ALLOC_FAILED;
    }

    //
    // Training kmeans for meso-clusters
    //
    int numIterations_2 = numIterations * 2;
    for (int iter = 0; iter < numIterations_2; iter += 2) {
      fprintf(stderr,
              "(%s) "
              "Training kmeans for meso-clusters: %.1f / %u    \r",
              __func__,
              (float)iter / 2,
              numIterations);
      _cuann_kmeans_predict(handle->cublasHandle,
                            mesoClusterCenters,
                            numMesoClusters,
                            desc->dimDataset,
                            trainset,
                            dtype,
                            numTrainset,
                            mesoClusterLabels,
                            desc->similarity,
                            (iter != 0),
                            NULL,
                            mesoClusterCentersTemp,
                            mesoClusterSize,
                            true);
      if ((iter + 1 < numIterations_2) && _cuann_kmeans_adjust_centers(mesoClusterCenters,
                                                                       numMesoClusters,
                                                                       desc->dimDataset,
                                                                       trainset,
                                                                       dtype,
                                                                       numTrainset,
                                                                       mesoClusterLabels,
                                                                       desc->similarity,
                                                                       mesoClusterSize,
                                                                       (float)1.0 / 4,
                                                                       nullptr)) {
        iter -= 1;
      }
    }
    fprintf(stderr, "\n");
    cudaDeviceSynchronize();

    // Number of centers in each meso cluster
    // [numMesoClusters,]
    uint32_t* numFineClusters = (uint32_t*)malloc(sizeof(uint32_t) * numMesoClusters);

    // [numMesoClusters + 1,]
    uint32_t* csumFineClusters = (uint32_t*)malloc(sizeof(uint32_t) * (numMesoClusters + 1));
    csumFineClusters[0]        = 0;

    uint32_t numClustersRemain  = desc->numClusters;
    uint32_t numTrainsetRemain  = numTrainset;
    uint32_t mesoClusterSizeSum = 0;  // check
    uint32_t mesoClusterSizeMax = 0;
    uint32_t numFineClustersMax = 0;
    for (uint32_t i = 0; i < numMesoClusters; i++) {
      if (i < numMesoClusters - 1) {
        numFineClusters[i] =
          (double)numClustersRemain * mesoClusterSize[i] / numTrainsetRemain + .5;
      } else {
        numFineClusters[i] = numClustersRemain;
      }
      csumFineClusters[i + 1] = csumFineClusters[i] + numFineClusters[i];

      numClustersRemain -= numFineClusters[i];
      numTrainsetRemain -= mesoClusterSize[i];
      mesoClusterSizeSum += mesoClusterSize[i];
      mesoClusterSizeMax = max(mesoClusterSizeMax, mesoClusterSize[i]);
      numFineClustersMax = max(numFineClustersMax, numFineClusters[i]);
    }
    assert(mesoClusterSizeSum == numTrainset);
    assert(csumFineClusters[numMesoClusters] == desc->numClusters);

    uint32_t** idsTrainset =
      _cuann_multi_device_malloc<uint32_t>(handle->numDevices, mesoClusterSizeMax, "idsTrainset");

    float** subTrainset = _cuann_multi_device_malloc<float>(
      handle->numDevices, mesoClusterSizeMax * desc->dimDataset, "subTrainset");

    // label (cluster ID) of each vector
    uint32_t** labelsMP =
      _cuann_multi_device_malloc<uint32_t>(handle->numDevices, mesoClusterSizeMax, "labelsMP");

    float** clusterCentersEach = _cuann_multi_device_malloc<float>(
      handle->numDevices, numFineClustersMax * desc->dimDataset, "clusterCentersEach");

    float** clusterCentersMP = _cuann_multi_device_malloc<float>(
      handle->numDevices, numFineClustersMax * desc->dimDataset, "clusterCentersMP");

    // number of vectors in each cluster
    uint32_t** clusterSizeMP =
      _cuann_multi_device_malloc<uint32_t>(handle->numDevices, numFineClustersMax, "clusterSizeMP");

    size_t sizePredictWorkspace = 0;
    for (uint32_t i = 0; i < numMesoClusters; i++) {
      sizePredictWorkspace =
        max(sizePredictWorkspace,
            _cuann_kmeans_predict_bufferSize(numFineClusters[i],  // number of centers
                                             desc->dimDataset,
                                             mesoClusterSize[i]  // number of vectors
                                             ));
    }
    void** predictWorkspace = (void**)_cuann_multi_device_malloc<uint8_t>(
      handle->numDevices, sizePredictWorkspace, "predictWorkspace");

    //
    // Training kmeans for clusters in each meso-clusters
    //
#pragma omp parallel for schedule(dynamic) num_threads(handle->numDevices)
    for (uint32_t i = 0; i < numMesoClusters; i++) {
      int devId = omp_get_thread_num();
      cudaSetDevice(devId);

      uint32_t k = 0;
      for (uint32_t j = 0; j < numTrainset; j++) {
        if (mesoClusterLabels[j] != i) continue;
        idsTrainset[devId][k++] = j;
      }
      assert(k == mesoClusterSize[i]);

      if (dtype == CUDA_R_32F) {
        _cuann_copy_with_list<float>(mesoClusterSize[i],
                                     desc->dimDataset,
                                     (const float*)trainset,
                                     (const uint32_t*)(idsTrainset[devId]),
                                     desc->dimDataset,
                                     subTrainset[devId],
                                     desc->dimDataset);
      } else if (dtype == CUDA_R_8U) {
        float divisor = 256.0;
        _cuann_copy_with_list<uint8_t>(mesoClusterSize[i],
                                       desc->dimDataset,
                                       (const uint8_t*)trainset,
                                       (const uint32_t*)(idsTrainset[devId]),
                                       desc->dimDataset,
                                       subTrainset[devId],
                                       desc->dimDataset,
                                       divisor);
      } else if (dtype == CUDA_R_8I) {
        float divisor = 128.0;
        _cuann_copy_with_list<int8_t>(mesoClusterSize[i],
                                      desc->dimDataset,
                                      (const int8_t*)trainset,
                                      (const uint32_t*)(idsTrainset[devId]),
                                      desc->dimDataset,
                                      subTrainset[devId],
                                      desc->dimDataset,
                                      divisor);
      }
      int numIterations_2 = numIterations * 2;
      for (int iter = 0; iter < numIterations_2; iter += 2) {
        if (devId == 0) {
          fprintf(stderr,
                  "(%s) Training kmeans for clusters in "
                  "meso-cluster %u (numClusters: %u): %.1f / %u    \r",
                  __func__,
                  i,
                  numFineClusters[i],
                  (float)iter / 2,
                  numIterations);
        }
        _cuann_kmeans_predict(handle->cublasHandles[devId],
                              clusterCentersEach[devId],
                              numFineClusters[i],
                              desc->dimDataset,
                              subTrainset[devId],
                              CUDA_R_32F,
                              mesoClusterSize[i],
                              labelsMP[devId],
                              desc->similarity,
                              (iter != 0),
                              predictWorkspace[devId],
                              clusterCentersMP[devId],
                              clusterSizeMP[devId],
                              true);
        if ((iter + 1 < numIterations_2) && _cuann_kmeans_adjust_centers(clusterCentersEach[devId],
                                                                         numFineClusters[i],
                                                                         desc->dimDataset,
                                                                         subTrainset[devId],
                                                                         CUDA_R_32F,
                                                                         mesoClusterSize[i],
                                                                         labelsMP[devId],
                                                                         desc->similarity,
                                                                         clusterSizeMP[devId],
                                                                         (float)1.0 / 4,
                                                                         wsKAC[devId])) {
          iter -= 1;
        }
      }
      cudaMemcpy(clusterCenters + (desc->dimDataset * csumFineClusters[i]),
                 clusterCentersEach[devId],
                 sizeof(float) * numFineClusters[i] * desc->dimDataset,
                 cudaMemcpyDeviceToDevice);
    }
    for (int devId = 0; devId < handle->numDevices; devId++) {
      cudaSetDevice(devId);
      cudaDeviceSynchronize();
    }
    fprintf(stderr, "\n");
    cudaSetDevice(cuannDevId);

    _cuann_multi_device_free<uint32_t>(idsTrainset, handle->numDevices);
    _cuann_multi_device_free<float>(subTrainset, handle->numDevices);
    _cuann_multi_device_free<uint32_t>(labelsMP, handle->numDevices);
    _cuann_multi_device_free<float>(clusterCentersEach, handle->numDevices);
    _cuann_multi_device_free<float>(clusterCentersMP, handle->numDevices);
    _cuann_multi_device_free<uint32_t>(clusterSizeMP, handle->numDevices);
    _cuann_multi_device_free<uint8_t>((uint8_t**)predictWorkspace, handle->numDevices);

    cudaFree(mesoClusterSize);
    cudaFree(mesoClusterLabels);
    cudaFree(mesoClusterCenters);
    cudaFree(mesoClusterCentersTemp);

    free(numFineClusters);
    free(csumFineClusters);

    //
    // Fine-tuning kmeans for whole clusters (with multipel GPUs)
    //
    // (*) Since the likely cluster centroids have been calculated
    // hierarchically already, the number of iteration for fine-tuning
    // kmeans for whole clusters should be reduced. However, there
    // is a possibility that the clusters could be unbalanced here,
    // in which case the actual number of iterations would be increased.
    //
    const int X         = 5;
    int numIterations_X = max(numIterations / 10, 2) * X;
    for (int iter = 0; iter < numIterations_X; iter += X) {
      fprintf(stderr,
              "(%s) "
              "Fine-tuning kmeans for whole clusters: %.1f / %d    \r",
              __func__,
              (float)iter / X,
              numIterations_X / X);
      _cuann_kmeans_predict_MP(handle->numDevices,
                               handle->cublasHandles,
                               clusterCenters,
                               desc->numClusters,
                               desc->dimDataset,
                               trainset,
                               dtype,
                               numTrainset,
                               trainsetLabels,
                               desc->similarity,
                               true,
                               clusterSize,
                               true /* to update clusterCenters */);
      if ((iter + 1 < numIterations_X) && _cuann_kmeans_adjust_centers(clusterCenters,
                                                                       desc->numClusters,
                                                                       desc->dimDataset,
                                                                       trainset,
                                                                       dtype,
                                                                       numTrainset,
                                                                       trainsetLabels,
                                                                       desc->similarity,
                                                                       clusterSize,
                                                                       (float)1.0 / 5,
                                                                       nullptr)) {
        iter -= (X - 1);
      }
    }
    fprintf(stderr, "\n");
  } else {
    // Flat kmeans
    int numIterations_2 = numIterations * 2;
    for (int iter = 0; iter < numIterations_2; iter += 2) {
      fprintf(
        stderr, "(%s) Training kmeans: %.1f / %u    \r", __func__, (float)iter / 2, numIterations);
      _cuann_kmeans_predict(handle->cublasHandle,
                            clusterCenters,
                            desc->numClusters,
                            desc->dimDataset,
                            trainset,
                            dtype,
                            numTrainset,
                            trainsetLabels,
                            desc->similarity,
                            (iter != 0),
                            NULL,
                            clusterCentersTemp,
                            clusterSize,
                            true);
      if ((iter + 1 < numIterations_2) && _cuann_kmeans_adjust_centers(clusterCenters,
                                                                       desc->numClusters,
                                                                       desc->dimDataset,
                                                                       trainset,
                                                                       dtype,
                                                                       numTrainset,
                                                                       trainsetLabels,
                                                                       desc->similarity,
                                                                       clusterSize,
                                                                       (float)1.0 / 4,
                                                                       nullptr)) {
        iter -= 1;
      }
    }
    fprintf(stderr, "\n");
  }

  uint32_t* datasetLabels;  // [numDataset]
  cudaError = cudaMallocManaged(&datasetLabels, sizeof(uint32_t) * desc->numDataset);
  if (cudaError != cudaSuccess) {
    fprintf(stderr, "(%s, %d) cudaMallocManaged() failed.\n", __func__, __LINE__);
    return CUANN_STATUS_ALLOC_FAILED;
  }

  //
  // Predict labels of whole dataset (with multiple GPUs)
  //
  fprintf(stderr, "(%s) Final fitting\n", __func__);
  _cuann_kmeans_predict_MP(handle->numDevices,
                           handle->cublasHandles,
                           clusterCenters,
                           desc->numClusters,
                           desc->dimDataset,
                           dataset,
                           dtype,
                           desc->numDataset,
                           datasetLabels,
                           desc->similarity,
                           true,
                           clusterSize,
                           true /* to update clusterCenters */);

#ifdef CUANN_DEBUG
  cudaDeviceSynchronize();
  _cuann_kmeans_show_centers(clusterCenters, desc->numClusters, desc->dimDataset, clusterSize);
#endif

  // Make rotation matrix
  fprintf(stderr, "# dimDataset: %u\n", desc->dimDataset);
  fprintf(stderr, "# dimRotDataset: %u\n", desc->dimRotDataset);
  fprintf(stderr, "# randomRotation: %u\n", randomRotation);
  _cuann_make_rotation_matrix(
    desc->dimRotDataset, desc->dimDataset, desc->lenPq, randomRotation, rotationMatrix);

  // Rotate clusterCenters
  cudaStream_t cublasStream  = _cuann_set_cublas_stream(handle->cublasHandle, NULL);
  float alpha                = 1.0;
  float beta                 = 0.0;
  cublasStatus_t cublasError = cublasGemmEx(handle->cublasHandle,
                                            CUBLAS_OP_T,
                                            CUBLAS_OP_N,
                                            desc->dimRotDataset,
                                            desc->numClusters,
                                            desc->dimDataset,
                                            &alpha,
                                            rotationMatrix,
                                            CUDA_R_32F,
                                            desc->dimDataset,
                                            clusterCenters,
                                            CUDA_R_32F,
                                            desc->dimDataset,
                                            &beta,
                                            clusterRotCenters,
                                            CUDA_R_32F,
                                            desc->dimRotDataset,
                                            CUBLAS_COMPUTE_32F,
                                            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (cublasError != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "(%s, %d) cublasGemmEx() failed.\n", __func__, __LINE__);
    return CUANN_STATUS_CUBLAS_ERROR;
  }
  _cuann_set_cublas_stream(handle->cublasHandle, cublasStream);

  //
  // Make indexPtr, originalNumbers and pqDataset
  //
  uint32_t maxClusterSize = 0;
  // indexPtr
  indexPtr[0] = 0;
  for (uint32_t l = 0; l < desc->numClusters; l++) {
    indexPtr[l + 1] = indexPtr[l] + clusterSize[l];
    if (maxClusterSize < clusterSize[l]) { maxClusterSize = clusterSize[l]; }
  }
  if (indexPtr[desc->numClusters] != desc->numDataset) {
    fprintf(stderr, "(%s, %d) Unexpected Error.\n", __func__, __LINE__);
    return CUANN_STATUS_INTERNAL_ERROR;
  }
  desc->maxClusterSize = maxClusterSize;
  // fprintf(stderr, "(%s) maxClusterSize: %u\n", __func__, maxClusterSize);

  // originalNumbers
  for (uint32_t i = 0; i < desc->numDataset; i++) {
    uint32_t l                   = datasetLabels[i];
    originalNumbers[indexPtr[l]] = i;
    indexPtr[l] += 1;
  }

  // Recover indexPtr
  for (uint32_t l = 0; l < desc->numClusters; l++) {
    indexPtr[l] -= clusterSize[l];
  }

  // [numDevices][1 << bitPq, lenPq]
  float** pqCentersTemp = _cuann_multi_device_malloc<float>(
    handle->numDevices, (1 << desc->bitPq) * desc->lenPq, "pqCentersTemp");

  // [numDevices][1 << bitPq,]
  uint32_t** pqClusterSize =
    _cuann_multi_device_malloc<uint32_t>(handle->numDevices, (1 << desc->bitPq), "pqClusterSize");

  // Allocate workspace for PQ codebook training
  size_t sizePqPredictWorkspace =
    _cuann_kmeans_predict_bufferSize((1 << desc->bitPq), desc->lenPq, numTrainset);
  sizePqPredictWorkspace    = max(sizePqPredictWorkspace,
                               _cuann_kmeans_predict_bufferSize(
                                 (1 << desc->bitPq), desc->lenPq, maxClusterSize * desc->dimPq));
  void** pqPredictWorkspace = (void**)_cuann_multi_device_malloc<uint8_t>(
    handle->numDevices, sizePqPredictWorkspace, "pqPredictWorkspace");

  if (desc->typePqCenter == CUANN_PQ_CENTER_PER_SUBSPACE) {
    //
    // Training PQ codebook (CUANN_PQ_CENTER_PER_SUBSPACE)
    // (*) PQ codebooks are trained for each subspace.
    //

    // Predict label of trainset again (with multiple GPUs)
    fprintf(stderr, "(%s) Predict label of trainset again\n", __func__);
    _cuann_kmeans_predict_MP(handle->numDevices,
                             handle->cublasHandles,
                             clusterCenters,
                             desc->numClusters,
                             desc->dimDataset,
                             trainset,
                             dtype,
                             numTrainset,
                             trainsetLabels,
                             desc->similarity,
                             true,
                             NULL,
                             false /* do not update clusterCenters */);

    // [dimPq, numTrainset, lenPq]
    size_t sizeModTrainset = sizeof(float) * desc->dimPq * numTrainset * desc->lenPq;
    float* modTrainset     = (float*)malloc(sizeModTrainset);
    memset(modTrainset, 0, sizeModTrainset);

    // modTrainset[] = transpose( rotate(trainset[]) - clusterRotCenters[] )
#pragma omp parallel for
    for (uint32_t i = 0; i < numTrainset; i++) {
      uint32_t l = trainsetLabels[i];
      for (uint32_t j = 0; j < desc->dimRotDataset; j++) {
        float val;
        if (dtype == CUDA_R_32F) {
          val =
            _cuann_dot<float, float, float>(desc->dimDataset,
                                            (float*)trainset + ((uint64_t)(desc->dimDataset) * i),
                                            1,
                                            rotationMatrix + ((uint64_t)(desc->dimDataset) * j),
                                            1);
        } else if (dtype == CUDA_R_8U) {
          float divisor = 256.0;
          val           = _cuann_dot<float, uint8_t, float>(
            desc->dimDataset,
            (uint8_t*)trainset + ((uint64_t)(desc->dimDataset) * i),
            1,
            rotationMatrix + ((uint64_t)(desc->dimDataset) * j),
            1,
            divisor);
        } else if (dtype == CUDA_R_8I) {
          float divisor = 128.0;
          val =
            _cuann_dot<float, int8_t, float>(desc->dimDataset,
                                             (int8_t*)trainset + ((uint64_t)(desc->dimDataset) * i),
                                             1,
                                             rotationMatrix + ((uint64_t)(desc->dimDataset) * j),
                                             1,
                                             divisor);
        }
        uint32_t j0 = j / (desc->lenPq);  // 0 <= j0 < dimPq
        uint32_t j1 = j % (desc->lenPq);  // 0 <= j1 < lenPq
        uint64_t idx =
          j1 + ((uint64_t)(desc->lenPq) * i) + ((uint64_t)(desc->lenPq) * numTrainset * j0);
        modTrainset[idx] = val - clusterRotCenters[j + (desc->dimRotDataset * l)];
      }
    }

    // [numDevices][numTrainset, lenPq]
    float** subTrainset = _cuann_multi_device_malloc<float>(
      handle->numDevices, numTrainset * desc->lenPq, "subTrainset");

    // [numDevices][numTrainset]
    uint32_t** subTrainsetLabels =
      _cuann_multi_device_malloc<uint32_t>(handle->numDevices, numTrainset, "subTrainsetLabels");

    float** pqCentersEach = _cuann_multi_device_malloc<float>(
      handle->numDevices, ((1 << desc->bitPq) * desc->lenPq), "pqCentersEach");

#pragma omp parallel for schedule(dynamic) num_threads(handle->numDevices)
    for (uint32_t j = 0; j < desc->dimPq; j++) {
      int devId = omp_get_thread_num();
      cudaSetDevice(devId);

      float* curPqCenters = pqCenters + ((1 << desc->bitPq) * desc->lenPq) * j;
      cudaMemcpy(subTrainset[devId],
                 modTrainset + ((uint64_t)numTrainset * desc->lenPq * j),
                 sizeof(float) * numTrainset * desc->lenPq,
                 cudaMemcpyHostToDevice);
      // Train kmeans for each PQ
      int numIterations_2 = numIterations * 2;
      for (int iter = 0; iter < numIterations_2; iter += 2) {
        if (devId == 0) {
          fprintf(stderr,
                  "(%s) Training PQ codebook %u (out of %u): "
                  "%.1f / %u    \r",
                  __func__,
                  j,
                  desc->dimPq,
                  (float)iter / 2,
                  numIterations);
        }
        _cuann_kmeans_predict(handle->cublasHandles[devId],
                              pqCentersEach[devId],
                              (1 << desc->bitPq),
                              desc->lenPq,
                              subTrainset[devId],
                              CUDA_R_32F,
                              numTrainset,
                              subTrainsetLabels[devId],
                              CUANN_SIMILARITY_L2,
                              (iter != 0),
                              pqPredictWorkspace[devId],
                              pqCentersTemp[devId],
                              pqClusterSize[devId],
                              true);
        if ((iter + 1 < numIterations_2) && _cuann_kmeans_adjust_centers(pqCentersEach[devId],
                                                                         (1 << desc->bitPq),
                                                                         desc->lenPq,
                                                                         subTrainset[devId],
                                                                         CUDA_R_32F,
                                                                         numTrainset,
                                                                         subTrainsetLabels[devId],
                                                                         CUANN_SIMILARITY_L2,
                                                                         pqClusterSize[devId],
                                                                         (float)1.0 / 4,
                                                                         wsKAC[devId])) {
          iter -= 1;
        }
      }
      cudaMemcpy(curPqCenters,
                 pqCentersEach[devId],
                 sizeof(float) * ((1 << desc->bitPq) * desc->lenPq),
                 cudaMemcpyDeviceToDevice);
#ifdef CUANN_DEBUG
      if (j == 0) {
        cudaDeviceSynchronize();
        _cuann_kmeans_show_centers(
          curPqCenters, (1 << desc->bitPq), desc->lenPq, pqClusterSize[devId]);
      }
#endif
    }
    fprintf(stderr, "\n");
    cudaSetDevice(cuannDevId);

    _cuann_multi_device_free<float>(subTrainset, handle->numDevices);
    _cuann_multi_device_free<uint32_t>(subTrainsetLabels, handle->numDevices);
    _cuann_multi_device_free<float>(pqCentersEach, handle->numDevices);
    free(modTrainset);
  }

  //
  // Compute PQ code for whole dataset
  //
  _cuann_compute_PQ_code(handle,
                         desc->numDataset,
                         desc->dimDataset,
                         desc->dimRotDataset,
                         desc->dimPq,
                         desc->lenPq,
                         desc->bitPq,
                         desc->numClusters,
                         dtype,
                         desc->typePqCenter,
                         maxClusterSize,
                         clusterCenters,
                         rotationMatrix,
                         dataset,
                         originalNumbers,
                         clusterSize,
                         indexPtr,
                         pqCenters,
                         numIterations,
                         pqDataset);
  cudaSetDevice(cuannDevId);

  //
  _cuann_get_inclusiveSumSortedClusterSize(
    desc, indexPtr, clusterCenters, &(desc->inclusiveSumSortedClusterSize));
  _cuann_get_sqsumClusters(desc, clusterCenters, &(desc->sqsumClusters));

  {
    // combine clusterCenters and sqsumClusters
    cudaDeviceSynchronize();
    float* tmpClusterCenters;  // [numClusters, dimDataset]
    cudaError =
      cudaMallocManaged(&tmpClusterCenters, sizeof(float) * desc->numClusters * desc->dimDataset);
    if (cudaError != cudaSuccess) {
      fprintf(stderr, "(%s, %d) cudaMallocManaged() failed.\n", __func__, __LINE__);
      return CUANN_STATUS_ALLOC_FAILED;
    }
    for (uint32_t i = 0; i < desc->numClusters * desc->dimDataset; i++) {
      tmpClusterCenters[i] = clusterCenters[i];
    }
    for (uint32_t i = 0; i < desc->numClusters; i++) {
      for (uint32_t j = 0; j < desc->dimDataset; j++) {
        clusterCenters[j + (desc->dimDatasetExt * i)] =
          tmpClusterCenters[j + (desc->dimDataset * i)];
      }
      clusterCenters[desc->dimDataset + (desc->dimDatasetExt * i)] = desc->sqsumClusters[i];
    }
    cudaFree(tmpClusterCenters);
  }

  //
  cuannIvfPqGetIndexSize(desc, &(header->indexSize));
  header->similarity      = desc->similarity;
  header->numClusters     = desc->numClusters;
  header->numDataset      = desc->numDataset;
  header->dimDataset      = desc->dimDataset;
  header->dimPq           = desc->dimPq;
  header->maxClusterSize  = maxClusterSize;
  header->dimRotDataset   = desc->dimRotDataset;
  header->bitPq           = desc->bitPq;
  header->typePqCenter    = desc->typePqCenter;
  header->dtypeDataset    = desc->dtypeDataset;
  header->dimDatasetExt   = desc->dimDatasetExt;
  header->numDatasetAdded = 0;

  //
  cudaFree(clusterSize);
  cudaFree(trainsetLabels);
  cudaFree(datasetLabels);
  cudaFree(clusterCentersTemp);

  _cuann_multi_device_free<uint32_t>(wsKAC, handle->numDevices);
  _cuann_multi_device_free<float>(pqCentersTemp, handle->numDevices);
  _cuann_multi_device_free<uint32_t>(pqClusterSize, handle->numDevices);
  _cuann_multi_device_free<uint8_t>((uint8_t**)pqPredictWorkspace, handle->numDevices);

  cuannSetDevice(handle, cuannDevId);
  _cuann_set_device(callerDevId);

  return CUANN_STATUS_SUCCESS;
}

// cuannIvfPqSaveIndex
cuannStatus_t cuannIvfPqSaveIndex(cuannHandle_t handle,
                                  cuannIvfPqDescriptor_t desc,
                                  const void* index,
                                  const char* fileName)
{
  if (handle == NULL || desc == NULL) { return CUANN_STATUS_NOT_INITIALIZED; }
  int orgDevId = _cuann_set_device(handle->devId);

  FILE* fp = fopen(fileName, "w");
  if (fp == NULL) {
    fprintf(stderr, "(%s) failed to open file (%s).\n", __func__, fileName);
    return CUANN_STATUS_FILEIO_ERROR;
  }
  struct cuannIvfPqIndexHeader* header = (struct cuannIvfPqIndexHeader*)index;
  fprintf(stderr, "(%s) indexSize: %lu\n", __func__, header->indexSize);
  if (fwrite(index, 1, header->indexSize, fp) != header->indexSize) {
    fprintf(stderr, "(%s) failed to save index to file (%s)\n", __func__, fileName);
    return CUANN_STATUS_FILEIO_ERROR;
  }
  fclose(fp);

  _cuann_set_device(orgDevId);
  return CUANN_STATUS_SUCCESS;
}

// cuannIvfPqLoadIndex
cuannStatus_t cuannIvfPqLoadIndex(cuannHandle_t handle,
                                  cuannIvfPqDescriptor_t desc,
                                  void** index,
                                  const char* fileName)
{
  if (handle == NULL || desc == NULL) { return CUANN_STATUS_NOT_INITIALIZED; }
  int orgDevId = _cuann_set_device(handle->devId);

  if (1 /* *index == NULL */) {
    FILE* fp = fopen(fileName, "r");
    if (fp == NULL) {
      fprintf(stderr, "(%s) failed to open file (%s)\n", __func__, fileName);
      return CUANN_STATUS_FILEIO_ERROR;
    }
    size_t indexSize;
    fread(&indexSize, sizeof(size_t), 1, fp);
    fprintf(stderr, "(%s) indexSize: %lu\n", __func__, indexSize);
    cudaError_t cudaError = cudaMallocManaged(index, indexSize);
    if (cudaError != cudaSuccess) {
      fprintf(stderr, "(%s) cudaMallocManaged() failed.\n", __func__);
      return CUANN_STATUS_ALLOC_FAILED;
    }
    fseek(fp, 0, SEEK_SET);
    if (fread(*index, 1, indexSize, fp) != indexSize) {
      fprintf(stderr, "(%s) failed to load index to from file (%s)\n", __func__, fileName);
      return CUANN_STATUS_FILEIO_ERROR;
    }
    fclose(fp);

    cudaMemAdvise(index, indexSize, cudaMemAdviseSetReadMostly, handle->devId);
  }

  struct cuannIvfPqIndexHeader* header = (struct cuannIvfPqIndexHeader*)(*index);
  desc->numClusters                    = header->numClusters;
  desc->numDataset                     = header->numDataset;
  desc->dimDataset                     = header->dimDataset;
  desc->dimPq                          = header->dimPq;
  desc->similarity                     = (cuannSimilarity_t)header->similarity;
  desc->maxClusterSize                 = header->maxClusterSize;
  desc->dimRotDataset                  = header->dimRotDataset;
  desc->lenPq                          = desc->dimRotDataset / desc->dimPq;
  desc->bitPq                          = header->bitPq;
  desc->typePqCenter                   = (cuannPqCenter_t)header->typePqCenter;
  desc->dtypeDataset                   = (cudaDataType_t)header->dtypeDataset;
  desc->dimDatasetExt                  = header->dimDatasetExt;
  desc->indexVersion                   = header->version;

  float* clusterCenters;      // [numClusters, dimDatasetExt]
  float* pqCenters;           // [dimPq, 1 << bitPq, lenPq], or
                              // [numClusters, 1 << bitPq, lenPq]
  uint8_t* pqDataset;         // [numDataset, dimPq * bitPq / 8]
  uint32_t* originalNumbers;  // [numDataset]
  uint32_t* indexPtr;         // [numClusters + 1]
  float* rotationMatrix;      // [dimDataset, dimRotDataset]
  float* clusterRotCenters;   // [numClusters, dimRotDataset]
  _cuann_get_index_pointers(desc,
                            *index,
                            &header,
                            &clusterCenters,
                            &pqCenters,
                            &pqDataset,
                            &originalNumbers,
                            &indexPtr,
                            &rotationMatrix,
                            &clusterRotCenters);

  //
  _cuann_get_inclusiveSumSortedClusterSize(
    desc, indexPtr, clusterCenters, &(desc->inclusiveSumSortedClusterSize));

  size_t size;
  // pqDataset
  size = sizeof(uint8_t) * desc->numDataset * desc->dimPq * desc->bitPq / 8;
  if (size < (handle->deviceProp).totalGlobalMem) {
    cudaMemPrefetchAsync(pqDataset, size, handle->devId);
  }
  // clusterCenters
  size = sizeof(float) * desc->numClusters * desc->dimDatasetExt;
  cudaMemPrefetchAsync(clusterCenters, size, handle->devId);
  // pqCenters
  if (desc->typePqCenter == CUANN_PQ_CENTER_PER_SUBSPACE) {
    size = sizeof(float) * desc->dimPq * (1 << desc->bitPq) * desc->lenPq;
  } else {
    size = sizeof(float) * desc->numClusters * (1 << desc->bitPq) * desc->lenPq;
  }
  cudaMemPrefetchAsync(pqCenters, size, handle->devId);
  // originalNumbers
  size = sizeof(uint32_t) * desc->numDataset;
  cudaMemPrefetchAsync(originalNumbers, size, handle->devId);
  // indexPtr
  size = sizeof(uint32_t) * (desc->numClusters + 1);
  cudaMemPrefetchAsync(indexPtr, size, handle->devId);
  // rotationMatrix
  if (rotationMatrix != NULL) {
    size = sizeof(float) * desc->dimDataset * desc->dimRotDataset;
    cudaMemPrefetchAsync(rotationMatrix, size, handle->devId);
  }
  // clusterRotCenters
  if (clusterRotCenters != NULL) {
    size = sizeof(float) * desc->numClusters * desc->dimRotDataset;
    cudaMemPrefetchAsync(clusterRotCenters, size, handle->devId);
  }

  _cuann_set_device(orgDevId);
  return CUANN_STATUS_SUCCESS;
}

// cuannIvfPqCreateNewIndexByAddingVectorsToOldIndex
cuannStatus_t cuannIvfPqCreateNewIndexByAddingVectorsToOldIndex(
  cuannHandle_t handle,
  const char* oldIndexFileName,
  const char* newIndexFileName,
  const void* newVectors, /* [numNewVectors, dimDataset] */
  uint32_t numNewVectors)
{
  cudaError_t cudaError;
  cuannStatus_t ret;
  if (handle == NULL) { return CUANN_STATUS_NOT_INITIALIZED; }
  cudaPointerAttributes attr;
  cudaPointerGetAttributes(&attr, newVectors);
  if (attr.type == cudaMemoryTypeDevice) {
    fprintf(stderr, "(%s, %d) newVectors must be accessible from the host.\n", __func__, __LINE__);
    return CUANN_STATUS_INVALID_POINTER;
  }
  int cuannDevId  = handle->devId;
  int callerDevId = _cuann_set_device(cuannDevId);

  //
  // Load old index
  //
  cuannIvfPqDescriptor_t oldDesc;
  ret = cuannIvfPqCreateDescriptor(&oldDesc);
  if (ret != CUANN_STATUS_SUCCESS) { return ret; }
  void* oldIndex;
  ret = cuannIvfPqLoadIndex(handle, oldDesc, &oldIndex, oldIndexFileName);
  if (ret != CUANN_STATUS_SUCCESS) { return ret; }
  cudaDataType_t dtype = oldDesc->dtypeDataset;
  char dtypeString[64];
  fprintf(stderr, "(%s) dtype: %s\n", __func__, _cuann_get_dtype_string(dtype, dtypeString));
  fprintf(stderr, "(%s) dimDataset: %u\n", __func__, oldDesc->dimDataset);
  struct cuannIvfPqIndexHeader* oldHeader;
  float* oldClusterCenters;      // [numClusters, dimDatasetExt]
  float* oldPqCenters;           // [dimPq, 1 << bitPq, lenPq], or
                                 // [numClusters, 1 << bitPq, lenPq]
  uint8_t* oldPqDataset;         // [numDataset, dimPq * bitPq / 8]
  uint32_t* oldOriginalNumbers;  // [numDataset]
  uint32_t* oldIndexPtr;         // [numClusters + 1]
  float* oldRotationMatrix;      // [dimDataset, dimRotDataset]
  float* oldClusterRotCenters;   // [numClusters, dimRotDataset]
  _cuann_get_index_pointers(oldDesc,
                            oldIndex,
                            &oldHeader,
                            &oldClusterCenters,
                            &oldPqCenters,
                            &oldPqDataset,
                            &oldOriginalNumbers,
                            &oldIndexPtr,
                            &oldRotationMatrix,
                            &oldClusterRotCenters);

  //
  // The clusterCenters stored in index contain data other than cluster
  // centroids to speed up the search. Here, only the cluster centroids
  // are extracted.
  //
  float* clusterCenters;  // [numClusters, dimDataset]
  cudaError =
    cudaMallocManaged(&clusterCenters, sizeof(float) * oldDesc->numClusters * oldDesc->dimDataset);
  if (cudaError != cudaSuccess) {
    fprintf(stderr, "(%s, %d) cudaMallocManaged() failed.\n", __func__, __LINE__);
    return CUANN_STATUS_ALLOC_FAILED;
  }
  for (int i = 0; i < oldDesc->numClusters; i++) {
    memcpy(clusterCenters + (uint64_t)i * oldDesc->dimDataset,
           oldClusterCenters + (uint64_t)i * oldDesc->dimDatasetExt,
           sizeof(float) * oldDesc->dimDataset);
  }

  //
  // Use the existing cluster centroids to find the label (cluster ID)
  // of the vector to be added.
  //
  uint32_t* newVectorLabels;  // [numNewVectors,]
  cudaError = cudaMallocManaged(&newVectorLabels, sizeof(uint32_t) * numNewVectors);
  if (cudaError != cudaSuccess) {
    fprintf(stderr, "(%s, %d) cudaMallocManaged() failed.\n", __func__, __LINE__);
    return CUANN_STATUS_ALLOC_FAILED;
  }
  cudaMemset(newVectorLabels, 0, sizeof(uint32_t) * numNewVectors);
  uint32_t* clusterSize;  // [numClusters,]
  cudaError = cudaMallocManaged(&clusterSize, sizeof(uint32_t) * oldDesc->numClusters);
  if (cudaError != cudaSuccess) {
    fprintf(stderr, "(%s, %d) cudaMallocManaged() failed.\n", __func__, __LINE__);
    return CUANN_STATUS_ALLOC_FAILED;
  }
  cudaMemset(clusterSize, 0, sizeof(uint32_t) * oldDesc->numClusters);
  fprintf(stderr, "(%s) Predict label of new vectors\n", __func__);
  _cuann_kmeans_predict_MP(handle->numDevices,
                           handle->cublasHandles,
                           clusterCenters,
                           oldDesc->numClusters,
                           oldDesc->dimDataset,
                           newVectors,
                           dtype,
                           numNewVectors,
                           newVectorLabels,
                           oldDesc->similarity,
                           true,
                           clusterSize,
                           false /* do not update clusterCenters */);

#ifdef CUANN_DEBUG
  if (1) {
    const int _num_show = 10;
    fprintf(stderr, "# numNewVectors: %u\n", numNewVectors);
    fprintf(stderr, "# newVectorLabels: ");
    for (int i = 0; i < numNewVectors; i++) {
      if ((i < _num_show) || (numNewVectors - i <= _num_show)) {
        fprintf(stderr, "%u, ", newVectorLabels[i]);
      } else if (i == _num_show) {
        fprintf(stderr, "..., ");
      }
    }
    fprintf(stderr, "\n");
  }
  if (1) {
    const int _num_show = 10;
    fprintf(stderr, "# oldDesc->numClusters: %u\n", oldDesc->numClusters);
    fprintf(stderr, "# clusterSize: ");
    int _sum = 0;
    for (int i = 0; i < oldDesc->numClusters; i++) {
      _sum += clusterSize[i];
      if ((i < _num_show) || (oldDesc->numClusters - i <= _num_show)) {
        fprintf(stderr, "%u, ", clusterSize[i]);
      } else if (i == _num_show) {
        fprintf(stderr, "..., ");
      }
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "# _sum: %d\n", _sum);
  }
#endif

  //
  // Make indexPtr, originalNumbers
  //
  uint32_t maxClusterSize = 0;
  uint32_t* indexPtr;         // [numClusters + 1]
  uint32_t* originalNumbers;  // [numNewVectors]
  indexPtr        = (uint32_t*)malloc(sizeof(uint32_t) * (oldDesc->numClusters + 1));
  originalNumbers = (uint32_t*)malloc(sizeof(uint32_t) * numNewVectors);
  // indexPtr
  indexPtr[0] = 0;
  for (uint32_t l = 0; l < oldDesc->numClusters; l++) {
    indexPtr[l + 1] = indexPtr[l] + clusterSize[l];
    maxClusterSize  = max(maxClusterSize, clusterSize[l]);
  }
  if (indexPtr[oldDesc->numClusters] != numNewVectors) {
    fprintf(stderr, "(%s, %d) Unexpected Error.\n", __func__, __LINE__);
    return CUANN_STATUS_INTERNAL_ERROR;
  }
  // originalNumbers
  for (uint32_t i = 0; i < numNewVectors; i++) {
    uint32_t l                   = newVectorLabels[i];
    originalNumbers[indexPtr[l]] = i;
    indexPtr[l] += 1;
  }
  // Recover indexPtr
  for (uint32_t l = 0; l < oldDesc->numClusters; l++) {
    indexPtr[l] -= clusterSize[l];
  }

  //
  // Compute PQ code for new vectors
  //
  uint8_t* pqDataset;  // [numNewVectors, dimPq * bitPq / 8]
  cudaError = cudaMallocManaged(
    &pqDataset, sizeof(uint8_t) * numNewVectors * oldDesc->dimPq * oldDesc->bitPq / 8);
  if (cudaError != cudaSuccess) {
    fprintf(stderr, "(%s, %d) cudaMallocManaged() failed.\n", __func__, __LINE__);
    return CUANN_STATUS_ALLOC_FAILED;
  }
  _cuann_compute_PQ_code(handle,
                         numNewVectors,
                         oldDesc->dimDataset,
                         oldDesc->dimRotDataset,
                         oldDesc->dimPq,
                         oldDesc->lenPq,
                         oldDesc->bitPq,
                         oldDesc->numClusters,
                         dtype,
                         oldDesc->typePqCenter,
                         maxClusterSize,
                         clusterCenters,
                         oldRotationMatrix,
                         newVectors,
                         originalNumbers,
                         clusterSize,
                         indexPtr,
                         oldPqCenters,
                         0,
                         pqDataset);
  cudaSetDevice(cuannDevId);

  //
  // Create descriptor for new index
  //
  cuannIvfPqDescriptor_t newDesc;
  ret = cuannIvfPqCreateDescriptor(&newDesc);
  if (ret != CUANN_STATUS_SUCCESS) { return ret; }
  memcpy(newDesc, oldDesc, sizeof(struct cuannIvfPqDescriptor));
  newDesc->numDataset += numNewVectors;
  fprintf(
    stderr, "(%s) numDataset: %u -> %u\n", __func__, oldDesc->numDataset, newDesc->numDataset);

  //
  // Allocate memory for new index
  //
  size_t newIndexSize;
  ret = cuannIvfPqGetIndexSize(newDesc, &newIndexSize);
  if (ret != CUANN_STATUS_SUCCESS) { return ret; }
  fprintf(stderr, "(%s) indexSize: %lu -> %lu\n", __func__, oldHeader->indexSize, newIndexSize);
  void* newIndex = malloc(newIndexSize);
  memset(newIndex, 0, newIndexSize);
  struct cuannIvfPqIndexHeader* newHeader;
  float* newClusterCenters;      // [numClusters, dimDatasetExt]
  float* newPqCenters;           // [dimPq, 1 << bitPq, lenPq], or
                                 // [numClusters, 1 << bitPq, lenPq]
  uint8_t* newPqDataset;         // [numDataset, dimPq * bitPq / 8]  ***
  uint32_t* newOriginalNumbers;  // [numDataset]  ***
  uint32_t* newIndexPtr;         // [numClusters + 1]  ***
  float* newRotationMatrix;      // [dimDataset, dimRotDataset]
  float* newClusterRotCenters;   // [numClusters, dimRotDataset]
  _cuann_get_index_pointers(newDesc,
                            newIndex,
                            &newHeader,
                            &newClusterCenters,
                            &newPqCenters,
                            &newPqDataset,
                            &newOriginalNumbers,
                            &newIndexPtr,
                            &newRotationMatrix,
                            &newClusterRotCenters);

  //
  // Copy the unchanged parts
  //    header, clusterCenters, pqCenters, rotationMatrix, clusterRotCenters
  //
  memcpy(newHeader, oldHeader, sizeof(struct cuannIvfPqIndexHeader));
  {
    cuannIvfPqGetIndexSize(newDesc, &(newHeader->indexSize));
    newHeader->numDataset = newDesc->numDataset;
    newHeader->numDatasetAdded += numNewVectors;
  }
  memcpy(newClusterCenters, oldClusterCenters, _cuann_getIndexSize_clusterCenters(oldDesc));
  memcpy(newPqCenters, oldPqCenters, _cuann_getIndexSize_pqCenters(oldDesc));
  memcpy(newRotationMatrix, oldRotationMatrix, _cuann_getIndexSize_rotationMatrix(oldDesc));
  memcpy(
    newClusterRotCenters, oldClusterRotCenters, _cuann_getIndexSize_clusterRotCenters(oldDesc));

  //
  // Make newIndexPtr
  //
  maxClusterSize = 0;
  newIndexPtr[0] = 0;
  for (uint32_t l = 0; l < newDesc->numClusters; l++) {
    uint32_t oldClusterSize = oldIndexPtr[l + 1] - oldIndexPtr[l];
    newIndexPtr[l + 1]      = newIndexPtr[l];
    newIndexPtr[l + 1] += oldClusterSize + clusterSize[l];
    maxClusterSize = max(maxClusterSize, oldClusterSize + clusterSize[l]);
  }
  {
    newDesc->maxClusterSize   = maxClusterSize;
    newHeader->maxClusterSize = maxClusterSize;
  }
  fprintf(stderr,
          "(%s) maxClusterSize: %u -> %u\n",
          __func__,
          oldDesc->maxClusterSize,
          newDesc->maxClusterSize);

  //
  // Make newOriginalNumbers
  //
  for (uint32_t i = 0; i < numNewVectors; i++) {
    originalNumbers[i] += oldDesc->numDataset;
  }
  for (uint32_t l = 0; l < newDesc->numClusters; l++) {
    uint32_t oldClusterSize = oldIndexPtr[l + 1] - oldIndexPtr[l];
    memcpy(newOriginalNumbers + newIndexPtr[l],
           oldOriginalNumbers + oldIndexPtr[l],
           sizeof(uint32_t) * oldClusterSize);
    memcpy(newOriginalNumbers + newIndexPtr[l] + oldClusterSize,
           originalNumbers + indexPtr[l],
           sizeof(uint32_t) * clusterSize[l]);
  }

  //
  // Make newPqDataset
  //
  size_t unitPqDataset = newDesc->dimPq * newDesc->bitPq / 8;
  for (uint32_t l = 0; l < newDesc->numClusters; l++) {
    uint32_t oldClusterSize = oldIndexPtr[l + 1] - oldIndexPtr[l];
    memcpy(newPqDataset + unitPqDataset * newIndexPtr[l],
           oldPqDataset + unitPqDataset * oldIndexPtr[l],
           sizeof(uint8_t) * unitPqDataset * oldClusterSize);
    memcpy(newPqDataset + unitPqDataset * (newIndexPtr[l] + oldClusterSize),
           pqDataset + unitPqDataset * indexPtr[l],
           sizeof(uint8_t) * unitPqDataset * clusterSize[l]);
  }

  //
  // Save new index
  //
  ret = cuannIvfPqSaveIndex(handle, newDesc, newIndex, newIndexFileName);
  if (ret != CUANN_STATUS_SUCCESS) { return ret; }
  if (newHeader->numDatasetAdded * 2 >= newHeader->numDataset) {
    fprintf(stderr,
            "(%s) The total number of vectors in the new index"
            " is now more than twice the initial number of vectors."
            " You may want to re-build the index from scratch."
            " (numVectors: %u, numVectorsAdded: %u)\n",
            __func__,
            newHeader->numDataset,
            newHeader->numDatasetAdded);
  }

  //
  // Finalize
  //
  cuannIvfPqDestroyDescriptor(oldDesc);
  cuannIvfPqDestroyDescriptor(newDesc);

  free(originalNumbers);
  free(indexPtr);
  free(newIndex);

  cudaFree(pqDataset);
  cudaFree(clusterSize);
  cudaFree(newVectorLabels);
  cudaFree(clusterCenters);
  cudaFree(oldIndex);

  cuannSetDevice(handle, cuannDevId);
  _cuann_set_device(callerDevId);

  return CUANN_STATUS_SUCCESS;
}

// cuannIvfPqSetSearchParameters
cuannStatus_t cuannIvfPqSetSearchParameters(cuannIvfPqDescriptor_t desc,
                                            const uint32_t numProbes,
                                            const uint32_t topK)
{
  if (desc == NULL) { return CUANN_STATUS_NOT_INITIALIZED; }
  if (numProbes == 0) {
    fprintf(
      stderr, "(%s) numProbes must be larger than zero (numProbes:%u).\n", __func__, numProbes);
    return CUANN_STATUS_INVALID_VALUE;
  }
  if (topK == 0) {
    fprintf(stderr, "(%s) topK must be larger than zero (topK:%u).\n", __func__, topK);
    return CUANN_STATUS_INVALID_VALUE;
  }
  if (numProbes > desc->numClusters) {
    fprintf(stderr,
            "(%s) numProbes must be smaller than or equal to numClusters (numProbes:%u, "
            "numClusters:%u).\n",
            __func__,
            numProbes,
            desc->numClusters);
    return CUANN_STATUS_INVALID_VALUE;
  }
  if (topK > desc->numDataset) {
    fprintf(stderr,
            "(%s) topK must be smaller than or equal to numDataset (topK:%u, numDataset:%u).\n",
            __func__,
            topK,
            desc->numDataset);
    return CUANN_STATUS_INVALID_VALUE;
  }
  uint32_t numSamplesWorstCase = desc->numDataset;
  if (numProbes < desc->numClusters) {
    numSamplesWorstCase =
      desc->numDataset -
      desc->inclusiveSumSortedClusterSize[desc->numClusters - 1 - numProbes -
                                          desc->_numClustersSize0];  // (*) urgent WA, need to be
                                                                     // fixed.
  }
  if (topK > numSamplesWorstCase) {
    fprintf(stderr,
            "(%s) numProbes is too small to get topK results reliably (numProbes:%u, topK:%u, "
            "numSamplesWorstCase:%u).\n",
            __func__,
            numProbes,
            topK,
            numSamplesWorstCase);
    return CUANN_STATUS_INVALID_VALUE;
  }
  desc->numProbes = numProbes;
  desc->topK      = topK;
  if (0) {
    char dtypeString[64];
    fprintf(
      stderr, "# dtypeDataset: %s\n", _cuann_get_dtype_string(desc->dtypeDataset, dtypeString));
  }
  desc->maxSamples = desc->inclusiveSumSortedClusterSize[numProbes - 1];
  if (desc->maxSamples % 128) { desc->maxSamples += 128 - (desc->maxSamples % 128); }
  desc->internalDistanceDtype    = CUDA_R_32F;
  desc->smemLutDtype             = CUDA_R_32F;
  desc->preferredThreadBlockSize = 0;
  // fprintf(stderr, "# maxSample: %u\n", desc->inclusiveSumSortedClusterSize[0]);
  return CUANN_STATUS_SUCCESS;
}

// cuannIvfPqSetSearchParameters
cuannStatus_t cuannIvfPqSetSearchTuningParameters(cuannIvfPqDescriptor_t desc,
                                                  cudaDataType_t internalDistanceDtype,
                                                  cudaDataType_t smemLutDtype,
                                                  const uint32_t preferredThreadBlockSize)
{
  if (desc == NULL) { return CUANN_STATUS_NOT_INITIALIZED; }
  if (internalDistanceDtype != CUDA_R_16F && internalDistanceDtype != CUDA_R_32F) {
    fprintf(
      stderr, "(%s) internalDistanceDtype must be either CUDA_R_16F or CUDA_R_32F\n", __func__);
    return CUANN_STATUS_UNSUPPORTED_DTYPE;
  }
  if (smemLutDtype != CUDA_R_16F && smemLutDtype != CUDA_R_32F && smemLutDtype != CUDA_R_8U) {
    fprintf(stderr, "(%s) smemLutDtype must be CUDA_R_16F, CUDA_R_32F or CUDA_R_8U\n", __func__);
    return CUANN_STATUS_UNSUPPORTED_DTYPE;
  }
  if (preferredThreadBlockSize != 256 && preferredThreadBlockSize != 512 &&
      preferredThreadBlockSize != 1024 && preferredThreadBlockSize != 0) {
    fprintf(stderr,
            "(%s) preferredThreadBlockSize must be 0, 256, 512 or 1024. %u is given.\n",
            __func__,
            preferredThreadBlockSize);
    return CUANN_STATUS_UNSUPPORTED_DTYPE;
  }
  desc->internalDistanceDtype = internalDistanceDtype;
  desc->smemLutDtype          = smemLutDtype;
  if (0) {
    char dtypeString[64];
    fprintf(stderr,
            "# internalDistanceDtype: %s\n",
            _cuann_get_dtype_string(desc->internalDistanceDtype, dtypeString));
  }
  desc->preferredThreadBlockSize = preferredThreadBlockSize;
  // fprintf(stderr, "# maxSample: %u\n", desc->inclusiveSumSortedClusterSize[0]);
  return CUANN_STATUS_SUCCESS;
}

// cuannIvfPqGetSearchParameters
cuannStatus_t cuannIvfPqGetSearchParameters(cuannIvfPqDescriptor_t desc,
                                            uint32_t* numProbes,
                                            uint32_t* topK)
{
  if (desc == NULL) { return CUANN_STATUS_NOT_INITIALIZED; }
  *numProbes = desc->numProbes;
  *topK      = desc->topK;
  return CUANN_STATUS_SUCCESS;
}

// cuannIvfPqGetSearchTuningParameters
cuannStatus_t cuannIvfPqGetSearchTuningParameters(cuannIvfPqDescriptor_t desc,
                                                  cudaDataType_t* internalDistanceDtype,
                                                  cudaDataType_t* smemLutDtype,
                                                  uint32_t* preferredThreadBlockSize)
{
  if (desc == NULL) { return CUANN_STATUS_NOT_INITIALIZED; }
  *internalDistanceDtype    = desc->internalDistanceDtype;
  *smemLutDtype             = desc->smemLutDtype;
  *preferredThreadBlockSize = desc->preferredThreadBlockSize;
  return CUANN_STATUS_SUCCESS;
}

// cuannIvfPqSearch
cuannStatus_t cuannIvfPqSearch_bufferSize(cuannHandle_t handle,
                                          cuannIvfPqDescriptor_t desc,
                                          const void* index,
                                          uint32_t maxQueries,
                                          size_t maxWorkspaceSize,
                                          size_t* workspaceSize)
{
  if (handle == NULL || desc == NULL) { return CUANN_STATUS_NOT_INITIALIZED; }

  size_t max_ws = maxWorkspaceSize;
  if (max_ws == 0) {
    max_ws = (size_t)1 * 1024 * 1024 * 1024;  // default, 1GB
  } else {
    max_ws = max(max_ws, (size_t)512 * 1024 * 1024);
  }

  size_t size_0 =
    _cuann_aligned(sizeof(float) * maxQueries * desc->dimDatasetExt) +  // devQueries
    _cuann_aligned(sizeof(float) * maxQueries * desc->dimDatasetExt) +  // curQueries
    _cuann_aligned(sizeof(float) * maxQueries * desc->dimRotDataset) +  // rotQueries
    _cuann_aligned(sizeof(uint32_t) * maxQueries * desc->numProbes) +   // clusterLabels..
    _cuann_aligned(sizeof(float) * maxQueries * desc->numClusters) +    // QCDistances
    _cuann_find_topk_bufferSize(handle, desc->numProbes, maxQueries, desc->numClusters);
  if (size_0 > max_ws) {
    maxQueries = maxQueries * max_ws / size_0;
    if (maxQueries > 32) { maxQueries -= (maxQueries % 32); }
    // fprintf(stderr, "(%s) maxQueries is reduced to %u.\n", __func__, maxQueries);
  }
  // maxQueries = min(max(maxQueries, 1), 1024);
  // maxQueries = min(max(maxQueries, 1), 2048);
  maxQueries       = min(max(maxQueries, 1), 4096);
  desc->maxQueries = maxQueries;

  *workspaceSize =
    _cuann_aligned(sizeof(float) * maxQueries * desc->dimDatasetExt) +  // devQueries
    _cuann_aligned(sizeof(float) * maxQueries * desc->dimDatasetExt) +  // curQueries
    _cuann_aligned(sizeof(float) * maxQueries * desc->dimRotDataset) +  // rotQueries
    _cuann_aligned(sizeof(uint32_t) * maxQueries * desc->numProbes);    // clusterLabels..

  max_ws -= *workspaceSize;
  desc->maxBatchSize = 1;
  while (1) {
    uint32_t nextBatchSize = desc->maxBatchSize * max_ws / ivfpq_search_bufferSize(handle, desc);
    if (desc->maxBatchSize >= nextBatchSize) break;
    desc->maxBatchSize = nextBatchSize;
  }
  desc->maxBatchSize = min(max(desc->maxBatchSize, 1), maxQueries);

  if (maxQueries > desc->maxBatchSize) {
    // Adjust maxBatchSize to reduce workspace size.
    uint32_t num = (maxQueries + desc->maxBatchSize - 1) / desc->maxBatchSize;
    if (1 < num && num < 5) { desc->maxBatchSize = (maxQueries + num - 1) / num; }
  }

  if (1) {
    // Adjust maxBatchSize to improve GPU occupancy of topk kernel.
    uint32_t numCta_total    = (handle->deviceProp).multiProcessorCount * 2;
    uint32_t numCta_perBatch = numCta_total / desc->maxBatchSize;
    float utilization        = (float)numCta_perBatch * desc->maxBatchSize / numCta_total;
    if (numCta_perBatch > 1 || (numCta_perBatch == 1 && utilization < 0.6)) {
      uint32_t numCta_perBatch_1 = numCta_perBatch + 1;
      uint32_t maxBatchSize_1    = numCta_total / numCta_perBatch_1;
      float utilization_1        = (float)numCta_perBatch_1 * maxBatchSize_1 / numCta_total;
      // fprintf(stderr, "# maxBatchSize  :%u, utilization  :%f\n", desc->maxBatchSize,
      // utilization); fprintf(stderr, "# maxBatchSize_1:%u, utilization_1:%f\n", maxBatchSize_1,
      // utilization_1);
      if (utilization < utilization_1) { desc->maxBatchSize = maxBatchSize_1; }
    }
  }

  size_t size_1 =
    _cuann_aligned(sizeof(float) * maxQueries * desc->numClusters) +  // QCDistance
    _cuann_find_topk_bufferSize(handle, desc->numProbes, maxQueries, desc->numClusters);
  size_t size_2 = ivfpq_search_bufferSize(handle, desc);
  *workspaceSize += max(size_1, size_2);

#ifdef CUANN_DEBUG
  fprintf(stderr, "# maxQueries: %u\n", maxQueries);
  fprintf(stderr, "# maxBatchSize: %u\n", desc->maxBatchSize);
  fprintf(stderr,
          "# workspaceSize: %lu (%.3f GiB)\n",
          *workspaceSize,
          (float)*workspaceSize / 1024 / 1024 / 1024);
#endif

  return CUANN_STATUS_SUCCESS;
}

// cuannIvfPqSearch
cuannStatus_t cuannIvfPqSearch(
  cuannHandle_t handle,
  cuannIvfPqDescriptor_t desc,
  const void* index,
  const void* queries,  // [numQueries, dimDataset], host or device pointer
  cudaDataType_t dtype,
  uint32_t numQueries,
  uint64_t* neighbors,  // [numQueries, topK], device pointer
  float* distances,     // [numQueries, topK], device pointer
  void* workspace)
{
  if (handle == NULL || desc == NULL) { return CUANN_STATUS_NOT_INITIALIZED; }
  int orgDevId = _cuann_set_device(handle->devId);

  if (dtype != CUDA_R_32F && dtype != CUDA_R_8U && dtype != CUDA_R_8I) {
    return CUANN_STATUS_UNSUPPORTED_DTYPE;
  }

  struct cuannIvfPqIndexHeader* header;
  float* clusterCenters;      // [numClusters, dimDatasetExt]
  float* pqCenters;           // [dimPq, 1 << bitPq, lenPq], or
                              // [numClusters, 1 << bitPq, lenPq]
  uint8_t* pqDataset;         // [numDataset, dimPq * bitPq / 8]
  uint32_t* originalNumbers;  // [numDataset]
  uint32_t* indexPtr;         // [numClusters + 1]
  float* rotationMatrix;      // [dimDataset, dimRotDataset]
  float* clusterRotCenters;   // [numClusters, dimRotDataset]
  _cuann_get_index_pointers(desc,
                            index,
                            &header,
                            &clusterCenters,
                            &pqCenters,
                            &pqDataset,
                            &originalNumbers,
                            &indexPtr,
                            &rotationMatrix,
                            &clusterRotCenters);
  //
  void* devQueries;                // [maxQueries, dimDatasetExt]
  float* curQueries;               // [maxQueries, dimDatasetExt]
  float* rotQueries;               // [maxQueries, dimRotDataset]
  uint32_t* clusterLabelsToProbe;  // [maxQueries, numProbes]
  float* QCDistances;              // [maxQueries, numClusters]
  void* topkWorkspace;
  void* searchWorkspace;
  devQueries = (void*)workspace;
  curQueries = (float*)((uint8_t*)devQueries +
                        _cuann_aligned(sizeof(float) * desc->maxQueries * desc->dimDatasetExt));
  rotQueries = (float*)((uint8_t*)curQueries +
                        _cuann_aligned(sizeof(float) * desc->maxQueries * desc->dimDatasetExt));
  clusterLabelsToProbe =
    (uint32_t*)((uint8_t*)rotQueries +
                _cuann_aligned(sizeof(float) * desc->maxQueries * desc->dimRotDataset));
  //
  QCDistances   = (float*)((uint8_t*)clusterLabelsToProbe +
                         _cuann_aligned(sizeof(uint32_t) * desc->maxQueries * desc->numProbes));
  topkWorkspace = (void*)((uint8_t*)QCDistances +
                          _cuann_aligned(sizeof(float) * desc->maxQueries * desc->numClusters));
  //
  searchWorkspace = (void*)((uint8_t*)clusterLabelsToProbe +
                            _cuann_aligned(sizeof(uint32_t) * desc->maxQueries * desc->numProbes));

  void (*_ivfpq_search)(cuannHandle_t,
                        cuannIvfPqDescriptor_t,
                        uint32_t,
                        const float*,
                        const float*,
                        const uint8_t*,
                        const uint32_t*,
                        const uint32_t*,
                        const uint32_t*,
                        const float*,
                        uint64_t*,
                        float*,
                        void*);
  if (desc->internalDistanceDtype == CUDA_R_16F) {
    if (desc->smemLutDtype == CUDA_R_16F) {
      _ivfpq_search = ivfpq_search<half, half>;
    } else if (desc->smemLutDtype == CUDA_R_8U) {
      _ivfpq_search = ivfpq_search<half, fp_8bit<5>>;
    } else {
      _ivfpq_search = ivfpq_search<half, float>;
    }
  } else {
    if (desc->smemLutDtype == CUDA_R_16F) {
      _ivfpq_search = ivfpq_search<float, half>;
    } else if (desc->smemLutDtype == CUDA_R_8U) {
      _ivfpq_search = ivfpq_search<float, fp_8bit<5>>;
    } else {
      _ivfpq_search = ivfpq_search<float, float>;
    }
  }

  cublasStatus_t cublasError;
  cudaPointerAttributes attr;
  cudaPointerGetAttributes(&attr, neighbors);
  if (attr.type != cudaMemoryTypeDevice && attr.type != cudaMemoryTypeManaged) {
    fprintf(stderr, "(%s) neighbors must be accessible from the device.\n", __func__);
    return CUANN_STATUS_INVALID_POINTER;
  }
  cudaPointerGetAttributes(&attr, distances);
  if (attr.type != cudaMemoryTypeDevice && attr.type != cudaMemoryTypeManaged) {
    fprintf(stderr, "(%s) distances must be accessible from the device.\n", __func__);
    return CUANN_STATUS_INVALID_POINTER;
  }
  cudaPointerGetAttributes(&attr, queries);

#ifdef CUANN_DEBUG
  cudaError_t cudaError;
#endif

  for (uint32_t i = 0; i < numQueries; i += desc->maxQueries) {
    uint32_t nQueries = min(desc->maxQueries, numQueries - i);

    float fillValue = 0.0;
    if (desc->similarity == CUANN_SIMILARITY_L2) { fillValue = 1.0 / -2.0; }
    float divisor = 1.0;
    if (desc->dtypeDataset == CUDA_R_8U) {
      divisor = 256.0;
    } else if (desc->dtypeDataset == CUDA_R_8I) {
      divisor = 128.0;
    }
    if (dtype == CUDA_R_32F) {
      float* ptrQueries = (float*)queries + ((uint64_t)(desc->dimDataset) * i);
      if (attr.type != cudaMemoryTypeDevice && attr.type != cudaMemoryTypeManaged) {
        cudaMemcpyAsync(devQueries,
                        ptrQueries,
                        sizeof(float) * nQueries * desc->dimDataset,
                        cudaMemcpyHostToDevice,
                        handle->stream);
        ptrQueries = (float*)devQueries;
      }
      _cuann_copy_fill<float, float>(nQueries,
                                     desc->dimDataset,
                                     ptrQueries,
                                     desc->dimDataset,
                                     curQueries,
                                     desc->dimDatasetExt,
                                     fillValue,
                                     divisor,
                                     handle->stream);
    } else if (dtype == CUDA_R_8U) {
      uint8_t* ptrQueries = (uint8_t*)queries + ((uint64_t)(desc->dimDataset) * i);
      if (attr.type != cudaMemoryTypeDevice && attr.type != cudaMemoryTypeManaged) {
        cudaMemcpyAsync(devQueries,
                        ptrQueries,
                        sizeof(uint8_t) * nQueries * desc->dimDataset,
                        cudaMemcpyHostToDevice,
                        handle->stream);
        ptrQueries = (uint8_t*)devQueries;
      }
      _cuann_copy_fill<uint8_t, float>(nQueries,
                                       desc->dimDataset,
                                       ptrQueries,
                                       desc->dimDataset,
                                       curQueries,
                                       desc->dimDatasetExt,
                                       fillValue,
                                       divisor,
                                       handle->stream);
    } else if (dtype == CUDA_R_8I) {
      int8_t* ptrQueries = (int8_t*)queries + ((uint64_t)(desc->dimDataset) * i);
      if (attr.type != cudaMemoryTypeDevice && attr.type != cudaMemoryTypeManaged) {
        cudaMemcpyAsync(devQueries,
                        ptrQueries,
                        sizeof(int8_t) * nQueries * desc->dimDataset,
                        cudaMemcpyHostToDevice,
                        handle->stream);
        ptrQueries = (int8_t*)devQueries;
      }
      _cuann_copy_fill<int8_t, float>(nQueries,
                                      desc->dimDataset,
                                      ptrQueries,
                                      desc->dimDataset,
                                      curQueries,
                                      desc->dimDatasetExt,
                                      fillValue,
                                      divisor,
                                      handle->stream);
    }

    float alpha;
    float beta;
    uint32_t gemmK = desc->dimDataset;
    if (desc->similarity == CUANN_SIMILARITY_INNER) {
      alpha = -1.0;
      beta  = 0.0;
    } else {
      alpha = -2.0;
      beta  = 0.0;
      gemmK = desc->dimDataset + 1;
      assert(gemmK <= desc->dimDatasetExt);
    }
    cublasError = cublasGemmEx(handle->cublasHandle,
                               CUBLAS_OP_T,
                               CUBLAS_OP_N,
                               desc->numClusters,
                               nQueries,
                               gemmK,
                               &alpha,
                               clusterCenters,
                               CUDA_R_32F,
                               desc->dimDatasetExt,
                               curQueries,
                               CUDA_R_32F,
                               desc->dimDatasetExt,
                               &beta,
                               QCDistances,
                               CUDA_R_32F,
                               desc->numClusters,
                               CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (cublasError != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "(%s, %d) cublasGemmEx() failed.\n", __func__, __LINE__);
      return CUANN_STATUS_CUBLAS_ERROR;
    }

    // Rotate queries
    alpha       = 1.0;
    beta        = 0.0;
    cublasError = cublasGemmEx(handle->cublasHandle,
                               CUBLAS_OP_T,
                               CUBLAS_OP_N,
                               desc->dimRotDataset,
                               nQueries,
                               desc->dimDataset,
                               &alpha,
                               rotationMatrix,
                               CUDA_R_32F,
                               desc->dimDataset,
                               curQueries,
                               CUDA_R_32F,
                               desc->dimDatasetExt,
                               &beta,
                               rotQueries,
                               CUDA_R_32F,
                               desc->dimRotDataset,
                               CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (cublasError != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "(%s, %d) cublasGemmEx() failed.\n", __func__, __LINE__);
      return CUANN_STATUS_CUBLAS_ERROR;
    }

    // Select neighbor clusters for each query.
    _cuann_find_topk(handle,
                     desc->numProbes,
                     nQueries,
                     desc->numClusters,
                     NULL,
                     QCDistances,
                     clusterLabelsToProbe,
                     topkWorkspace,
                     false);
#ifdef CUANN_DEBUG
    cudaError = cudaDeviceSynchronize();
    if (cudaError != cudaSuccess) {
      fprintf(stderr, "(%s, %d) cudaDeviceSynchronize() failed.\n", __func__, __LINE__);
      return CUANN_STATUS_CUDA_ERROR;
    }
#endif
    //
    for (uint32_t j = 0; j < nQueries; j += desc->maxBatchSize) {
      uint32_t batchSize = min(desc->maxBatchSize, nQueries - j);
      _ivfpq_search(handle,
                    desc,
                    batchSize,
                    clusterRotCenters,
                    pqCenters,
                    pqDataset,
                    originalNumbers,
                    indexPtr,
                    clusterLabelsToProbe + ((uint64_t)(desc->numProbes) * j),
                    rotQueries + ((uint64_t)(desc->dimRotDataset) * j),
                    neighbors + ((uint64_t)(desc->topK) * (i + j)),
                    distances + ((uint64_t)(desc->topK) * (i + j)),
                    searchWorkspace);
#ifdef CUANN_DEBUG
      cudaError = cudaDeviceSynchronize();
      if (cudaError != cudaSuccess) {
        fprintf(
          stderr, "(%s, %d) cudaDeviceSynchronize() failed (%d)\n", __func__, __LINE__, cudaError);
        fprintf(stderr, "# i:%u, nQueries:%u, j:%u, batchSize:%u\n", i, nQueries, j, batchSize);
        return CUANN_STATUS_CUDA_ERROR;
      }
#endif
    }
  }

#ifdef CUANN_DEBUG
  cudaError = cudaDeviceSynchronize();
  if (cudaError != cudaSuccess) {
    fprintf(stderr, "(%s, %d) cudaDeviceSynchronize() failed.\n", __func__, __LINE__);
    return CUANN_STATUS_CUDA_ERROR;
  }
#endif

  _cuann_set_device(orgDevId);
  return CUANN_STATUS_SUCCESS;
}

}  // namespace raft::spatial::knn::ivf_pq
