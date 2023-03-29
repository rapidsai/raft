/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <raft/core/logger.hpp>
#include <raft/neighbors/detail/cagra/topk_for_cagra/topk.h>
#include <raft/neighbors/detail/cagra/topk_for_cagra/topk_core.cuh>

namespace raft::neighbors::experimental::cagra::detail {

namespace {

//
constexpr std::uint32_t NUM_THREADS      = 1024;  // DO NOT CHANGE
constexpr std::uint32_t STATE_BIT_LENGTH = 8;     // 0: state not used,  8: state used
constexpr std::uint32_t MAX_VEC_LENGTH   = 4;     // 1, 2, 4 or 8

//
//
int _get_vecLen(uint32_t maxSamples, int maxVecLen = MAX_VEC_LENGTH)
{
  int vecLen = min(maxVecLen, MAX_VEC_LENGTH);
  while ((maxSamples % vecLen) != 0) {
    vecLen /= 2;
  }
  return vecLen;
}
}  // unnamed namespace

template <int blockDim_x, int stateBitLen, int vecLen, int maxTopk, int numSortThreads>
__launch_bounds__(1024, 1) __global__
  void kern_topk_cta_11(uint32_t topk,
                        uint32_t size_batch,
                        uint32_t len_x,
                        const uint32_t* _x,  // [size_batch, ld_x,]
                        uint32_t ld_x,
                        const uint32_t* _in_vals,  // [size_batch, ld_iv,]
                        uint32_t ld_iv,
                        uint32_t* _y,  // [size_batch, ld_y,]
                        uint32_t ld_y,
                        uint32_t* _out_vals,  // [size_batch, ld_ov,]
                        uint32_t ld_ov,
                        uint8_t* _state,   // [size_batch, ...,]
                        uint32_t* _hints,  // [size_batch,]
                        bool sort)
{
  uint32_t i_batch = blockIdx.x;
  if (i_batch >= size_batch) return;
  __shared__ uint32_t _smem[2 * maxTopk + 2048 + 8];

  topk_cta_11_core<blockDim_x, stateBitLen, vecLen, maxTopk, numSortThreads>(
    topk,
    len_x,
    (_x == NULL ? NULL : _x + i_batch * ld_x),
    (_in_vals == NULL ? NULL : _in_vals + i_batch * ld_iv),
    (_y == NULL ? NULL : _y + i_batch * ld_y),
    (_out_vals == NULL ? NULL : _out_vals + i_batch * ld_ov),
    (_state == NULL ? NULL : _state + i_batch * get_state_size<blockDim_x, stateBitLen>(len_x)),
    (_hints == NULL ? NULL : _hints + i_batch),
    sort,
    _smem);
}

//
size_t _cuann_find_topk_bufferSize(uint32_t topK,
                                   uint32_t sizeBatch,
                                   uint32_t numElements,
                                   cudaDataType_t sampleDtype)
{
  constexpr int numThreads  = NUM_THREADS;
  constexpr int stateBitLen = STATE_BIT_LENGTH;
  assert(stateBitLen == 0 || stateBitLen == 8);

  size_t workspaceSize = 1;
  // state
  if (stateBitLen == 8) {
    workspaceSize = _cuann_aligned(
      sizeof(uint8_t) * get_state_size<numThreads, stateBitLen>(numElements) * sizeBatch);
  }

  return workspaceSize;
}

//
void _cuann_find_topk(uint32_t topK,
                      uint32_t sizeBatch,
                      uint32_t numElements,
                      const float* inputKeys,     // [sizeBatch, ldIK,]
                      uint32_t ldIK,              // (*) ldIK >= numElements
                      const uint32_t* inputVals,  // [sizeBatch, ldIV,]
                      uint32_t ldIV,              // (*) ldIV >= numElements
                      float* outputKeys,          // [sizeBatch, ldOK,]
                      uint32_t ldOK,              // (*) ldOK >= topK
                      uint32_t* outputVals,       // [sizeBatch, ldOV,]
                      uint32_t ldOV,              // (*) ldOV >= topK
                      void* workspace,
                      bool sort,
                      uint32_t* hints,
                      cudaStream_t stream)
{
  assert(ldIK >= numElements);
  assert(ldIV >= numElements);
  assert(ldOK >= topK);
  assert(ldOV >= topK);

  constexpr int numThreads  = NUM_THREADS;
  constexpr int stateBitLen = STATE_BIT_LENGTH;
  assert(stateBitLen == 0 || stateBitLen == 8);

  uint8_t* state = NULL;
  if (stateBitLen == 8) { state = (uint8_t*)workspace; }

  dim3 threads(numThreads, 1, 1);
  dim3 blocks(sizeBatch, 1, 1);

  void (*cta_kernel)(uint32_t,
                     uint32_t,
                     uint32_t,
                     const uint32_t*,
                     uint32_t,
                     const uint32_t*,
                     uint32_t,
                     uint32_t*,
                     uint32_t,
                     uint32_t*,
                     uint32_t,
                     uint8_t*,
                     uint32_t*,
                     bool) = nullptr;

  // V:vecLen, K:maxTopk, T:numSortThreads
#define SET_KERNEL_VKT(V, K, T)                                      \
  do {                                                               \
    assert(numThreads >= T);                                         \
    assert((K % T) == 0);                                            \
    assert((K / T) <= 4);                                            \
    cta_kernel = kern_topk_cta_11<numThreads, stateBitLen, V, K, T>; \
  } while (0)

  // V: vecLen
#define SET_KERNEL_V(V)                                                                      \
  do {                                                                                       \
    if (topK <= 32) {                                                                        \
      SET_KERNEL_VKT(V, 32, 32);                                                             \
    } else if (topK <= 64) {                                                                 \
      SET_KERNEL_VKT(V, 64, 32);                                                             \
    } else if (topK <= 96) {                                                                 \
      SET_KERNEL_VKT(V, 96, 32);                                                             \
    } else if (topK <= 128) {                                                                \
      SET_KERNEL_VKT(V, 128, 32);                                                            \
    } else if (topK <= 192) {                                                                \
      SET_KERNEL_VKT(V, 192, 64);                                                            \
    } else if (topK <= 256) {                                                                \
      SET_KERNEL_VKT(V, 256, 64);                                                            \
    } else if (topK <= 384) {                                                                \
      SET_KERNEL_VKT(V, 384, 128);                                                           \
    } else if (topK <= 512) {                                                                \
      SET_KERNEL_VKT(V, 512, 128);                                                           \
    } else if (topK <= 768) {                                                                \
      SET_KERNEL_VKT(V, 768, 256);                                                           \
    } else if (topK <= 1024) {                                                               \
      SET_KERNEL_VKT(V, 1024, 256);                                                          \
    } \
        /* else if (topK <= 1536) { SET_KERNEL_VKT(V, 1536, 512); } */ \
        /* else if (topK <= 2048) { SET_KERNEL_VKT(V, 2048, 512); } */ \
        /* else if (topK <= 3072) { SET_KERNEL_VKT(V, 3072, 1024); } */ \
        /* else if (topK <= 4096) { SET_KERNEL_VKT(V, 4096, 1024); } */ \
        else {                                                                                      \
      RAFT_LOG_DEBUG(                                                                        \
        "[ERROR] (%s, %d) topk must be lower than or equla to 1024.\n", __func__, __LINE__); \
      exit(-1);                                                                              \
    }                                                                                        \
  } while (0)

  int _vecLen = _get_vecLen(ldIK, 2);
  if (_vecLen == 2) {
    SET_KERNEL_V(2);
  } else if (_vecLen == 1) {
    SET_KERNEL_V(1);
  }

  cta_kernel<<<blocks, threads, 0, stream>>>(topK,
                                             sizeBatch,
                                             numElements,
                                             (const uint32_t*)inputKeys,
                                             ldIK,
                                             inputVals,
                                             ldIV,
                                             (uint32_t*)outputKeys,
                                             ldOK,
                                             outputVals,
                                             ldOV,
                                             state,
                                             hints,
                                             sort);

  return;
}
}  // namespace raft::neighbors::experimental::cagra::detail
