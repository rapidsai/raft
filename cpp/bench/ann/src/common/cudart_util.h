/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#ifndef CUDART_UTIL_H_
#define CUDART_UTIL_H_
#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>

#define ANN_CUDA_CHECK(call)                                   \
  {                                                            \
    raft::bench::ann::cuda_check_((call), __FILE__, __LINE__); \
  }

#ifndef NDEBUG
#define ANN_CUDA_CHECK_LAST_ERROR()                               \
  {                                                               \
    raft::bench::ann::cuda_check_last_error_(__FILE__, __LINE__); \
  }
#else
#define ANN_CUDA_CHECK_LAST_ERROR()
#endif

namespace raft::bench::ann {

constexpr unsigned int WARP_FULL_MASK = 0xffffffff;
constexpr int WARP_SIZE               = 32;

class CudaException : public std::runtime_error {
 public:
  explicit CudaException(const std::string& what) : runtime_error(what) {}
};

inline void cuda_check_(cudaError_t val, const char* file, int line)
{
  if (val != cudaSuccess) {
    throw CudaException(std::string(file) + ":" + std::to_string(line) + ": CUDA error " +
                        std::to_string(val) + ": " + cudaGetErrorName(val) + ": " +
                        cudaGetErrorString(val));
  }
}

inline void cuda_check_last_error_(const char* file, int line)
{
  cudaDeviceSynchronize();
  cudaError_t err = cudaPeekAtLastError();
  cuda_check_(err, file, line);
}

}  // namespace raft::bench::ann
#endif
