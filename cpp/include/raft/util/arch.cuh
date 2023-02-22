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
#pragma once

namespace raft::arch {

/* raft::arch provides the following facilities:
 *
 * - raft::arch::SM_XX : hardcoded compile-time constants for various compute
 *   architectures. The values raft::arch::SM_min and raft::arch::SM_future
 *   represent architectures that are always smaller and larger (respectively)
 *   than any architecture that can be encountered in practice.
 *
 * - raft::arch::SM_compute_arch : a compile-time value for the *current*
 *   compute architecture that a kernel is compiled with. It can only be used
 *   inside kernels with a template argument.
 *
 * - raft::arch::kernel_runtime_arch : a function that computes at *run-time*
 *   which version of a kernel will launch (i.e., it will return the compute
 *   architecture of the version of the kernel that will be launched by the
 *   driver).
 *
 * - raft::arch::SM_range : a compile-time value to represent an open interval
 *   of compute architectures. This can be used to check if the current
 *   compile-time architecture is in a specified compatibility range.
 */

// inner::SM_generic is a template to create a generic compile-time SM
// architecture constant.
namespace inner {
template <int n>
struct SM_generic {
public:
  __host__ __device__ constexpr int value() const {
    return n;
  }
};

// A
__global__ inline void dummy_runtime_kernel() {}
}

// A list of architectures that RAPIDS explicitly builds for (SM60, ..., SM90)
// and SM_MIN and SM_FUTURE, that allow specifying an open interval of
// compatible compute architectures.
using SM_min = inner::SM_generic<350>;
using SM_60 = inner::SM_generic<600>;
using SM_70 = inner::SM_generic<700>;
using SM_75 = inner::SM_generic<750>;
using SM_80 = inner::SM_generic<800>;
using SM_86 = inner::SM_generic<860>;
using SM_90 = inner::SM_generic<900>;
using SM_future = inner::SM_generic<99999>;

// This is a type that uses the __CUDA_ARCH__ macro to obtain the compile-time
// compute architecture. It can only be used where __CUDA_ARCH__ is defined,
// i.e., inside a __global__ function template.
struct SM_compute_arch {
  template <int dummy = 0>
  __host__ __device__ constexpr int value() const {
#ifdef __CUDA_ARCH__
    return __CUDA_ARCH__;
#else
    static_assert(dummy != 0,
                  "SM_compute_arch.value() is only callable from a __global__ function template. "
                  "A way to create a function template is by adding 'template <int dummy = 0>'.");
    return -1;
#endif
  }
};

// A runtime value for the actual compute architecture of a kernel.
//
// A single kernel can be compiled for several "virtual" compute architectures.
// When a program runs, the driver picks the version of the kernel that most
// closely matches the current hardware. This struct reflects the virtual
// compute architecture of the version of the kernel that the driver picks when
// the kernel runs.
struct SM_runtime {
  friend SM_runtime kernel_runtime_arch();
private:
  const int _version;
  SM_runtime(int version)
    : _version (version) {}

public:
  __host__ __device__ int value() const  {
    return _version;
  }
};

// Computes which compute architecture of a kernel will run
//
// Semantics are described above in the documentation of SM_runtime.
SM_runtime kernel_runtime_arch() {
  auto kernel = inner::dummy_runtime_kernel;
  cudaFuncAttributes attributes;
  cudaFuncGetAttributes(&attributes, kernel);

  return SM_runtime(10 * attributes.ptxVersion);
}

// SM_range represents a range of SM architectures. It can be used to
// conditionally compile a kernel.
template <typename SM_MIN, typename SM_MAX>
struct SM_range {
private:
  const SM_MIN _min;
  const SM_MAX _max;
public:
  __host__ __device__ constexpr SM_range(SM_MIN min, SM_MAX max)
    : _min(min), _max(max) {}

  template <typename SM_t>
  __host__ __device__ constexpr bool contains(SM_t current) const {
    return _min.value() <= current.value() && current.value() < _max.value();
  }
};

} // namespace raft::arch
