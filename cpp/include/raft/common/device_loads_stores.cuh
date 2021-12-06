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

namespace raft {

/**
 * @defgroup SmemStores Shared memory store operations
 * @{
 * @brief Stores to shared memory (both vectorized and non-vectorized forms)
 *        requires the given shmem pointer to be aligned by the vector
          length, like for float4 lds/sts shmem pointer should be aligned
          by 16 bytes else it might silently fail or can also give
          runtime error.
 * @param[out] addr shared memory address (should be aligned to vector size)
 * @param[in]  x    data to be stored at this address
 */
DI void sts(float* addr, const float& x)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<float*>(addr));
  asm volatile("st.shared.f32 [%0], {%1};" : : "l"(s1), "f"(x));
}
DI void sts(float* addr, const float (&x)[1])
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<float*>(addr));
  asm volatile("st.shared.f32 [%0], {%1};" : : "l"(s1), "f"(x[0]));
}
DI void sts(float* addr, const float (&x)[2])
{
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<float2*>(addr));
  asm volatile("st.shared.v2.f32 [%0], {%1, %2};" : : "l"(s2), "f"(x[0]), "f"(x[1]));
}
DI void sts(float* addr, const float (&x)[4])
{
  auto s4 = __cvta_generic_to_shared(reinterpret_cast<float4*>(addr));
  asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(s4), "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]));
}

DI void sts(double* addr, const double& x)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<double*>(addr));
  asm volatile("st.shared.f64 [%0], {%1};" : : "l"(s1), "d"(x));
}
DI void sts(double* addr, const double (&x)[1])
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<double*>(addr));
  asm volatile("st.shared.f64 [%0], {%1};" : : "l"(s1), "d"(x[0]));
}
DI void sts(double* addr, const double (&x)[2])
{
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<double2*>(addr));
  asm volatile("st.shared.v2.f64 [%0], {%1, %2};" : : "l"(s2), "d"(x[0]), "d"(x[1]));
}
/** @} */

/**
 * @defgroup SmemLoads Shared memory load operations
 * @{
 * @brief Loads from shared memory (both vectorized and non-vectorized forms)
          requires the given shmem pointer to be aligned by the vector
          length, like for float4 lds/sts shmem pointer should be aligned
          by 16 bytes else it might silently fail or can also give
          runtime error.
 * @param[out] x    the data to be loaded
 * @param[in]  addr shared memory address from where to load
 *                  (should be aligned to vector size)
 */
DI void lds(float& x, float* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<float*>(addr));
  asm volatile("ld.shared.f32 {%0}, [%1];" : "=f"(x) : "l"(s1));
}
DI void lds(float (&x)[1], float* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<float*>(addr));
  asm volatile("ld.shared.f32 {%0}, [%1];" : "=f"(x[0]) : "l"(s1));
}
DI void lds(float (&x)[2], float* addr)
{
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<float2*>(addr));
  asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(x[0]), "=f"(x[1]) : "l"(s2));
}
DI void lds(float (&x)[4], float* addr)
{
  auto s4 = __cvta_generic_to_shared(reinterpret_cast<float4*>(addr));
  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(x[0]), "=f"(x[1]), "=f"(x[2]), "=f"(x[3])
               : "l"(s4));
}
DI void lds(double& x, double* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<double*>(addr));
  asm volatile("ld.shared.f64 {%0}, [%1];" : "=d"(x) : "l"(s1));
}
DI void lds(double (&x)[1], double* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<double*>(addr));
  asm volatile("ld.shared.f64 {%0}, [%1];" : "=d"(x[0]) : "l"(s1));
}
DI void lds(double (&x)[2], double* addr)
{
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<double2*>(addr));
  asm volatile("ld.shared.v2.f64 {%0, %1}, [%2];" : "=d"(x[0]), "=d"(x[1]) : "l"(s2));
}
/** @} */

/**
 * @defgroup GlobalLoads Global cached load operations
 * @{
 * @brief Load from global memory with caching at L1 level
 * @param[out] x    data to be loaded from global memory
 * @param[in]  addr address in global memory from where to load
 */
DI void ldg(float& x, const float* addr)
{
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(x) : "l"(addr));
}
DI void ldg(float (&x)[1], const float* addr)
{
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(x[0]) : "l"(addr));
}
DI void ldg(float (&x)[2], const float* addr)
{
  asm volatile("ld.global.cg.v2.f32 {%0, %1}, [%2];" : "=f"(x[0]), "=f"(x[1]) : "l"(addr));
}
DI void ldg(float (&x)[4], const float* addr)
{
  asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(x[0]), "=f"(x[1]), "=f"(x[2]), "=f"(x[3])
               : "l"(addr));
}
DI void ldg(double& x, const double* addr)
{
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(x) : "l"(addr));
}
DI void ldg(double (&x)[1], const double* addr)
{
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(x[0]) : "l"(addr));
}
DI void ldg(double (&x)[2], const double* addr)
{
  asm volatile("ld.global.cg.v2.f64 {%0, %1}, [%2];" : "=d"(x[0]), "=d"(x[1]) : "l"(addr));
}
/** @} */

}  // namespace raft
