/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <cstdint>  // uintX_t
#include <raft/core/device_span.hpp>
#include <raft/util/cuda_dev_essentials.cuh>  // DI

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
DI void sts(uint8_t* addr, const uint8_t& x)
{
  uint32_t x_int;
  x_int   = x;
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<uint8_t*>(addr));
  asm volatile("st.shared.u8 [%0], {%1};" : : "l"(s1), "r"(x_int));
}
DI void sts(uint8_t* addr, const uint8_t (&x)[1])
{
  uint32_t x_int[1];
  x_int[0] = x[0];
  auto s1  = __cvta_generic_to_shared(reinterpret_cast<uint8_t*>(addr));
  asm volatile("st.shared.u8 [%0], {%1};" : : "l"(s1), "r"(x_int[0]));
}
DI void sts(uint8_t* addr, const uint8_t (&x)[2])
{
  uint32_t x_int[2];
  x_int[0] = x[0];
  x_int[1] = x[1];
  auto s2  = __cvta_generic_to_shared(reinterpret_cast<uint8_t*>(addr));
  asm volatile("st.shared.v2.u8 [%0], {%1, %2};" : : "l"(s2), "r"(x_int[0]), "r"(x_int[1]));
}
DI void sts(uint8_t* addr, const uint8_t (&x)[4])
{
  uint32_t x_int[4];
  x_int[0] = x[0];
  x_int[1] = x[1];
  x_int[2] = x[2];
  x_int[3] = x[3];
  auto s4  = __cvta_generic_to_shared(reinterpret_cast<uint8_t*>(addr));
  asm volatile("st.shared.v4.u8 [%0], {%1, %2, %3, %4};"
               :
               : "l"(s4), "r"(x_int[0]), "r"(x_int[1]), "r"(x_int[2]), "r"(x_int[3]));
}

DI void sts(int8_t* addr, const int8_t& x)
{
  int32_t x_int = x;
  auto s1       = __cvta_generic_to_shared(reinterpret_cast<int8_t*>(addr));
  asm volatile("st.shared.s8 [%0], {%1};" : : "l"(s1), "r"(x_int));
}
DI void sts(int8_t* addr, const int8_t (&x)[1])
{
  int32_t x_int[1];
  x_int[0] = x[0];
  auto s1  = __cvta_generic_to_shared(reinterpret_cast<int8_t*>(addr));
  asm volatile("st.shared.s8 [%0], {%1};" : : "l"(s1), "r"(x_int[0]));
}
DI void sts(int8_t* addr, const int8_t (&x)[2])
{
  int32_t x_int[2];
  x_int[0] = x[0];
  x_int[1] = x[1];
  auto s2  = __cvta_generic_to_shared(reinterpret_cast<int8_t*>(addr));
  asm volatile("st.shared.v2.s8 [%0], {%1, %2};" : : "l"(s2), "r"(x_int[0]), "r"(x_int[1]));
}
DI void sts(int8_t* addr, const int8_t (&x)[4])
{
  int32_t x_int[4];
  x_int[0] = x[0];
  x_int[1] = x[1];
  x_int[2] = x[2];
  x_int[3] = x[3];
  auto s4  = __cvta_generic_to_shared(reinterpret_cast<int8_t*>(addr));
  asm volatile("st.shared.v4.s8 [%0], {%1, %2, %3, %4};"
               :
               : "l"(s4), "r"(x_int[0]), "r"(x_int[1]), "r"(x_int[2]), "r"(x_int[3]));
}

DI void sts(uint32_t* addr, const uint32_t& x)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<uint32_t*>(addr));
  asm volatile("st.shared.u32 [%0], {%1};" : : "l"(s1), "r"(x));
}
DI void sts(uint32_t* addr, const uint32_t (&x)[1])
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<uint32_t*>(addr));
  asm volatile("st.shared.u32 [%0], {%1};" : : "l"(s1), "r"(x[0]));
}
DI void sts(uint32_t* addr, const uint32_t (&x)[2])
{
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<uint2*>(addr));
  asm volatile("st.shared.v2.u32 [%0], {%1, %2};" : : "l"(s2), "r"(x[0]), "r"(x[1]));
}
DI void sts(uint32_t* addr, const uint32_t (&x)[4])
{
  auto s4 = __cvta_generic_to_shared(reinterpret_cast<uint4*>(addr));
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(s4), "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]));
}

DI void sts(int32_t* addr, const int32_t& x)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<int32_t*>(addr));
  asm volatile("st.shared.u32 [%0], {%1};" : : "l"(s1), "r"(x));
}
DI void sts(int32_t* addr, const int32_t (&x)[1])
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<int32_t*>(addr));
  asm volatile("st.shared.u32 [%0], {%1};" : : "l"(s1), "r"(x[0]));
}
DI void sts(int32_t* addr, const int32_t (&x)[2])
{
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<int2*>(addr));
  asm volatile("st.shared.v2.u32 [%0], {%1, %2};" : : "l"(s2), "r"(x[0]), "r"(x[1]));
}
DI void sts(int32_t* addr, const int32_t (&x)[4])
{
  auto s4 = __cvta_generic_to_shared(reinterpret_cast<int4*>(addr));
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(s4), "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]));
}

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

DI void lds(uint8_t& x, const uint8_t* addr)
{
  uint32_t x_int;
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const uint8_t*>(addr));
  asm volatile("ld.shared.u8 {%0}, [%1];" : "=r"(x_int) : "l"(s1));
  x = x_int;
}
DI void lds(uint8_t (&x)[1], const uint8_t* addr)
{
  uint32_t x_int[1];
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const uint8_t*>(addr));
  asm volatile("ld.shared.u8 {%0}, [%1];" : "=r"(x_int[0]) : "l"(s1));
  x[0] = x_int[0];
}
DI void lds(uint8_t (&x)[2], const uint8_t* addr)
{
  uint32_t x_int[2];
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<const uint8_t*>(addr));
  asm volatile("ld.shared.v2.u8 {%0, %1}, [%2];" : "=r"(x_int[0]), "=r"(x_int[1]) : "l"(s2));
  x[0] = x_int[0];
  x[1] = x_int[1];
}
DI void lds(uint8_t (&x)[4], const uint8_t* addr)
{
  uint32_t x_int[4];
  auto s4 = __cvta_generic_to_shared(reinterpret_cast<const uint8_t*>(addr));
  asm volatile("ld.shared.v4.u8 {%0, %1, %2, %3}, [%4];"
               : "=r"(x_int[0]), "=r"(x_int[1]), "=r"(x_int[2]), "=r"(x_int[3])
               : "l"(s4));
  x[0] = x_int[0];
  x[1] = x_int[1];
  x[2] = x_int[2];
  x[3] = x_int[3];
}

DI void lds(int8_t& x, const int8_t* addr)
{
  int32_t x_int;
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const int8_t*>(addr));
  asm volatile("ld.shared.s8 {%0}, [%1];" : "=r"(x_int) : "l"(s1));
  x = x_int;
}
DI void lds(int8_t (&x)[1], const int8_t* addr)
{
  int32_t x_int[1];
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const int8_t*>(addr));
  asm volatile("ld.shared.s8 {%0}, [%1];" : "=r"(x_int[0]) : "l"(s1));
  x[0] = x_int[0];
}
DI void lds(int8_t (&x)[2], const int8_t* addr)
{
  int32_t x_int[2];
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<const int8_t*>(addr));
  asm volatile("ld.shared.v2.s8 {%0, %1}, [%2];" : "=r"(x_int[0]), "=r"(x_int[1]) : "l"(s2));
  x[0] = x_int[0];
  x[1] = x_int[1];
}
DI void lds(int8_t (&x)[4], const int8_t* addr)
{
  int32_t x_int[4];
  auto s4 = __cvta_generic_to_shared(reinterpret_cast<const int8_t*>(addr));
  asm volatile("ld.shared.v4.s8 {%0, %1, %2, %3}, [%4];"
               : "=r"(x_int[0]), "=r"(x_int[1]), "=r"(x_int[2]), "=r"(x_int[3])
               : "l"(s4));
  x[0] = x_int[0];
  x[1] = x_int[1];
  x[2] = x_int[2];
  x[3] = x_int[3];
}

DI void lds(uint32_t (&x)[4], const uint32_t* addr)
{
  auto s4 = __cvta_generic_to_shared(reinterpret_cast<const uint32_t*>(addr));
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3])
               : "l"(s4));
}

DI void lds(uint32_t (&x)[2], const uint32_t* addr)
{
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<const uint32_t*>(addr));
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(x[0]), "=r"(x[1]) : "l"(s2));
}

DI void lds(uint32_t (&x)[1], const uint32_t* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const uint32_t*>(addr));
  asm volatile("ld.shared.u32 {%0}, [%1];" : "=r"(x[0]) : "l"(s1));
}

DI void lds(uint32_t& x, const uint32_t* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const uint32_t*>(addr));
  asm volatile("ld.shared.u32 {%0}, [%1];" : "=r"(x) : "l"(s1));
}

DI void lds(int32_t (&x)[4], const int32_t* addr)
{
  auto s4 = __cvta_generic_to_shared(reinterpret_cast<const int32_t*>(addr));
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3])
               : "l"(s4));
}

DI void lds(int32_t (&x)[2], const int32_t* addr)
{
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<const int32_t*>(addr));
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(x[0]), "=r"(x[1]) : "l"(s2));
}

DI void lds(int32_t (&x)[1], const int32_t* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const int32_t*>(addr));
  asm volatile("ld.shared.u32 {%0}, [%1];" : "=r"(x[0]) : "l"(s1));
}

DI void lds(int32_t& x, const int32_t* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const int32_t*>(addr));
  asm volatile("ld.shared.u32 {%0}, [%1];" : "=r"(x) : "l"(s1));
}

DI void lds(float& x, const float* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const float*>(addr));
  asm volatile("ld.shared.f32 {%0}, [%1];" : "=f"(x) : "l"(s1));
}
DI void lds(float (&x)[1], const float* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const float*>(addr));
  asm volatile("ld.shared.f32 {%0}, [%1];" : "=f"(x[0]) : "l"(s1));
}
DI void lds(float (&x)[2], const float* addr)
{
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<const float2*>(addr));
  asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(x[0]), "=f"(x[1]) : "l"(s2));
}
DI void lds(float (&x)[4], const float* addr)
{
  auto s4 = __cvta_generic_to_shared(reinterpret_cast<const float4*>(addr));
  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(x[0]), "=f"(x[1]), "=f"(x[2]), "=f"(x[3])
               : "l"(s4));
}

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

DI void ldg(uint32_t (&x)[4], const uint32_t* const& addr)
{
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3])
               : "l"(addr));
}

DI void ldg(uint32_t (&x)[2], const uint32_t* const& addr)
{
  asm volatile("ld.global.cg.v2.u32 {%0, %1}, [%2];" : "=r"(x[0]), "=r"(x[1]) : "l"(addr));
}

DI void ldg(uint32_t (&x)[1], const uint32_t* const& addr)
{
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x[0]) : "l"(addr));
}

DI void ldg(uint32_t& x, const uint32_t* const& addr)
{
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x) : "l"(addr));
}

DI void ldg(int32_t (&x)[4], const int32_t* const& addr)
{
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3])
               : "l"(addr));
}

DI void ldg(int32_t (&x)[2], const int32_t* const& addr)
{
  asm volatile("ld.global.cg.v2.u32 {%0, %1}, [%2];" : "=r"(x[0]), "=r"(x[1]) : "l"(addr));
}

DI void ldg(int32_t (&x)[1], const int32_t* const& addr)
{
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x[0]) : "l"(addr));
}

DI void ldg(int32_t& x, const int32_t* const& addr)
{
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x) : "l"(addr));
}

DI void ldg(uint8_t (&x)[4], const uint8_t* const& addr)
{
  uint32_t x_int[4];
  asm volatile("ld.global.cg.v4.u8 {%0, %1, %2, %3}, [%4];"
               : "=r"(x_int[0]), "=r"(x_int[1]), "=r"(x_int[2]), "=r"(x_int[3])
               : "l"(addr));
  x[0] = x_int[0];
  x[1] = x_int[1];
  x[2] = x_int[2];
  x[3] = x_int[3];
}

DI void ldg(uint8_t (&x)[2], const uint8_t* const& addr)
{
  uint32_t x_int[2];
  asm volatile("ld.global.cg.v2.u8 {%0, %1}, [%2];" : "=r"(x_int[0]), "=r"(x_int[1]) : "l"(addr));
  x[0] = x_int[0];
  x[1] = x_int[1];
}

DI void ldg(uint8_t (&x)[1], const uint8_t* const& addr)
{
  uint32_t x_int;
  asm volatile("ld.global.cg.u8 %0, [%1];" : "=r"(x_int) : "l"(addr));
  x[0] = x_int;
}

DI void ldg(uint8_t& x, const uint8_t* const& addr)
{
  uint32_t x_int;
  asm volatile("ld.global.cg.u8 %0, [%1];" : "=r"(x_int) : "l"(addr));
  x = x_int;
}

DI void ldg(int8_t (&x)[4], const int8_t* const& addr)
{
  int x_int[4];
  asm volatile("ld.global.cg.v4.s8 {%0, %1, %2, %3}, [%4];"
               : "=r"(x_int[0]), "=r"(x_int[1]), "=r"(x_int[2]), "=r"(x_int[3])
               : "l"(addr));
  x[0] = x_int[0];
  x[1] = x_int[1];
  x[2] = x_int[2];
  x[3] = x_int[3];
}

DI void ldg(int8_t (&x)[2], const int8_t* const& addr)
{
  int x_int[2];
  asm volatile("ld.global.cg.v2.s8 {%0, %1}, [%2];" : "=r"(x_int[0]), "=r"(x_int[1]) : "l"(addr));
  x[0] = x_int[0];
  x[1] = x_int[1];
}

DI void ldg(int8_t& x, const int8_t* const& addr)
{
  int x_int;
  asm volatile("ld.global.cg.s8 %0, [%1];" : "=r"(x_int) : "l"(addr));
  x = x_int;
}

DI void ldg(int8_t (&x)[1], const int8_t* const& addr)
{
  int x_int;
  asm volatile("ld.global.cg.s8 %0, [%1];" : "=r"(x_int) : "l"(addr));
  x[0] = x_int;
}

/**
 * @brief Executes a 1D block strided copy
 * @param dst destination pointer
 * @param src source pointer
 * @param size number of items to copy
 */
template <typename T>
DI void block_copy(T* dst, const T* src, const size_t size)
{
  for (auto i = threadIdx.x; i < size; i += blockDim.x) {
    dst[i] = src[i];
  }
}

/**
 * @brief Executes a 1D block strided copy
 * @param dst span of destination pointer
 * @param src span of source pointer
 * @param size number of items to copy
 */
template <typename T>
DI void block_copy(raft::device_span<T> dst,
                   const raft::device_span<const T> src,
                   const size_t size)
{
  assert(src.size() >= size);
  assert(dst.size() >= size);
  block_copy(dst.data(), src.data(), size);
}

/**
 * @brief Executes a 1D block strided copy
 * @param dst span of destination pointer
 * @param src span of source pointer
 * @param size number of items to copy
 */
template <typename T>
DI void block_copy(raft::device_span<T> dst, const raft::device_span<T> src, const size_t size)
{
  assert(src.size() >= size);
  assert(dst.size() >= size);
  block_copy(dst.data(), src.data(), size);
}

/**
 * @brief Executes a 1D block strided copy
 * @param dst span of destination pointer
 * @param src span of source pointer
 */
template <typename T>
DI void block_copy(raft::device_span<T> dst, const raft::device_span<T> src)
{
  assert(dst.size() >= src.size());
  block_copy(dst, src, src.size());
}

/** @} */

}  // namespace raft
