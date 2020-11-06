/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include <math_constants.h>
#include <cstdint>

#include <raft/cudart_utils.h>

#ifndef ENABLE_MEMCPY_ASYNC
// enable memcpy_async interface by default for newer GPUs
#if __CUDA_ARCH__ >= 800
#define ENABLE_MEMCPY_ASYNC 1
#endif
#else  // ENABLE_MEMCPY_ASYNC
// disable memcpy_async for all older GPUs
#if __CUDA_ARCH__ < 800
#define ENABLE_MEMCPY_ASYNC 0
#endif
#endif  // ENABLE_MEMCPY_ASYNC

namespace raft {

/** helper macro for device inlined functions */
#define DI inline __device__
#define HDI inline __host__ __device__
#define HD __host__ __device__

/**
 * @brief Provide a ceiling division operation ie. ceil(a / b)
 * @tparam IntType supposed to be only integers for now!
 */
template <typename IntType>
constexpr HDI auto ceildiv(IntType a, IntType b) -> IntType {
  return (a + b - 1) / b;
}

/**
 * @brief Provide an alignment function ie. ceil(a / b) * b
 * @tparam IntType supposed to be only integers for now!
 */
template <typename IntType>
constexpr HDI auto alignTo(IntType a, IntType b) -> IntType {  // NOLINT
  return ceildiv(a, b) * b;
}

/**
 * @brief Provide an alignment function ie. (a / b) * b
 * @tparam IntType supposed to be only integers for now!
 */
template <typename IntType>
constexpr HDI auto alignDown(IntType a, IntType b) -> IntType {  // NOLINT
  return (a / b) * b;
}

/**
 * @brief Check if the input is a power of 2
 * @tparam IntType data type (checked only for integers)
 */
template <typename IntType>
constexpr HDI auto isPo2(IntType num) -> bool {  // NOLINT
  return (num && !(num & (num - 1)));
}

/**
 * @brief Give logarithm of the number to base-2
 * @tparam IntType data type (checked only for integers)
 */
template <typename IntType>
constexpr HDI auto log2(IntType num, IntType ret = IntType(0)) -> IntType {
  return num <= IntType(1) ? ret : log2(num >> IntType(1), ++ret);
}

/** Device function to apply the input lambda across threads in the grid */
template <int ItemsPerThread, typename L>
DI void forEach(int num, L lambda) {  // NOLINT
  int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
  const int num_threads = blockDim.x * gridDim.x;
#pragma unroll
  for (int itr = 0; itr < ItemsPerThread; ++itr, idx += num_threads) {
    if (idx < num) lambda(idx, itr);
  }
}

/** number of threads per warp */
static const int WarpSize = 32;  // NOLINT

/** get the laneId of the current thread */
DI auto laneId() -> int {  // NOLINT
  int id;
  asm("mov.s32 %0, %laneid;" : "=r"(id));
  return id;
}

/**
 * @brief Swap two values
 * @tparam T the datatype of the values
 * @param a first input
 * @param b second input
 */
template <typename T>
HDI void swapVals(T &a, T &b) {  // NOLINT
  T tmp = a;
  a = b;
  b = tmp;
}

/** Device function to have atomic add support for older archs */
template <typename Type>
DI void myAtomicAdd(Type *address, Type val) {  // NOLINT
  atomicAdd(address, val);
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
// Ref:
// http://on-demand.gputechconf.com/gtc/2013/presentations/S3101-Atomic-Memory-Operations.pdf
template <>
DI void myAtomicAdd(double *address, double val) {  // NOLINT
  auto *address_as_ull =
    reinterpret_cast<unsigned long long *>(address);  // NOLINT
  auto old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
}
#endif

template <typename T, typename ReduceLambda>
DI void myAtomicReduce(T *address, T val, ReduceLambda op);  // NOLINT

template <typename ReduceLambda>
DI void myAtomicReduce(double *address, double val,  // NOLINT
                       ReduceLambda op) {
  auto *address_as_ull =
    reinterpret_cast<unsigned long long *>(address);  // NOLINT
  unsigned long long old = *address_as_ull, assumed;  // NOLINT
  do {
    assumed = old;
    old =
      atomicCAS(address_as_ull, assumed,
                __double_as_longlong(op(val, __longlong_as_double(assumed))));
  } while (assumed != old);
}

template <typename ReduceLambda>
DI void myAtomicReduce(float *address, float val, ReduceLambda op) {  // NOLINT
  auto *address_as_uint = reinterpret_cast<unsigned *>(address);
  unsigned old = *address_as_uint, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_uint, assumed,
                    __float_as_uint(op(val, __uint_as_float(assumed))));
  } while (assumed != old);
}

template <typename ReduceLambda>
DI void myAtomicReduce(int *address, int val, ReduceLambda op) {  // NOLINT
  int old = *address, assumed;
  do {
    assumed = old;
    old = atomicCAS(address, assumed, op(val, assumed));
  } while (assumed != old);
}

template <typename ReduceLambda>
DI void myAtomicReduce(long long *address, long long val,  // NOLINT
                       ReduceLambda op) {
  long long old = *address, assumed;  // NOLINT
  do {
    assumed = old;
    old = atomicCAS(address, assumed, op(val, assumed));
  } while (assumed != old);
}

template <typename ReduceLambda>
DI void myAtomicReduce(unsigned long long *address,                // NOLINT
                       unsigned long long val, ReduceLambda op) {  // NOLINT
  unsigned long long old = *address, assumed;                      // NOLINT
  do {
    assumed = old;
    old = atomicCAS(address, assumed, op(val, assumed));
  } while (assumed != old);
}

/**
 * @brief Provide atomic min operation.
 * @tparam T: data type for input data (float or double).
 * @param[in] address: address to read old value from, and to atomically update w/ min(old value, val)
 * @param[in] val: new value to compare with old
 */
template <typename T>
DI T myAtomicMin(T *address, T val);  // NOLINT

/**
 * @brief Provide atomic max operation.
 * @tparam T: data type for input data (float or double).
 * @param[in] address: address to read old value from, and to atomically update w/ max(old value, val)
 * @param[in] val: new value to compare with old
 */
template <typename T>
DI auto myAtomicMax(T *address, T val) -> T;  // NOLINT

DI auto myAtomicMin(float *address, float val) -> float {  // NOLINT
  myAtomicReduce(address, val, fminf);
  return *address;
}

DI auto myAtomicMax(float *address, float val) -> float {  // NOLINT
  myAtomicReduce(address, val, fmaxf);
  return *address;
}

DI auto myAtomicMin(double *address, double val) -> double {  // NOLINT
  myAtomicReduce<double(double, double)>(address, val, fmin);
  return *address;
}

DI auto myAtomicMax(double *address, double val) -> double {  // NOLINT
  myAtomicReduce<double(double, double)>(address, val, fmax);
  return *address;
}

/**
 * @defgroup Max maximum of two numbers
 * @{
 */
template <typename T>
HDI auto myMax(T x, T y) -> T;  // NOLINT
template <>
HDI auto myMax<float>(float x, float y) -> float {  // NOLINT
  return fmaxf(x, y);
}
template <>
HDI auto myMax<double>(double x, double y) -> double {  // NOLINT
  return fmax(x, y);
}
/** @} */

/**
 * @defgroup Min minimum of two numbers
 * @{
 */
template <typename T>
HDI auto myMin(T x, T y) -> T;  // NOLINT
template <>
HDI auto myMin<float>(float x, float y) -> float {  // NOLINT
  return fminf(x, y);
}
template <>
HDI auto myMin<double>(double x, double y) -> double {  // NOLINT
  return fmin(x, y);
}
/** @} */

/**
 * @brief Provide atomic min operation.
 * @tparam T: data type for input data (float or double).
 * @param[in] address: address to read old value from, and to atomically update w/ min(old value, val)
 * @param[in] val: new value to compare with old
 */
template <typename T>
DI auto myAtomicMin(T *address, T val) -> T {  // NOLINT
  myAtomicReduce(address, val, myMin<T>);
  return *address;
}

/**
 * @brief Provide atomic max operation.
 * @tparam T: data type for input data (float or double).
 * @param[in] address: address to read old value from, and to atomically update w/ max(old value, val)
 * @param[in] val: new value to compare with old
 */
template <typename T>
DI auto myAtomicMax(T *address, T val) -> T {  // NOLINT
  myAtomicReduce(address, val, myMax<T>);
  return *address;
}

/**
 * Sign function
 */
template <typename T>
HDI auto sgn(const T val) -> int {
  return (T(0) < val) - (val < T(0));
}

/**
 * @defgroup Exp Exponential function
 * @{
 */
template <typename T>
HDI auto myExp(T x) -> T;  // NOLINT
template <>
HDI auto myExp(float x) -> float {  // NOLINT
  return expf(x);
}
template <>
HDI auto myExp(double x) -> double {  // NOLINT
  return exp(x);
}
/** @} */

/**
 * @defgroup Cuda infinity values
 * @{
 */
template <typename T>
DI auto myInf() -> T;  // NOLINT
template <>
DI auto myInf<float>() -> float {  // NOLINT
  return CUDART_INF_F;
}
template <>
DI auto myInf<double>() -> double {  // NOLINT
  return CUDART_INF;
}
/** @} */

/**
 * @defgroup Log Natural logarithm
 * @{
 */
template <typename T>
HDI auto myLog(T x)-> T;  // NOLINT
template <>
HDI auto myLog(float x) -> float {  // NOLINT
  return logf(x);
}
template <>
HDI auto myLog(double x) -> double {  // NOLINT
  return log(x);
}
/** @} */

/**
 * @defgroup Sqrt Square root
 * @{
 */
template <typename T>
HDI auto mySqrt(T x) -> T;  // NOLINT
template <>
HDI auto mySqrt(float x) -> float {  // NOLINT
  return sqrtf(x);
}
template <>
HDI auto mySqrt(double x) -> double {  // NOLINT
  return sqrt(x);
}
/** @} */

/**
 * @defgroup SineCosine Sine and cosine calculation
 * @{
 */
template <typename T>
DI void mySinCos(T x, T &s, T &c);  // NOLINT
template <>
DI void mySinCos(float x, float &s, float &c) {  // NOLINT
  sincosf(x, &s, &c);
}
template <>
DI void mySinCos(double x, double &s, double &c) {  // NOLINT
  sincos(x, &s, &c);
}
/** @} */

/**
 * @defgroup Sine Sine calculation
 * @{
 */
template <typename T>
DI auto mySin(T x) -> T;  // NOLINT
template <>
DI auto mySin(float x) -> float {  // NOLINT
  return sinf(x);
}
template <>
DI auto mySin(double x) -> double {  // NOLINT
  return sin(x);
}
/** @} */

/**
 * @defgroup Abs Absolute value
 * @{
 */
template <typename T>
DI auto myAbs(T x) -> T {  // NOLINT
  return x < 0 ? -x : x;
}
template <>
DI auto myAbs(float x) -> float {  // NOLINT
  return fabsf(x);
}
template <>
DI auto myAbs(double x) -> double {  // NOLINT
  return fabs(x);
}
/** @} */

/**
 * @defgroup Pow Power function
 * @{
 */
template <typename T>
HDI auto myPow(T x, T power) -> T;  // NOLINT
template <>
HDI auto myPow(float x, float power) -> float {  // NOLINT
  return powf(x, power);
}
template <>
HDI auto myPow(double x, double power) -> double {  // NOLINT
  return pow(x, power);
}
/** @} */

/**
 * @defgroup myTanh tanh function
 * @{
 */
template <typename T>
HDI auto myTanh(T x) -> T;  // NOLINT
template <>
HDI auto myTanh(float x) -> float {  // NOLINT
  return tanhf(x);
}
template <>
HDI auto myTanh(double x) -> double {  // NOLINT
  return tanh(x);
}
/** @} */

/**
 * @defgroup myATanh arctanh function
 * @{
 */
template <typename T>
HDI auto myATanh(T x) -> T;  // NOLINT
template <>
HDI auto myATanh(float x) -> float {  // NOLINT
  return atanhf(x);
}
template <>
HDI auto myATanh(double x) -> double {  // NOLINT
  return atanh(x);
}
/** @} */

/**
 * @defgroup LambdaOps Lambda operations in reduction kernels
 * @{
 */
// IdxType mostly to be used for MainLambda in *Reduction kernels
template <typename Type, typename IdxType = int>
struct Nop {  // NOLINT
  HDI auto operator()(Type in, IdxType i = 0) -> Type { return in; }
};

template <typename Type, typename IdxType = int>
struct L1Op {  // NOLINT
  HDI auto operator()(Type in, IdxType i = 0) -> Type { return myAbs(in); }
};

template <typename Type, typename IdxType = int>
struct L2Op {  // NOLINT
  HDI auto operator()(Type in, IdxType i = 0) -> Type { return in * in; }
};

template <typename Type>
struct Sum {  // NOLINT
  HDI auto operator()(Type a, Type b) -> Type { return a + b; }
};
/** @} */

/**
 * @defgroup Sign Obtain sign value
 * @brief Obtain sign of x
 * @param x input
 * @return +1 if x >= 0 and -1 otherwise
 * @{
 */
template <typename T>
DI auto signPrim(T x) -> T {  // NOLINT
  return x < 0 ? -1 : +1;
}
template <>
DI auto signPrim(float x) -> float {  // NOLINT
  return signbit(x) == true ? -1.0f : +1.0f;
}
template <>
DI auto signPrim(double x) -> double {  // NOLINT
  return signbit(x) == true ? -1.0 : +1.0;
}
/** @} */

/**
 * @defgroup Max maximum of two numbers
 * @brief Obtain maximum of two values
 * @param x one item
 * @param y second item
 * @return maximum of two items
 * @{
 */
template <typename T>
DI auto maxPrim(T x, T y) -> T {  // NOLINT
  return x > y ? x : y;
}
template <>
DI auto maxPrim(float x, float y) -> float {  // NOLINT
  return fmaxf(x, y);
}
template <>
DI auto maxPrim(double x, double y) -> double {  // NOLINT
  return fmax(x, y);
}
/** @} */

/** apply a warp-wide fence (useful from Volta+ archs) */
DI void warpFence() {  // NOLINT
#if __CUDA_ARCH__ >= 700
  __syncwarp();
#endif
}

/** warp-wide any boolean aggregator */
DI auto any(bool inFlag, uint32_t mask = 0xffffffffu) -> bool {
#if CUDART_VERSION >= 9000
  inFlag = __any_sync(mask, inFlag);
#else
  inFlag = __any(inFlag);
#endif
  return inFlag;
}

/** warp-wide all boolean aggregator */
DI auto all(bool inFlag, uint32_t mask = 0xffffffffu) -> bool {
#if CUDART_VERSION >= 9000
  inFlag = __all_sync(mask, inFlag);
#else
  inFlag = __all(inFlag);
#endif
  return inFlag;
}

/**
 * @brief Shuffle the data inside a warp
 * @tparam T the data type (currently assumed to be 4B)
 * @param val value to be shuffled
 * @param srcLane lane from where to shuffle
 * @param width lane width
 * @param mask mask of participating threads (Volta+)
 * @return the shuffled data
 */
template <typename T>
DI auto shfl(T val, int srcLane, int width = WarpSize,
          uint32_t mask = 0xffffffffu) -> T {
#if CUDART_VERSION >= 9000
  return __shfl_sync(mask, val, srcLane, width);
#else
  return __shfl(val, srcLane, width);
#endif
}

/**
 * @brief Shuffle the data inside a warp
 * @tparam T the data type (currently assumed to be 4B)
 * @param val value to be shuffled
 * @param laneMask mask to be applied in order to perform xor shuffle
 * @param width lane width
 * @param mask mask of participating threads (Volta+)
 * @return the shuffled data
 */
template <typename T>
DI auto shfl_xor(T val, int laneMask, int width = WarpSize,
              uint32_t mask = 0xffffffffu) -> T {
#if CUDART_VERSION >= 9000
  return __shfl_xor_sync(mask, val, laneMask, width);
#else
  return __shfl_xor(val, laneMask, width);
#endif
}

/**
 * @brief Warp-level sum reduction
 * @param val input value
 * @return only the lane0 will contain valid reduced result
 * @note Why not cub? Because cub doesn't seem to allow working with arbitrary
 *       number of warps in a block. All threads in the warp must enter this
 *       function together
 * @todo Expand this to support arbitrary reduction ops
 */
template <typename T>
DI auto warpReduce(T val) -> T {  // NOLINT
#pragma unroll
  for (int i = WarpSize / 2; i > 0; i >>= 1) {
    T tmp = shfl(val, laneId() + i);
    val += tmp;
  }
  return val;
}

/**
 * @brief 1-D block-level sum reduction
 * @param val input value
 * @param smem shared memory region needed for storing intermediate results. It
 *             must alteast be of size: `sizeof(T) * nWarps`
 * @return only the thread0 will contain valid reduced result
 * @note Why not cub? Because cub doesn't seem to allow working with arbitrary
 *       number of warps in a block. All threads in the block must enter this
 *       function together
 * @todo Expand this to support arbitrary reduction ops
 */
template <typename T>
DI auto blockReduce(T val, char *smem) -> T {  // NOLINT
  auto *s_temp = reinterpret_cast<T *>(smem);
  int n_warps = (blockDim.x + WarpSize - 1) / WarpSize;
  int lid = laneId();
  int wid = threadIdx.x / WarpSize;
  val = warpReduce(val);
  if (lid == 0) s_temp[wid] = val;
  __syncthreads();
  val = lid < n_warps ? s_temp[lid] : T(0);
  return warpReduce(val);
}

/**
 * @brief Simple utility function to determine whether user_stream or one of the
 * internal streams should be used.
 * @param user_stream main user stream
 * @param int_streams array of internal streams
 * @param n_int_streams number of internal streams
 * @param idx the index for which to query the stream
 */
inline auto select_stream(cudaStream_t user_stream,
                          cudaStream_t *int_streams, int n_int_streams,
                          int idx) -> cudaStream_t {
  return n_int_streams > 0 ? int_streams[idx % n_int_streams] : user_stream;
}

}  // namespace raft
