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

#include <cstdio>
#include <cstdlib>
#include <curand_kernel.h>
#include <raft/common/cub_wrappers.cuh>
#include <raft/common/scatter.cuh>
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <random>
#include <rmm/device_uvector.hpp>
#include <stdint.h>
#include <type_traits>

namespace raft {
namespace random {
namespace detail {

/** all different generator types used */
enum GeneratorType {
  /** curand-based philox generator */
  GenPhilox = 0,
  /** LFSR taps generator */
  GenTaps,
  /** kiss99 generator (currently the fastest) */
  GenKiss99
};

template <typename Type>
DI void box_muller_transform(Type& val1, Type& val2, Type sigma1, Type mu1, Type sigma2, Type mu2)
{
  constexpr Type twoPi  = Type(2.0) * Type(3.141592654);
  constexpr Type minus2 = -Type(2.0);
  Type R                = raft::mySqrt(minus2 * raft::myLog(val1));
  Type theta            = twoPi * val2;
  Type s, c;
  raft::mySinCos(theta, s, c);
  val1 = R * c * sigma1 + mu1;
  val2 = R * s * sigma2 + mu2;
}
template <typename Type>
DI void box_muller_transform(Type& val1, Type& val2, Type sigma1, Type mu1)
{
  box_muller_transform<Type>(val1, val2, sigma1, mu1, sigma1, mu1);
}

/**
 * @brief generator-agnostic way of generating random numbers
 * @tparam GenType the generator object that expose 'next' method
 */
template <typename GenType>
struct Generator {
  DI Generator(uint64_t seed, uint64_t subsequence, uint64_t offset)
    : gen(seed, subsequence, offset)
  {
  }

  template <typename Type>
  DI void next(Type& ret)
  {
    gen.next(ret);
  }

 private:
  /** the actual generator */
  GenType gen;
};

template <typename OutType, typename MathType, typename GenType, typename LenType, typename Lambda>
__global__ void randKernel(uint64_t seed, uint64_t offset, OutType* ptr, LenType len, Lambda randOp)
{
  LenType tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  Generator<GenType> gen(seed, (uint64_t)tid, offset);
  const LenType stride = gridDim.x * blockDim.x;
  for (LenType idx = tid; idx < len; idx += stride) {
    MathType val;
    gen.next(val);
    ptr[idx] = randOp(val, idx);
  }
}

// used for Box-Muller type transformations
template <typename OutType, typename MathType, typename GenType, typename LenType, typename Lambda2>
__global__ void rand2Kernel(
  uint64_t seed, uint64_t offset, OutType* ptr, LenType len, Lambda2 rand2Op)
{
  LenType tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  Generator<GenType> gen(seed, (uint64_t)tid, offset);
  const LenType stride = gridDim.x * blockDim.x;
  for (LenType idx = tid; idx < len; idx += stride) {
    MathType val1, val2;
    gen.next(val1);
    gen.next(val2);
    rand2Op(val1, val2, idx, idx + stride);
    if (idx < len) ptr[idx] = (OutType)val1;
    idx += stride;
    if (idx < len) ptr[idx] = (OutType)val2;
  }
}

template <typename Type>
__global__ void constFillKernel(Type* ptr, int len, Type val)
{
  unsigned tid          = (blockIdx.x * blockDim.x) + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;
  for (unsigned idx = tid; idx < len; idx += stride) {
    ptr[idx] = val;
  }
}

/** Philox-based random number generator */
// Courtesy: Jakub Szuppe
struct PhiloxGenerator {
  /**
   * @brief ctor. Initializes the state for RNG
   * @param seed random seed (can be same across all threads)
   * @param subsequence as found in curand docs
   * @param offset as found in curand docs
   */
  DI PhiloxGenerator(uint64_t seed, uint64_t subsequence, uint64_t offset)
  {
    curand_init(seed, subsequence, offset, &state);
  }

  /**
   * @defgroup NextRand Generate the next random number
   * @{
   */
  DI void next(float& ret) { ret = curand_uniform(&(this->state)); }
  DI void next(double& ret) { ret = curand_uniform_double(&(this->state)); }
  DI void next(uint32_t& ret) { ret = curand(&(this->state)); }
  DI void next(uint64_t& ret)
  {
    uint32_t a, b;
    next(a);
    next(b);
    ret = (uint64_t)a | ((uint64_t)b << 32);
  }
  DI void next(int32_t& ret)
  {
    uint32_t val;
    next(val);
    ret = int32_t(val & 0x7fffffff);
  }
  DI void next(int64_t& ret)
  {
    uint64_t val;
    next(val);
    ret = int64_t(val & 0x7fffffffffffffff);
  }
  /** @} */

 private:
  /** the state for RNG */
  curandStatePhilox4_32_10_t state;
};

/** LFSR taps-filter for generating random numbers. */
// Courtesy: Vinay Deshpande
struct TapsGenerator {
  /**
   * @brief ctor. Initializes the state for RNG
   * @param seed the seed (can be same across all threads)
   * @param subsequence unused
   * @param offset unused
   */
  DI TapsGenerator(uint64_t seed, uint64_t subsequence, uint64_t offset)
  {
    uint64_t delta  = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint64_t stride = blockDim.x * gridDim.x;
    delta += ((blockIdx.y * blockDim.y) + threadIdx.y) * stride;
    stride *= blockDim.y * gridDim.y;
    delta += ((blockIdx.z * blockDim.z) + threadIdx.z) * stride;
    state = seed + delta + 1;
  }

  /**
   * @defgroup NextRand Generate the next random number
   * @{
   */
  template <typename Type>
  DI void next(Type& ret)
  {
    constexpr double ULL_LARGE = 1.8446744073709551614e19;
    uint64_t val;
    next(val);
    ret = static_cast<Type>(val);
    ret /= static_cast<Type>(ULL_LARGE);
  }
  DI void next(uint64_t& ret)
  {
    constexpr uint64_t TAPS = 0x8000100040002000ULL;
    constexpr int ROUNDS    = 128;
    for (int i = 0; i < ROUNDS; i++)
      state = (state >> 1) ^ (-(state & 1ULL) & TAPS);
    ret = state;
  }
  DI void next(uint32_t& ret)
  {
    uint64_t val;
    next(val);
    ret = (uint32_t)val;
  }
  DI void next(int32_t& ret)
  {
    uint32_t val;
    next(val);
    ret = int32_t(val & 0x7fffffff);
  }
  DI void next(int64_t& ret)
  {
    uint64_t val;
    next(val);
    ret = int64_t(val & 0x7fffffffffffffff);
  }
  /** @} */

 private:
  /** the state for RNG */
  uint64_t state;
};

/** Kiss99-based random number generator */

struct Kiss99Generator {
  /**
   * @brief ctor. Initializes the state for RNG
   * @param seed the seed (can be same across all threads)
   * @param subsequence unused
   * @param offset unused
   */
  DI Kiss99Generator(uint64_t seed, uint64_t subsequence, uint64_t offset) { initKiss99(seed); }

  /**
   * @defgroup NextRand Generate the next random number
   * @{
   */
  template <typename Type>
  DI void next(Type& ret)
  {
    constexpr double U_LARGE = 4.294967295e9;
    uint32_t val;
    next(val);
    ret = static_cast<Type>(val);
    ret /= static_cast<Type>(U_LARGE);
  }
  DI void next(uint32_t& ret)
  {
    uint32_t MWC;
    z   = 36969 * (z & 65535) + (z >> 16);
    w   = 18000 * (w & 65535) + (w >> 16);
    MWC = ((z << 16) + w);
    jsr ^= (jsr << 17);
    jsr ^= (jsr >> 13);
    jsr ^= (jsr << 5);
    jcong = 69069 * jcong + 1234567;
    MWC   = ((MWC ^ jcong) + jsr);
    ret   = MWC;
  }
  DI void next(uint64_t& ret)
  {
    uint32_t a, b;
    next(a);
    next(b);
    ret = (uint64_t)a | ((uint64_t)b << 32);
  }
  DI void next(int32_t& ret)
  {
    uint32_t val;
    next(val);
    ret = int32_t(val & 0x7fffffff);
  }
  DI void next(int64_t& ret)
  {
    uint64_t val;
    next(val);
    ret = int64_t(val & 0x7fffffffffffffff);
  }
  /** @} */

 private:
  /** one of the kiss99 states */
  uint32_t z;
  /** one of the kiss99 states */
  uint32_t w;
  /** one of the kiss99 states */
  uint32_t jsr;
  /** one of the kiss99 states */
  uint32_t jcong;

  // This function multiplies 128-bit hash by 128-bit FNV prime and returns lower
  // 128 bits. It uses 32-bit wide multiply only.
  DI void mulByFnv1a128Prime(uint32_t* h)
  {
    typedef union {
      uint32_t u32[2];
      uint64_t u64[1];
    } words64;

    // 128-bit FNV prime = p3 * 2^96 + p2 * 2^64 + p1 * 2^32 + p0
    // Here p0 = 315, p2 = 16777216, p1 = p3 = 0
    const uint32_t p0 = uint32_t(315), p2 = uint32_t(16777216);
    // Partial products
    words64 h0p0, h1p0, h2p0, h0p2, h3p0, h1p2;

    h0p0.u64[0] = uint64_t(h[0]) * p0;
    h1p0.u64[0] = uint64_t(h[1]) * p0;
    h2p0.u64[0] = uint64_t(h[2]) * p0;
    h0p2.u64[0] = uint64_t(h[0]) * p2;
    h3p0.u64[0] = uint64_t(h[3]) * p0;
    h1p2.u64[0] = uint64_t(h[1]) * p2;

    // h_n[0] = LO(h[0]*p[0]);
    // h_n[1] = HI(h[0]*p[0]) + LO(h[1]*p[0]);
    // h_n[2] = HI(h[1]*p[0]) + LO(h[2]*p[0]) + LO(h[0]*p[2]);
    // h_n[3] = HI(h[2]*p[0]) + HI(h[0]*p[2]) + LO(h[3]*p[0]) + LO(h[1]*p[2]);
    uint32_t carry = 0;
    h[0]           = h0p0.u32[0];

    h[1]  = h0p0.u32[1] + h1p0.u32[0];
    carry = h[1] < h0p0.u32[1] ? 1 : 0;

    h[2]  = h1p0.u32[1] + carry;
    carry = h[2] < h1p0.u32[1] ? 1 : 0;
    h[2] += h2p0.u32[0];
    carry = h[2] < h2p0.u32[0] ? carry + 1 : carry;
    h[2] += h0p2.u32[0];
    carry = h[2] < h0p2.u32[0] ? carry + 1 : carry;

    h[3] = h2p0.u32[1] + h0p2.u32[1] + h3p0.u32[0] + h1p2.u32[0] + carry;
    return;
  }

  DI void fnv1a128(uint32_t* hash, uint32_t txt)
  {
    hash[0] ^= (txt >> 0) & 0xFF;
    mulByFnv1a128Prime(hash);
    hash[0] ^= (txt >> 8) & 0xFF;
    mulByFnv1a128Prime(hash);
    hash[0] ^= (txt >> 16) & 0xFF;
    mulByFnv1a128Prime(hash);
    hash[0] ^= (txt >> 24) & 0xFF;
    mulByFnv1a128Prime(hash);
  }

  DI void initKiss99(uint64_t seed)
  {
    // Initialize hash to 128-bit FNV1a basis
    uint32_t hash[4] = {1653982605UL, 1656234357UL, 129696066UL, 1818371886UL};

    // Digest threadIdx, blockIdx and seed
    fnv1a128(hash, threadIdx.x);
    fnv1a128(hash, threadIdx.y);
    fnv1a128(hash, threadIdx.z);
    fnv1a128(hash, blockIdx.x);
    fnv1a128(hash, blockIdx.y);
    fnv1a128(hash, blockIdx.z);
    fnv1a128(hash, uint32_t(seed));
    fnv1a128(hash, uint32_t(seed >> 32));

    // Initialize KISS99 state with hash
    z     = hash[0];
    w     = hash[1];
    jsr   = hash[2];
    jcong = hash[3];
  }
};

/** The main random number generator class, fully on GPUs */
class RngImpl {
 public:
  RngImpl(uint64_t _s, GeneratorType _t = GenPhilox)
    : type(_t),
      offset(0),
      // simple heuristic to make sure all SMs will be occupied properly
      // and also not too many initialization calls will be made by each thread
      nBlocks(4 * getMultiProcessorCount()),
      gen()
  {
    seed(_s);
  }

  void seed(uint64_t _s)
  {
    gen.seed(_s);
    offset = 0;
  }

  template <typename IdxT>
  void affine_transform_params(IdxT n, IdxT& a, IdxT& b)
  {
    // always keep 'a' to be coprime to 'n'
    a = gen() % n;
    while (gcd(a, n) != 1) {
      ++a;
      if (a >= n) a = 0;
    }
    // the bias term 'b' can be any number in the range of [0, n)
    b = gen() % n;
  }

  template <typename Type, typename LenType = int>
  void uniform(Type* ptr, LenType len, Type start, Type end, cudaStream_t stream)
  {
    static_assert(std::is_floating_point<Type>::value,
                  "Type for 'uniform' can only be floating point!");
    custom_distribution(
      ptr,
      len,
      [=] __device__(Type val, LenType idx) { return (val * (end - start)) + start; },
      stream);
  }
  template <typename IntType, typename LenType = int>
  void uniformInt(IntType* ptr, LenType len, IntType start, IntType end, cudaStream_t stream)
  {
    static_assert(std::is_integral<IntType>::value, "Type for 'uniformInt' can only be integer!");
    custom_distribution(
      ptr,
      len,
      [=] __device__(IntType val, LenType idx) { return (val % (end - start)) + start; },
      stream);
  }

  template <typename Type, typename LenType = int>
  void normal(Type* ptr, LenType len, Type mu, Type sigma, cudaStream_t stream)
  {
    static_assert(std::is_floating_point<Type>::value,
                  "Type for 'normal' can only be floating point!");
    rand2Impl(
      offset,
      ptr,
      len,
      [=] __device__(Type & val1, Type & val2, LenType idx1, LenType idx2) {
        box_muller_transform<Type>(val1, val2, sigma, mu);
      },
      NumThreads,
      nBlocks,
      type,
      stream);
  }
  template <typename IntType, typename LenType = int>
  void normalInt(IntType* ptr, LenType len, IntType mu, IntType sigma, cudaStream_t stream)
  {
    static_assert(std::is_integral<IntType>::value, "Type for 'normalInt' can only be integer!");
    rand2Impl<IntType, double>(
      offset,
      ptr,
      len,
      [=] __device__(double& val1, double& val2, LenType idx1, LenType idx2) {
        box_muller_transform<double>(val1, val2, sigma, mu);
      },
      NumThreads,
      nBlocks,
      type,
      stream);
  }

  template <typename Type, typename LenType = int>
  void normalTable(Type* ptr,
                   LenType n_rows,
                   LenType n_cols,
                   const Type* mu,
                   const Type* sigma_vec,
                   Type sigma,
                   cudaStream_t stream)
  {
    rand2Impl(
      offset,
      ptr,
      n_rows * n_cols,
      [=] __device__(Type & val1, Type & val2, LenType idx1, LenType idx2) {
        // yikes! use fast-int-div
        auto col1  = idx1 % n_cols;
        auto col2  = idx2 % n_cols;
        auto mean1 = mu[col1];
        auto mean2 = mu[col2];
        auto sig1  = sigma_vec == nullptr ? sigma : sigma_vec[col1];
        auto sig2  = sigma_vec == nullptr ? sigma : sigma_vec[col2];
        box_muller_transform<Type>(val1, val2, sig1, mean1, sig2, mean2);
      },
      NumThreads,
      nBlocks,
      type,
      stream);
  }

  template <typename Type, typename LenType = int>
  void fill(Type* ptr, LenType len, Type val, cudaStream_t stream)
  {
    detail::constFillKernel<Type><<<nBlocks, NumThreads, 0, stream>>>(ptr, len, val);
    CUDA_CHECK(cudaPeekAtLastError());
  }

  template <typename Type, typename OutType = bool, typename LenType = int>
  void bernoulli(OutType* ptr, LenType len, Type prob, cudaStream_t stream)
  {
    custom_distribution<OutType, Type>(
      ptr, len, [=] __device__(Type val, LenType idx) { return val > prob; }, stream);
  }

  template <typename Type, typename LenType = int>
  void scaled_bernoulli(Type* ptr, LenType len, Type prob, Type scale, cudaStream_t stream)
  {
    static_assert(std::is_floating_point<Type>::value,
                  "Type for 'scaled_bernoulli' can only be floating point!");
    custom_distribution(
      ptr,
      len,
      [=] __device__(Type val, LenType idx) { return val > prob ? -scale : scale; },
      stream);
  }

  template <typename Type, typename LenType = int>
  void gumbel(Type* ptr, LenType len, Type mu, Type beta, cudaStream_t stream)
  {
    custom_distribution(
      ptr,
      len,
      [=] __device__(Type val, LenType idx) { return mu - beta * raft::myLog(-raft::myLog(val)); },
      stream);
  }

  template <typename Type, typename LenType = int>
  void lognormal(Type* ptr, LenType len, Type mu, Type sigma, cudaStream_t stream)
  {
    rand2Impl(
      offset,
      ptr,
      len,
      [=] __device__(Type & val1, Type & val2, LenType idx1, LenType idx2) {
        box_muller_transform<Type>(val1, val2, sigma, mu);
        val1 = raft::myExp(val1);
        val2 = raft::myExp(val2);
      },
      NumThreads,
      nBlocks,
      type,
      stream);
  }

  template <typename Type, typename LenType = int>
  void logistic(Type* ptr, LenType len, Type mu, Type scale, cudaStream_t stream)
  {
    custom_distribution(
      ptr,
      len,
      [=] __device__(Type val, LenType idx) {
        constexpr Type one = (Type)1.0;
        return mu - scale * raft::myLog(one / val - one);
      },
      stream);
  }

  template <typename Type, typename LenType = int>
  void exponential(Type* ptr, LenType len, Type lambda, cudaStream_t stream)
  {
    custom_distribution(
      ptr,
      len,
      [=] __device__(Type val, LenType idx) {
        constexpr Type one = (Type)1.0;
        return -raft::myLog(one - val) / lambda;
      },
      stream);
  }

  template <typename Type, typename LenType = int>
  void rayleigh(Type* ptr, LenType len, Type sigma, cudaStream_t stream)
  {
    custom_distribution(
      ptr,
      len,
      [=] __device__(Type val, LenType idx) {
        constexpr Type one = (Type)1.0;
        constexpr Type two = (Type)2.0;
        return raft::mySqrt(-two * raft::myLog(one - val)) * sigma;
      },
      stream);
  }

  template <typename Type, typename LenType = int>
  void laplace(Type* ptr, LenType len, Type mu, Type scale, cudaStream_t stream)
  {
    custom_distribution(
      ptr,
      len,
      [=] __device__(Type val, LenType idx) {
        constexpr Type one     = (Type)1.0;
        constexpr Type two     = (Type)2.0;
        constexpr Type oneHalf = (Type)0.5;
        Type out;
        if (val <= oneHalf) {
          out = mu + scale * raft::myLog(two * val);
        } else {
          out = mu - scale * raft::myLog(two * (one - val));
        }
        return out;
      },
      stream);
  }

  template <typename DataT, typename WeightsT, typename IdxT = int>
  void sampleWithoutReplacement(const raft::handle_t& handle,
                                DataT* out,
                                IdxT* outIdx,
                                const DataT* in,
                                const WeightsT* wts,
                                IdxT sampledLen,
                                IdxT len,
                                cudaStream_t stream)
  {
    ASSERT(sampledLen <= len, "sampleWithoutReplacement: 'sampledLen' cant be more than 'len'.");

    rmm::device_uvector<WeightsT> expWts(len, stream);
    rmm::device_uvector<WeightsT> sortedWts(len, stream);
    rmm::device_uvector<IdxT> inIdx(len, stream);
    rmm::device_uvector<IdxT> outIdxBuff(len, stream);
    auto* inIdxPtr = inIdx.data();
    // generate modified weights
    custom_distribution(
      expWts.data(),
      len,
      [wts, inIdxPtr] __device__(WeightsT val, IdxT idx) {
        inIdxPtr[idx]          = idx;
        constexpr WeightsT one = (WeightsT)1.0;
        auto exp               = -raft::myLog(one - val);
        if (wts != nullptr) { return exp / wts[idx]; }
        return exp;
      },
      stream);
    ///@todo: use a more efficient partitioning scheme instead of full sort
    // sort the array and pick the top sampledLen items
    IdxT* outIdxPtr = outIdxBuff.data();
    rmm::device_uvector<char> workspace(0, stream);
    sortPairs(workspace, expWts.data(), sortedWts.data(), inIdxPtr, outIdxPtr, (int)len, stream);
    if (outIdx != nullptr) {
      CUDA_CHECK(cudaMemcpyAsync(
        outIdx, outIdxPtr, sizeof(IdxT) * sampledLen, cudaMemcpyDeviceToDevice, stream));
    }
    raft::scatter<DataT, IdxT>(out, in, outIdxPtr, sampledLen, stream);
  }

  template <typename OutType, typename MathType = OutType, typename LenType = int, typename Lambda>
  void custom_distribution(OutType* ptr, LenType len, Lambda randOp, cudaStream_t stream)
  {
    randImpl<OutType, MathType, LenType, Lambda>(
      offset, ptr, len, randOp, NumThreads, nBlocks, type, stream);
  }
  template <typename OutType, typename MathType = OutType, typename LenType = int, typename Lambda>
  void custom_distribution2(OutType* ptr, LenType len, Lambda randOp, cudaStream_t stream)
  {
    rand2Impl<OutType, MathType, LenType, Lambda>(
      offset, ptr, len, randOp, NumThreads, nBlocks, type, stream);
  }
  /** @} */

 private:
  /** generator type */
  GeneratorType type;
  /**
   * offset is also used to initialize curand state.
   * Limits period of Philox RNG from (4 * 2^128) to (Blocks * Threads * 2^64),
   * but is still a large period.
   */
  uint64_t offset;
  /** number of blocks to launch */
  int nBlocks;
  /** next seed generator for device-side RNG */
  std::mt19937_64 gen;

  static const int NumThreads = 256;

  template <bool IsNormal, typename Type, typename LenType>
  uint64_t _setupSeeds(uint64_t& seed, uint64_t& offset, LenType len, int nThreads, int nBlocks)
  {
    LenType itemsPerThread = raft::ceildiv(len, LenType(nBlocks * nThreads));
    if (IsNormal && itemsPerThread % 2 == 1) { ++itemsPerThread; }
    // curand uses 2 32b uint's to generate one double
    uint64_t factor = sizeof(Type) / sizeof(float);
    if (factor == 0) ++factor;
    // Check if there are enough random numbers left in sequence
    // If not, then generate new seed and start from zero offset
    uint64_t newOffset = offset + LenType(itemsPerThread) * factor;
    if (newOffset < offset) {
      offset    = 0;
      seed      = gen();
      newOffset = itemsPerThread * factor;
    }
    return newOffset;
  }

  template <typename OutType, typename MathType = OutType, typename LenType = int, typename Lambda>
  void randImpl(uint64_t& offset,
                OutType* ptr,
                LenType len,
                Lambda randOp,
                int nThreads,
                int nBlocks,
                GeneratorType type,
                cudaStream_t stream)
  {
    if (len <= 0) return;
    uint64_t seed  = gen();
    auto newOffset = _setupSeeds<false, MathType, LenType>(seed, offset, len, nThreads, nBlocks);
    switch (type) {
      case GenPhilox:
        detail::randKernel<OutType, MathType, detail::PhiloxGenerator, LenType, Lambda>
          <<<nBlocks, nThreads, 0, stream>>>(seed, offset, ptr, len, randOp);
        break;
      case GenTaps:
        detail::randKernel<OutType, MathType, detail::TapsGenerator, LenType, Lambda>
          <<<nBlocks, nThreads, 0, stream>>>(seed, offset, ptr, len, randOp);
        break;
      case GenKiss99:
        detail::randKernel<OutType, MathType, detail::Kiss99Generator, LenType, Lambda>
          <<<nBlocks, nThreads, 0, stream>>>(seed, offset, ptr, len, randOp);
        break;
      default: ASSERT(false, "randImpl: Incorrect generator type! %d", type);
    };
    CUDA_CHECK(cudaGetLastError());
    offset = newOffset;
  }

  template <typename OutType, typename MathType = OutType, typename LenType = int, typename Lambda2>
  void rand2Impl(uint64_t& offset,
                 OutType* ptr,
                 LenType len,
                 Lambda2 rand2Op,
                 int nThreads,
                 int nBlocks,
                 GeneratorType type,
                 cudaStream_t stream)
  {
    if (len <= 0) return;
    auto seed      = gen();
    auto newOffset = _setupSeeds<true, MathType, LenType>(seed, offset, len, nThreads, nBlocks);
    switch (type) {
      case GenPhilox:
        detail::rand2Kernel<OutType, MathType, detail::PhiloxGenerator, LenType, Lambda2>
          <<<nBlocks, nThreads, 0, stream>>>(seed, offset, ptr, len, rand2Op);
        break;
      case GenTaps:
        detail::rand2Kernel<OutType, MathType, detail::TapsGenerator, LenType, Lambda2>
          <<<nBlocks, nThreads, 0, stream>>>(seed, offset, ptr, len, rand2Op);
        break;
      case GenKiss99:
        detail::rand2Kernel<OutType, MathType, detail::Kiss99Generator, LenType, Lambda2>
          <<<nBlocks, nThreads, 0, stream>>>(seed, offset, ptr, len, rand2Op);
        break;
      default: ASSERT(false, "rand2Impl: Incorrect generator type! %d", type);
    };
    CUDA_CHECK(cudaGetLastError());
    offset = newOffset;
  }
};

};  // end namespace detail
};  // end namespace random
};  // end namespace raft
