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

#include <raft/random/rng_state.hpp>
#include <raft/util/cuda_utils.cuh>

#include <curand_kernel.h>

#include <rmm/device_uvector.hpp>

#include <random>

namespace raft {
namespace random {
namespace detail {

/**
 * The device state used to communicate RNG state from host to device.
 * As of now, it is just a templated version of `RngState`.
 */
template <typename GenType>
struct DeviceState {
  using gen_t                    = GenType;
  static constexpr auto GEN_TYPE = gen_t::GEN_TYPE;

  explicit DeviceState(const RngState& rng_state)
    : seed(rng_state.seed), base_subsequence(rng_state.base_subsequence)
  {
  }

  uint64_t seed;
  uint64_t base_subsequence;
};

template <typename OutType>
struct InvariantDistParams {
  OutType const_val;
};

template <typename OutType>
struct UniformDistParams {
  OutType start;
  OutType end;
};

template <typename OutType, typename DiffType>
struct UniformIntDistParams {
  OutType start;
  OutType end;
  DiffType diff;
};

template <typename OutType>
struct NormalDistParams {
  OutType mu;
  OutType sigma;
};

template <typename IntType>
struct NormalIntDistParams {
  IntType mu;
  IntType sigma;
};

template <typename OutType, typename LenType>
struct NormalTableDistParams {
  LenType n_rows;
  LenType n_cols;
  const OutType* mu_vec;
  OutType sigma;
  const OutType* sigma_vec;
};

template <typename OutType>
struct BernoulliDistParams {
  OutType prob;
};

template <typename OutType>
struct ScaledBernoulliDistParams {
  OutType prob;
  OutType scale;
};

template <typename OutType>
struct GumbelDistParams {
  OutType mu;
  OutType beta;
};

template <typename OutType>
struct LogNormalDistParams {
  OutType mu;
  OutType sigma;
};

template <typename OutType>
struct LogisticDistParams {
  OutType mu;
  OutType scale;
};

template <typename OutType>
struct ExponentialDistParams {
  OutType lambda;
};

template <typename OutType>
struct RayleighDistParams {
  OutType sigma;
};

template <typename OutType>
struct LaplaceDistParams {
  OutType mu;
  OutType scale;
};

// Not really a distro, useful for sample without replacement function
template <typename WeightsT, typename IdxT>
struct SamplingParams {
  IdxT* inIdxPtr;
  const WeightsT* wts;
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

template <typename GenType, typename OutType, typename LenType>
DI void custom_next(GenType& gen,
                    OutType* val,
                    InvariantDistParams<OutType> params,
                    LenType idx    = 0,
                    LenType stride = 0)
{
  *val = params.const_val;
}

template <typename GenType, typename OutType, typename LenType>
DI void custom_next(GenType& gen,
                    OutType* val,
                    UniformDistParams<OutType> params,
                    LenType idx    = 0,
                    LenType stride = 0)
{
  OutType res;
  gen.next(res);
  *val = (res * (params.end - params.start)) + params.start;
}

template <typename GenType, typename OutType, typename LenType>
DI void custom_next(GenType& gen,
                    OutType* val,
                    UniformIntDistParams<OutType, uint32_t> params,
                    LenType idx    = 0,
                    LenType stride = 0)
{
  uint32_t x = 0;
  uint32_t s = params.diff;
  gen.next(x);
  uint64_t m = uint64_t(x) * s;
  uint32_t l = uint32_t(m);
  if (l < s) {
    uint32_t t = (-s) % s;  // (2^32 - s) mod s
    while (l < t) {
      gen.next(x);
      m = uint64_t(x) * s;
      l = uint32_t(m);
    }
  }
  *val = OutType(m >> 32) + params.start;
}

template <typename GenType, typename OutType, typename LenType>
DI void custom_next(GenType& gen,
                    OutType* val,
                    UniformIntDistParams<OutType, uint64_t> params,
                    LenType idx    = 0,
                    LenType stride = 0)
{
  uint64_t x = 0;
  gen.next(x);
  uint64_t s = params.diff;
  uint64_t m_lo, m_hi;
  // m = x * s;
  asm("mul.hi.u64 %0, %1, %2;" : "=l"(m_hi) : "l"(x), "l"(s));
  asm("mul.lo.u64 %0, %1, %2;" : "=l"(m_lo) : "l"(x), "l"(s));
  if (m_lo < s) {
    uint64_t t = (-s) % s;  // (2^64 - s) mod s
    while (m_lo < t) {
      gen.next(x);
      asm("mul.hi.u64 %0, %1, %2;" : "=l"(m_hi) : "l"(x), "l"(s));
      asm("mul.lo.u64 %0, %1, %2;" : "=l"(m_lo) : "l"(x), "l"(s));
    }
  }
  *val = OutType(m_hi) + params.start;
}

template <typename GenType, typename OutType, typename LenType>
DI void custom_next(
  GenType& gen, OutType* val, NormalDistParams<OutType> params, LenType idx = 0, LenType stride = 0)
{
  OutType res1, res2;

  do {
    gen.next(res1);
  } while (res1 == OutType(0.0));

  gen.next(res2);

  box_muller_transform<OutType>(res1, res2, params.sigma, params.mu);
  *val       = res1;
  *(val + 1) = res2;
}

template <typename GenType, typename IntType, typename LenType>
DI void custom_next(GenType& gen,
                    IntType* val,
                    NormalIntDistParams<IntType> params,
                    LenType idx    = 0,
                    LenType stride = 0)
{
  IntType res1_int, res2_int;

  do {
    gen.next(res1_int);
  } while (res1_int == 0);

  gen.next(res2_int);
  double res1  = static_cast<double>(res1_int);
  double res2  = static_cast<double>(res2_int);
  double mu    = static_cast<double>(params.mu);
  double sigma = static_cast<double>(params.sigma);
  box_muller_transform<double>(res1, res2, sigma, mu);
  *val       = static_cast<IntType>(res1);
  *(val + 1) = static_cast<IntType>(res2);
}

template <typename GenType, typename OutType, typename LenType>
DI void custom_next(GenType& gen,
                    OutType* val,
                    NormalTableDistParams<OutType, LenType> params,
                    LenType idx,
                    LenType stride)
{
  OutType res1, res2;

  do {
    gen.next(res1);
  } while (res1 == OutType(0.0));

  gen.next(res2);
  LenType col1  = idx % params.n_cols;
  LenType col2  = (idx + stride) % params.n_cols;
  OutType mean1 = params.mu_vec[col1];
  OutType mean2 = params.mu_vec[col2];
  OutType sig1  = params.sigma_vec == nullptr ? params.sigma : params.sigma_vec[col1];
  OutType sig2  = params.sigma_vec == nullptr ? params.sigma : params.sigma_vec[col2];
  box_muller_transform<OutType>(res1, res2, sig1, mean1, sig2, mean2);
  *val       = res1;
  *(val + 1) = res2;
}

template <typename GenType, typename OutType, typename Type, typename LenType>
DI void custom_next(
  GenType& gen, OutType* val, BernoulliDistParams<Type> params, LenType idx = 0, LenType stride = 0)
{
  Type res = 0;
  gen.next(res);
  *val = res < params.prob;
}

template <typename GenType, typename OutType, typename LenType>
DI void custom_next(GenType& gen,
                    OutType* val,
                    ScaledBernoulliDistParams<OutType> params,
                    LenType idx,
                    LenType stride)
{
  OutType res = 0;
  gen.next(res);
  *val = res < params.prob ? -params.scale : params.scale;
}

template <typename GenType, typename OutType, typename LenType>
DI void custom_next(
  GenType& gen, OutType* val, GumbelDistParams<OutType> params, LenType idx = 0, LenType stride = 0)
{
  OutType res = 0;

  do {
    gen.next(res);
  } while (res == OutType(0.0));

  *val = params.mu - params.beta * raft::myLog(-raft::myLog(res));
}

template <typename GenType, typename OutType, typename LenType>
DI void custom_next(GenType& gen,
                    OutType* val,
                    LogNormalDistParams<OutType> params,
                    LenType idx    = 0,
                    LenType stride = 0)
{
  OutType res1 = 0, res2 = 0;
  do {
    gen.next(res1);
  } while (res1 == OutType(0.0));

  gen.next(res2);
  box_muller_transform<OutType>(res1, res2, params.sigma, params.mu);
  *val       = raft::myExp(res1);
  *(val + 1) = raft::myExp(res2);
}

template <typename GenType, typename OutType, typename LenType>
DI void custom_next(GenType& gen,
                    OutType* val,
                    LogisticDistParams<OutType> params,
                    LenType idx    = 0,
                    LenType stride = 0)
{
  OutType res;

  do {
    gen.next(res);
  } while (res == OutType(0.0));

  constexpr OutType one = (OutType)1.0;
  *val                  = params.mu - params.scale * raft::myLog(one / res - one);
}

template <typename GenType, typename OutType, typename LenType>
DI void custom_next(GenType& gen,
                    OutType* val,
                    ExponentialDistParams<OutType> params,
                    LenType idx    = 0,
                    LenType stride = 0)
{
  OutType res;
  gen.next(res);
  constexpr OutType one = (OutType)1.0;
  *val                  = -raft::myLog(one - res) / params.lambda;
}

template <typename GenType, typename OutType, typename LenType>
DI void custom_next(GenType& gen,
                    OutType* val,
                    RayleighDistParams<OutType> params,
                    LenType idx    = 0,
                    LenType stride = 0)
{
  OutType res;
  gen.next(res);

  constexpr OutType one = (OutType)1.0;
  constexpr OutType two = (OutType)2.0;
  *val                  = raft::mySqrt(-two * raft::myLog(one - res)) * params.sigma;
}

template <typename GenType, typename OutType, typename LenType>
DI void custom_next(GenType& gen,
                    OutType* val,
                    LaplaceDistParams<OutType> params,
                    LenType idx    = 0,
                    LenType stride = 0)
{
  OutType res, out;

  do {
    gen.next(res);
  } while (res == OutType(0.0));

  constexpr OutType one     = (OutType)1.0;
  constexpr OutType two     = (OutType)2.0;
  constexpr OutType oneHalf = (OutType)0.5;

  // The <= comparison here means, number of samples going in `if` branch are more by 1 than `else`
  // branch. However it does not matter as for 0.5 both branches evaluate to same result.
  if (res <= oneHalf) {
    out = params.mu + params.scale * raft::myLog(two * res);
  } else {
    out = params.mu - params.scale * raft::myLog(two * (one - res));
  }
  *val = out;
}

template <typename GenType, typename OutType, typename LenType>
DI void custom_next(
  GenType& gen, OutType* val, SamplingParams<OutType, LenType> params, LenType idx, LenType stride)
{
  OutType res;
  gen.next(res);
  params.inIdxPtr[idx]  = idx;
  constexpr OutType one = (OutType)1.0;
  auto exp              = -raft::myLog(one - res);
  if (params.wts != nullptr) {
    *val = exp / params.wts[idx];
  } else {
    *val = exp;
  }
}

/** Philox-based random number generator */
// Courtesy: Jakub Szuppe
struct PhiloxGenerator {
  static constexpr auto GEN_TYPE = GeneratorType::GenPhilox;

  /**
   * @brief ctor. Initializes the state for RNG
   * @param seed random seed (can be same across all threads)
   * @param subsequence as found in curand docs
   * @param offset as found in curand docs
   */
  DI PhiloxGenerator(uint64_t seed, uint64_t subsequence, uint64_t offset)
  {
    curand_init(seed, subsequence, offset, &philox_state);
  }

  DI PhiloxGenerator(const DeviceState<PhiloxGenerator>& rng_state, const uint64_t subsequence)
  {
    curand_init(rng_state.seed, rng_state.base_subsequence + subsequence, 0, &philox_state);
  }

  /**
   * @defgroup NextRand Generate the next random number
   * @{
   */
  DI uint32_t next_u32()
  {
    uint32_t ret = curand(&(this->philox_state));
    return ret;
  }

  DI uint64_t next_u64()
  {
    uint64_t ret;
    uint32_t a, b;
    a   = next_u32();
    b   = next_u32();
    ret = uint64_t(a) | (uint64_t(b) << 32);
    return ret;
  }

  DI int32_t next_i32()
  {
    int32_t ret;
    uint32_t val;
    val = next_u32();
    ret = int32_t(val & 0x7fffffff);
    return ret;
  }

  DI int64_t next_i64()
  {
    int64_t ret;
    uint64_t val;
    val = next_u64();
    ret = int64_t(val & 0x7fffffffffffffff);
    return ret;
  }

  DI float next_float()
  {
    float ret;
    uint32_t val = next_u32() >> 8;
    ret          = static_cast<float>(val) / float(uint32_t(1) << 24);
    return ret;
  }

  DI double next_double()
  {
    double ret;
    uint64_t val = next_u64() >> 11;
    ret          = static_cast<double>(val) / double(uint64_t(1) << 53);
    return ret;
  }

  DI void next(float& ret)
  {
    // ret = curand_uniform(&(this->philox_state));
    ret = next_float();
  }

  DI void next(double& ret)
  {
    // ret = curand_uniform_double(&(this->philox_state));
    ret = next_double();
  }

  DI void next(uint32_t& ret) { ret = next_u32(); }
  DI void next(uint64_t& ret) { ret = next_u64(); }
  DI void next(int32_t& ret) { ret = next_i32(); }
  DI void next(int64_t& ret) { ret = next_i64(); }

  /** @} */

 private:
  /** the state for RNG */
  curandStatePhilox4_32_10_t philox_state;
};

/** PCG random number generator */
struct PCGenerator {
  static constexpr auto GEN_TYPE = GeneratorType::GenPC;

  /**
   * @brief ctor. Initializes the state for RNG. This code is derived from PCG basic code
   * @param seed the seed (can be same across all threads). Same as PCG's initstate
   * @param subsequence is same as PCG's initseq
   * @param offset unused
   */
  DI PCGenerator(uint64_t seed, uint64_t subsequence, uint64_t offset)
  {
    pcg_state = uint64_t(0);
    inc       = (subsequence << 1u) | 1u;
    uint32_t discard;
    next(discard);
    pcg_state += seed;
    next(discard);
    skipahead(offset);
  }

  DI PCGenerator(const DeviceState<PCGenerator>& rng_state, const uint64_t subsequence)
  {
    pcg_state = uint64_t(0);
    inc       = ((rng_state.base_subsequence + subsequence) << 1u) | 1u;
    uint32_t discard;
    next(discard);
    pcg_state += rng_state.seed;
    next(discard);
    skipahead(subsequence);
  }

  // Based on "Random Number Generation with Arbitrary Strides" F. B. Brown
  // Link https://mcnp.lanl.gov/pdf_files/anl-rn-arb-stride.pdf
  DI void skipahead(uint64_t offset)
  {
    uint64_t G = 1;
    uint64_t h = 6364136223846793005ULL;
    uint64_t C = 0;
    uint64_t f = inc;
    while (offset) {
      if (offset & 1) {
        G = G * h;
        C = C * h + f;
      }
      f = f * (h + 1);
      h = h * h;
      offset >>= 1;
    }
    pcg_state = pcg_state * G + C;
  }

  /**
   * @defgroup NextRand Generate the next random number
   * @brief This code is derived from PCG basic code
   * @{
   */
  DI uint32_t next_u32()
  {
    uint32_t ret;
    uint64_t oldstate   = pcg_state;
    pcg_state           = oldstate * 6364136223846793005ULL + inc;
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot        = oldstate >> 59u;
    ret                 = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    return ret;
  }
  DI uint64_t next_u64()
  {
    uint64_t ret;
    uint32_t a, b;
    a   = next_u32();
    b   = next_u32();
    ret = uint64_t(a) | (uint64_t(b) << 32);
    return ret;
  }

  DI int32_t next_i32()
  {
    int32_t ret;
    uint32_t val;
    val = next_u32();
    ret = int32_t(val & 0x7fffffff);
    return ret;
  }

  DI int64_t next_i64()
  {
    int64_t ret;
    uint64_t val;
    val = next_u64();
    ret = int64_t(val & 0x7fffffffffffffff);
    return ret;
  }

  DI float next_float()
  {
    float ret;
    uint32_t val = next_u32() >> 8;
    ret          = static_cast<float>(val) / (1U << 24);
    return ret;
  }

  DI double next_double()
  {
    double ret;
    uint64_t val = next_u64() >> 11;
    ret          = static_cast<double>(val) / (1LU << 53);
    return ret;
  }

  DI void next(uint32_t& ret) { ret = next_u32(); }
  DI void next(uint64_t& ret) { ret = next_u64(); }
  DI void next(int32_t& ret) { ret = next_i32(); }
  DI void next(int64_t& ret) { ret = next_i64(); }

  DI void next(float& ret) { ret = next_float(); }
  DI void next(double& ret) { ret = next_double(); }

  /** @} */

 private:
  uint64_t pcg_state;
  uint64_t inc;
};

template <int ITEMS_PER_CALL,
          typename OutType,
          typename LenType,
          typename GenType,
          typename ParamType>
__global__ void rngKernel(DeviceState<GenType> rng_state,
                          OutType* ptr,
                          LenType len,
                          ParamType params)
{
  LenType tid = threadIdx.x + static_cast<LenType>(blockIdx.x) * blockDim.x;
  GenType gen(rng_state, (uint64_t)tid);
  const LenType stride = gridDim.x * blockDim.x;
  for (LenType idx = tid; idx < len; idx += stride * ITEMS_PER_CALL) {
    OutType val[ITEMS_PER_CALL];
    custom_next(gen, val, params, idx, stride);
#pragma unroll
    for (int i = 0; i < ITEMS_PER_CALL; i++) {
      if ((idx + i * stride) < len) ptr[idx + i * stride] = val[i];
    }
  }
  return;
}

/**
 * This kernel is deprecated and should be removed in a future release
 */
template <typename OutType,
          typename LenType,
          typename GenType,
          int ITEMS_PER_CALL,
          typename ParamType>
__global__ void fillKernel(
  uint64_t seed, uint64_t adv_subs, uint64_t offset, OutType* ptr, LenType len, ParamType params)
{
  LenType tid = threadIdx.x + static_cast<LenType>(blockIdx.x) * blockDim.x;
  GenType gen(seed, adv_subs + (uint64_t)tid, offset);
  const LenType stride = gridDim.x * blockDim.x;
  for (LenType idx = tid; idx < len; idx += stride * ITEMS_PER_CALL) {
    OutType val[ITEMS_PER_CALL];
    custom_next(gen, val, params, idx, stride);
#pragma unroll
    for (int i = 0; i < ITEMS_PER_CALL; i++) {
      if ((idx + i * stride) < len) ptr[idx + i * stride] = val[i];
    }
  }
  return;
}

#undef POTENTIAL_DEPR

};  // end namespace detail
};  // end namespace random
};  // end namespace raft
