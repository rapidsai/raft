/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

#include <raft/core/detail/macros.hpp>
#include <raft/random/rng_device.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/detail/cub_wrappers.cuh>
#include <raft/util/scatter.cuh>

namespace raft {
namespace random {
namespace detail {

/**
 * This macro will invoke function `func` with the correct instantiation of
 * device state as the first parameter, and passes all subsequent macro
 * arguments through to the function.
 * Note that you can call this macro with incomplete template specializations
 * as well as triple chevron kernel calls, see the following code example
 * @code
 * template <int C1, int C2, typename GenType>
 * RAFT_KERNEL my_kernel(DeviceState<GenType> state, int arg1) { ... }
 *
 * template <int C1, typename GenType, int C2 = 2>
 * void foo(DeviceState<GenType> state, int arg1) {
 *   my_kernel<C1, C2><<<1, 1>>>(state, arg1);
 * }
 *
 * RAFT_CALL_RNG_FUNC(rng_state, foo<1>, 5);
 * RAFT_CALL_RNG_FUNC(rng_state, (my_kernel<1, 2><<<1, 1>>>), 5);
 * @endcode
 */
#define RAFT_CALL_RNG_FUNC(rng_state, func, ...)                                    \
  switch ((rng_state).type) {                                                       \
    case raft::random::GeneratorType::GenPhilox: {                                  \
      raft::random::DeviceState<raft::random::PhiloxGenerator> r_phil{(rng_state)}; \
      RAFT_DEPAREN(func)(r_phil, ##__VA_ARGS__);                                    \
      break;                                                                        \
    }                                                                               \
    case raft::random::GeneratorType::GenPC: {                                      \
      raft::random::DeviceState<raft::random::PCGenerator> r_pc{(rng_state)};       \
      RAFT_DEPAREN(func)(r_pc, ##__VA_ARGS__);                                      \
      break;                                                                        \
    }                                                                               \
    default: RAFT_FAIL("Unexpected generator type '%d'", int((rng_state).type));    \
  }

template <int ITEMS_PER_CALL, typename GenType, typename... ArgsT>
void call_rng_kernel(DeviceState<GenType> const& dev_state,
                     RngState& rng_state,
                     cudaStream_t stream,
                     ArgsT... args)
{
  auto n_threads = 256;
  auto n_blocks  = 4 * getMultiProcessorCount();
  rngKernel<ITEMS_PER_CALL><<<n_blocks, n_threads, 0, stream>>>(dev_state, args...);
  rng_state.advance(uint64_t(n_blocks) * n_threads, 16);
}

template <typename OutType, typename LenType = int>
void uniform(
  RngState& rng_state, OutType* ptr, LenType len, OutType start, OutType end, cudaStream_t stream)
{
  static_assert(std::is_floating_point<OutType>::value,
                "Type for 'uniform' can only be floating point!");
  UniformDistParams<OutType> params;
  params.start = start;
  params.end   = end;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, rng_state, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void uniformInt(
  RngState& rng_state, OutType* ptr, LenType len, OutType start, OutType end, cudaStream_t stream)
{
  static_assert(std::is_integral<OutType>::value, "Type for 'uniformInt' can only be integer!");
  ASSERT(end > start, "'end' must be greater than 'start'");
  if (sizeof(OutType) == 4) {
    UniformIntDistParams<OutType, uint32_t> params;
    params.start = start;
    params.end   = end;
    params.diff  = uint32_t(params.end - params.start);
    RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, rng_state, stream, ptr, len, params);
  } else {
    UniformIntDistParams<OutType, uint64_t> params;
    params.start = start;
    params.end   = end;
    params.diff  = uint64_t(params.end - params.start);
    RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, rng_state, stream, ptr, len, params);
  }
}

template <typename OutType, typename LenType = int>
void normal(
  RngState& rng_state, OutType* ptr, LenType len, OutType mu, OutType sigma, cudaStream_t stream)
{
  static_assert(std::is_floating_point<OutType>::value,
                "Type for 'normal' can only be floating point!");
  NormalDistParams<OutType> params;
  params.mu    = mu;
  params.sigma = sigma;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<2>, rng_state, stream, ptr, len, params);
}

template <typename IntType, typename LenType = int>
void normalInt(
  RngState& rng_state, IntType* ptr, LenType len, IntType mu, IntType sigma, cudaStream_t stream)
{
  static_assert(std::is_integral<IntType>::value, "Type for 'normalInt' can only be integer!");
  NormalIntDistParams<IntType> params;
  params.mu    = mu;
  params.sigma = sigma;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<2>, rng_state, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void normalTable(RngState& rng_state,
                 OutType* ptr,
                 LenType n_rows,
                 LenType n_cols,
                 const OutType* mu_vec,
                 const OutType* sigma_vec,
                 OutType sigma,
                 cudaStream_t stream)
{
  NormalTableDistParams<OutType, LenType> params;
  params.n_rows    = n_rows;
  params.n_cols    = n_cols;
  params.mu_vec    = mu_vec;
  params.sigma     = sigma;
  params.sigma_vec = sigma_vec;
  LenType len      = n_rows * n_cols;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<2>, rng_state, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void fill(RngState& rng_state, OutType* ptr, LenType len, OutType val, cudaStream_t stream)
{
  InvariantDistParams<OutType> params;
  params.const_val = val;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, rng_state, stream, ptr, len, params);
}

template <typename Type, typename OutType = bool, typename LenType = int>
void bernoulli(RngState& rng_state, OutType* ptr, LenType len, Type prob, cudaStream_t stream)
{
  BernoulliDistParams<Type> params;
  params.prob = prob;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, rng_state, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void scaled_bernoulli(
  RngState& rng_state, OutType* ptr, LenType len, OutType prob, OutType scale, cudaStream_t stream)
{
  static_assert(std::is_floating_point<OutType>::value,
                "Type for 'scaled_bernoulli' can only be floating point!");
  ScaledBernoulliDistParams<OutType> params;
  params.prob  = prob;
  params.scale = scale;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, rng_state, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void gumbel(
  RngState& rng_state, OutType* ptr, LenType len, OutType mu, OutType beta, cudaStream_t stream)
{
  GumbelDistParams<OutType> params;
  params.mu   = mu;
  params.beta = beta;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, rng_state, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void lognormal(
  RngState& rng_state, OutType* ptr, LenType len, OutType mu, OutType sigma, cudaStream_t stream)
{
  LogNormalDistParams<OutType> params;
  params.mu    = mu;
  params.sigma = sigma;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<2>, rng_state, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void logistic(
  RngState& rng_state, OutType* ptr, LenType len, OutType mu, OutType scale, cudaStream_t stream)
{
  LogisticDistParams<OutType> params;
  params.mu    = mu;
  params.scale = scale;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, rng_state, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void exponential(
  RngState& rng_state, OutType* ptr, LenType len, OutType lambda, cudaStream_t stream)
{
  ExponentialDistParams<OutType> params;
  params.lambda = lambda;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, rng_state, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void rayleigh(RngState& rng_state, OutType* ptr, LenType len, OutType sigma, cudaStream_t stream)
{
  RayleighDistParams<OutType> params;
  params.sigma = sigma;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, rng_state, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void laplace(
  RngState& rng_state, OutType* ptr, LenType len, OutType mu, OutType scale, cudaStream_t stream)
{
  LaplaceDistParams<OutType> params;
  params.mu    = mu;
  params.scale = scale;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, rng_state, stream, ptr, len, params);
}

template <typename GenType, typename OutType, typename WeightType, typename IdxType>
void call_sample_with_replacement_kernel(DeviceState<GenType> const& dev_state,
                                         RngState& rng_state,
                                         cudaStream_t stream,
                                         OutType* out,
                                         const WeightType* weights_csum,
                                         IdxType sampledLen,
                                         IdxType len)
{
  IdxType n_threads = 256;
  IdxType n_blocks  = raft::ceildiv(sampledLen, n_threads);
  sample_with_replacement_kernel<<<n_blocks, n_threads, 0, stream>>>(
    dev_state, out, weights_csum, sampledLen, len);
  rng_state.advance(uint64_t(n_blocks) * n_threads, 1);
}

template <typename OutType, typename WeightType, typename IndexType = OutType>
std::enable_if_t<std::is_integral_v<OutType>> discrete(RngState& rng_state,
                                                       OutType* ptr,
                                                       const WeightType* weights,
                                                       IndexType sampledLen,
                                                       IndexType len,
                                                       cudaStream_t stream)
{
  // Compute the cumulative sums of the weights
  size_t temp_storage_bytes = 0;
  rmm::device_uvector<WeightType> weights_csum(len, stream);
  cub::DeviceScan::InclusiveSum(
    nullptr, temp_storage_bytes, weights, weights_csum.data(), len, stream);
  rmm::device_uvector<uint8_t> temp_storage(temp_storage_bytes, stream);
  cub::DeviceScan::InclusiveSum(
    temp_storage.data(), temp_storage_bytes, weights, weights_csum.data(), len, stream);

  // Sample indices with replacement
  RAFT_CALL_RNG_FUNC(rng_state,
                     call_sample_with_replacement_kernel,
                     rng_state,
                     stream,
                     ptr,
                     weights_csum.data(),
                     sampledLen,
                     len);
}

template <typename DataT, typename WeightsT, typename IdxT = int>
void sampleWithoutReplacement(RngState& rng_state,
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
  SamplingParams<WeightsT, IdxT> params;
  params.inIdxPtr = inIdxPtr;
  params.wts      = wts;

  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, rng_state, stream, expWts.data(), len, params);

  ///@todo: use a more efficient partitioning scheme instead of full sort
  // sort the array and pick the top sampledLen items
  IdxT* outIdxPtr = outIdxBuff.data();
  rmm::device_uvector<char> workspace(0, stream);
  sortPairs(workspace, expWts.data(), sortedWts.data(), inIdxPtr, outIdxPtr, (int)len, stream);
  if (outIdx != nullptr) {
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      outIdx, outIdxPtr, sizeof(IdxT) * sampledLen, cudaMemcpyDeviceToDevice, stream));
  }
  scatter<DataT, IdxT>(out, in, outIdxPtr, sampledLen, stream);
}

template <typename IdxT>
void affine_transform_params(RngState const& rng_state, IdxT n, IdxT& a, IdxT& b)
{
  // always keep 'a' to be coprime to 'n'
  std::mt19937_64 mt_rng(rng_state.seed + rng_state.base_subsequence);
  a = mt_rng() % n;
  while (gcd(a, n) != 1) {
    ++a;
    if (a >= n) a = 0;
  }
  // the bias term 'b' can be any number in the range of [0, n)
  b = mt_rng() % n;
}

};  // end namespace detail
};  // end namespace random
};  // end namespace raft
