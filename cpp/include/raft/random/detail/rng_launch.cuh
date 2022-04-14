/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

#include "rng_state.hpp"
#include "rng_device.cuh"

#include <raft/cudart_utils.h>

namespace raft {
namespace random {
namespace detail {

#define RAFT_CALL_RNG_FUNC(rng_state, func, ...) \
  switch ((rng_state).type) { \
    case GeneratorType::Philox: \
      DeviceState<PhiloxGenerator> device_state{(rng_state)}; \
      func(device_state, __VA_ARGS__);  \
      break; \
    case GeneratorType::PCG: \
      DeviceState<PCGenerator> device_state{(rng_state)}; \
      func(device_state, __VA_ARGS__); \
      break; \
    default: \
      RAFT_FAIL("Unepxected generator type '%d'", int((rng_state).type)); \
  }

/**
 * This function is useful if all template arguments to `func` can be inferred
 * by the compiler, otherwise use the MACRO `RAFT_CALL_RNG_FUNC` which
 * can accept incomplete template specializations for the function
 */
template <typename FuncT, typename... ArgsT>
void call_rng_func(RngState const &rng_state, FuncT func, ArgsT... args) {
  switch (rng_state.type) {
    case GeneratorType::Philox:
      DeviceState<PhiloxGenerator> device_state{rng_state};
      func(device_state, args...);
      break;
    case GeneratorType::PCG:
      DeviceState<PCGenerator> device_state{rng_state};
      func(device_state, args...);
      break;
    default:
      RAFT_FAIL("Unepxected generator type '%d'", int(rng_state.type));
  }
}

template <int ITEMS_PER_CALL, typename... ArgsT>
void call_rng_kernel(DeviceState const &state, cudaStream_t stream, ArgsT... args) {
  static constexpr auto N_THREADS = 256;
  auto n_blocks = 4 * getMultiProcessorCount();
  rngKernel<ITEMS_PER_CALL><<<n_blocks, N_THREADS, 0, stream>>>(state, args...);
}

template <typename OutType, typename LenType = int>
void uniform(RngState& rng_state, OutType* ptr, LenType len, OutType start, OutType end, cudaStream_t stream)
{
  static_assert(std::is_floating_point<OutType>::value,
                "Type for 'uniform' can only be floating point!");
  UniformDistParams<OutType> params;
  params.start = start;
  params.end   = end;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void uniformInt(RngState& rng_state, OutType* ptr, LenType len, OutType start, OutType end, cudaStream_t stream)
{
  static_assert(std::is_integral<OutType>::value, "Type for 'uniformInt' can only be integer!");
  ASSERT(end > start, "'end' must be greater than 'start'");
  if (sizeof(OutType) == 4) {
    UniformIntDistParams<OutType, uint32_t> params;
    params.start = start;
    params.end   = end;
    params.diff  = uint32_t(params.end - params.start);
    RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, stream, ptr, len, params);
  } else {
    UniformIntDistParams<OutType, uint64_t> params;
    params.start = start;
    params.end   = end;
    params.diff  = uint64_t(params.end - params.start);
    RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, stream, ptr, len, params);
  }
}

template <typename OutType, typename LenType = int>
void normal(OutType* ptr, LenType len, OutType mu, OutType sigma, cudaStream_t stream)
{
  static_assert(std::is_floating_point<OutType>::value,
                "Type for 'normal' can only be floating point!");
  NormalDistParams<OutType> params;
  params.mu    = mu;
  params.sigma = sigma;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<2>, stream, ptr, len, params);
}

template <typename IntType, typename LenType = int>
void normalInt(IntType* ptr, LenType len, IntType mu, IntType sigma, cudaStream_t stream)
{
  static_assert(std::is_integral<IntType>::value, "Type for 'normalInt' can only be integer!");
  NormalIntDistParams<IntType> params;
  params.mu    = mu;
  params.sigma = sigma;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<2>, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void normalTable(OutType* ptr,
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
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<2>, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void fill(OutType* ptr, LenType len, OutType val, cudaStream_t stream)
{
  InvariantDistParams<OutType> params;
  params.const_val = val;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, stream, ptr, len, params);
}

template <typename Type, typename OutType = bool, typename LenType = int>
void bernoulli(OutType* ptr, LenType len, Type prob, cudaStream_t stream)
{
  BernoulliDistParams<Type> params;
  params.prob = prob;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void scaled_bernoulli(OutType* ptr, LenType len, OutType prob, OutType scale, cudaStream_t stream)
{
  static_assert(std::is_floating_point<OutType>::value,
                "Type for 'scaled_bernoulli' can only be floating point!");
  ScaledBernoulliDistParams<OutType> params;
  params.prob  = prob;
  params.scale = scale;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void gumbel(OutType* ptr, LenType len, OutType mu, OutType beta, cudaStream_t stream)
{
  GumbelDistParams<OutType> params;
  params.mu   = mu;
  params.beta = beta;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void lognormal(OutType* ptr, LenType len, OutType mu, OutType sigma, cudaStream_t stream)
{
  LogNormalDistParams<OutType> params;
  params.mu    = mu;
  params.sigma = sigma;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<2>, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void logistic(OutType* ptr, LenType len, OutType mu, OutType scale, cudaStream_t stream)
{
  LogisticDistParams<OutType> params;
  params.mu    = mu;
  params.scale = scale;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void exponential(OutType* ptr, LenType len, OutType lambda, cudaStream_t stream)
{
  ExponentialDistParams<OutType> params;
  params.lambda = lambda;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void rayleigh(OutType* ptr, LenType len, OutType sigma, cudaStream_t stream)
{
  RayleighDistParams<OutType> params;
  params.sigma = sigma;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void laplace(OutType* ptr, LenType len, OutType mu, OutType scale, cudaStream_t stream)
{
  LaplaceDistParams<OutType> params;
  params.mu    = mu;
  params.scale = scale;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, stream, ptr, len, params);
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
  SamplingParams<WeightsT, IdxT> params;
  params.inIdxPtr = inIdxPtr;
  params.wts      = wts;

  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, stream, ptr, len, params);

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

};  // end namespace detail
};  // end namespace random
};  // end namespace raft
