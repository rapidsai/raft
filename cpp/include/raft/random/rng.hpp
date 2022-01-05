/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include "rng_impl.cuh"
#include <raft/common/cub_wrappers.cuh>
#include <raft/common/scatter.cuh>
#include <raft/handle.hpp>
#include <random>
#include <rmm/device_uvector.hpp>

namespace raft {
namespace random {

template <typename OutType,
          typename LenType,
          typename GenType,
          int ITEMS_PER_CALL,
          typename ParamType>
__global__ void fillKernel(
  uint64_t seed, uint64_t adv_subs, uint64_t offset, OutType* ptr, LenType len, ParamType params)
{
  LenType tid = threadIdx.x + blockIdx.x * blockDim.x;
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

class Rng {
 public:
  Rng(uint64_t _s, GeneratorType _t = GenPhilox)
    : seed(_s),
      type(_t),
      offset(0),
      _base_subsequence(0),
      // simple heuristic to make sure all SMs will be occupied properly
      // and also not too many initialization calls will be made by each thread
      nBlocks(4 * getMultiProcessorCount())
  {
  }

  /**
   * @brief Generates the 'a' and 'b' parameters for a modulo affine
   *        transformation equation: `(ax + b) % n`
   *
   * @tparam IdxT integer type
   *
   * @param[in]  n the modulo range
   * @param[out] a slope parameter
   * @param[out] b intercept parameter
   */
  template <typename IdxT>
  void affine_transform_params(IdxT n, IdxT& a, IdxT& b)
  {
    // always keep 'a' to be coprime to 'n'
    std::mt19937_64 mt_rng(seed + _base_subsequence);
    a = mt_rng() % n;
    while (gcd(a, n) != 1) {
      ++a;
      if (a >= n) a = 0;
    }
    // the bias term 'b' can be any number in the range of [0, n)
    b = mt_rng() % n;
  }

  /**
   * @brief Generate uniformly distributed numbers in the given range
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr the output array
   * @param len the number of elements in the output
   * @param start start of the range
   * @param end end of the range
   * @param stream stream where to launch the kernel
   * @{
   */
  template <typename OutType, typename LenType = int>
  void uniform(OutType* ptr, LenType len, OutType start, OutType end, cudaStream_t stream)
  {
    static_assert(std::is_floating_point<OutType>::value,
                  "Type for 'uniform' can only be floating point!");
    UniformDistParams<OutType> params;
    params.start = start;
    params.end   = end;
    kernel_dispatch<OutType, LenType, 1, UniformDistParams<OutType>>(ptr, len, stream, params);
  }

  template <typename OutType, typename LenType = int>
  void uniformInt(OutType* ptr, LenType len, OutType start, OutType end, cudaStream_t stream)
  {
    static_assert(std::is_integral<OutType>::value, "Type for 'uniformInt' can only be integer!");
    ASSERT(end > start, "'end' must be greater than 'start'");
    if (sizeof(OutType) == 4) {
      UniformIntDistParams<OutType, uint32_t> params;
      params.start = start;
      params.end   = end;
      params.diff  = uint32_t(params.end - params.start);
      kernel_dispatch<OutType, LenType, 1, UniformIntDistParams<OutType, uint32_t>>(
        ptr, len, stream, params);
    } else {
      UniformIntDistParams<OutType, uint64_t> params;
      params.start = start;
      params.end   = end;
      params.diff  = uint64_t(params.end - params.start);
      kernel_dispatch<OutType, LenType, 1, UniformIntDistParams<OutType, uint64_t>>(
        ptr, len, stream, params);
    }
  }
  /** @} */

  /**
   * @brief Generate normal distributed numbers
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr the output array
   * @param len the number of elements in the output
   * @param mu mean of the distribution
   * @param sigma std-dev of the distribution
   * @param stream stream where to launch the kernel
   * @{
   */
  template <typename OutType, typename LenType = int>
  void normal(OutType* ptr, LenType len, OutType mu, OutType sigma, cudaStream_t stream)
  {
    static_assert(std::is_floating_point<OutType>::value,
                  "Type for 'normal' can only be floating point!");
    NormalDistParams<OutType> params;
    params.mu    = mu;
    params.sigma = sigma;
    kernel_dispatch<OutType, LenType, 2, NormalDistParams<OutType>>(ptr, len, stream, params);
  }

  template <typename IntType, typename LenType = int>
  void normalInt(IntType* ptr, LenType len, IntType mu, IntType sigma, cudaStream_t stream)
  {
    static_assert(std::is_integral<IntType>::value, "Type for 'normalInt' can only be integer!");
    NormalIntDistParams<IntType> params;
    params.mu    = mu;
    params.sigma = sigma;
    kernel_dispatch<IntType, LenType, 2, NormalIntDistParams<IntType>>(ptr, len, stream, params);
  }
  /** @} */

  /**
   * @brief Generate normal distributed table according to the given set of
   * means and scalar standard deviations.
   *
   * Each row in this table conforms to a normally distributed n-dim vector
   * whose mean is the input vector and standard deviation is the corresponding
   * vector or scalar. Correlations among the dimensions itself is assumed to
   * be absent.
   *
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr the output table (dim = n_rows x n_cols)
   * @param n_rows number of rows in the table
   * @param n_cols number of columns in the table
   * @param mu_vec mean vector (dim = n_cols x 1).
   * @param sigma_vec std-dev vector of each component (dim = n_cols x 1). Pass
   * a nullptr to use the same scalar 'sigma' across all components
   * @param sigma scalar sigma to be used if 'sigma_vec' is nullptr
   * @param stream stream where to launch the kernel
   */
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
    kernel_dispatch<OutType, LenType, 2, NormalTableDistParams<OutType, LenType>>(
      ptr, len, stream, params);
  }

  /**
   * @brief Fill an array with the given value
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr the output array
   * @param len the number of elements in the output
   * @param val value to be filled
   * @param stream stream where to launch the kernel
   */
  template <typename OutType, typename LenType = int>
  void fill(OutType* ptr, LenType len, OutType val, cudaStream_t stream)
  {
    InvariantDistParams<OutType> params;
    params.const_val = val;
    kernel_dispatch<OutType, LenType, 1, InvariantDistParams<OutType>>(ptr, len, stream, params);
  }

  /**
   * @brief Generate bernoulli distributed boolean array
   *
   * @tparam Type    data type in which to compute the probabilities
   * @tparam OutType output data type
   * @tparam LenType data type used to represent length of the arrays
   *
   * @param[out] ptr    the output array
   * @param[in]  len    the number of elements in the output
   * @param[in]  prob   coin-toss probability for heads
   * @param[in]  stream stream where to launch the kernel
   */
  template <typename Type, typename OutType = bool, typename LenType = int>
  void bernoulli(OutType* ptr, LenType len, Type prob, cudaStream_t stream)
  {
    BernoulliDistParams<Type> params;
    params.prob = prob;
    kernel_dispatch<OutType, LenType, 1, BernoulliDistParams<Type>>(ptr, len, stream, params);
  }

  /**
   * @brief Generate bernoulli distributed array and applies scale
   * @tparam Type data type in which to compute the probabilities
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr the output array
   * @param len the number of elements in the output
   * @param prob coin-toss probability for heads
   * @param scale scaling factor
   * @param stream stream where to launch the kernel
   */
  template <typename OutType, typename LenType = int>
  void scaled_bernoulli(OutType* ptr, LenType len, OutType prob, OutType scale, cudaStream_t stream)
  {
    static_assert(std::is_floating_point<OutType>::value,
                  "Type for 'scaled_bernoulli' can only be floating point!");
    ScaledBernoulliDistParams<OutType> params;
    params.prob  = prob;
    params.scale = scale;
    kernel_dispatch<OutType, LenType, 1, ScaledBernoulliDistParams<OutType>>(
      ptr, len, stream, params);
  }

  /**
   * @brief Generate Gumbel distributed random numbers
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr output array
   * @param len number of elements in the output array
   * @param mu mean value
   * @param beta scale value
   * @param stream stream where to launch the kernel
   * @note https://en.wikipedia.org/wiki/Gumbel_distribution
   */
  template <typename OutType, typename LenType = int>
  void gumbel(OutType* ptr, LenType len, OutType mu, OutType beta, cudaStream_t stream)
  {
    GumbelDistParams<OutType> params;
    params.mu   = mu;
    params.beta = beta;
    kernel_dispatch<OutType, LenType, 1, GumbelDistParams<OutType>>(ptr, len, stream, params);
  }

  /**
   * @brief Generate lognormal distributed numbers
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr the output array
   * @param len the number of elements in the output
   * @param mu mean of the distribution
   * @param sigma std-dev of the distribution
   * @param stream stream where to launch the kernel
   */
  template <typename OutType, typename LenType = int>
  void lognormal(OutType* ptr, LenType len, OutType mu, OutType sigma, cudaStream_t stream)
  {
    LogNormalDistParams<OutType> params;
    params.mu    = mu;
    params.sigma = sigma;
    kernel_dispatch<OutType, LenType, 2, LogNormalDistParams<OutType>>(ptr, len, stream, params);
  }

  /**
   * @brief Generate logistic distributed random numbers
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr output array
   * @param len number of elements in the output array
   * @param mu mean value
   * @param scale scale value
   * @param stream stream where to launch the kernel
   */
  template <typename OutType, typename LenType = int>
  void logistic(OutType* ptr, LenType len, OutType mu, OutType scale, cudaStream_t stream)
  {
    LogisticDistParams<OutType> params;
    params.mu    = mu;
    params.scale = scale;
    kernel_dispatch<OutType, LenType, 1, LogisticDistParams<OutType>>(ptr, len, stream, params);
  }

  /**
   * @brief Generate exponentially distributed random numbers
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr output array
   * @param len number of elements in the output array
   * @param lambda the lambda
   * @param stream stream where to launch the kernel
   */
  template <typename OutType, typename LenType = int>
  void exponential(OutType* ptr, LenType len, OutType lambda, cudaStream_t stream)
  {
    ExponentialDistParams<OutType> params;
    params.lambda = lambda;
    kernel_dispatch<OutType, LenType, 1, ExponentialDistParams<OutType>>(ptr, len, stream, params);
  }

  /**
   * @brief Generate rayleigh distributed random numbers
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr output array
   * @param len number of elements in the output array
   * @param sigma the sigma
   * @param stream stream where to launch the kernel
   */
  template <typename OutType, typename LenType = int>
  void rayleigh(OutType* ptr, LenType len, OutType sigma, cudaStream_t stream)
  {
    RayleighDistParams<OutType> params;
    params.sigma = sigma;
    kernel_dispatch<OutType, LenType, 1, RayleighDistParams<OutType>>(ptr, len, stream, params);
  }

  /**
   * @brief Generate laplace distributed random numbers
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr output array
   * @param len number of elements in the output array
   * @param mu the mean
   * @param scale the scale
   * @param stream stream where to launch the kernel
   */
  template <typename OutType, typename LenType = int>
  void laplace(OutType* ptr, LenType len, OutType mu, OutType scale, cudaStream_t stream)
  {
    LaplaceDistParams<OutType> params;
    params.mu    = mu;
    params.scale = scale;
    kernel_dispatch<OutType, LenType, 1, LaplaceDistParams<OutType>>(ptr, len, stream, params);
  }

  void advance(uint64_t max_streams, uint64_t max_calls_per_subsequence)
  {
    _base_subsequence += max_streams;
  }

  template <typename OutType, typename LenType, int ITEMS_PER_CALL, typename ParamType>
  void kernel_dispatch(OutType* ptr, LenType len, cudaStream_t stream, ParamType params)
  {
    switch (type) {
      case GenPhilox:
        fillKernel<OutType, LenType, PhiloxGenerator, ITEMS_PER_CALL>
          <<<nBlocks, nThreads, 0, stream>>>(seed, _base_subsequence, offset, ptr, len, params);
        break;
      case GenPC:
        fillKernel<OutType, LenType, PCGenerator, ITEMS_PER_CALL>
          <<<nBlocks, nThreads, 0, stream>>>(seed, _base_subsequence, offset, ptr, len, params);
        break;
      default: break;
    }
    // The max_calls_per_subsequence parameter does not matter for now, using 16 for now
    advance(uint64_t(nBlocks) * nThreads, 16);
    return;
  }

  /**
   * @brief Sample the input array without replacement, optionally based on the
   * input weight vector for each element in the array
   *
   * Implementation here is based on the `one-pass sampling` algo described here:
   * https://www.ethz.ch/content/dam/ethz/special-interest/baug/ivt/ivt-dam/vpl/reports/1101-1200/ab1141.pdf
   *
   * @note In the sampled array the elements which are picked will always appear
   * in the increasing order of their weights as computed using the exponential
   * distribution. So, if you're particular about the order (for eg. array
   * permutations), then this might not be the right choice!
   *
   * @tparam DataT data type
   * @tparam WeightsT weights type
   * @tparam IdxT index type
   * @param handle
   * @param out output sampled array (of length 'sampledLen')
   * @param outIdx indices of the sampled array (of length 'sampledLen'). Pass
   * a nullptr if this is not required.
   * @param in input array to be sampled (of length 'len')
   * @param wts weights array (of length 'len'). Pass a nullptr if uniform
   * sampling is desired
   * @param sampledLen output sampled array length
   * @param len input array length
   * @param stream cuda stream
   */
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
    kernel_dispatch<WeightsT, IdxT, 1, SamplingParams<WeightsT, IdxT>>(
      expWts.data(), len, stream, params);
    ///@todo: use a more efficient partitioning scheme instead of full sort
    // sort the array and pick the top sampledLen items
    IdxT* outIdxPtr = outIdxBuff.data();
    rmm::device_uvector<char> workspace(0, stream);
    sortPairs(workspace, expWts.data(), sortedWts.data(), inIdxPtr, outIdxPtr, (int)len, stream);
    if (outIdx != nullptr) {
      CUDA_CHECK(cudaMemcpyAsync(
        outIdx, outIdxPtr, sizeof(IdxT) * sampledLen, cudaMemcpyDeviceToDevice, stream));
    }
    scatter<DataT, IdxT>(out, in, outIdxPtr, sampledLen, stream);
  }

  GeneratorType type;
  uint64_t offset;
  uint64_t seed;
  uint64_t _base_subsequence;
  /** number of blocks to launch */
  int nBlocks;
  static const int nThreads = 256;
};

};  // end namespace random
};  // end namespace raft
