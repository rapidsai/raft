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

#ifndef __RNG_H
#define __RNG_H

#pragma once

#include "detail/rng_impl.cuh"

namespace raft {
namespace random {

using detail::RngState;

using detail::GeneratorType;
using detail::GenPC;
using detail::GenPhilox;

using detail::PCGenerator;
using detail::PhiloxGenerator;

using detail::BernoulliDistParams;
using detail::ExponentialDistParams;
using detail::GumbelDistParams;
using detail::InvariantDistParams;
using detail::LaplaceDistParams;
using detail::LogisticDistParams;
using detail::LogNormalDistParams;
using detail::NormalDistParams;
using detail::NormalIntDistParams;
using detail::NormalTableDistParams;
using detail::RayleighDistParams;
using detail::SamplingParams;
using detail::ScaledBernoulliDistParams;
using detail::UniformDistParams;
using detail::UniformIntDistParams;

// Not strictly needed due to C++ ADL rules
using detail::custom_next;

/**
 * @brief Helper method to compute Box Muller transform
 *
 * @tparam Type data type
 *
 * @param[inout] val1   first value
 * @param[inout] val2   second value
 * @param[in]    sigma1 standard deviation of output gaussian for first value
 * @param[in]    mu1    mean of output gaussian for first value
 * @param[in]    sigma2 standard deviation of output gaussian for second value
 * @param[in]    mu2    mean of output gaussian for second value
 * @{
 */
template <typename Type>
DI void box_muller_transform(Type& val1, Type& val2, Type sigma1, Type mu1, Type sigma2, Type mu2)
{
  detail::box_muller_transform(val1, val2, sigma1, mu1, sigma2, mu2);
}

template <typename Type>
DI void box_muller_transform(Type& val1, Type& val2, Type sigma1, Type mu1)
{
  detail::box_muller_transform(val1, val2, sigma1, mu1);
}
/** @} */

class Rng : public detail::RngImpl {
 public:
  /**
   * @brief ctor
   * @param _s 64b seed used to initialize the RNG
   * @param _t core device RNG generator type
   * @note Refer to the `Rng::seed` method for details about seeding the engine
   */
  Rng(uint64_t _s, GeneratorType _t = GenPhilox) : detail::RngImpl(_s, _t) {}

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
    detail::RngImpl::affine_transform_params(n, a, b);
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
    detail::RngImpl::uniform(ptr, len, start, end, stream);
  }

  template <typename OutType, typename LenType = int>
  void uniformInt(OutType* ptr, LenType len, OutType start, OutType end, cudaStream_t stream)
  {
    detail::RngImpl::uniformInt(ptr, len, start, end, stream);
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
    detail::RngImpl::normal(ptr, len, mu, sigma, stream);
  }

  template <typename IntType, typename LenType = int>
  void normalInt(IntType* ptr, LenType len, IntType mu, IntType sigma, cudaStream_t stream)
  {
    detail::RngImpl::normalInt(ptr, len, mu, sigma, stream);
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
    detail::RngImpl::normalTable(ptr, n_rows, n_cols, mu_vec, sigma_vec, sigma, stream);
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
    detail::RngImpl::fill(ptr, len, val, stream);
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
    detail::RngImpl::bernoulli(ptr, len, prob, stream);
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
    detail::RngImpl::scaled_bernoulli(ptr, len, prob, scale, stream);
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
    detail::RngImpl::gumbel(ptr, len, mu, beta, stream);
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
    detail::RngImpl::lognormal(ptr, len, mu, sigma, stream);
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
    detail::RngImpl::logistic(ptr, len, mu, scale, stream);
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
    detail::RngImpl::exponential(ptr, len, lambda, stream);
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
    detail::RngImpl::rayleigh(ptr, len, sigma, stream);
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
    detail::RngImpl::laplace(ptr, len, mu, scale, stream);
  }

  void advance(uint64_t max_streams, uint64_t max_calls_per_subsequence)
  {
    detail::RngImpl::advance(max_streams, max_calls_per_subsequence);
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
    detail::RngImpl::sampleWithoutReplacement(
      handle, out, outIdx, in, wts, sampledLen, len, stream);
  }
};

};  // end namespace random
};  // end namespace raft

#endif