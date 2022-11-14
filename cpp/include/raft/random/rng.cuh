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

#include "detail/rng_impl.cuh"
#include "detail/rng_impl_deprecated.cuh"  // necessary for now (to be removed)
#include "rng_state.hpp"
#include <cassert>
#include <optional>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <type_traits>
#include <variant>

namespace raft::random {

/**
 * @brief Generate uniformly distributed numbers in the given range
 *
 * @tparam OutputValueType Data type of output random number
 * @tparam Index Data type used to represent length of the arrays
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] out the output array
 * @param[in] start start of the range
 * @param[in] end end of the range
 */
template <typename OutputValueType, typename IndexType>
void uniform(const raft::handle_t& handle,
             RngState& rng_state,
             raft::device_vector_view<OutputValueType, IndexType> out,
             OutputValueType start,
             OutputValueType end)
{
  detail::uniform(rng_state, out.data_handle(), out.extent(0), start, end, handle.get_stream());
}

/**
 * @brief Legacy overload of `uniform` taking raw pointers
 *
 * @tparam OutType data type of output random number
 * @tparam LenType data type used to represent length of the arrays
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] ptr the output array
 * @param[in] len the number of elements in the output
 * @param[in] start start of the range
 * @param[in] end end of the range
 */
template <typename OutType, typename LenType = int>
void uniform(const raft::handle_t& handle,
             RngState& rng_state,
             OutType* ptr,
             LenType len,
             OutType start,
             OutType end)
{
  detail::uniform(rng_state, ptr, len, start, end, handle.get_stream());
}

/**
 * @brief Generate uniformly distributed integers in the given range
 *
 * @tparam OutputValueType Integral type; value type of the output vector
 * @tparam IndexType Type used to represent length of the output vector
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] out the output vector of random numbers
 * @param[in] start start of the range
 * @param[in] end end of the range
 */
template <typename OutputValueType, typename IndexType>
void uniformInt(const raft::handle_t& handle,
                RngState& rng_state,
                raft::device_vector_view<OutputValueType, IndexType> out,
                OutputValueType start,
                OutputValueType end)
{
  static_assert(
    std::is_same<OutputValueType, typename std::remove_cv<OutputValueType>::type>::value,
    "uniformInt: The output vector must be a view of nonconst, "
    "so that we can write to it.");
  static_assert(std::is_integral<OutputValueType>::value,
                "uniformInt: The elements of the output vector must have integral type.");
  detail::uniformInt(rng_state, out.data_handle(), out.extent(0), start, end, handle.get_stream());
}

/**
 * @brief Legacy raw pointer overload of `uniformInt`
 *
 * @tparam OutType data type of output random number
 * @tparam LenType data type used to represent length of the arrays
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] ptr the output array
 * @param[in] len the number of elements in the output
 * @param[in] start start of the range
 * @param[in] end end of the range
 */
template <typename OutType, typename LenType = int>
void uniformInt(const raft::handle_t& handle,
                RngState& rng_state,
                OutType* ptr,
                LenType len,
                OutType start,
                OutType end)
{
  detail::uniformInt(rng_state, ptr, len, start, end, handle.get_stream());
}

/**
 * @brief Generate normal distributed numbers
 *   with a given mean and standard deviation
 *
 * @tparam OutputValueType data type of output random number
 * @tparam IndexType data type used to represent length of the arrays
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] out the output array
 * @param[in] mu mean of the distribution
 * @param[in] sigma std-dev of the distribution
 */
template <typename OutputValueType, typename IndexType>
void normal(const raft::handle_t& handle,
            RngState& rng_state,
            raft::device_vector_view<OutputValueType, IndexType> out,
            OutputValueType mu,
            OutputValueType sigma)
{
  detail::normal(rng_state, out.data_handle(), out.extent(0), mu, sigma, handle.get_stream());
}

/**
 * @brief Legacy raw pointer overload of `normal`.
 *
 * @tparam OutType data type of output random number
 * @tparam LenType data type used to represent length of the arrays
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] ptr the output array
 * @param[in] len the number of elements in the output
 * @param[in] mu mean of the distribution
 * @param[in] sigma std-dev of the distribution
 */
template <typename OutType, typename LenType = int>
void normal(const raft::handle_t& handle,
            RngState& rng_state,
            OutType* ptr,
            LenType len,
            OutType mu,
            OutType sigma)
{
  detail::normal(rng_state, ptr, len, mu, sigma, handle.get_stream());
}

/**
 * @brief Generate normal distributed integers
 *
 * @tparam OutputValueType Integral type; value type of the output vector
 * @tparam IndexType Integral type of the output vector's length
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] out the output array
 * @param[in] mu mean of the distribution
 * @param[in] sigma standard deviation of the distribution
 */
template <typename OutputValueType, typename IndexType>
void normalInt(const raft::handle_t& handle,
               RngState& rng_state,
               raft::device_vector_view<OutputValueType, IndexType> out,
               OutputValueType mu,
               OutputValueType sigma)
{
  static_assert(
    std::is_same<OutputValueType, typename std::remove_cv<OutputValueType>::type>::value,
    "normalInt: The output vector must be a view of nonconst, "
    "so that we can write to it.");
  static_assert(std::is_integral<OutputValueType>::value,
                "normalInt: The output vector's value type must be an integer.");

  detail::normalInt(rng_state, out.data_handle(), out.extent(0), mu, sigma, handle.get_stream());
}

/**
 * @brief Legacy raw pointer overload of `normalInt`
 *
 * @tparam OutType data type of output random number
 * @tparam LenType data type used to represent length of the arrays
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] ptr the output array
 * @param[in] len the number of elements in the output
 * @param[in] mu mean of the distribution
 * @param[in] sigma std-dev of the distribution
 */
template <typename IntType, typename LenType = int>
void normalInt(const raft::handle_t& handle,
               RngState& rng_state,
               IntType* ptr,
               LenType len,
               IntType mu,
               IntType sigma)
{
  detail::normalInt(rng_state, ptr, len, mu, sigma, handle.get_stream());
}

/**
 * @brief Generate normal distributed table according to the given set of
 * means and scalar standard deviations.
 *
 * Each row in this table conforms to a normally distributed n-dim vector
 * whose mean is the input vector and standard deviation is the corresponding
 * vector or scalar. Correlations among the dimensions itself are assumed to
 * be absent.
 *
 * @tparam OutputValueType data type of output random number
 * @tparam IndexType data type used to represent length of the arrays
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[in] mu_vec mean vector (of length `out.extent(1)`)
 * @param[in] sigma Either the standard-deviation vector
 *            (of length `out.extent(1)`) of each component,
 *            or a scalar standard deviation for all components.
 * @param[out] out the output table
 */
template <typename OutputValueType, typename IndexType>
void normalTable(
  const raft::handle_t& handle,
  RngState& rng_state,
  raft::device_vector_view<const OutputValueType, IndexType> mu_vec,
  std::variant<raft::device_vector_view<const OutputValueType, IndexType>, OutputValueType> sigma,
  raft::device_matrix_view<OutputValueType, IndexType, raft::row_major> out)
{
  const OutputValueType* sigma_vec_ptr = nullptr;
  OutputValueType sigma_value{};

  using sigma_vec_type = raft::device_vector_view<const OutputValueType, IndexType>;
  if (std::holds_alternative<sigma_vec_type>(sigma)) {
    auto sigma_vec = std::get<sigma_vec_type>(sigma);
    RAFT_EXPECTS(sigma_vec.extent(0) == out.extent(1),
                 "normalTable: The sigma vector "
                 "has length %zu, which does not equal the number of columns "
                 "in the output table %zu.",
                 static_cast<size_t>(sigma_vec.extent(0)),
                 static_cast<size_t>(out.extent(1)));
    // The extra length check makes this work even if sigma_vec views a std::vector,
    // where .data() need not return nullptr even if .size() is zero.
    sigma_vec_ptr = sigma_vec.extent(0) == 0 ? nullptr : sigma_vec.data_handle();
  } else {
    sigma_value = std::get<OutputValueType>(sigma);
  }

  RAFT_EXPECTS(mu_vec.extent(0) == out.extent(1),
               "normalTable: The mu vector "
               "has length %zu, which does not equal the number of columns "
               "in the output table %zu.",
               static_cast<size_t>(mu_vec.extent(0)),
               static_cast<size_t>(out.extent(1)));

  detail::normalTable(rng_state,
                      out.data_handle(),
                      out.extent(0),
                      out.extent(1),
                      mu_vec.data_handle(),
                      sigma_vec_ptr,
                      sigma_value,
                      handle.get_stream());
}

/**
 * @brief Legacy raw pointer overload of `normalTable`.
 *
 * Each row in this table conforms to a normally distributed n-dim vector
 * whose mean is the input vector and standard deviation is the corresponding
 * vector or scalar. Correlations among the dimensions itself are assumed to
 * be absent.
 *
 * @tparam OutType data type of output random number
 * @tparam LenType data type used to represent length of the arrays
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] ptr the output table (dim = n_rows x n_cols)
 * @param[in] n_rows number of rows in the table
 * @param[in] n_cols number of columns in the table
 * @param[in] mu_vec mean vector (dim = n_cols x 1).
 * @param[in] sigma_vec std-dev vector of each component (dim = n_cols x 1). Pass
 * a nullptr to use the same scalar 'sigma' across all components
 * @param[in] sigma scalar sigma to be used if 'sigma_vec' is nullptr
 */
template <typename OutType, typename LenType = int>
void normalTable(const raft::handle_t& handle,
                 RngState& rng_state,
                 OutType* ptr,
                 LenType n_rows,
                 LenType n_cols,
                 const OutType* mu_vec,
                 const OutType* sigma_vec,
                 OutType sigma)
{
  detail::normalTable(
    rng_state, ptr, n_rows, n_cols, mu_vec, sigma_vec, sigma, handle.get_stream());
}

/**
 * @brief Fill a vector with the given value
 *
 * @tparam OutputValueType Value type of the output vector
 * @tparam IndexType Integral type used to represent length of the output vector
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[in] val value with which to fill the output vector
 * @param[out] out the output vector
 */
template <typename OutputValueType, typename IndexType>
void fill(const raft::handle_t& handle,
          RngState& rng_state,
          OutputValueType val,
          raft::device_vector_view<OutputValueType, IndexType> out)
{
  detail::fill(rng_state, out.data_handle(), out.extent(0), val, handle.get_stream());
}

/**
 * @brief Legacy raw pointer overload of `fill`
 *
 * @tparam OutType data type of output random number
 * @tparam LenType data type used to represent length of the arrays
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] ptr the output array
 * @param[in] len the number of elements in the output
 * @param[in] val value to be filled
 */
template <typename OutType, typename LenType = int>
void fill(const raft::handle_t& handle, RngState& rng_state, OutType* ptr, LenType len, OutType val)
{
  detail::fill(rng_state, ptr, len, val, handle.get_stream());
}

/**
 * @brief Generate bernoulli distributed boolean array
 *
 * @tparam OutputValueType Type of each element of the output vector;
 *         must be able to represent boolean values (e.g., `bool`)
 * @tparam IndexType Integral type of the output vector's length
 * @tparam Type Data type in which to compute the probabilities
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] out the output vector
 * @param[in] prob coin-toss probability for heads
 */
template <typename OutputValueType, typename IndexType, typename Type>
void bernoulli(const raft::handle_t& handle,
               RngState& rng_state,
               raft::device_vector_view<OutputValueType, IndexType> out,
               Type prob)
{
  detail::bernoulli(rng_state, out.data_handle(), out.extent(0), prob, handle.get_stream());
}

/**
 * @brief Legacy raw pointer overload of `bernoulli`
 *
 * @tparam Type    data type in which to compute the probabilities
 * @tparam OutType output data type
 * @tparam LenType data type used to represent length of the arrays
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] ptr    the output array
 * @param[in]  len    the number of elements in the output
 * @param[in]  prob   coin-toss probability for heads
 */
template <typename Type, typename OutType = bool, typename LenType = int>
void bernoulli(
  const raft::handle_t& handle, RngState& rng_state, OutType* ptr, LenType len, Type prob)
{
  detail::bernoulli(rng_state, ptr, len, prob, handle.get_stream());
}

/**
 * @brief Generate bernoulli distributed array and applies scale
 *
 * @tparam OutputValueType Data type in which to compute the probabilities
 * @tparam IndexType Integral type of the output vector's length
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] out the output vector
 * @param[in] prob coin-toss probability for heads
 * @param[in] scale scaling factor
 */
template <typename OutputValueType, typename IndexType>
void scaled_bernoulli(const raft::handle_t& handle,
                      RngState& rng_state,
                      raft::device_vector_view<OutputValueType, IndexType> out,
                      OutputValueType prob,
                      OutputValueType scale)
{
  detail::scaled_bernoulli(
    rng_state, out.data_handle(), out.extent(0), prob, scale, handle.get_stream());
}

/**
 * @brief Legacy raw pointer overload of `scaled_bernoulli`
 *
 * @tparam OutType data type in which to compute the probabilities
 * @tparam LenType data type used to represent length of the arrays
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] ptr the output array
 * @param[in] len the number of elements in the output
 * @param[in] prob coin-toss probability for heads
 * @param[in] scale scaling factor
 */
template <typename OutType, typename LenType = int>
void scaled_bernoulli(const raft::handle_t& handle,
                      RngState& rng_state,
                      OutType* ptr,
                      LenType len,
                      OutType prob,
                      OutType scale)
{
  detail::scaled_bernoulli(rng_state, ptr, len, prob, scale, handle.get_stream());
}

/**
 * @brief Generate Gumbel distributed random numbers
 *
 * @tparam OutputValueType data type of output random number
 * @tparam IndexType data type used to represent length of the arrays
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] out output array
 * @param[in] mu mean value
 * @param[in] beta scale value
 * @note https://en.wikipedia.org/wiki/Gumbel_distribution
 */
template <typename OutputValueType, typename IndexType = int>
void gumbel(const raft::handle_t& handle,
            RngState& rng_state,
            raft::device_vector_view<OutputValueType, IndexType> out,
            OutputValueType mu,
            OutputValueType beta)
{
  detail::gumbel(rng_state, out.data_handle(), out.extent(0), mu, beta, handle.get_stream());
}

/**
 * @brief Legacy raw pointer overload of `gumbel`.
 *
 * @tparam OutType data type of output random number
 * @tparam LenType data type used to represent length of the arrays
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] ptr output array
 * @param[in] len number of elements in the output array
 * @param[in] mu mean value
 * @param[in] beta scale value
 * @note https://en.wikipedia.org/wiki/Gumbel_distribution
 */
template <typename OutType, typename LenType = int>
void gumbel(const raft::handle_t& handle,
            RngState& rng_state,
            OutType* ptr,
            LenType len,
            OutType mu,
            OutType beta)
{
  detail::gumbel(rng_state, ptr, len, mu, beta, handle.get_stream());
}

/**
 * @brief Generate lognormal distributed numbers
 *
 * @tparam OutputValueType data type of output random number
 * @tparam IndexType data type used to represent length of the arrays
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] out the output array
 * @param[in] mu mean of the distribution
 * @param[in] sigma standard deviation of the distribution
 */
template <typename OutputValueType, typename IndexType>
void lognormal(const raft::handle_t& handle,
               RngState& rng_state,
               raft::device_vector_view<OutputValueType, IndexType> out,
               OutputValueType mu,
               OutputValueType sigma)
{
  detail::lognormal(rng_state, out.data_handle(), out.extent(0), mu, sigma, handle.get_stream());
}

/**
 * @brief Legacy raw pointer overload of `lognormal`.
 *
 * @tparam OutType data type of output random number
 * @tparam LenType data type used to represent length of the arrays
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] ptr the output array
 * @param[in] len the number of elements in the output
 * @param[in] mu mean of the distribution
 * @param[in] sigma standard deviation of the distribution
 */
template <typename OutType, typename LenType = int>
void lognormal(const raft::handle_t& handle,
               RngState& rng_state,
               OutType* ptr,
               LenType len,
               OutType mu,
               OutType sigma)
{
  detail::lognormal(rng_state, ptr, len, mu, sigma, handle.get_stream());
}

/**
 * @brief Generate logistic distributed random numbers
 *
 * @tparam OutputValueType data type of output random number
 * @tparam IndexType data type used to represent length of the arrays
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] out output array
 * @param[in] mu mean value
 * @param[in] scale scale value
 */
template <typename OutputValueType, typename IndexType = int>
void logistic(const raft::handle_t& handle,
              RngState& rng_state,
              raft::device_vector_view<OutputValueType, IndexType> out,
              OutputValueType mu,
              OutputValueType scale)
{
  detail::logistic(rng_state, out.data_handle(), out.extent(0), mu, scale, handle.get_stream());
}

/**
 * @brief Legacy raw pointer overload of `logistic`.
 *
 * @tparam OutType data type of output random number
 * @tparam LenType data type used to represent length of the arrays
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] ptr output array
 * @param[in] len number of elements in the output array
 * @param[in] mu mean value
 * @param[in] scale scale value
 */
template <typename OutType, typename LenType = int>
void logistic(const raft::handle_t& handle,
              RngState& rng_state,
              OutType* ptr,
              LenType len,
              OutType mu,
              OutType scale)
{
  detail::logistic(rng_state, ptr, len, mu, scale, handle.get_stream());
}

/**
 * @brief Generate exponentially distributed random numbers
 *
 * @tparam OutputValueType data type of output random number
 * @tparam IndexType data type used to represent length of the arrays
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] out output array
 * @param[in] lambda the exponential distribution's lambda parameter
 */
template <typename OutputValueType, typename IndexType>
void exponential(const raft::handle_t& handle,
                 RngState& rng_state,
                 raft::device_vector_view<OutputValueType, IndexType> out,
                 OutputValueType lambda)
{
  detail::exponential(rng_state, out.data_handle(), out.extent(0), lambda, handle.get_stream());
}

/**
 * @brief Legacy raw pointer overload of `exponential`.
 *
 * @tparam OutType data type of output random number
 * @tparam LenType data type used to represent length of the arrays
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] ptr output array
 * @param[in] len number of elements in the output array
 * @param[in] lambda the exponential distribution's lambda parameter
 */
template <typename OutType, typename LenType = int>
void exponential(
  const raft::handle_t& handle, RngState& rng_state, OutType* ptr, LenType len, OutType lambda)
{
  detail::exponential(rng_state, ptr, len, lambda, handle.get_stream());
}

/**
 * @brief Generate rayleigh distributed random numbers
 *
 * @tparam OutputValueType data type of output random number
 * @tparam IndexType data type used to represent length of the arrays
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] out output array
 * @param[in] sigma the distribution's sigma parameter
 */
template <typename OutputValueType, typename IndexType>
void rayleigh(const raft::handle_t& handle,
              RngState& rng_state,
              raft::device_vector_view<OutputValueType, IndexType> out,
              OutputValueType sigma)
{
  detail::rayleigh(rng_state, out.data_handle(), out.extent(0), sigma, handle.get_stream());
}

/**
 * @brief Legacy raw pointer overload of `rayleigh`.
 *
 * @tparam OutType data type of output random number
 * @tparam LenType data type used to represent length of the arrays
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] ptr output array
 * @param[in] len number of elements in the output array
 * @param[in] sigma the distribution's sigma parameter
 */
template <typename OutType, typename LenType = int>
void rayleigh(
  const raft::handle_t& handle, RngState& rng_state, OutType* ptr, LenType len, OutType sigma)
{
  detail::rayleigh(rng_state, ptr, len, sigma, handle.get_stream());
}

/**
 * @brief Generate laplace distributed random numbers
 *
 * @tparam OutputValueType data type of output random number
 * @tparam IndexType data type used to represent length of the arrays
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] out output array
 * @param[in] mu the mean
 * @param[in] scale the scale
 */
template <typename OutputValueType, typename IndexType>
void laplace(const raft::handle_t& handle,
             RngState& rng_state,
             raft::device_vector_view<OutputValueType, IndexType> out,
             OutputValueType mu,
             OutputValueType scale)
{
  detail::laplace(rng_state, out.data_handle(), out.extent(0), mu, scale, handle.get_stream());
}

/**
 * @brief Legacy raw pointer overload of `laplace`.
 *
 * @tparam OutType data type of output random number
 * @tparam LenType data type used to represent length of the arrays
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] ptr output array
 * @param[in] len number of elements in the output array
 * @param[in] mu the mean
 * @param[in] scale the scale
 */
template <typename OutType, typename LenType = int>
void laplace(const raft::handle_t& handle,
             RngState& rng_state,
             OutType* ptr,
             LenType len,
             OutType mu,
             OutType scale)
{
  detail::laplace(rng_state, ptr, len, mu, scale, handle.get_stream());
}

namespace sample_without_replacement_impl {
template <typename T>
struct weight_alias {
};

template <>
struct weight_alias<std::nullopt_t> {
  using type = double;
};

template <typename ElementType, typename IndexType>
struct weight_alias<std::optional<raft::device_vector_view<ElementType, IndexType>>> {
  using type = typename raft::device_vector_view<ElementType, IndexType>::value_type;
};

template <typename T>
using weight_t = typename weight_alias<T>::type;
}  // namespace sample_without_replacement_impl

/**
 * @brief Sample the input vector without replacement, optionally based on the
 * input weight vector for each element in the array.
 *
 * The implementation is based on the `one-pass sampling` algorithm described in
 * ["Accelerating weighted random sampling without
 * replacement,"](https://www.ethz.ch/content/dam/ethz/special-interest/baug/ivt/ivt-dam/vpl/reports/1101-1200/ab1141.pdf)
 * a technical report by Kirill Mueller.
 *
 * If no input weight vector is provided, then input elements will be
 * sampled uniformly.  Otherwise, the elements sampled from the input
 * vector will always appear in increasing order of their weights as
 * computed using the exponential distribution. So, if you are
 * particular about the order (for e.g., array permutations), then
 * this might not be the right choice.
 *
 * @tparam DataT type of each element of the input array @c in
 * @tparam IdxT type of the dimensions of the arrays; output index type
 * @tparam WeightsVectorType std::optional<raft::device_vector_view<const weight_type, IdxT>> of
 * each elements of the weights array @c weights_opt
 * @tparam OutIndexVectorType std::optional<raft::device_vector_view<IdxT, IdxT>> of output indices
 * @c outIdx_opt
 *
 * @note Please do not specify template parameters explicitly,
 *   as the compiler can deduce them from the arguments.
 *
 * @param[in] handle RAFT handle containing (among other resources)
 *   the CUDA stream on which to run.
 * @param[inout] rng_state Pseudorandom number generator state.
 * @param[in] in Input vector to be sampled.
 * @param[in] weights_opt std::optional weights vector.
 *        If not provided, uniform sampling will be used.
 * @param[out] out Vector of samples from the input vector.
 * @param[out] outIdx_opt std::optional vector of the indices
 *   sampled from the input array.
 *
 * @pre The number of samples `out.extent(0)`
 *   is less than or equal to the number of inputs `in.extent(0)`.
 *
 * @pre The number of weights `wts.extent(0)`
 *   equals the number of inputs `in.extent(0)`.
 */
template <typename DataT, typename IdxT, typename WeightsVectorType, class OutIndexVectorType>
void sample_without_replacement(const raft::handle_t& handle,
                                RngState& rng_state,
                                raft::device_vector_view<const DataT, IdxT> in,
                                WeightsVectorType&& weights_opt,
                                raft::device_vector_view<DataT, IdxT> out,
                                OutIndexVectorType&& outIdx_opt)
{
  using weight_type = sample_without_replacement_impl::weight_t<
    std::remove_const_t<std::remove_reference_t<WeightsVectorType>>>;

  std::optional<raft::device_vector_view<const weight_type, IdxT>> wts =
    std::forward<WeightsVectorType>(weights_opt);
  std::optional<raft::device_vector_view<IdxT, IdxT>> outIdx =
    std::forward<OutIndexVectorType>(outIdx_opt);

  static_assert(std::is_integral<IdxT>::value, "IdxT must be an integral type.");
  const IdxT sampledLen = out.extent(0);
  const IdxT len        = in.extent(0);
  RAFT_EXPECTS(sampledLen <= len,
               "sampleWithoutReplacement: "
               "sampledLen (out.extent(0)) must be <= len (in.extent(0))");
  RAFT_EXPECTS(len == 0 || in.data_handle() != nullptr,
               "sampleWithoutReplacement: "
               "If in.extents(0) != 0, then in.data_handle() must be nonnull");
  RAFT_EXPECTS(sampledLen == 0 || out.data_handle() != nullptr,
               "sampleWithoutReplacement: "
               "If out.extents(0) != 0, then out.data_handle() must be nonnull");

  const bool outIdx_has_value = outIdx.has_value();
  if (outIdx_has_value) {
    RAFT_EXPECTS((*outIdx).extent(0) == sampledLen,
                 "sampleWithoutReplacement: "
                 "If outIdx is provided, its extent(0) must equal out.extent(0)");
  }
  IdxT* outIdx_ptr = outIdx_has_value ? (*outIdx).data_handle() : nullptr;

  const bool wts_has_value = wts.has_value();
  if (wts_has_value) {
    RAFT_EXPECTS((*wts).extent(0) == len,
                 "sampleWithoutReplacement: "
                 "If wts is provided, its extent(0) must equal in.extent(0)");
  }
  const weight_type* wts_ptr = wts_has_value ? (*wts).data_handle() : nullptr;

  detail::sampleWithoutReplacement(rng_state,
                                   out.data_handle(),
                                   outIdx_ptr,
                                   in.data_handle(),
                                   wts_ptr,
                                   sampledLen,
                                   len,
                                   handle.get_stream());
}

/**
 * @brief Overload of `sample_without_replacement` to help the
 *   compiler find the above overload, in case users pass in
 *   `std::nullopt` for one or both of the optional arguments.
 *
 * Please see above for documentation of `sample_without_replacement`.
 */
template <typename... Args, typename = std::enable_if_t<sizeof...(Args) == 5>>
void sample_without_replacement(Args... args)
{
  sample_without_replacement(std::forward<Args>(args)..., std::nullopt);
}

/**
 * @brief Legacy version of @c sample_without_replacement (see above)
 *   that takes raw arrays instead of device mdspan.
 *
 * @tparam DataT data type
 * @tparam WeightsT weights type
 * @tparam IdxT index type
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] out output sampled array (of length 'sampledLen')
 * @param[out] outIdx indices of the sampled array (of length 'sampledLen'). Pass
 * a nullptr if this is not required.
 * @param[in] in input array to be sampled (of length 'len')
 * @param[in] wts weights array (of length 'len'). Pass a nullptr if uniform
 * sampling is desired
 * @param[in] sampledLen output sampled array length
 * @param[in] len input array length
 */
template <typename DataT, typename WeightsT, typename IdxT = int>
void sampleWithoutReplacement(const raft::handle_t& handle,
                              RngState& rng_state,
                              DataT* out,
                              IdxT* outIdx,
                              const DataT* in,
                              const WeightsT* wts,
                              IdxT sampledLen,
                              IdxT len)
{
  detail::sampleWithoutReplacement(
    rng_state, out, outIdx, in, wts, sampledLen, len, handle.get_stream());
}

/**
 * @brief Generates the 'a' and 'b' parameters for a modulo affine
 *        transformation equation: `(ax + b) % n`
 *
 * @tparam IdxT integer type
 *
 * @param[in] rng_state random number generator state
 * @param[in]  n the modulo range
 * @param[out] a slope parameter
 * @param[out] b intercept parameter
 */
template <typename IdxT>
void affine_transform_params(RngState const& rng_state, IdxT n, IdxT& a, IdxT& b)
{
  detail::affine_transform_params(rng_state, n, a, b);
}

///////////////////////////////////////////////////////////////////////////
// Everything below this point is deprecated and will be removed         //
///////////////////////////////////////////////////////////////////////////

// without the macro, clang-format seems to go insane
#define DEPR [[deprecated("Use 'RngState' with the new flat functions instead")]]

using detail::bernoulli;
using detail::exponential;
using detail::fill;
using detail::gumbel;
using detail::laplace;
using detail::logistic;
using detail::lognormal;
using detail::normal;
using detail::normalInt;
using detail::normalTable;
using detail::rayleigh;
using detail::scaled_bernoulli;
using detail::uniform;
using detail::uniformInt;

using detail::sampleWithoutReplacement;

class DEPR Rng : public detail::RngImpl {
 public:
  /**
   * @brief ctor
   * @param _s 64b seed used to initialize the RNG
   * @param _t backend device RNG generator type
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

#undef DEPR

};  // end namespace raft::random
