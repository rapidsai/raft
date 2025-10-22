/*
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <curand.h>

namespace raft::random {
namespace detail {

// @todo: We probably want to scrape through and replace any consumers of
// these wrappers with our RNG
/** check for curand runtime API errors and assert accordingly */
#define CURAND_CHECK(call)                                                                         \
  do {                                                                                             \
    curandStatus_t status = call;                                                                  \
    ASSERT(status == CURAND_STATUS_SUCCESS, "FAIL: curand-call='%s'. Reason:%d\n", #call, status); \
  } while (0)

/**
 * @defgroup normal curand normal random number generation operations
 * @{
 */
template <typename T>
curandStatus_t curandGenerateNormal(
  curandGenerator_t generator, T* outputPtr, size_t n, T mean, T stddev);

template <>
inline curandStatus_t curandGenerateNormal(
  curandGenerator_t generator, float* outputPtr, size_t n, float mean, float stddev)
{
  return curandGenerateNormal(generator, outputPtr, n, mean, stddev);
}

template <>
inline curandStatus_t curandGenerateNormal(
  curandGenerator_t generator, double* outputPtr, size_t n, double mean, double stddev)
{
  return curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev);
}
/** @} */

};  // end namespace detail
};  // end namespace raft::random
