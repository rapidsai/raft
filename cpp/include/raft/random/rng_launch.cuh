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

#ifndef __RNG_LAUNCH_H
#define __RNG_LAUNCH_H

#pragma once

#include "detail/rng_launch.cuh"
#include "rng_state.hpp"

namespace raft {
namespace random {

using detail::call_rng_func;

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

};  // end namespace random
};  // end namespace raft

#endif
