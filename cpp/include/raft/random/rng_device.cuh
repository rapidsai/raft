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

#ifndef __RNG_DEVICE_H
#define __RNG_DEVICE_H

#pragma once

#include "detail/rng_device.cuh"
#include "rng_state.hpp"

namespace raft {
namespace random {

using detail::DeviceState;

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
// this is necessary again since all arguments are primitive types
using detail::box_muller_transform;

};  // end namespace random
};  // end namespace raft

#endif
