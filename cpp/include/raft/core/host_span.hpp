/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <raft/core/span.hpp>

namespace raft {

/**
 * @defgroup host_span one-dimensional device span type
 * @{
 */

/**
 * @brief A span class for host pointer.
 */
template <typename T, size_t extent = std::experimental::dynamic_extent>
using host_span = span<T, false, extent>;

/**
 * @}
 */

}  // end namespace raft