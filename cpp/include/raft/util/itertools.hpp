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

#include <raft/util/detail/itertools.hpp>

/**
 * Helpers inspired by the Python itertools library
 *
 */

namespace raft::util::itertools {

/**
 * @brief Cartesian product of the given initializer lists.
 *
 * This helper can be used to easily define input parameters in tests/benchmarks.
 * Note that it's not optimized for use with large lists / many lists in performance-critical code!
 *
 * @tparam S    Type of the output structures.
 * @tparam Args Types of the elements of the initilizer lists, matching the types of the first
 *              fields of the structure (if the structure has more fields, some might be initialized
 *              with their default value).
 * @param lists One or more initializer lists.
 * @return std::vector<S> A vector of structures containing the cartesian product.
 */
template <typename S, typename... Args>
std::vector<S> product(std::initializer_list<Args>... lists)
{
  return detail::product<S>(std::index_sequence_for<Args...>(), (std::vector<Args>(lists))...);
}

}  // namespace raft::util::itertools
