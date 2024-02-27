/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "detail/hnsw_serialize.hpp"
#include "hnsw_types.hpp"

#include <raft/core/resources.hpp>
#include <raft/distance/distance_types.hpp>

namespace raft::neighbors::hnsw {

/**
 * @defgroup hnsw_serialize HNSW Serialize
 * @{
 */

/**
 * Load an hnswlib index which was serialized from a CAGRA index
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an an unallocated pointer
 * int dim = 10;
 * raft::distance::DistanceType = raft::distance::L2Expanded
 * auto index = raft::deserialize(handle, filename, dim, metric);
 * @endcode
 *
 * @tparam T data element type
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] dim dimensionality of the index
 * @param[in] metric metric used to build the index
 *
 * @return std::unique_ptr<index<T>>
 *
 */
template <typename T>
std::unique_ptr<index<T>> deserialize(raft::resources const& handle,
                                      const std::string& filename,
                                      int dim,
                                      raft::distance::DistanceType metric)
{
  return detail::deserialize<T>(handle, filename, dim, metric);
}

/**@}*/

}  // namespace raft::neighbors::hnsw
