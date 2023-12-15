/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "detail/hnsw_serialize.cuh"
#include "hnsw_types.hpp"
#include <raft/distance/distance_types.hpp>

#include <raft/core/resources.hpp>

namespace raft::neighbors::hnsw {

/**
 * @defgroup hnsw_serialize HNSW Serialize
 * @{
 */

/**
 * Write the CAGRA built index as a base layer HNSW index to an output stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 *
 * raft::resources handle;
 *
 * // create an output stream
 * std::ostream os(std::cout.rdbuf());
 * // create an index with `auto index = cagra::build(...);`
 * raft::serialize(handle, os, index);
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index CAGRA index
 *
 */
template <typename T, typename IdxT>
void serialize(raft::resources const& handle,
               std::ostream& os,
               const raft::neighbors::cagra::index<T, IdxT>& index)
{
  detail::serialize<T, IdxT>(handle, os, index);
}

/**
 * Save a CAGRA build index in hnswlib base-layer-only serialized format
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
 * // create an index with `auto index = cagra::build(...);`
 * raft::serialize(handle, filename, index);
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index CAGRA index
 *
 */
template <typename T, typename IdxT>
void serialize(raft::resources const& handle,
               const std::string& filename,
               const raft::neighbors::cagra::index<T, IdxT>& index)
{
  detail::serialize<T, IdxT>(handle, filename, index);
}

/**
 * Load an hnswlib index which was serialized from a CAGRA index
 *
 * NOTE: This function allocates the index on the heap, and it is
 * the user's responsibility to de-allocate the index
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
 * raft::neighbors::hnsw* index;
 * raft::deserialize(handle, filename, index);
 * // use the index, then delete when done
 * delete index;
 * @endcode
 *
 * @tparam T data element type
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[out] index CAGRA index
 * @param[in] dim dimensionality of the index
 * @param[in] metric metric used to build the index
 *
 */
template <typename T>
void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 index<T>*& index,
                 int dim,
                 raft::distance::DistanceType metric)
{
  detail::deserialize<T>(handle, filename, index, dim, metric);
}

/**@}*/

}  // namespace raft::neighbors::hnsw
