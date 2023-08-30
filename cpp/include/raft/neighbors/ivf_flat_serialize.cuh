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

#include "detail/ivf_flat_serialize.cuh"

namespace raft::neighbors::ivf_flat {

/**
 * \defgroup ivf_flat_serialize IVF-Flat Serialize
 * @{
 */

/**
 * Write the index to an output stream
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
 * // create an index with `auto index = ivf_flat::build(...);`
 * raft::serialize(handle, os, index);
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index IVF-Flat index
 *
 */
template <typename T, typename IdxT>
void serialize(raft::resources const& handle, std::ostream& os, const index<T, IdxT>& index)
{
  detail::serialize(handle, os, index);
}

/**
 * Save the index to file.
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
 * // create an index with `auto index = ivf_flat::build(...);`
 * raft::serialize(handle, filename, index);
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index IVF-Flat index
 *
 */
template <typename T, typename IdxT>
void serialize(raft::resources const& handle,
               const std::string& filename,
               const index<T, IdxT>& index)
{
  detail::serialize(handle, filename, index);
}

/**
 * Load index from input stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 *
 * raft::resources handle;
 *
 * // create an input stream
 * std::istream is(std::cin.rdbuf());
 * using T    = float; // data element type
 * using IdxT = int; // type of the index
 * auto index = raft::deserialize<T, IdxT>(handle, is);
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices
 *
 * @param[in] handle the raft handle
 * @param[in] is input stream
 *
 * @return raft::neighbors::ivf_flat::index<T, IdxT>
 */
template <typename T, typename IdxT>
index<T, IdxT> deserialize(raft::resources const& handle, std::istream& is)
{
  return detail::deserialize<T, IdxT>(handle, is);
}

/**
 * Load index from file.
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
 * using T    = float; // data element type
 * using IdxT = int; // type of the index
 * auto index = raft::deserialize<T, IdxT>(handle, filename);
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 *
 * @return raft::neighbors::ivf_flat::index<T, IdxT>
 */
template <typename T, typename IdxT>
index<T, IdxT> deserialize(raft::resources const& handle, const std::string& filename)
{
  return detail::deserialize<T, IdxT>(handle, filename);
}

/**@}*/

}  // namespace raft::neighbors::ivf_flat
