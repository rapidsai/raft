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

#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>
#include <raft/neighbors/brute_force_types.hpp>

namespace raft::neighbors::brute_force {

auto static constexpr serialization_version = 0;

/**
 * \defgroup brute_force_serialize Brute Force Serialize
 * @{
 */

/**
 * Write the index to an output stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/neighbors/brute_force_serialize.cuh>
 *
 * raft::resources handle;
 *
 * // create an output stream
 * std::ostream os(std::cout.rdbuf());
 * // create an index with `auto index = brute_force::build(...);`
 * raft::neighbors::brute_force::serialize(handle, os, index);
 * @endcode
 *
 * @tparam T data element type
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index brute force index
 * @param[in] include_dataset whether to include the dataset in the serialized
 * output
 *
 */
template <typename T>
void serialize(raft::resources const& handle,
               std::ostream& os,
               const index<T>& index,
               bool include_dataset = true)
{
  RAFT_LOG_DEBUG(
    "Saving brute force index, size %zu, dim %u", static_cast<size_t>(index.size()), index.dim());

  auto dtype_string = raft::detail::numpy_serializer::get_numpy_dtype<T>().to_string();
  dtype_string.resize(4);
  os << dtype_string;

  serialize_scalar(handle, os, serialization_version);
  serialize_scalar(handle, os, index.size());
  serialize_scalar(handle, os, index.dim());
  serialize_scalar(handle, os, index.metric());
  serialize_scalar(handle, os, index.metric_arg());
  serialize_scalar(handle, os, include_dataset);
  if (include_dataset) { serialize_mdspan(handle, os, index.dataset()); }
  auto has_norms = index.has_norms();
  serialize_scalar(handle, os, has_norms);
  if (has_norms) { serialize_mdspan(handle, os, index.norms()); }
  resource::sync_stream(handle);
}

/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/neighbors/brute_force_serialize.cuh>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = brute_force::build(...);`
 * raft::neighbors::brute_force::serialize(handle, filename, index);
 * @endcode
 *
 * @tparam T data element type
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index brute force index
 * @param[in] include_dataset whether to include the dataset in the serialized
 * output
 *
 */
template <typename T>
void serialize(raft::resources const& handle,
               const std::string& filename,
               const index<T>& index,
               bool include_dataset = true)
{
  auto os = std::ofstream{filename, std::ios::out | std::ios::binary};
  RAFT_EXPECTS(os, "Cannot open file %s", filename.c_str());
  serialize(handle, os, index, include_dataset);
}

/**
 * Load index from input stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/neighbors/brute_force_serialize.cuh>
 *
 * raft::resources handle;
 *
 * // create an input stream
 * std::istream is(std::cin.rdbuf());
 * using T    = float; // data element type
 * auto index = raft::neighbors::brute_force::deserialize<T>(handle, is);
 * @endcode
 *
 * @tparam T data element type
 *
 * @param[in] handle the raft handle
 * @param[in] is input stream
 *
 * @return raft::neighbors::brute_force::index<T>
 */
template <typename T>
auto deserialize(raft::resources const& handle, std::istream& is)
{
  auto dtype_string = std::array<char, 4>{};
  is.read(dtype_string.data(), 4);

  auto ver = deserialize_scalar<int>(handle, is);
  if (ver != serialization_version) {
    RAFT_FAIL("serialization version mismatch, expected %d, got %d ", serialization_version, ver);
  }
  auto rows       = deserialize_scalar<std::int64_t>(handle, is);
  auto dim        = deserialize_scalar<std::int64_t>(handle, is);
  auto metric     = deserialize_scalar<raft::distance::DistanceType>(handle, is);
  auto metric_arg = deserialize_scalar<T>(handle, is);

  auto dataset_storage = raft::make_host_matrix<T>(std::int64_t{}, std::int64_t{});
  auto include_dataset = deserialize_scalar<bool>(handle, is);
  if (include_dataset) {
    dataset_storage = raft::make_host_matrix<T>(rows, dim);
    deserialize_mdspan(handle, is, dataset_storage.view());
  }

  auto has_norms     = deserialize_scalar<bool>(handle, is);
  auto norms_storage = has_norms ? std::optional{raft::make_host_vector<T, std::int64_t>(rows)}
                                 : std::optional<raft::host_vector<T, std::int64_t>>{};
  // TODO(wphicks): Use mdbuffer here when available
  auto norms_storage_dev =
    has_norms ? std::optional{raft::make_device_vector<T, std::int64_t>(handle, rows)}
              : std::optional<raft::device_vector<T, std::int64_t>>{};
  if (has_norms) {
    deserialize_mdspan(handle, is, norms_storage->view());
    raft::copy(handle, norms_storage_dev->view(), norms_storage->view());
  }

  auto result = index(handle,
                      raft::make_const_mdspan(dataset_storage.view()),
                      std::move(norms_storage_dev),
                      metric,
                      metric_arg);
  resource::sync_stream(handle);

  return result;
}

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/neighbors/brute_force_serialize.cuh>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * using T    = float; // data element type
 * auto index = raft::neighbors::brute_force::deserialize<T>(handle, filename);
 * @endcode
 *
 * @tparam T data element type
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 *
 * @return raft::neighbors::brute_force::index<T>
 */
template <typename T>
auto deserialize(raft::resources const& handle, const std::string& filename)
{
  auto is = std::ifstream{filename, std::ios::in | std::ios::binary};
  RAFT_EXPECTS(is, "Cannot open file %s", filename.c_str());

  return deserialize<T>(handle, is);
}

/**@}*/

}  // namespace raft::neighbors::brute_force
