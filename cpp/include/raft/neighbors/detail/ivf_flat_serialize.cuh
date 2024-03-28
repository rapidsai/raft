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

#include <raft/core/detail/mdspan_numpy_serializer.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/serialize.hpp>
#include <raft/neighbors/detail/ivf_flat_build.cuh>
#include <raft/neighbors/ivf_flat_types.hpp>
#include <raft/neighbors/ivf_list.hpp>
#include <raft/neighbors/ivf_list_types.hpp>
#include <raft/util/pow2_utils.cuh>

#include <fstream>

namespace raft::neighbors::ivf_flat::detail {

// Serialization version
// No backward compatibility yet; that is, can't add additional fields without breaking
// backward compatibility.
// TODO(hcho3) Implement next-gen serializer for IVF that allows for expansion in a backward
//             compatible fashion.
constexpr int serialization_version = 4;

/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index_ IVF-Flat index
 *
 */
template <typename T, typename IdxT>
void serialize(raft::resources const& handle, std::ostream& os, const index<T, IdxT>& index_)
{
  RAFT_LOG_DEBUG(
    "Saving IVF-Flat index, size %zu, dim %u", static_cast<size_t>(index_.size()), index_.dim());

  std::string dtype_string = raft::detail::numpy_serializer::get_numpy_dtype<T>().to_string();
  dtype_string.resize(4);
  os << dtype_string;

  serialize_scalar(handle, os, serialization_version);
  serialize_scalar(handle, os, index_.size());
  serialize_scalar(handle, os, index_.dim());
  serialize_scalar(handle, os, index_.n_lists());
  serialize_scalar(handle, os, index_.metric());
  serialize_scalar(handle, os, index_.adaptive_centers());
  serialize_scalar(handle, os, index_.conservative_memory_allocation());
  serialize_mdspan(handle, os, index_.centers());
  if (index_.center_norms()) {
    bool has_norms = true;
    serialize_scalar(handle, os, has_norms);
    serialize_mdspan(handle, os, *index_.center_norms());
  } else {
    bool has_norms = false;
    serialize_scalar(handle, os, has_norms);
  }
  auto sizes_host = make_host_vector<uint32_t, uint32_t>(index_.list_sizes().extent(0));
  copy(sizes_host.data_handle(),
       index_.list_sizes().data_handle(),
       sizes_host.size(),
       resource::get_cuda_stream(handle));
  resource::sync_stream(handle);
  serialize_mdspan(handle, os, sizes_host.view());

  list_spec<uint32_t, T, IdxT> list_store_spec{index_.dim(), true};
  for (uint32_t label = 0; label < index_.n_lists(); label++) {
    ivf::serialize_list(handle,
                        os,
                        index_.lists()[label],
                        list_store_spec,
                        Pow2<kIndexGroupSize>::roundUp(sizes_host(label)));
  }
  resource::sync_stream(handle);
}

template <typename T, typename IdxT>
void serialize(raft::resources const& handle,
               const std::string& filename,
               const index<T, IdxT>& index_)
{
  std::ofstream of(filename, std::ios::out | std::ios::binary);
  if (!of) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  detail::serialize(handle, of, index_);

  of.close();
  if (!of) { RAFT_FAIL("Error writing output %s", filename.c_str()); }
}

/** Load an index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[in] index_ IVF-Flat index
 *
 */
template <typename T, typename IdxT>
auto deserialize(raft::resources const& handle, std::istream& is) -> index<T, IdxT>
{
  char dtype_string[4];
  is.read(dtype_string, 4);

  auto ver = deserialize_scalar<int>(handle, is);
  if (ver != serialization_version) {
    RAFT_FAIL("serialization version mismatch, expected %d, got %d ", serialization_version, ver);
  }
  auto n_rows           = deserialize_scalar<IdxT>(handle, is);
  auto dim              = deserialize_scalar<std::uint32_t>(handle, is);
  auto n_lists          = deserialize_scalar<std::uint32_t>(handle, is);
  auto metric           = deserialize_scalar<raft::distance::DistanceType>(handle, is);
  bool adaptive_centers = deserialize_scalar<bool>(handle, is);
  bool cma              = deserialize_scalar<bool>(handle, is);

  index<T, IdxT> index_ = index<T, IdxT>(handle, metric, n_lists, adaptive_centers, cma, dim);

  deserialize_mdspan(handle, is, index_.centers());
  bool has_norms = deserialize_scalar<bool>(handle, is);
  if (has_norms) {
    index_.allocate_center_norms(handle);
    if (!index_.center_norms()) {
      RAFT_FAIL("Error inconsistent center norms");
    } else {
      auto center_norms = index_.center_norms().value();
      deserialize_mdspan(handle, is, center_norms);
    }
  }
  deserialize_mdspan(handle, is, index_.list_sizes());

  list_spec<uint32_t, T, IdxT> list_device_spec{index_.dim(), cma};
  list_spec<uint32_t, T, IdxT> list_store_spec{index_.dim(), true};
  for (uint32_t label = 0; label < index_.n_lists(); label++) {
    ivf::deserialize_list(handle, is, index_.lists()[label], list_store_spec, list_device_spec);
  }
  resource::sync_stream(handle);

  ivf::detail::recompute_internal_state(handle, index_);

  return index_;
}

template <typename T, typename IdxT>
auto deserialize(raft::resources const& handle, const std::string& filename) -> index<T, IdxT>
{
  std::ifstream is(filename, std::ios::in | std::ios::binary);

  if (!is) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  auto index = detail::deserialize<T, IdxT>(handle, is);

  is.close();

  return index;
}
}  // namespace raft::neighbors::ivf_flat::detail
