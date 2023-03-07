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

#include <raft/core/mdarray.hpp>
#include <raft/core/serialize.hpp>
#include <raft/neighbors/ivf_flat_types.hpp>
#include <raft/neighbors/ivf_list.hpp>
#include <raft/neighbors/ivf_list_types.hpp>

#include <fstream>

namespace raft::neighbors::ivf_flat::detail {

// Serialization version 3
// No backward compatibility yet; that is, can't add additional fields without breaking
// backward compatibility.
// TODO(hcho3) Implement next-gen serializer for IVF that allows for expansion in a backward
//             compatible fashion.
constexpr int serialization_version = 3;

// NB: we wrap this check in a struct, so that the updated RealSize is easy to see in the error
// message.
template <size_t RealSize, size_t ExpectedSize>
struct check_index_layout {
  static_assert(RealSize == ExpectedSize,
                "The size of the index struct has changed since the last update; "
                "paste in the new size and consider updating the serialization logic");
};

template struct check_index_layout<sizeof(index<double, std::uint64_t>), 368>;

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
void serialize(raft::device_resources const& handle,
               const std::string& filename,
               const index<T, IdxT>& index_)
{
  std::ofstream of(filename, std::ios::out | std::ios::binary);
  if (!of) { RAFT_FAIL("Cannot open %s", filename.c_str()); }

  RAFT_LOG_DEBUG(
    "Saving IVF-Flat index, size %zu, dim %u", static_cast<size_t>(index_.size()), index_.dim());

  serialize_scalar(handle, of, serialization_version);
  serialize_scalar(handle, of, index_.size());
  serialize_scalar(handle, of, index_.dim());
  serialize_scalar(handle, of, index_.n_lists());
  serialize_scalar(handle, of, index_.metric());
  serialize_scalar(handle, of, index_.adaptive_centers());
  serialize_scalar(handle, of, index_.conservative_memory_allocation());
  serialize_mdspan(handle, of, index_.centers());
  if (index_.center_norms()) {
    bool has_norms = true;
    serialize_scalar(handle, of, has_norms);
    serialize_mdspan(handle, of, *index_.center_norms());
  } else {
    bool has_norms = false;
    serialize_scalar(handle, of, has_norms);
  }
  auto sizes_host = make_host_vector<uint32_t, uint32_t>(index_.list_sizes().extent(0));
  copy(sizes_host.data_handle(),
       index_.list_sizes().data_handle(),
       sizes_host.size(),
       handle.get_stream());
  handle.sync_stream();
  serialize_mdspan(handle, of, sizes_host.view());

  list_spec<uint32_t, T, IdxT> list_store_spec{index_.dim(), true};
  for (uint32_t label = 0; label < index_.n_lists(); label++) {
    ivf::serialize_list(handle, of, index_.lists()[label], list_store_spec, sizes_host(label));
  }
  handle.sync_stream();
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
auto deserialize(raft::device_resources const& handle, const std::string& filename)
  -> index<T, IdxT>
{
  std::ifstream infile(filename, std::ios::in | std::ios::binary);

  if (!infile) { RAFT_FAIL("Cannot open %s", filename.c_str()); }

  auto ver = deserialize_scalar<int>(handle, infile);
  if (ver != serialization_version) {
    RAFT_FAIL("serialization version mismatch, expected %d, got %d ", serialization_version, ver);
  }
  auto n_rows           = deserialize_scalar<IdxT>(handle, infile);
  auto dim              = deserialize_scalar<std::uint32_t>(handle, infile);
  auto n_lists          = deserialize_scalar<std::uint32_t>(handle, infile);
  auto metric           = deserialize_scalar<raft::distance::DistanceType>(handle, infile);
  bool adaptive_centers = deserialize_scalar<bool>(handle, infile);
  bool cma              = deserialize_scalar<bool>(handle, infile);

  index<T, IdxT> index_ = index<T, IdxT>(handle, metric, n_lists, adaptive_centers, cma, dim);

  deserialize_mdspan(handle, infile, index_.centers());
  bool has_norms = deserialize_scalar<bool>(handle, infile);
  if (has_norms) {
    if (!index_.center_norms()) {
      RAFT_FAIL("Error inconsistent center norms");
    } else {
      auto center_norms = index_.center_norms().value();
      deserialize_mdspan(handle, infile, center_norms);
    }
  }
  deserialize_mdspan(handle, infile, index_.list_sizes());

  list_spec<uint32_t, T, IdxT> list_device_spec{index_.dim(), cma};
  list_spec<uint32_t, T, IdxT> list_store_spec{index_.dim(), true};
  for (uint32_t label = 0; label < index_.n_lists(); label++) {
    ivf::deserialize_list(handle, infile, index_.lists()[label], list_store_spec, list_device_spec);
  }
  handle.sync_stream();
  infile.close();

  index_.recompute_internal_state(handle);

  return index_;
}
}  // namespace raft::neighbors::ivf_flat::detail
