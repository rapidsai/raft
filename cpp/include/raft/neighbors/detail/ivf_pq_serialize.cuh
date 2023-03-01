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

#include <raft/neighbors/detail/ivf_pq_build.cuh>
#include <raft/neighbors/ivf_list.hpp>
#include <raft/neighbors/ivf_pq_types.hpp>

#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/serialize.hpp>

#include <fstream>
#include <memory>

namespace raft::neighbors::ivf_pq::detail {

// Serialization version
// No backward compatibility yet; that is, can't add additional fields without breaking
// backward compatibility.
// TODO(hcho3) Implement next-gen serializer for IVF that allows for expansion in a backward
//             compatible fashion.
constexpr int kSerializationVersion = 3;

// NB: we wrap this check in a struct, so that the updated RealSize is easy to see in the error
// message.
template <size_t RealSize, size_t ExpectedSize>
struct check_index_layout {
  static_assert(RealSize == ExpectedSize,
                "The size of the index struct has changed since the last update; "
                "paste in the new size and consider updating the serialization logic");
};

// TODO: Recompute this and come back to it.
// template struct check_index_layout<sizeof(index<std::uint64_t>), 536>;

/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index IVF-PQ index
 *
 */
template <typename IdxT>
void serialize(raft::device_resources const& handle_,
               const std::string& filename,
               const index<IdxT>& index)
{
  std::ofstream of(filename, std::ios::out | std::ios::binary);
  if (!of) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  RAFT_LOG_DEBUG("Size %zu, dim %d, pq_dim %d, pq_bits %d",
                 static_cast<size_t>(index.size()),
                 static_cast<int>(index.dim()),
                 static_cast<int>(index.pq_dim()),
                 static_cast<int>(index.pq_bits()));

  serialize_scalar(handle_, of, kSerializationVersion);
  serialize_scalar(handle_, of, index.size());
  serialize_scalar(handle_, of, index.dim());
  serialize_scalar(handle_, of, index.pq_bits());
  serialize_scalar(handle_, of, index.pq_dim());
  serialize_scalar(handle_, of, index.conservative_memory_allocation());

  serialize_scalar(handle_, of, index.metric());
  serialize_scalar(handle_, of, index.codebook_kind());
  serialize_scalar(handle_, of, index.n_lists());

  serialize_mdspan(handle_, of, index.pq_centers());
  serialize_mdspan(handle_, of, index.centers());
  serialize_mdspan(handle_, of, index.centers_rot());
  serialize_mdspan(handle_, of, index.rotation_matrix());

  auto sizes_host = make_host_mdarray<uint32_t, uint32_t, row_major>(index.list_sizes().extents());
  copy(sizes_host.data_handle(),
       index.list_sizes().data_handle(),
       sizes_host.size(),
       handle_.get_stream());
  handle_.sync_stream();
  serialize_mdspan(handle_, of, sizes_host.view());
  auto list_store_spec = list_spec<uint32_t>{index.pq_bits(), index.pq_dim(), true};
  for (uint32_t label = 0; label < index.n_lists(); label++) {
    ivf::serialize_list<list_spec, IdxT, uint32_t>(
      handle_, of, index.lists()[label], list_store_spec, sizes_host(label));
  }

  of.close();
  if (!of) { RAFT_FAIL("Error writing output %s", filename.c_str()); }
  return;
}

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[in] index IVF-PQ index
 *
 */
template <typename IdxT>
auto deserialize(raft::device_resources const& handle_, const std::string& filename) -> index<IdxT>
{
  std::ifstream infile(filename, std::ios::in | std::ios::binary);

  if (!infile) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  auto ver = deserialize_scalar<int>(handle_, infile);
  if (ver != kSerializationVersion) {
    RAFT_FAIL("serialization version mismatch %d vs. %d", ver, kSerializationVersion);
  }
  auto n_rows  = deserialize_scalar<IdxT>(handle_, infile);
  auto dim     = deserialize_scalar<std::uint32_t>(handle_, infile);
  auto pq_bits = deserialize_scalar<std::uint32_t>(handle_, infile);
  auto pq_dim  = deserialize_scalar<std::uint32_t>(handle_, infile);
  auto cma     = deserialize_scalar<bool>(handle_, infile);

  auto metric        = deserialize_scalar<raft::distance::DistanceType>(handle_, infile);
  auto codebook_kind = deserialize_scalar<raft::neighbors::ivf_pq::codebook_gen>(handle_, infile);
  auto n_lists       = deserialize_scalar<std::uint32_t>(handle_, infile);

  RAFT_LOG_DEBUG("n_rows %zu, dim %d, pq_dim %d, pq_bits %d, n_lists %d",
                 static_cast<std::size_t>(n_rows),
                 static_cast<int>(dim),
                 static_cast<int>(pq_dim),
                 static_cast<int>(pq_bits),
                 static_cast<int>(n_lists));

  auto index = raft::neighbors::ivf_pq::index<IdxT>(
    handle_, metric, codebook_kind, n_lists, dim, pq_bits, pq_dim, cma);

  deserialize_mdspan(handle_, infile, index.pq_centers());
  deserialize_mdspan(handle_, infile, index.centers());
  deserialize_mdspan(handle_, infile, index.centers_rot());
  deserialize_mdspan(handle_, infile, index.rotation_matrix());
  deserialize_mdspan(handle_, infile, index.list_sizes());
  auto list_device_spec = list_spec<uint32_t>{pq_bits, pq_dim, cma};
  auto list_store_spec  = list_spec<uint32_t>{pq_bits, pq_dim, true};
  for (auto& list : index.lists()) {
    ivf::deserialize_list<list_spec, IdxT, uint32_t>(
      handle_, infile, list, list_store_spec, list_device_spec);
  }

  handle_.sync_stream();
  infile.close();

  recompute_internal_state(handle_, index);

  return index;
}

}  // namespace raft::neighbors::ivf_pq::detail
