/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <raft/core/host_mdarray.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>
#include <raft/neighbors/detail/ivf_pq_build.cuh>
#include <raft/neighbors/ivf_list.hpp>
#include <raft/neighbors/ivf_pq_types.hpp>

#include <fstream>
#include <memory>

namespace raft::neighbors::ivf_pq::detail {

// Serialization version
// No backward compatibility yet; that is, can't add additional fields without breaking
// backward compatibility.
// TODO(hcho3) Implement next-gen serializer for IVF that allows for expansion in a backward
//             compatible fashion.
constexpr int kSerializationVersion = 3;

/**
 * Write the index to an output stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index IVF-PQ index
 *
 */
template <typename IdxT>
void serialize(raft::resources const& handle_, std::ostream& os, const index<IdxT>& index)
{
  RAFT_LOG_DEBUG("Size %zu, dim %d, pq_dim %d, pq_bits %d",
                 static_cast<size_t>(index.size()),
                 static_cast<int>(index.dim()),
                 static_cast<int>(index.pq_dim()),
                 static_cast<int>(index.pq_bits()));

  serialize_scalar(handle_, os, kSerializationVersion);
  serialize_scalar(handle_, os, index.size());
  serialize_scalar(handle_, os, index.dim());
  serialize_scalar(handle_, os, index.pq_bits());
  serialize_scalar(handle_, os, index.pq_dim());
  serialize_scalar(handle_, os, index.conservative_memory_allocation());

  serialize_scalar(handle_, os, index.metric());
  serialize_scalar(handle_, os, index.codebook_kind());
  serialize_scalar(handle_, os, index.n_lists());

  serialize_mdspan(handle_, os, index.pq_centers());
  serialize_mdspan(handle_, os, index.centers());
  serialize_mdspan(handle_, os, index.centers_rot());
  serialize_mdspan(handle_, os, index.rotation_matrix());

  auto sizes_host = make_host_mdarray<uint32_t, uint32_t, row_major>(index.list_sizes().extents());
  copy(sizes_host.data_handle(),
       index.list_sizes().data_handle(),
       sizes_host.size(),
       resource::get_cuda_stream(handle_));
  resource::sync_stream(handle_);
  serialize_mdspan(handle_, os, sizes_host.view());
  auto list_store_spec = list_spec<uint32_t, IdxT>{index.pq_bits(), index.pq_dim(), true};
  for (uint32_t label = 0; label < index.n_lists(); label++) {
    ivf::serialize_list(handle_, os, index.lists()[label], list_store_spec, sizes_host(label));
  }
}

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
void serialize(raft::resources const& handle_,
               const std::string& filename,
               const index<IdxT>& index)
{
  std::ofstream of(filename, std::ios::out | std::ios::binary);
  if (!of) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  detail::serialize(handle_, of, index);

  of.close();
  if (!of) { RAFT_FAIL("Error writing output %s", filename.c_str()); }
  return;
}

/**
 * Load index from input stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] handle the raft handle
 * @param[in] is input stream
 *
 */
template <typename IdxT>
auto deserialize(raft::resources const& handle_, std::istream& is) -> index<IdxT>
{
  auto ver = deserialize_scalar<int>(handle_, is);
  if (ver != kSerializationVersion) {
    RAFT_FAIL("serialization version mismatch %d vs. %d", ver, kSerializationVersion);
  }
  auto n_rows  = deserialize_scalar<IdxT>(handle_, is);
  auto dim     = deserialize_scalar<std::uint32_t>(handle_, is);
  auto pq_bits = deserialize_scalar<std::uint32_t>(handle_, is);
  auto pq_dim  = deserialize_scalar<std::uint32_t>(handle_, is);
  auto cma     = deserialize_scalar<bool>(handle_, is);

  auto metric        = deserialize_scalar<raft::distance::DistanceType>(handle_, is);
  auto codebook_kind = deserialize_scalar<raft::neighbors::ivf_pq::codebook_gen>(handle_, is);
  auto n_lists       = deserialize_scalar<std::uint32_t>(handle_, is);

  RAFT_LOG_DEBUG("n_rows %zu, dim %d, pq_dim %d, pq_bits %d, n_lists %d",
                 static_cast<std::size_t>(n_rows),
                 static_cast<int>(dim),
                 static_cast<int>(pq_dim),
                 static_cast<int>(pq_bits),
                 static_cast<int>(n_lists));

  auto index = raft::neighbors::ivf_pq::index<IdxT>(
    handle_, metric, codebook_kind, n_lists, dim, pq_bits, pq_dim, cma);

  deserialize_mdspan(handle_, is, index.pq_centers());
  deserialize_mdspan(handle_, is, index.centers());
  deserialize_mdspan(handle_, is, index.centers_rot());
  deserialize_mdspan(handle_, is, index.rotation_matrix());
  deserialize_mdspan(handle_, is, index.list_sizes());
  auto list_device_spec = list_spec<uint32_t, IdxT>{pq_bits, pq_dim, cma};
  auto list_store_spec  = list_spec<uint32_t, IdxT>{pq_bits, pq_dim, true};
  for (auto& list : index.lists()) {
    ivf::deserialize_list(handle_, is, list, list_store_spec, list_device_spec);
  }

  resource::sync_stream(handle_);

  ivf::detail::recompute_internal_state(handle_, index);

  return index;
}

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 *
 */
template <typename IdxT>
auto deserialize(raft::resources const& handle_, const std::string& filename) -> index<IdxT>
{
  std::ifstream infile(filename, std::ios::in | std::ios::binary);

  if (!infile) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  auto index = detail::deserialize<IdxT>(handle_, infile);

  infile.close();

  return index;
}

}  // namespace raft::neighbors::ivf_pq::detail
