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
#include <raft/neighbors/cagra_types.hpp>

#include <fstream>

namespace raft::neighbors::experimental::cagra::detail {

// Serialization version 1.
constexpr int serialization_version = 2;

// NB: we wrap this check in a struct, so that the updated RealSize is easy to see in the error
// message.
template <size_t RealSize, size_t ExpectedSize>
struct check_index_layout {
  static_assert(RealSize == ExpectedSize,
                "The size of the index struct has changed since the last update; "
                "paste in the new size and consider updating the serialization logic");
};

constexpr size_t expected_size = 200;
template struct check_index_layout<sizeof(index<double, std::uint64_t>), expected_size>;

/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] res the raft resource handle
 * @param[in] filename the file name for saving the index
 * @param[in] index_ CAGRA index
 *
 */
template <typename T, typename IdxT>
void serialize(raft::resources const& res, std::ostream& os, const index<T, IdxT>& index_)
{
  RAFT_LOG_DEBUG(
    "Saving CAGRA index, size %zu, dim %u", static_cast<size_t>(index_.size()), index_.dim());

  serialize_scalar(res, os, serialization_version);
  serialize_scalar(res, os, index_.size());
  serialize_scalar(res, os, index_.dim());
  serialize_scalar(res, os, index_.graph_degree());
  serialize_scalar(res, os, index_.metric());
  auto dataset = index_.dataset();
  // Remove padding before saving the dataset
  auto host_dataset = make_host_matrix<T, IdxT>(dataset.extent(0), dataset.extent(1));
  RAFT_CUDA_TRY(cudaMemcpy2DAsync(host_dataset.data_handle(),
                                  sizeof(T) * host_dataset.extent(1),
                                  dataset.data_handle(),
                                  sizeof(T) * dataset.stride(0),
                                  sizeof(T) * host_dataset.extent(1),
                                  dataset.extent(0),
                                  cudaMemcpyDefault,
                                  resource::get_cuda_stream(res)));
  resource::sync_stream(res);
  serialize_mdspan(res, os, host_dataset.view());
  serialize_mdspan(res, os, index_.graph());
}

template <typename T, typename IdxT>
void serialize(raft::resources const& res,
               const std::string& filename,
               const index<T, IdxT>& index_)
{
  std::ofstream of(filename, std::ios::out | std::ios::binary);
  if (!of) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  detail::serialize(res, of, index_);

  of.close();
  if (!of) { RAFT_FAIL("Error writing output %s", filename.c_str()); }
}

/** Load an index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] res the raft resource handle
 * @param[in] filename the name of the file that stores the index
 * @param[in] index_ CAGRA index
 *
 */
template <typename T, typename IdxT>
auto deserialize(raft::resources const& res, std::istream& is) -> index<T, IdxT>
{
  auto ver = deserialize_scalar<int>(res, is);
  if (ver != serialization_version) {
    RAFT_FAIL("serialization version mismatch, expected %d, got %d ", serialization_version, ver);
  }
  auto n_rows       = deserialize_scalar<IdxT>(res, is);
  auto dim          = deserialize_scalar<std::uint32_t>(res, is);
  auto graph_degree = deserialize_scalar<std::uint32_t>(res, is);
  auto metric       = deserialize_scalar<raft::distance::DistanceType>(res, is);

  auto dataset = raft::make_host_matrix<T, IdxT>(n_rows, dim);
  auto graph   = raft::make_host_matrix<IdxT, IdxT>(n_rows, graph_degree);
  deserialize_mdspan(res, is, dataset.view());
  deserialize_mdspan(res, is, graph.view());

  return index<T, IdxT>(
    res, metric, raft::make_const_mdspan(dataset.view()), raft::make_const_mdspan(graph.view()));
}

template <typename T, typename IdxT>
auto deserialize(raft::resources const& res, const std::string& filename) -> index<T, IdxT>
{
  std::ifstream is(filename, std::ios::in | std::ios::binary);

  if (!is) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  auto index = detail::deserialize<T, IdxT>(res, is);

  is.close();

  return index;
}
}  // namespace raft::neighbors::experimental::cagra::detail
