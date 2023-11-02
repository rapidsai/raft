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

#include "raft/core/host_mdarray.hpp"
#include "raft/core/mdspan_types.hpp"
#include "raft/core/resource/cuda_stream.hpp"
#include <raft/core/resource/thrust_policy.hpp>
#include <cstddef>
#include <cstdint>
#include <raft/core/mdarray.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/serialize.hpp>
#include <raft/neighbors/cagra_types.hpp>

#include <fstream>
#include <type_traits>

namespace raft::neighbors::cagra::detail {

constexpr int serialization_version = 3;

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
void serialize(raft::resources const& res,
               std::ostream& os,
               const index<T, IdxT>& index_,
               bool include_dataset)
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope("cagra::serialize");

  RAFT_LOG_DEBUG(
    "Saving CAGRA index, size %zu, dim %u", static_cast<size_t>(index_.size()), index_.dim());

  std::string dtype_string = raft::detail::numpy_serializer::get_numpy_dtype<T>().to_string();
  dtype_string.resize(4);
  os << dtype_string;

  serialize_scalar(res, os, serialization_version);
  serialize_scalar(res, os, index_.size());
  serialize_scalar(res, os, index_.dim());
  serialize_scalar(res, os, index_.graph_degree());
  serialize_scalar(res, os, index_.metric());
  serialize_mdspan(res, os, index_.graph());

  serialize_scalar(res, os, include_dataset);
  if (include_dataset) {
    auto dataset = index_.dataset();
    // Remove padding before saving the dataset
    auto host_dataset = make_host_matrix<T, int64_t>(dataset.extent(0), dataset.extent(1));
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
  }
}

template <typename T, typename IdxT>
void serialize(raft::resources const& res,
               const std::string& filename,
               const index<T, IdxT>& index_,
               bool include_dataset)
{
  std::ofstream of(filename, std::ios::out | std::ios::binary);
  if (!of) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  detail::serialize(res, of, index_, include_dataset);

  of.close();
  if (!of) { RAFT_FAIL("Error writing output %s", filename.c_str()); }
}

template <typename T, typename IdxT>
void serialize_to_hnswlib(raft::resources const& res,
               std::ostream& os,
               const index<T, IdxT>& index_)
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope("cagra::serialize_to_hnswlib");
  RAFT_LOG_DEBUG(
    "Saving CAGRA index to hnswlib format, size %zu, dim %u", static_cast<size_t>(index_.size()), index_.dim());

  serialize_scalar(res, os, std::size_t{0});
  serialize_scalar(res, os, static_cast<std::size_t>(index_.size()));
  serialize_scalar(res, os, static_cast<std::size_t>(index_.size()));
  // Example:M: 16, dim = 128, data_t = float, index_t = uint32_t, list_size_type = uint32_t, labeltype: size_t
  // size_data_per_element_ = M * 2 * sizeof(index_t) + sizeof(list_size_type) + dim * sizeof(data_t) + sizeof(labeltype)
  auto size_data_per_element = static_cast<std::size_t>(index_.graph_degree() * 4 + 4 + index_.size() * 4 + 8);
  serialize_scalar(res, os, size_data_per_element);
  serialize_scalar(res, os, size_data_per_element - 8);
  serialize_scalar(res, os, static_cast<std::size_t>(index_.graph_degree() * 4 + 4));
  serialize_scalar(res, os, std::int32_t{1});
  serialize_scalar(res, os, static_cast<std::int32_t>(index_.size() / 2));
  serialize_scalar(res, os, static_cast<std::size_t>(index_.graph_degree() / 2));
  serialize_scalar(res, os, static_cast<std::size_t>(index_.graph_degree()));
  serialize_scalar(res, os, static_cast<std::size_t>(index_.graph_degree() / 2));
  serialize_scalar(res, os, static_cast<double>(0.42424242));
  serialize_scalar(res, os, std::size_t{500});

  auto dataset = index_.dataset();
  // Remove padding before saving the dataset
  auto host_dataset = make_host_matrix<T, int64_t>(dataset.extent(0), dataset.extent(1));
  RAFT_CUDA_TRY(cudaMemcpy2DAsync(host_dataset.data_handle(),
                                  sizeof(T) * host_dataset.extent(1),
                                  dataset.data_handle(),
                                  sizeof(T) * dataset.stride(0),
                                  sizeof(T) * host_dataset.extent(1),
                                  dataset.extent(0),
                                  cudaMemcpyDefault,
                                  resource::get_cuda_stream(res)));
  resource::sync_stream(res);

  auto graph = index_.graph();
  // auto host_graph = raft::make_host_matrix<std::uint32_t, int64_t, raft::row_major>(graph.extent(0), graph.extent(1));
  // std::vector<uint32_t> host_graph_t(graph.size());
  IdxT* host_graph = new IdxT[graph.size()];
  // thrust::copy(raft::resource::get_thrust_policy(res), graph.data_handle(), graph.data_handle() + graph.size(), host_graph.data_handle());
  raft::copy(host_graph, graph.data_handle(), graph.size(), raft::resource::get_cuda_stream(res));

  // Write one dataset and graph row at a time
  for (std::size_t i = 0; i < index_.size(); i++) {
    serialize_scalar(res, os, static_cast<std::size_t>(index_.graph_degree()));

    auto graph_row = host_graph + (index_.graph_degree() * i);
    auto graph_row_mds = raft::make_host_vector_view<IdxT, int64_t>(graph_row, index_.graph_degree());
    serialize_mdspan(res, os, graph_row_mds);

    auto data_row = host_dataset.data_handle() + (index_.dim() * i);
    if constexpr (std::is_same_v<T, float>) {
      auto data_row_mds = raft::make_host_vector_view<T, int64_t>(data_row, index_.dim());
      serialize_mdspan(res, os, data_row_mds);
    }
    else if constexpr (std::is_same_v<T, std::int8_t> or std::is_same_v<T, std::uint8_t>) {
      auto data_row_int = raft::make_host_vector<std::int32_t, std::int64_t>(index_.dim());
      std::copy(data_row, data_row + index_.size(), data_row_int.data_handle());
      serialize_mdspan(res, os, data_row_int.view());
    }

    serialize_scalar(res, os, i);
  }

  for (std::size_t i = 0; i < index_.size(); i++) {
    serialize_scalar(res, os, std::int32_t{0});
  }
  delete [] host_graph;
}

template <typename T, typename IdxT>
void serialize_to_hnswlib(raft::resources const& res,
               const std::string& filename,
               const index<T, IdxT>& index_) {
  std::ofstream of(filename, std::ios::out | std::ios::binary);
  if (!of) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  detail::serialize_to_hnswlib<T, IdxT>(res, of, index_);

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
  common::nvtx::range<common::nvtx::domain::raft> fun_scope("cagra::deserialize");

  char dtype_string[4];
  is.read(dtype_string, 4);

  auto ver = deserialize_scalar<int>(res, is);
  if (ver != serialization_version) {
    RAFT_FAIL("serialization version mismatch, expected %d, got %d ", serialization_version, ver);
  }
  auto n_rows       = deserialize_scalar<IdxT>(res, is);
  auto dim          = deserialize_scalar<std::uint32_t>(res, is);
  auto graph_degree = deserialize_scalar<std::uint32_t>(res, is);
  auto metric       = deserialize_scalar<raft::distance::DistanceType>(res, is);

  auto graph = raft::make_host_matrix<IdxT, int64_t>(n_rows, graph_degree);
  deserialize_mdspan(res, is, graph.view());

  bool has_dataset = deserialize_scalar<bool>(res, is);
  if (has_dataset) {
    auto dataset = raft::make_host_matrix<T, int64_t>(n_rows, dim);
    deserialize_mdspan(res, is, dataset.view());
    return index<T, IdxT>(
      res, metric, raft::make_const_mdspan(dataset.view()), raft::make_const_mdspan(graph.view()));
  } else {
    // create a new index with no dataset - the user must supply via update_dataset themselves
    // later (this avoids allocating GPU memory in the meantime)
    index<T, IdxT> idx(res, metric);
    idx.update_graph(res, raft::make_const_mdspan(graph.view()));
    return idx;
  }
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
}  // namespace raft::neighbors::cagra::detail
