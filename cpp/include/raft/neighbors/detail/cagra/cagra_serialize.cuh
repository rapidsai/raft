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

#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/serialize.hpp>
#include <raft/neighbors/cagra_types.hpp>
#include <raft/neighbors/detail/dataset_serialize.hpp>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <type_traits>

namespace raft::neighbors::cagra::detail {

constexpr int serialization_version = 4;

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

  include_dataset &= (index_.data().n_rows() > 0);

  serialize_scalar(res, os, include_dataset);
  if (include_dataset) {
    RAFT_LOG_INFO("Saving CAGRA index with dataset");
    neighbors::detail::serialize(res, os, index_.data());
  } else {
    RAFT_LOG_DEBUG("Saving CAGRA index WITHOUT dataset");
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
                          const raft::neighbors::cagra::index<T, IdxT>& index_)
{
  // static_assert(std::is_same_v<IdxT, int> or std::is_same_v<IdxT, uint32_t>,
  //               "An hnswlib index can only be trained with int32 or uint32 IdxT");
  common::nvtx::range<common::nvtx::domain::raft> fun_scope("cagra::serialize");
  RAFT_LOG_DEBUG("Saving CAGRA index to hnswlib format, size %zu, dim %u",
                 static_cast<size_t>(index_.size()),
                 index_.dim());

  // offset_level_0
  std::size_t offset_level_0 = 0;
  os.write(reinterpret_cast<char*>(&offset_level_0), sizeof(std::size_t));
  // max_element
  std::size_t max_element = index_.size();
  os.write(reinterpret_cast<char*>(&max_element), sizeof(std::size_t));
  // curr_element_count
  std::size_t curr_element_count = index_.size();
  os.write(reinterpret_cast<char*>(&curr_element_count), sizeof(std::size_t));
  // Example:M: 16, dim = 128, data_t = float, index_t = uint32_t, list_size_type = uint32_t,
  // labeltype: size_t size_data_per_element_ = M * 2 * sizeof(index_t) + sizeof(list_size_type) +
  // dim * sizeof(data_t) + sizeof(labeltype)
  auto size_data_per_element = static_cast<std::size_t>(index_.graph_degree() * sizeof(IdxT) + 4 +
                                                        index_.dim() * sizeof(T) + 8);
  os.write(reinterpret_cast<char*>(&size_data_per_element), sizeof(std::size_t));
  // label_offset
  std::size_t label_offset = size_data_per_element - 8;
  os.write(reinterpret_cast<char*>(&label_offset), sizeof(std::size_t));
  // offset_data
  auto offset_data = static_cast<std::size_t>(index_.graph_degree() * sizeof(IdxT) + 4);
  os.write(reinterpret_cast<char*>(&offset_data), sizeof(std::size_t));
  // max_level
  int max_level = 1;
  os.write(reinterpret_cast<char*>(&max_level), sizeof(int));
  // entrypoint_node
  auto entrypoint_node = static_cast<int>(index_.size() / 2);
  os.write(reinterpret_cast<char*>(&entrypoint_node), sizeof(int));
  // max_M
  auto max_M = static_cast<std::size_t>(index_.graph_degree() / 2);
  os.write(reinterpret_cast<char*>(&max_M), sizeof(std::size_t));
  // max_M0
  std::size_t max_M0 = index_.graph_degree();
  os.write(reinterpret_cast<char*>(&max_M0), sizeof(std::size_t));
  // M
  auto M = static_cast<std::size_t>(index_.graph_degree() / 2);
  os.write(reinterpret_cast<char*>(&M), sizeof(std::size_t));
  // mult, can be anything
  double mult = 0.42424242;
  os.write(reinterpret_cast<char*>(&mult), sizeof(double));
  // efConstruction, can be anything
  std::size_t efConstruction = 500;
  os.write(reinterpret_cast<char*>(&efConstruction), sizeof(std::size_t));

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
  auto host_graph =
    raft::make_host_matrix<IdxT, int64_t, raft::row_major>(graph.extent(0), graph.extent(1));
  raft::copy(host_graph.data_handle(),
             graph.data_handle(),
             graph.size(),
             raft::resource::get_cuda_stream(res));
  resource::sync_stream(res);

  // Write one dataset and graph row at a time
  for (std::size_t i = 0; i < index_.size(); i++) {
    auto graph_degree = static_cast<int>(index_.graph_degree());
    os.write(reinterpret_cast<char*>(&graph_degree), sizeof(int));

    for (std::size_t j = 0; j < index_.graph_degree(); ++j) {
      auto graph_elem = host_graph(i, j);
      os.write(reinterpret_cast<char*>(&graph_elem), sizeof(IdxT));
    }

    auto data_row = host_dataset.data_handle() + (index_.dim() * i);
    // if constexpr (std::is_same_v<T, float>) {
    for (std::size_t j = 0; j < index_.dim(); ++j) {
      auto data_elem = host_dataset(i, j);
      os.write(reinterpret_cast<char*>(&data_elem), sizeof(T));
    }
    // } else if constexpr (std::is_same_v<T, std::int8_t> or std::is_same_v<T, std::uint8_t>) {
    //   for (std::size_t j = 0; j < index_.dim(); ++j) {
    //     auto data_elem = static_cast<int>(host_dataset(i, j));
    //     os.write(reinterpret_cast<char*>(&data_elem), sizeof(int));
    //   }
    // }

    os.write(reinterpret_cast<char*>(&i), sizeof(std::size_t));
  }

  for (std::size_t i = 0; i < index_.size(); i++) {
    // zeroes
    auto zero = 0;
    os.write(reinterpret_cast<char*>(&zero), sizeof(int));
  }
}

template <typename T, typename IdxT>
void serialize_to_hnswlib(raft::resources const& res,
                          const std::string& filename,
                          const raft::neighbors::cagra::index<T, IdxT>& index_)
{
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

  index<T, IdxT> idx(res, metric);
  idx.update_graph(res, raft::make_const_mdspan(graph.view()));
  bool has_dataset = deserialize_scalar<bool>(res, is);
  if (has_dataset) {
    idx.update_dataset(res, neighbors::detail::deserialize_dataset<int64_t>(res, is));
  }
  return idx;
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
