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

#include "detail/hnsw.hpp"
#include "hnsw.hpp"

#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/neighbors/cagra_types.hpp>

#include <cstddef>
#include <cstdint>

namespace raft::neighbors::hnsw {

/**
 * @addtogroup hnsw Build CAGRA index and search with hnswlib
 * @{
 */

/**
 * @brief Construct an hnswlib base-layer-only index from a CAGRA index
 * NOTE: 1. This method uses the filesystem to write the CAGRA index in `/tmp/<random_number>.bin`
 * before reading it as an hnswlib index, then deleting the temporary file.
 *       2. This function is only offered as a compiled symbol in `libraft.so`
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices
 *
 * @param[in] res raft resources
 * @param[in] cagra_index cagra index
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace raft::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *
 *   // Load CAGRA index as base-layer-only hnswlib index
 *   auto hnsw_index = hnsw::from_cagra(res, index);
 * @endcode
 */
template <typename T, typename IdxT>
std::unique_ptr<index<T>> from_cagra(raft::resources const& res,
                                     raft::neighbors::cagra::index<T, IdxT> cagra_index);

template <>
std::unique_ptr<index<float>> from_cagra(
  raft::resources const& res, raft::neighbors::cagra::index<float, uint32_t> cagra_index);

template <>
std::unique_ptr<index<int8_t>> from_cagra(
  raft::resources const& res, raft::neighbors::cagra::index<int8_t, uint32_t> cagra_index);

template <>
std::unique_ptr<index<uint8_t>> from_cagra(
  raft::resources const& res, raft::neighbors::cagra::index<uint8_t, uint32_t> cagra_index);

/**
 * @brief Search hnswlib base-layer-only index constructed from a CAGRA index
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] idx cagra index
 * @param[in] queries a host matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a host matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a host matrix view to the distances to the selected neighbors [n_queries,
 * k]
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace raft::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *
 *   // Save CAGRA index as base layer only hnswlib index
 *   hnsw::serialize(res, "my_index.bin", index);
 *
 *   // Load CAGRA index as base layer only hnswlib index
 *   raft::neighbors::hnsw::index* hnsw_index;
 *   auto hnsw_index = hnsw::deserialize(res, "my_index.bin", D, raft::distance::L2Expanded);
 *
 *   // Search K nearest neighbors as an hnswlib index
 *   // using host threads for concurrency
 *   hnsw::search_params search_params;
 *   search_params.ef = 50 // ef >= K;
 *   search_params.num_threads = 10;
 *   auto neighbors = raft::make_host_matrix<uint32_t>(res, n_queries, k);
 *   auto distances = raft::make_host_matrix<float>(res, n_queries, k);
 *   hnsw::search(res, search_params, *index, queries, neighbors, distances);
 *   // de-allocate hnsw_index
 *   delete hnsw_index;
 * @endcode
 */
template <typename T>
void search(raft::resources const& res,
            const search_params& params,
            const index<T>& idx,
            raft::host_matrix_view<const T, int64_t, row_major> queries,
            raft::host_matrix_view<uint64_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances)
{
  RAFT_EXPECTS(
    queries.extent(0) == neighbors.extent(0) && queries.extent(0) == distances.extent(0),
    "Number of rows in output neighbors and distances matrices must equal the number of queries.");

  RAFT_EXPECTS(neighbors.extent(1) == distances.extent(1),
               "Number of columns in output neighbors and distances matrices must equal k");
  RAFT_EXPECTS(queries.extent(1) == idx.dim(),
               "Number of query dimensions should equal number of dimensions in the index.");

  detail::search(res, params, idx, queries, neighbors, distances);
}

/**@}*/

}  // namespace raft::neighbors::hnsw
