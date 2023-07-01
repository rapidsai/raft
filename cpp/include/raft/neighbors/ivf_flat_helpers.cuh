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

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/neighbors/detail/ivf_flat_build.cuh>
#include <raft/neighbors/ivf_flat_types.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace raft::neighbors::ivf_flat::helpers {
/**
 * @defgroup ivf_flat_helpers Helper functions for manipulationg IVF Flat Index
 * @{
 */

namespace codepacker {
/**
 * @brief Unpack `n_take` consecutive records of a single non-interleaved list (cluster) starting at given `row_offset`.
 *
 *
 * Usage example:
 * @code{.cpp}
 *   auto list_data = index.lists()[label]->data.view();
 *   // allocate the buffer for the output
 *   uint32_t n_take = 4;
 *   auto codes = raft::make_device_matrix<uint8_t>(res, n_take, index.pq_dim());
 *   uint32_t offset = 0;
 *   // unpack n_take elements from the list
 *   ivf_flat::helpers::codepacker::unpack(res, list_data, index.pq_bits(), offset, codes.view());
 * @endcode
 *
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] res raft resource
 * @param[in] list_data block to read from
 * @param[in] offset
 *   How many records in the list to skip.
 * @param[out] codes
 */
template <typename value_t, typename idx_t>
inline void unpack(
  raft::resources const& res,
  device_matrix_view<value_t> list_data,
  uint32_t offset,
  device_matrix_view<value_t, uint32_t, row_major> codes)
{
  ivf_flat::detail::unpack_list_data_float32(res, codes, list_data, offset);
}

// /**
//  * Write flat PQ codes into an existing list by the given offset.
//  *
//  * NB: no memory allocation happens here; the list must fit the data (offset + n_vec).
//  *
//  * Usage example:
//  * @code{.cpp}
//  *   auto list_data  = index.lists()[label]->data.view();
//  *   // allocate the buffer for the input codes
//  *   auto codes = raft::make_device_matrix<uint8_t>(res, n_vec, index.pq_dim());
//  *   ... prepare n_vecs to pack into the list in codes ...
//  *   // write codes into the list starting from the 42nd position
//  *   ivf_flat::helpers::codepacker::pack(
//  *       res, make_const_mdspan(codes.view()), index.pq_bits(), 42, list_data);
//  * @endcode
//  *
//  * @param[in] res
//  * @param[in] codes flat PQ codes, one code per byte [n_vec, pq_dim]
//  * @param[in] offset how many records to skip before writing the data into the list
//  * @param[in] list_data block to write into
//  */
//  template <typename value_t, typename idx_t>
// inline void pack(
//   raft::resources const& res,
//   device_matrix_view<const value_t, uint32_t, row_major> codes,
//   uint32_t offset,
//   device_mdspan<value_t, typename list_spec<uint32_t, value_t, idx_t>::list_extents, row_major> list_data)
// {
//   ivf_flat::detail::pack_list_data(list_data, codes, offset, resource::get_cuda_stream(res));
// }
// }  // namespace codepacker

// /**
//  * Write flat PQ codes into an existing list by the given offset.
//  *
//  * The list is identified by its label.
//  *
//  * NB: no memory allocation happens here; the list must fit the data (offset + n_vec).
//  *
//  * Usage example:
//  * @code{.cpp}
//  *   // We will write into the 137th cluster
//  *   uint32_t label = 137;
//  *   // allocate the buffer for the input codes
//  *   auto codes = raft::make_device_matrix<const uint8_t>(res, n_vec, index.pq_dim());
//  *   ... prepare n_vecs to pack into the list in codes ...
//  *   // write codes into the list starting from the 42nd position
//  *   ivf_flat::helpers::pack_list_data(res, &index, codes_to_pack, label, 42);
//  * @endcode
//  *
//  * @param[in] res
//  * @param[inout] index IVF-PQ index.
//  * @param[in] codes flat PQ codes, one code per byte [n_rows, pq_dim]
//  * @param[in] label The id of the list (cluster) into which we write.
//  * @param[in] offset how many records to skip before writing the data into the list
//  */
// template <typename IdxT>
// void pack_list_data(raft::resources const& res,
//                     index<IdxT>* index,
//                     device_matrix_view<const uint8_t, uint32_t, row_major> codes,
//                     uint32_t label,
//                     uint32_t offset)
// {
//   ivf_flat::detail::pack_list_data(res, index, codes, label, offset);
// }

// /**
//  * @brief Unpack `n_take` consecutive records of a single list (cluster) in the compressed index
//  * starting at given `offset`, one code per byte (independently of pq_bits).
//  *
//  * Usage example:
//  * @code{.cpp}
//  *   // We will unpack the fourth cluster
//  *   uint32_t label = 3;
//  *   // Get the list size
//  *   uint32_t list_size = 0;
//  *   raft::copy(&list_size, index.list_sizes().data_handle() + label, 1,
//  * resource::get_cuda_stream(res)); resource::sync_stream(res);
//  *   // allocate the buffer for the output
//  *   auto codes = raft::make_device_matrix<float>(res, list_size, index.pq_dim());
//  *   // unpack the whole list
//  *   ivf_flat::helpers::unpack_list_data(res, index, codes.view(), label, 0);
//  * @endcode
//  *
//  * @tparam IdxT type of the indices in the source dataset
//  *
//  * @param[in] res
//  * @param[in] index
//  * @param[out] out_codes
//  *   the destination buffer [n_take, index.pq_dim()].
//  *   The length `n_take` defines how many records to unpack,
//  *   it must be smaller than the list size.
//  * @param[in] label
//  *   The id of the list (cluster) to decode.
//  * @param[in] offset
//  *   How many records in the list to skip.
//  */
// template <typename IdxT>
// void unpack_list_data(raft::resources const& res,
//                       const index<IdxT>& index,
//                       device_matrix_view<uint8_t, uint32_t, row_major> out_codes,
//                       uint32_t label,
//                       uint32_t offset)
// {
//   return ivf_flat::detail::unpack_list_data<IdxT>(res, index, out_codes, label, offset);
// }

// /**
//  * @brief Unpack a series of records of a single list (cluster) in the compressed index
//  * by their in-list offsets, one code per byte (independently of pq_bits).
//  *
//  * Usage example:
//  * @code{.cpp}
//  *   // We will unpack the fourth cluster
//  *   uint32_t label = 3;
//  *   // Create the selection vector
//  *   auto selected_indices = raft::make_device_vector<uint32_t>(res, 4);
//  *   ... fill the indices ...
//  *   resource::sync_stream(res);
//  *   // allocate the buffer for the output
//  *   auto codes = raft::make_device_matrix<float>(res, selected_indices.size(), index.pq_dim());
//  *   // decode the whole list
//  *   ivf_flat::helpers::unpack_list_data(
//  *       res, index, selected_indices.view(), codes.view(), label);
//  * @endcode
//  *
//  * @tparam IdxT type of the indices in the source dataset
//  *
//  * @param[in] res
//  * @param[in] index
//  * @param[in] in_cluster_indices
//  *   The offsets of the selected indices within the cluster.
//  * @param[out] out_codes
//  *   the destination buffer [n_take, index.pq_dim()].
//  *   The length `n_take` defines how many records to unpack,
//  *   it must be smaller than the list size.
//  * @param[in] label
//  *   The id of the list (cluster) to decode.
//  */
// template <typename IdxT>
// void unpack_list_data(raft::resources const& res,
//                       const index<IdxT>& index,
//                       device_vector_view<const uint32_t> in_cluster_indices,
//                       device_matrix_view<uint8_t, uint32_t, row_major> out_codes,
//                       uint32_t label)
// {
//   return ivf_flat::detail::unpack_list_data<IdxT>(res, index, out_codes, label, in_cluster_indices);
// }

/** @} */
}  // namespace raft::neighbors::ivf_flat::helpers
