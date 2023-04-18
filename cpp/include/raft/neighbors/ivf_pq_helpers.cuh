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

#include <raft/neighbors/detail/ivf_pq_build.cuh>
#include <raft/neighbors/ivf_pq_types.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>

namespace raft::neighbors::ivf_pq::helpers {
/**
 * @defgroup ivf_pq_helpers Helper functions for manipulationg IVF PQ Index
 * @{
 */

namespace codepacker {
/**
 * @brief Unpack `n_take` consecutive records of a single list (cluster) in the compressed index
 * starting at given `offset`.
 *
 * Bit compression is removed, which means output will have pq_dim dimensional vectors (one code per
 * byte, instead of ceildiv(pq_dim * pq_bits, 8) bytes of pq codes).
 *
 * Usage example:
 * @code{.cpp}
 *   auto list_data = index.lists()[label]->data.view();
 *   // allocate the buffer for the output
 *   uint32_t n_take = 4;
 *   auto codes = raft::make_device_matrix<uint8_t>(res, n_take, index.pq_dim());
 *   uint32_t offset = 0;
 *   // unpack n_take elements from the list
 *   ivf_pq::helpers::codepacker::unpack(res, list_data, index.pq_bits(), offset, codes.view());
 * @endcode
 *
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] res raft resource
 * @param[in] list_data block to read from
 * @param[in] pq_bits bit length of encoded vector elements
 * @param[in] offset
 *   How many records in the list to skip.
 * @param[out] codes
 *   the destination buffer [n_take, index.pq_dim()].
 *   The length `n_take` defines how many records to unpack,
 *   it must be smaller than the list size.
 */
inline void unpack(
  raft::device_resources const& res,
  device_mdspan<const uint8_t, list_spec<uint32_t, uint32_t>::list_extents, row_major> list_data,
  uint32_t pq_bits,
  uint32_t offset,
  device_matrix_view<uint8_t, uint32_t, row_major> codes)
{
  ivf_pq::detail::unpack_list_data(codes, list_data, offset, pq_bits, res.get_stream());
}

/**
 * Write flat PQ codes into an existing list by the given offset.
 *
 * NB: no memory allocation happens here; the list must fit the data (offset + n_vec).
 *
 * Usage example:
 * @code{.cpp}
 *   auto list_data  = index.lists()[label]->data.view();
 *   // allocate the buffer for the input codes
 *   auto codes = raft::make_device_matrix<uint8_t>(res, n_vec, index.pq_dim());
 *   ... prepare n_vecs to pack into the list in codes ...
 *   // write codes into the list starting from the 42nd position
 *   ivf_pq::helpers::codepacker::pack(
 *       res, make_const_mdspan(codes.view()), index.pq_bits(), 42, list_data);
 * @endcode
 *
 * @param[in] res
 * @param[in] codes flat PQ codes, one code per byte [n_vec, pq_dim]
 * @param[in] pq_bits bit length of encoded vector elements
 * @param[in] offset how many records to skip before writing the data into the list
 * @param[in] list_data block to write into
 */
inline void pack(
  raft::device_resources const& res,
  device_matrix_view<const uint8_t, uint32_t, row_major> codes,
  uint32_t pq_bits,
  uint32_t offset,
  device_mdspan<uint8_t, list_spec<uint32_t, uint32_t>::list_extents, row_major> list_data)
{
  ivf_pq::detail::pack_list_data(list_data, codes, offset, pq_bits, res.get_stream());
}
}  // namespace codepacker

/**
 * Write flat PQ codes into an existing list by the given offset.
 *
 * The list is identified by its label.
 *
 * NB: no memory allocation happens here; the list must fit the data (offset + n_vec).
 *
 * Usage example:
 * @code{.cpp}
 *   // We will write into the 137th cluster
 *   uint32_t label = 137;
 *   // allocate the buffer for the input codes
 *   auto codes = raft::make_device_matrix<const uint8_t>(res, n_vec, index.pq_dim());
 *   ... prepare n_vecs to pack into the list in codes ...
 *   // write codes into the list starting from the 42nd position
 *   ivf_pq::helpers::pack_list_data(res, &index, codes_to_pack, label, 42);
 * @endcode
 *
 * @param[in] res
 * @param[inout] index IVF-PQ index.
 * @param[in] codes flat PQ codes, one code per byte [n_rows, pq_dim]
 * @param[in] label The id of the list (cluster) into which we write.
 * @param[in] offset how many records to skip before writing the data into the list
 */
template <typename IdxT>
void pack_list_data(raft::device_resources const& res,
                    index<IdxT>* index,
                    device_matrix_view<const uint8_t, uint32_t, row_major> codes,
                    uint32_t label,
                    uint32_t offset)
{
  ivf_pq::detail::pack_list_data(res, index, codes, label, offset);
}

/**
 * @brief Unpack `n_take` consecutive records of a single list (cluster) in the compressed index
 * starting at given `offset`, one code per byte (independently of pq_bits).
 *
 * Usage example:
 * @code{.cpp}
 *   // We will unpack the fourth cluster
 *   uint32_t label = 3;
 *   // Get the list size
 *   uint32_t list_size = 0;
 *   raft::copy(&list_size, index.list_sizes().data_handle() + label, 1, res.get_stream());
 *   res.sync_stream();
 *   // allocate the buffer for the output
 *   auto codes = raft::make_device_matrix<float>(res, list_size, index.pq_dim());
 *   // unpack the whole list
 *   ivf_pq::helpers::unpack_list_data(res, index, codes.view(), label, 0);
 * @endcode
 *
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] res
 * @param[in] index
 * @param[out] out_codes
 *   the destination buffer [n_take, index.pq_dim()].
 *   The length `n_take` defines how many records to unpack,
 *   it must be smaller than the list size.
 * @param[in] label
 *   The id of the list (cluster) to decode.
 * @param[in] offset
 *   How many records in the list to skip.
 */
template <typename IdxT>
void unpack_list_data(raft::device_resources const& res,
                      const index<IdxT>& index,
                      device_matrix_view<uint8_t, uint32_t, row_major> out_codes,
                      uint32_t label,
                      uint32_t offset)
{
  return ivf_pq::detail::unpack_list_data<IdxT>(res, index, out_codes, label, offset);
}

/**
 * @brief Unpack a series of records of a single list (cluster) in the compressed index
 * by their in-list offsets, one code per byte (independently of pq_bits).
 *
 * Usage example:
 * @code{.cpp}
 *   // We will unpack the fourth cluster
 *   uint32_t label = 3;
 *   // Create the selection vector
 *   auto selected_indices = raft::make_device_vector<uint32_t>(res, 4);
 *   ... fill the indices ...
 *   res.sync_stream();
 *   // allocate the buffer for the output
 *   auto codes = raft::make_device_matrix<float>(res, selected_indices.size(), index.pq_dim());
 *   // decode the whole list
 *   ivf_pq::helpers::unpack_list_data(
 *       res, index, selected_indices.view(), codes.view(), label);
 * @endcode
 *
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] res
 * @param[in] index
 * @param[in] in_cluster_indices
 *   The offsets of the selected indices within the cluster.
 * @param[out] out_codes
 *   the destination buffer [n_take, index.pq_dim()].
 *   The length `n_take` defines how many records to unpack,
 *   it must be smaller than the list size.
 * @param[in] label
 *   The id of the list (cluster) to decode.
 */
template <typename IdxT>
void unpack_list_data(raft::device_resources const& res,
                      const index<IdxT>& index,
                      device_vector_view<const uint32_t> in_cluster_indices,
                      device_matrix_view<uint8_t, uint32_t, row_major> out_codes,
                      uint32_t label)
{
  return ivf_pq::detail::unpack_list_data<IdxT>(res, index, out_codes, label, in_cluster_indices);
}

/**
 * @brief Decode `n_take` consecutive records of a single list (cluster) in the compressed index
 * starting at given `offset`.
 *
 * Usage example:
 * @code{.cpp}
 *   // We will reconstruct the fourth cluster
 *   uint32_t label = 3;
 *   // Get the list size
 *   uint32_t list_size = 0;
 *   raft::copy(&list_size, index.list_sizes().data_handle() + label, 1, res.get_stream());
 *   res.sync_stream();
 *   // allocate the buffer for the output
 *   auto decoded_vectors = raft::make_device_matrix<float>(res, list_size, index.dim());
 *   // decode the whole list
 *   ivf_pq::helpers::reconstruct_list_data(res, index, decoded_vectors.view(), label, 0);
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] res
 * @param[in] index
 * @param[out] out_vectors
 *   the destination buffer [n_take, index.dim()].
 *   The length `n_take` defines how many records to reconstruct,
 *   it must be smaller than the list size.
 * @param[in] label
 *   The id of the list (cluster) to decode.
 * @param[in] offset
 *   How many records in the list to skip.
 */
template <typename T, typename IdxT>
void reconstruct_list_data(raft::device_resources const& res,
                           const index<IdxT>& index,
                           device_matrix_view<T, uint32_t, row_major> out_vectors,
                           uint32_t label,
                           uint32_t offset)
{
  return ivf_pq::detail::reconstruct_list_data(res, index, out_vectors, label, offset);
}

/**
 * @brief Decode a series of records of a single list (cluster) in the compressed index
 * by their in-list offsets.
 *
 * Usage example:
 * @code{.cpp}
 *   // We will reconstruct the fourth cluster
 *   uint32_t label = 3;
 *   // Create the selection vector
 *   auto selected_indices = raft::make_device_vector<uint32_t>(res, 4);
 *   ... fill the indices ...
 *   res.sync_stream();
 *   // allocate the buffer for the output
 *   auto decoded_vectors = raft::make_device_matrix<float>(
 *                             res, selected_indices.size(), index.dim());
 *   // decode the whole list
 *   ivf_pq::helpers::reconstruct_list_data(
 *       res, index, selected_indices.view(), decoded_vectors.view(), label);
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] res
 * @param[in] index
 * @param[in] in_cluster_indices
 *   The offsets of the selected indices within the cluster.
 * @param[out] out_vectors
 *   the destination buffer [n_take, index.dim()].
 *   The length `n_take` defines how many records to reconstruct,
 *   it must be smaller than the list size.
 * @param[in] label
 *   The id of the list (cluster) to decode.
 */
template <typename T, typename IdxT>
void reconstruct_list_data(raft::device_resources const& res,
                           const index<IdxT>& index,
                           device_vector_view<const uint32_t> in_cluster_indices,
                           device_matrix_view<T, uint32_t, row_major> out_vectors,
                           uint32_t label)
{
  return ivf_pq::detail::reconstruct_list_data(res, index, out_vectors, label, in_cluster_indices);
}

/**
 * @brief Extend one list of the index in-place, by the list label, skipping the classification and
 * encoding steps.
 *
 * Usage example:
 * @code{.cpp}
 *   // We will extend the fourth cluster
 *   uint32_t label = 3;
 *   // We will fill 4 new vectors
 *   uint32_t n_vec = 4;
 *   // Indices of the new vectors
 *   auto indices = raft::make_device_vector<uint32_t>(res, n_vec);
 *   ... fill the indices ...
 *   auto new_codes = raft::make_device_matrix<uint8_t, uint32_t, row_major> new_codes(
 *       res, n_vec, index.pq_dim());
 *   ... fill codes ...
 *   // extend list with new codes
 *   ivf_pq::helpers::extend_list_with_codes(
 *       res, &index, codes.view(), indices.view(), label);
 * @endcode
 *
 * @tparam IdxT
 *
 * @param[in] res
 * @param[inout] index
 * @param[in] new_codes flat PQ codes, one code per byte [n_rows, index.pq_dim()]
 * @param[in] new_indices source indices [n_rows]
 * @param[in] label the id of the target list (cluster).
 */
template <typename IdxT>
void extend_list_with_codes(raft::device_resources const& res,
                            index<IdxT>* index,
                            device_matrix_view<const uint8_t, uint32_t, row_major> new_codes,
                            device_vector_view<const IdxT, uint32_t, row_major> new_indices,
                            uint32_t label)
{
  ivf_pq::detail::extend_list_with_codes(res, index, new_codes, new_indices, label);
}

/**
 * @brief Extend one list of the index in-place, by the list label, skipping the classification
 * step.
 *
 *  Usage example:
 * @code{.cpp}
 *   // We will extend the fourth cluster
 *   uint32_t label = 3;
 *   // We will extend with 4 new vectors
 *   uint32_t n_vec = 4;
 *   // Indices of the new vectors
 *   auto indices = raft::make_device_vector<uint32_t>(res, n_vec);
 *   ... fill the indices ...
 *   auto new_vectors = raft::make_device_matrix<float, uint32_t, row_major> new_codes(
 *       res, n_vec, index.dim());
 *   ... fill vectors ...
 *   // extend list with new vectors
 *   ivf_pq::helpers::extend_list(
 *       res, &index, new_vectors.view(), indices.view(), label);
 * @endcode
 *
 * @tparam T
 * @tparam IdxT
 *
 * @param[in] res
 * @param[inout] index
 * @param[in] new_vectors data to encode [n_rows, index.dim()]
 * @param[in] new_indices source indices [n_rows]
 * @param[in] label the id of the target list (cluster).
 *
 */
template <typename T, typename IdxT>
void extend_list(raft::device_resources const& res,
                 index<IdxT>* index,
                 device_matrix_view<const T, uint32_t, row_major> new_vectors,
                 device_vector_view<const IdxT, uint32_t, row_major> new_indices,
                 uint32_t label)
{
  ivf_pq::detail::extend_list(res, index, new_vectors, new_indices, label);
}

/**
 * @brief Remove all data from a single list (cluster) in the index.
 *
 * Usage example:
 * @code{.cpp}
 *   // We will erase the fourth cluster (label = 3)
 *   ivf_pq::helpers::erase_list(res, &index, 3);
 * @endcode
 *
 * @tparam IdxT
 * @param[in] res
 * @param[inout] index
 * @param[in] label the id of the target list (cluster).
 */
template <typename IdxT>
void erase_list(raft::device_resources const& res, index<IdxT>* index, uint32_t label)
{
  ivf_pq::detail::erase_list(res, index, label);
}

/** @} */
}  // namespace raft::neighbors::ivf_pq::helpers
