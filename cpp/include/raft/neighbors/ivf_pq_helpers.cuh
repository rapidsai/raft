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

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/neighbors/detail/ivf_pq_build.cuh>
#include <raft/neighbors/ivf_pq_types.hpp>
#include <raft/spatial/knn/detail/ann_utils.cuh>

#include <cstdio>

namespace raft::neighbors::ivf_pq::helpers {
using namespace raft::spatial::knn::detail;  // NOLINT
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
  raft::resources const& res,
  device_mdspan<const uint8_t, list_spec<uint32_t, uint32_t>::list_extents, row_major> list_data,
  uint32_t pq_bits,
  uint32_t offset,
  device_matrix_view<uint8_t, uint32_t, row_major> codes)
{
  ivf_pq::detail::unpack_list_data(
    codes, list_data, offset, pq_bits, resource::get_cuda_stream(res));
}

/**
 * @brief Unpack `n_rows` consecutive records of a single list (cluster) in the compressed index
 * starting at given `offset`. The output codes of a single vector are contiguous, not expanded to
 * one code per byte, which means the output has ceildiv(pq_dim * pq_bits, 8) bytes per PQ encoded
 * vector.
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   auto list_data = index.lists()[label]->data.view();
 *   // allocate the buffer for the output
 *   uint32_t n_rows = 4;
 *   auto codes = raft::make_device_matrix<uint8_t>(
 *     res, n_rows, raft::ceildiv(index.pq_dim() * index.pq_bits(), 8));
 *   uint32_t offset = 0;
 *   // unpack n_rows elements from the list
 *   ivf_pq::helpers::codepacker::unpack_contiguous(
 *     res, list_data, index.pq_bits(), offset, n_rows, index.pq_dim(), codes.data_handle());
 * @endcode
 *
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] res raft resource
 * @param[in] list_data block to read from
 * @param[in] pq_bits bit length of encoded vector elements
 * @param[in] offset
 *   How many records in the list to skip.
 * @param[in] n_rows How many records to unpack
 * @param[in] pq_dim The dimensionality of the PQ compressed records
 * @param[out] codes
 *   the destination buffer [n_rows, ceildiv(pq_dim * pq_bits, 8)].
 *   The length `n_rows` defines how many records to unpack,
 *   it must be smaller than the list size.
 */
inline void unpack_contiguous(
  raft::resources const& res,
  device_mdspan<const uint8_t, list_spec<uint32_t, uint32_t>::list_extents, row_major> list_data,
  uint32_t pq_bits,
  uint32_t offset,
  uint32_t n_rows,
  uint32_t pq_dim,
  uint8_t* codes)
{
  ivf_pq::detail::unpack_contiguous_list_data(
    codes, list_data, n_rows, pq_dim, offset, pq_bits, resource::get_cuda_stream(res));
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
 * @param[in] res raft resource
 * @param[in] codes flat PQ codes, one code per byte [n_vec, pq_dim]
 * @param[in] pq_bits bit length of encoded vector elements
 * @param[in] offset how many records to skip before writing the data into the list
 * @param[in] list_data block to write into
 */
inline void pack(
  raft::resources const& res,
  device_matrix_view<const uint8_t, uint32_t, row_major> codes,
  uint32_t pq_bits,
  uint32_t offset,
  device_mdspan<uint8_t, list_spec<uint32_t, uint32_t>::list_extents, row_major> list_data)
{
  ivf_pq::detail::pack_list_data(list_data, codes, offset, pq_bits, resource::get_cuda_stream(res));
}

/**
 * Write flat PQ codes into an existing list by the given offset. The input codes of a single vector
 * are contiguous (not expanded to one code per byte).
 *
 * NB: no memory allocation happens here; the list must fit the data (offset + n_rows records).
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   auto list_data  = index.lists()[label]->data.view();
 *   // allocate the buffer for the input codes
 *   auto codes = raft::make_device_matrix<uint8_t>(
 *     res, n_rows, raft::ceildiv(index.pq_dim() * index.pq_bits(), 8));
 *   ... prepare compressed vectors to pack into the list in codes ...
 *   // write codes into the list starting from the 42nd position. If the current size of the list
 *   // is greater than 42, this will overwrite the codes starting at this offset.
 *   ivf_pq::helpers::codepacker::pack_contiguous(
 *     res, codes.data_handle(), n_rows, index.pq_dim(), index.pq_bits(), 42, list_data);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] codes flat PQ codes, [n_vec, ceildiv(pq_dim * pq_bits, 8)]
 * @param[in] n_rows number of records
 * @param[in] pq_dim
 * @param[in] pq_bits bit length of encoded vector elements
 * @param[in] offset how many records to skip before writing the data into the list
 * @param[in] list_data block to write into
 */
inline void pack_contiguous(
  raft::resources const& res,
  const uint8_t* codes,
  uint32_t n_rows,
  uint32_t pq_dim,
  uint32_t pq_bits,
  uint32_t offset,
  device_mdspan<uint8_t, list_spec<uint32_t, uint32_t>::list_extents, row_major> list_data)
{
  ivf_pq::detail::pack_contiguous_list_data(
    list_data, codes, n_rows, pq_dim, offset, pq_bits, resource::get_cuda_stream(res));
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
 * @param[in] res raft resource
 * @param[inout] index IVF-PQ index.
 * @param[in] codes flat PQ codes, one code per byte [n_rows, pq_dim]
 * @param[in] label The id of the list (cluster) into which we write.
 * @param[in] offset how many records to skip before writing the data into the list
 */
template <typename IdxT>
void pack_list_data(raft::resources const& res,
                    index<IdxT>* index,
                    device_matrix_view<const uint8_t, uint32_t, row_major> codes,
                    uint32_t label,
                    uint32_t offset)
{
  ivf_pq::detail::pack_list_data(res, index, codes, label, offset);
}

/**
 * Write flat PQ codes into an existing list by the given offset. Use this when the input
 * vectors are PQ encoded and not expanded to one code per byte.
 *
 * The list is identified by its label.
 *
 * NB: no memory allocation happens here; the list into which the vectors are packed must fit offset
 * + n_rows rows.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace raft::neighbors;
 *   raft::resources res;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(res, index_params, dataset, N, D);
 *   // allocate the buffer for n_rows input codes. Each vector occupies
 *   // raft::ceildiv(index.pq_dim() * index.pq_bits(), 8) bytes because
 *   // codes are compressed and without gaps.
 *   auto codes = raft::make_device_matrix<const uint8_t>(
 *     res, n_rows, raft::ceildiv(index.pq_dim() * index.pq_bits(), 8));
 *   ... prepare the compressed vectors to pack into the list in codes ...
 *   // the first n_rows codes in the fourth IVF list are to be overwritten.
 *   uint32_t label = 3;
 *   // write codes into the list starting from the 0th position
 *   ivf_pq::helpers::pack_contiguous_list_data(
 *     res, &index, codes.data_handle(), n_rows, label, 0);
 * @endcode
 *
 * @tparam IdxT
 *
 * @param[in] res raft resource
 * @param[inout] index pointer to IVF-PQ index
 * @param[in] codes flat contiguous PQ codes [n_rows, ceildiv(pq_dim * pq_bits, 8)]
 * @param[in] n_rows how many records to pack
 * @param[in] label The id of the list (cluster) into which we write.
 * @param[in] offset how many records to skip before writing the data into the list
 */
template <typename IdxT>
void pack_contiguous_list_data(raft::resources const& res,
                               index<IdxT>* index,
                               uint8_t* codes,
                               uint32_t n_rows,
                               uint32_t label,
                               uint32_t offset)
{
  ivf_pq::detail::pack_contiguous_list_data(res, index, codes, n_rows, label, offset);
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
 *   raft::copy(&list_size, index.list_sizes().data_handle() + label, 1,
 * resource::get_cuda_stream(res)); resource::sync_stream(res);
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
void unpack_list_data(raft::resources const& res,
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
 *   resource::sync_stream(res);
 *   // allocate the buffer for the output
 *   auto codes = raft::make_device_matrix<float>(res, selected_indices.size(), index.pq_dim());
 *   // decode the whole list
 *   ivf_pq::helpers::unpack_list_data(
 *       res, index, selected_indices.view(), codes.view(), label);
 * @endcode
 *
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] res raft resource
 * @param[in] index IVF-PQ index (passed by reference)
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
void unpack_list_data(raft::resources const& res,
                      const index<IdxT>& index,
                      device_vector_view<const uint32_t> in_cluster_indices,
                      device_matrix_view<uint8_t, uint32_t, row_major> out_codes,
                      uint32_t label)
{
  return ivf_pq::detail::unpack_list_data<IdxT>(res, index, out_codes, label, in_cluster_indices);
}

/**
 * @brief Unpack `n_rows` consecutive PQ encoded vectors of a single list (cluster) in the
 * compressed index starting at given `offset`, not expanded to one code per byte. Each code in the
 * output buffer occupies ceildiv(index.pq_dim() * index.pq_bits(), 8) bytes.
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   // We will unpack the whole fourth cluster
 *   uint32_t label = 3;
 *   // Get the list size
 *   uint32_t list_size = 0;
 *   raft::update_host(&list_size, index.list_sizes().data_handle() + label, 1,
 * raft::resource::get_cuda_stream(res)); raft::resource::sync_stream(res);
 *   // allocate the buffer for the output
 *   auto codes = raft::make_device_matrix<float>(res, list_size, raft::ceildiv(index.pq_dim() *
 * index.pq_bits(), 8));
 *   // unpack the whole list
 *   ivf_pq::helpers::unpack_list_data(res, index, codes.data_handle(), list_size, label, 0);
 * @endcode
 *
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] res raft resource
 * @param[in] index IVF-PQ index (passed by reference)
 * @param[out] out_codes
 *   the destination buffer [n_rows, ceildiv(index.pq_dim() * index.pq_bits(), 8)].
 *   The length `n_rows` defines how many records to unpack,
 *   offset + n_rows must be smaller than or equal to the list size.
 * @param[in] n_rows how many codes to unpack
 * @param[in] label
 *   The id of the list (cluster) to decode.
 * @param[in] offset
 *   How many records in the list to skip.
 */
template <typename IdxT>
void unpack_contiguous_list_data(raft::resources const& res,
                                 const index<IdxT>& index,
                                 uint8_t* out_codes,
                                 uint32_t n_rows,
                                 uint32_t label,
                                 uint32_t offset)
{
  return ivf_pq::detail::unpack_contiguous_list_data<IdxT>(
    res, index, out_codes, n_rows, label, offset);
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
 *   raft::copy(&list_size, index.list_sizes().data_handle() + label, 1,
 *   resource::get_cuda_stream(res)); resource::sync_stream(res);
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
void reconstruct_list_data(raft::resources const& res,
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
 *   resource::sync_stream(res);
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
void reconstruct_list_data(raft::resources const& res,
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
void extend_list_with_codes(raft::resources const& res,
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
void extend_list(raft::resources const& res,
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
 *
 * @param[in] res
 * @param[inout] index
 * @param[in] label the id of the target list (cluster).
 */
template <typename IdxT>
void erase_list(raft::resources const& res, index<IdxT>* index, uint32_t label)
{
  ivf_pq::detail::erase_list(res, index, label);
}

/**
 * @brief Public helper API to reset the data and indices ptrs, and the list sizes. Useful for
 * externally modifying the index without going through the build stage. The data and indices of the
 * IVF lists will be lost.
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   using namespace raft::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // initialize an empty index
 *   ivf_pq::index<int64_t> index(res, index_params, D);
 *   // reset the index's state and list sizes
 *   ivf_pq::helpers::reset_index(res, &index);
 * @endcode
 *
 * @tparam IdxT
 *
 * @param[in] res raft resource
 * @param[inout] index pointer to IVF-PQ index
 */
template <typename IdxT>
void reset_index(const raft::resources& res, index<IdxT>* index)
{
  auto stream = resource::get_cuda_stream(res);

  utils::memzero(
    index->accum_sorted_sizes().data_handle(), index->accum_sorted_sizes().size(), stream);
  utils::memzero(index->list_sizes().data_handle(), index->list_sizes().size(), stream);
  utils::memzero(index->data_ptrs().data_handle(), index->data_ptrs().size(), stream);
  utils::memzero(index->inds_ptrs().data_handle(), index->inds_ptrs().size(), stream);
}

/**
 * @brief Public helper API exposing the computation of the index's rotation matrix.
 * NB: This is to be used only when the rotation matrix is not already computed through
 * raft::neighbors::ivf_pq::build.
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // force random rotation
 *   index_params.force_random_rotation = true;
 *   // initialize an empty index
 *   raft::neighbors::ivf_pq::index<int64_t> index(res, index_params, D);
 *   // reset the index
 *   reset_index(res, &index);
 *   // compute the rotation matrix with random_rotation
 *   raft::neighbors::ivf_pq::helpers::make_rotation_matrix(
 *     res, &index, index_params.force_random_rotation);
 * @endcode
 *
 * @tparam IdxT
 *
 * @param[in] res raft resource
 * @param[inout] index pointer to IVF-PQ index
 * @param[in] force_random_rotation whether to apply a random rotation matrix on the input data. See
 * raft::neighbors::ivf_pq::index_params for more details.
 */
template <typename IdxT>
void make_rotation_matrix(raft::resources const& res,
                          index<IdxT>* index,
                          bool force_random_rotation)
{
  raft::neighbors::ivf_pq::detail::make_rotation_matrix(res,
                                                        force_random_rotation,
                                                        index->rot_dim(),
                                                        index->dim(),
                                                        index->rotation_matrix().data_handle());
}

/**
 * @brief Public helper API for externally modifying the index's IVF centroids.
 * NB: The index must be reset before this. Use raft::neighbors::ivf_pq::extend to construct IVF
 lists according to new centroids.
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   // allocate the buffer for the input centers
 *   auto cluster_centers = raft::make_device_matrix<float, uint32_t>(res, index.n_lists(),
 index.dim());
 *   ... prepare ivf centroids in cluster_centers ...
 *   // reset the index
 *   reset_index(res, &index);
 *   // recompute the state of the index
 *   raft::neighbors::ivf_pq::helpers::recompute_internal_state(res, index);
 *   // Write the IVF centroids
 *   raft::neighbors::ivf_pq::helpers::set_centers(
                    res,
                    &index,
                    cluster_centers);
 * @endcode
 *
 * @tparam IdxT
 *
 * @param[in] res raft resource
 * @param[inout] index pointer to IVF-PQ index
 * @param[in] cluster_centers new cluster centers [index.n_lists(), index.dim()]
 */
template <typename IdxT>
void set_centers(raft::resources const& res,
                 index<IdxT>* index,
                 device_matrix_view<const float, uint32_t> cluster_centers)
{
  RAFT_EXPECTS(cluster_centers.extent(0) == index->n_lists(),
               "Number of rows in the new centers must be equal to the number of IVF lists");
  RAFT_EXPECTS(cluster_centers.extent(1) == index->dim(),
               "Number of columns in the new cluster centers and index dim are different");
  RAFT_EXPECTS(index->size() == 0, "Index must be empty");
  ivf_pq::detail::set_centers(res, index, cluster_centers.data_handle());
}

/**
 * @brief Helper exposing the re-computation of list sizes and related arrays if IVF lists have been
 * modified.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace raft::neighbors;
 *   raft::resources res;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // initialize an empty index
 *   ivf_pq::index<int64_t> index(res, index_params, D);
 *   ivf_pq::helpers::reset_index(res, &index);
 *   // resize the first IVF list to hold 5 records
 *   auto spec = list_spec<uint32_t, int64_t>{
 *     index->pq_bits(), index->pq_dim(), index->conservative_memory_allocation()};
 *   uint32_t new_size = 5;
 *   ivf::resize_list(res, list, spec, new_size, 0);
 *   raft::update_device(index.list_sizes(), &new_size, 1, stream);
 *   // recompute the internal state of the index
 *   ivf_pq::helpers::recompute_internal_state(res, &index);
 * @endcode
 *
 * @tparam IdxT
 *
 * @param[in] res raft resource
 * @param[inout] index pointer to IVF-PQ index
 */
template <typename IdxT>
void recompute_internal_state(const raft::resources& res, index<IdxT>* index)
{
  auto& list = index->lists()[0];
  ivf::detail::recompute_internal_state(res, *index);
}

/**
 * @brief Public helper API for fetching a trained index's IVF centroids into a buffer that may be
 * allocated on either host or device.
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   // allocate the buffer for the output centers
 *   auto cluster_centers = raft::make_device_matrix<float, uint32_t>(
 *     res, index.n_lists(), index.dim());
 *   // Extract the IVF centroids into the buffer
 *   raft::neighbors::ivf_pq::helpers::extract_centers(res, index, cluster_centers.data_handle());
 * @endcode
 *
 * @tparam IdxT
 *
 * @param[in] res raft resource
 * @param[in] index IVF-PQ index (passed by reference)
 * @param[out] cluster_centers IVF cluster centers [index.n_lists(), index.dim]
 */
template <typename IdxT>
void extract_centers(raft::resources const& res,
                     const index<IdxT>& index,
                     raft::device_matrix_view<float> cluster_centers)
{
  RAFT_EXPECTS(cluster_centers.extent(0) == index.n_lists(),
               "Number of rows in the output buffer for cluster centers must be equal to the "
               "number of IVF lists");
  RAFT_EXPECTS(
    cluster_centers.extent(1) == index.dim(),
    "Number of columns in the output buffer for cluster centers and index dim are different");
  auto stream = resource::get_cuda_stream(res);
  RAFT_CUDA_TRY(cudaMemcpy2DAsync(cluster_centers.data_handle(),
                                  sizeof(float) * index.dim(),
                                  index.centers().data_handle(),
                                  sizeof(float) * index.dim_ext(),
                                  sizeof(float) * index.dim(),
                                  index.n_lists(),
                                  cudaMemcpyDefault,
                                  stream));
}
/** @} */
}  // namespace raft::neighbors::ivf_pq::helpers
