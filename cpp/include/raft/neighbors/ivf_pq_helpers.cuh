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

#include <cstdio>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/neighbors/detail/ivf_pq_build.cuh>
#include <raft/neighbors/ivf_pq_types.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <raft/spatial/knn/detail/ann_utils.cuh>

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
 * @brief Unpack `n_take` consecutive records of a single list (cluster) in the compressed index
 * starting at given `offset`. The output codes are compressed, which means the output has
 * ceildiv(pq_dim * pq_bits, 8) bytes of pq codes.
 *
 * Usage example:
 * @code{.cpp}
 *   auto list_data = index.lists()[label]->data.view();
 *   // allocate the buffer for the output
 *   uint32_t n_take = 4;
 *   auto codes = raft::make_device_vector<uint8_t>(res, raft::ceildiv(n_take * index.pq_dim() *
 * index.pq_bits(), 8)); uint32_t offset = 0;
 *   // unpack n_take elements from the list
 *   ivf_pq::helpers::codepacker::unpack(res, list_data, index.pq_bits(), offset, n_take,
 * index.pq_dim(), codes.data_handle());
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
 *   the destination buffer [ceildiv(n_rows * pq_dim * pq_bits, 8)].
 *   The length `n_rows` defines how many records to unpack,
 *   it must be smaller than the list size.
 */
inline void unpack_compressed(
  raft::resources const& res,
  device_mdspan<const uint8_t, list_spec<uint32_t, uint32_t>::list_extents, row_major> list_data,
  uint32_t pq_bits,
  uint32_t offset,
  uint32_t n_rows,
  uint32_t pq_dim,
  uint8_t* codes)
{
  ivf_pq::detail::unpack_compressed_list_data(
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
 * @param[in] res
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
 * Write flat PQ codes into an existing list by the given offset. The input codes are compressed
 * (not expanded to one code per byte).
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
 * @param[in] codes flat PQ codes, [n_vec, ceildiv(pq_dim * pq_bits, 8)]
 * @param[in] n_rows number of records
 * @param[in] pq_dim
 * @param[in] pq_bits bit length of encoded vector elements
 * @param[in] offset how many records to skip before writing the data into the list
 * @param[in] list_data block to write into
 */
inline void pack_compressed(
  raft::resources const& res,
  const uint8_t* codes,
  uint32_t n_rows,
  uint32_t pq_dim,
  uint32_t pq_bits,
  uint32_t offset,
  device_mdspan<uint8_t, list_spec<uint32_t, uint32_t>::list_extents, row_major> list_data)
{
  ivf_pq::detail::pack_compressed_list_data(
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
 * @param[in] res
 * @param[inout] index IVF-PQ index.
 * @param[in] codes flat PQ codes, one code per byte [n_rows, pq_dim]
 * @param[in] label The id of the list (cluster) into which we write.
 * @param[in] offset how many records to skip before writing the data into the list
 */
template <typename IdxT>
void pack_list_data(raft::resources const& res,
                    index<IdxT>* index,
                    uint8_t* codes,
                    uint32_t n_rows,
                    uint32_t label,
                    uint32_t offset)
{
  ivf_pq::detail::pack_list_data(res, index, codes, n_rows, label, offset);
}

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
void pack_list_data(raft::resources const& res,
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
                      uint8_t* out_codes,
                      uint32_t n_rows,
                      uint32_t label,
                      uint32_t offset)
{
  return ivf_pq::detail::unpack_list_data<IdxT>(res, index, out_codes, n_rows, label, offset);
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
void unpack_list_data(raft::resources const& res,
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
 *   raft::copy(&list_size, index.list_sizes().data_handle() + label, 1,
 * resource::get_cuda_stream(res)); resource::sync_stream(res);
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
 * @brief Public helper API for computing the index's rotation matrix.
 *
 * @tparam IdxT
 * @param[in] res
 * @param[inout] index
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
 *
 * @tparam IdxT
 * @param[in] res
 * @param[inout] index
 * @param[in] cluster_centers the new cluster centers
 */
template <typename IdxT>
void set_centers(raft::resources const& res,
                 index<IdxT>* index,
                 device_matrix_view<const float, uint32_t> cluster_centers)
{
  RAFT_EXPECTS(cluster_centers.extent(0) == index->n_lists(),
               "Number of rows in cluster centers and IVF centers are different");
  RAFT_EXPECTS(cluster_centers.extent(1) == index->dim(),
               "Number of columns in cluster centers and index dim are different");
  ivf_pq::detail::set_centers(res, index, cluster_centers.data_handle());
}

template <typename IdxT>
void set_pq_centers(raft::resources const& res, index<IdxT>* index, float* pq_centers)
{
  ivf_pq::detail::transpose_pq_centers(res, *index, pq_centers);
}

template <typename IdxT>
auto get_list_size_in_bytes(const index<IdxT>* index, uint32_t label) -> uint32_t
{
  RAFT_EXPECTS(label < index->n_lists(),
               "Expected label to be less than number of lists in the index");
  auto list_data = index->lists()[label]->data;
  return list_data.size();
}

template <typename IdxT>
void recompute_internal_state(const raft::resources& res, index<IdxT>* index)
{
  ivf_pq::detail::recompute_internal_state(res, *index);
}

template <typename IdxT>
void reset_index(const raft::resources& res, index<IdxT>& index)
{
  auto stream = resource::get_cuda_stream(res);

  utils::memzero(
    index.accum_sorted_sizes().data_handle(), index.accum_sorted_sizes().size(), stream);
  utils::memzero(index.list_sizes().data_handle(), index.list_sizes().size(), stream);
  utils::memzero(index.data_ptrs().data_handle(), index.data_ptrs().size(), stream);
  utils::memzero(index.inds_ptrs().data_handle(), index.inds_ptrs().size(), stream);
}
/** @} */
}  // namespace raft::neighbors::ivf_pq::helpers
