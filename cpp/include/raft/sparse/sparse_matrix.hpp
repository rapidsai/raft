/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <raft/core/mdarray.hpp>

#include <optional>
#include <tuple>

#pragma once

#ifndef _RAFT_SPARSE_HD
#ifdef __NVCC__
#define _RAFT_SPARSE_HD __host__ __device__
#else
#define _RAFT_SPARSE_HD
#endif
#endif
namespace raft {

template<typename T, ContainerPolicy>
using array_t = raft::mdarray<T, raft::detail::vector_extent, raft::layout_c_contiguous, ContainerPolicy>;

// TODO: We need to be able to accept anything which is an mdspan<T, vector_extent>
template<typename T>
using array_view_t = raft::mdspan<T, raft::detail::vector_extent>;

namespace sparse {



/**
 * Compressed sparse view object which can be used to represent an adjacency list for both
 * graph and bigraph in (b)csr/csc formats. The distinction here is made very clear-
 * since in libraries like cuGraph, the matrix is almost always assumed to be square
 * (e.g. nvec == vdim) but in many uses in cuML, the matrix might not be square.
 *
 * This object can be used to represent structural associations (e.g. no values array)
 * or weighed / valued associations.
 *
 * @tparam idx_t
 * @tparam offset_t
 * @tparam value_t
 *
 * @param [in] nvec total number of vectors for axis==0 (e.g. rows for csr, cols for csc)
 * @param [in] vdim total number of dimensions (e.g. cols for csr, rows for csc)
 * @param [in] offsets pointer of offsets into indices array for each vector idx
 * @param [in] indices pointer of indices from vector offsets (size nnz)
 * @param [in] values optional pointer of values for each vector offset (size nnz)
 * @param [in] nnz number of nonzero offsets
 *
 */
template <typename idx_t, typename offset_t, typename value_t>
class comp_sparse_view_t {
 public:
    comp_sparse_view_t(idx_t nvec,  // total number of row vectors for csr or col vectors for csc
                        idx_t vdim,  // total number of col vectors for csr or row vectors for csc
                        array_view_t<offset_t> const offsets,
                        array_view_t<idx_t> const indices,
                        std::optional<array_view_t<value_t> const> values,
                        offset_t nnz)
    : offsets_(offsets),
      indices_(indices),
      values_(values.has_value() ? std::optional<array_view_t<value_t> const>(*values) : std::nullopt),
      nvec_(nvec),
      vdim_(vdim),
      nnz_(nnz)
  {
  }

  /**
   * Using blas terminology here to abstract away the notion of graph
   */
    _RAFT_SPARSE_HD idx_t nvec() const { return nvec_; }
    _RAFT_SPARSE_HD idx_t vdim() const { return vdim_; }

    _RAFT_SPARSE_HD offset_t nnz() const { return nnz_; }

    _RAFT_SPARSE_HD array_view_t<offset_t> const offsets() const { return offsets_; }
    _RAFT_SPARSE_HD array_view_t<idx_t> const indices() const { return indices_; }
    _RAFT_SPARSE_HD std::optional<array_view_t<value_t> const> values() const { return values_; }

  // major_idx == major offset if CSR/CSC, major_offset != major_idx if DCSR/DCSC
    _RAFT_SPARSE_HD std::tuple<idx_t const*, std::optional<value_t const*>, offset_t> local_vecs(
    idx_t major_idx) const noexcept
  {
    auto offset  = local_offset(major_idx);
    auto degree  = local_offset(major_idx + 1) - offset;
    auto indices = indices_ + offset;
    auto values  = values_ ? std::optional<value_t const*>{*values_ + offset} : std::nullopt;
    return std::make_tuple(indices, values, degree);
  }

  // major_idx == major offset if CSR/CSC, major_offset != major_idx if DCSR/DCSC
    _RAFT_SPARSE_HD offset_t local_nnz(idx_t major_idx) const noexcept
  {
    return *(offsets_ + (major_idx + 1)) - *(offsets_ + major_idx);
  }

  // major_idx == major offset if CSR/CSC, major_offset != major_idx if DCSR/DCSC
    _RAFT_SPARSE_HD offset_t local_offset(idx_t major_idx) const noexcept
  {
    return *(offsets_ + major_idx);
  }

 private:
  // TODO: Use span/mdspan here
  // should be trivially copyable to device
  idx_t nvec_{0};
  idx_t vdim_{0};
  array_view_t<offset_t> const offsets_{nullptr};
  array_view_t<idx_t> const indices_{nullptr};
  std::optional<array_view_t<value_t> const> values_{std::nullopt};
  offset_t nnz_{0};
};

/**
 * Sparse object which can be used to represent an edge list or sparse matrix in coo format.
 *
 * From a graph perspective: This is a simple edge list that supports both graphs and bigraphs
 * and allows both the source and destination vertices to have different indexing data types.
 *
 * From a linear algebra perspective: This supports both square and non-square sparse matrices in
 * the coordinate (COO) format, allowing for different indexing data types for both the row and
 * column indices.
 *
 * @tparam src_idx_t
 * @tparam dst_idx_t
 * @tparam value_t
 *
 * @param [in] nvec total number of vectors for axis==0 (e.g. rows for csr, cols for csc)
 * @param [in] vdim total number of dimensions (e.g. cols for csr, rows for csc)
 * @param [in] src pointer of source indices for each edge (size nnz)
 * @param [in] dst pointer of destination indices for each edge (size nnz)
 * @param [in] values optional pointer of values or weights for each edge (size nnz)
 * @param [in] nnz number of nonzero values or edges
 *
 */
template <typename src_idx_t, typename dst_idx_t, typename value_t>
class coo_device_view_t {
 public:
  // TODO: Use a strategy / policy to separate host from device views
  coo_view_t(src_idx_t nvec,         // total number of source vectors
                    dst_idx_t vdim,  // total number of destination vectors
                    array_view_t<src_idx_t> const src,
                    array_view_t<dst_idx_t> const dst,
                    std::optional<array_view_t<value_t> const> values,
                    size_t nnz)
    : src_(src),
      dst_(dst),
      values_(values),
      nvec_(nvec),
      vdim_(vdim),
      nnz_(nnz)
  {
  }

  /**
   * Using blas terminology here to abstract away the notion of graph
   */
    _RAFT_SPARSE_HD src_idx_t nvec() const { return nvec_; }
    _RAFT_SPARSE_HD dst_idx_t vdim() const { return vdim_; }

    _RAFT_SPARSE_HD size_t nnz() const { return nnz_; }

    _RAFT_SPARSE_HD array_view_t<src_idx_t> const src() const { return src_; }
    _RAFT_SPARSE_HD array_view_t<dst_idx_t> const dst() const { return dst_; }
    _RAFT_SPARSE_HD std::optional<array_view_t<value_t> const> values() const { return values_; }

 private:
  // TODO: Use span/mdspan here
  src_idx_t nvec_{0};
  dst_idx_t vdim_{0};
  array_view_t<src_idx_t> const src_;
  array_view_t<dst_idx_t> const dst_{nullptr};
  std::optional<array_view_t<value_t> const> values_{std::nullopt};
  offset_t nnz_{0};
};



/**
 * Compressed sparse object which can be used to represent an adjacency list for both
 * graph and bigraph in (b)csr/csc formats. The distinction here is made very clear-
 * since in libraries like cuGraph, the matrix is almost always assumed to be square
 * (e.g. nvec == vdim) but in many uses in cuML, the matrix might not be square.
 *
 * This object can be used to represent structural associations (e.g. no values array)
 * or weighed / valued associations.
 *
 * @tparam idx_t
 * @tparam offset_t
 * @tparam value_t
 *
 * @param [in] nvec total number of vectors for axis==0 (e.g. rows for csr, cols for csc)
 * @param [in] vdim total number of dimensions (e.g. cols for csr, rows for csc)
 * @param [in] offsets pointer of offsets into indices array for each vector idx
 * @param [in] indices pointer of indices from vector offsets (size nnz)
 * @param [in] values optional pointer of values for each vector offset (size nnz)
 * @param [in] nnz number of nonzero offsets
 *
 */
template <typename idx_t, typename offset_t, typename value_t, typename ContainerPolicy>
class comp_sparse_t {
public:
    comp_sparse_t(idx_t nvec,  // total number of row vectors for csr or col vectors for csc
                       idx_t vdim,  // total number of col vectors for csr or row vectors for csc
                       bool use_values = false,
                       offset_t nnz) : nvec_(nvec), vdim_(vdim), nnz_(nnz) {}

    /**
     * Using blas terminology here to abstract away the notion of graph
     */
    _RAFT_SPARSE_HD idx_t nvec() const { return nvec_; }
    _RAFT_SPARSE_HD idx_t vdim() const { return vdim_; }

    _RAFT_SPARSE_HD offset_t nnz() const { return nnz_; }

    _RAFT_SPARSE_HD array_view_t<offset_t> const offsets() const { return offsets_; }
    _RAFT_SPARSE_HD array_view_t<idx_t> const indices() const { return indices_; }
    _RAFT_SPARSE_HD std::optional<array_view_t<value_t> const> values() const { return values_; }

    // major_idx == major offset if CSR/CSC, major_offset != major_idx if DCSR/DCSC
    _RAFT_SPARSE_HD std::tuple<idx_t const*, std::optional<value_t const*>, offset_t> local_vecs(
            idx_t major_idx) const noexcept
    {
        auto offset  = local_offset(major_idx);
        auto degree  = local_offset(major_idx + 1) - offset;
        auto indices = indices_ + offset;
        auto values  = values_ ? std::optional<value_t const*>{*values_ + offset} : std::nullopt;
        return std::make_tuple(indices, values, degree);
    }

    // major_idx == major offset if CSR/CSC, major_offset != major_idx if DCSR/DCSC
    _RAFT_SPARSE_HD offset_t local_nnz(idx_t major_idx) const noexcept
    {
        return *(offsets_ + (major_idx + 1)) - *(offsets_ + major_idx);
    }

    // major_idx == major offset if CSR/CSC, major_offset != major_idx if DCSR/DCSC
    _RAFT_SPARSE_HD offset_t local_offset(idx_t major_idx) const noexcept
    {
        return *(offsets_ + major_idx);
    }

private:
    bool use_values{false};
    idx_t nvec_{0};
    idx_t vdim_{0};
    array_t<offset_t, ContainerPolicy> const offsets_;
    array_t<idx_t, ContainerPolicy> const indices_;
    std::optional<array_t<value_t, ContainerPolicy> const> values_{std::nullopt};
    offset_t nnz_{0};
};

};  // namespace sparse
};  // namespace raft
