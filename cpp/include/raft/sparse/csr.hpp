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

#pragma once

#include <thrust/optional.h>
#include <thrust/tuple.h>

#include <raft/sparse/detail/csr.cuh>

namespace raft {
namespace sparse {

constexpr int TPB_X = 256;

using WeakCCState = detail::WeakCCState;

/**
 * @brief Partial calculation of the weakly connected components in the
 * context of a batched algorithm: the labels are computed wrt the sub-graph
 * represented by the given CSR matrix of dimensions batch_size * N.
 * Note that this overwrites the labels array and it is the responsibility of
 * the caller to combine the results from different batches
 * (cf label/merge_labels.cuh)
 *
 * @tparam Index_ the numeric type of non-floating point elements
 * @tparam TPB_X the threads to use per block when configuring the kernel
 * @param labels an array for the output labels
 * @param row_ind the compressed row index of the CSR array
 * @param row_ind_ptr the row index pointer of the CSR array
 * @param nnz the size of row_ind_ptr array
 * @param N number of vertices
 * @param start_vertex_id the starting vertex index for the current batch
 * @param batch_size number of vertices for current batch
 * @param state instance of inter-batch state management
 * @param stream the cuda stream to use
 * @param filter_op an optional filtering function to determine which points
 * should get considered for labeling. It gets global indexes (not batch-wide!)
 */
template <typename Index_, typename Lambda>
void weak_cc_batched(Index_* labels,
                     const Index_* row_ind,
                     const Index_* row_ind_ptr,
                     Index_ nnz,
                     Index_ N,
                     Index_ start_vertex_id,
                     Index_ batch_size,
                     WeakCCState* state,
                     cudaStream_t stream,
                     Lambda filter_op)
{
  detail::weak_cc_batched<Index_, TPB_X, Lambda>(
    labels, row_ind, row_ind_ptr, nnz, N, start_vertex_id, batch_size, state, stream, filter_op);
}

/**
 * @brief Partial calculation of the weakly connected components in the
 * context of a batched algorithm: the labels are computed wrt the sub-graph
 * represented by the given CSR matrix of dimensions batch_size * N.
 * Note that this overwrites the labels array and it is the responsibility of
 * the caller to combine the results from different batches
 * (cf label/merge_labels.cuh)
 *
 * @tparam Index_ the numeric type of non-floating point elements
 * @tparam TPB_X the threads to use per block when configuring the kernel
 * @param labels an array for the output labels
 * @param row_ind the compressed row index of the CSR array
 * @param row_ind_ptr the row index pointer of the CSR array
 * @param nnz the size of row_ind_ptr array
 * @param N number of vertices
 * @param start_vertex_id the starting vertex index for the current batch
 * @param batch_size number of vertices for current batch
 * @param state instance of inter-batch state management
 * @param stream the cuda stream to use
 */
template <typename Index_>
void weak_cc_batched(Index_* labels,
                     const Index_* row_ind,
                     const Index_* row_ind_ptr,
                     Index_ nnz,
                     Index_ N,
                     Index_ start_vertex_id,
                     Index_ batch_size,
                     WeakCCState* state,
                     cudaStream_t stream)
{
  weak_cc_batched(labels,
                  row_ind,
                  row_ind_ptr,
                  nnz,
                  N,
                  start_vertex_id,
                  batch_size,
                  state,
                  stream,
                  [] __device__(Index_ tid) { return true; });
}

/**
 * @brief Compute weakly connected components. Note that the resulting labels
 * may not be taken from a monotonically increasing set (eg. numbers may be
 * skipped). The MLCommon::Label package contains a primitive `make_monotonic`,
 * which will make a monotonically increasing set of labels.
 *
 * This implementation comes from [1] and solves component labeling problem in
 * parallel on CSR-indexes based upon the vertex degree and adjacency graph.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 *
 * @tparam Type the numeric type of non-floating point elements
 * @tparam TPB_X the threads to use per block when configuring the kernel
 * @tparam Lambda the type of an optional filter function (int)->bool
 * @param labels an array for the output labels
 * @param row_ind the compressed row index of the CSR array
 * @param row_ind_ptr the row index pointer of the CSR array
 * @param nnz the size of row_ind_ptr array
 * @param N number of vertices
 * @param stream the cuda stream to use
 * @param filter_op an optional filtering function to determine which points
 * should get considered for labeling. It gets global indexes (not batch-wide!)
 */
template <typename Index_ = int, typename Lambda>
void weak_cc(Index_* labels,
             const Index_* row_ind,
             const Index_* row_ind_ptr,
             Index_ nnz,
             Index_ N,
             cudaStream_t stream,
             Lambda filter_op)
{
  rmm::device_scalar<bool> m(stream);
  WeakCCState state(m.data());
  weak_cc_batched<Index_, TPB_X>(labels, row_ind, row_ind_ptr, nnz, N, 0, N, stream, filter_op);
}

/**
 * @brief Compute weakly connected components. Note that the resulting labels
 * may not be taken from a monotonically increasing set (eg. numbers may be
 * skipped). The MLCommon::Label package contains a primitive `make_monotonic`,
 * which will make a monotonically increasing set of labels.
 *
 * This implementation comes from [1] and solves component labeling problem in
 * parallel on CSR-indexes based upon the vertex degree and adjacency graph.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 *
 * @tparam Type the numeric type of non-floating point elements
 * @tparam TPB_X the threads to use per block when configuring the kernel
 * @tparam Lambda the type of an optional filter function (int)->bool
 * @param labels an array for the output labels
 * @param row_ind the compressed row index of the CSR array
 * @param row_ind_ptr the row index pointer of the CSR array
 * @param nnz the size of row_ind_ptr array
 * @param N number of vertices
 * @param stream the cuda stream to use
 */
template <typename Index_>
void weak_cc(Index_* labels,
             const Index_* row_ind,
             const Index_* row_ind_ptr,
             Index_ nnz,
             Index_ N,
             cudaStream_t stream)
{
  rmm::device_scalar<bool> m(stream);
  WeakCCState state(m.data());
  weak_cc_batched<Index_, TPB_X>(
    labels, row_ind, row_ind_ptr, nnz, N, 0, N, stream, [](Index_) { return true; });
}

template <typename idx_t, typename offset_t, typename value_t>
class sparse_host_view_t {
public:
    sparse_host_view_t(offset_t const* offsets,
                               idx_t const* indices,
                               std::optional<value_t const*> weights,
                               offset_t nnz)
            : offsets_(offsets), indices_(indices), weights_(weights), nnz_(nnz)
    {
    }

    offset_t nnz() const { return number_of_edges_; }

    offset_t const* offsets() const { return offsets_; }
    idx_t const* indices() const { return indices_; }
    std::optional<value_t const*> weights() const { return weights_; }

private:
    offset_t const* offsets_{nullptr};
    idx_t const* indices_{nullptr};
    std::optional<value_t const*> weights_{std::nullopt};
    offset_t nnz_{0};
};


template <typename idx_t, typename offset_t, typename value_t>
class sparse_device_view_t {
public:
    sparse_device_view_t(
               idx_t nvec,   // total number of row vectors for csr or col vectors for csc
               idx_t vdim,   // total number of col vectors for csr or row vectors for csc
               offset_t const* offsets,
               idx_t const* indices,
               std::optional<value_t const*> values,
               offset_t nnz)
            : offsets_(offsets),
              indices_(indices),
              values_(values_ ? thrust::optional<value_t const*>(*values) : thrust::nullopt),
              nvec_(nvec),
              vdim_(vdim),
              nnz_(nnz) {}

    __host__ __device__ offset_t nnz() const { return nnz_; }
    __host__ __device__ offset_t const* offsets() const { return offsets_; }
    __host__ __device__ idx_t const* indices() const { return indices_; }
    __host__ __device__ thrust::optional<value_t const*> values() const { return values_; }

    // major_idx == major offset if CSR/CSC, major_offset != major_idx if DCSR/DCSC
    __device__ thrust::tuple<idx_t const*, thrust::optional<value_t const*>, offset_t> local_vecs(
            idx_t major_idx) const noexcept
    {
        auto offset = local_offset(major_idx);
        auto degree = local_offset(major_idx+1) - offset;
        auto indices      = indices_ + offset;
        auto values =
        values_ ? thrust::optional<value_t const*>{*values_ + offset} : thrust::nullopt;
        return thrust::make_tuple(indices, values, degree);
    }

    // major_idx == major offset if CSR/CSC, major_offset != major_idx if DCSR/DCSC
    __device__ offset_t local_degree(idx_t major_idx) const noexcept
    {
        return *(offsets_ + (major_idx + 1)) - *(offsets_ + major_idx);
    }

    // major_idx == major offset if CSR/CSC, major_offset != major_idx if DCSR/DCSC
    __device__ offset_t local_offset(idx_t major_idx) const noexcept
    {
        return *(offsets_ + major_idx);
    }

private:

    // TODO: Use span/mdspan here
    // should be trivially copyable to device
    idx_t nvec_{0};
    idx_t vdim_{0};
    offset_t const* offsets_{nullptr};
    idx_t const* indices_{nullptr};
    thrust::optional<value_t const*> values_{thrust::nullopt};
    offset_t nnz_{0};
};


};  // namespace sparse
};  // namespace raft
