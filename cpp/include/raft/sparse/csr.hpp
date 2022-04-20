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

template <typename vertex_t, typename edge_t, typename weight_t>
class csr_view_t {
public:
    csr_host_view_t(edge_t const* offsets,
                               vertex_t const* indices,
                               std::optional<weight_t const*> weights,
                               edge_t nnz)
            : offsets_(offsets), indices_(indices), weights_(weights), nnz_(nnz)
    {
    }

    edge_t nnz() const { return number_of_edges_; }

    edge_t const* offsets() const { return offsets_; }
    vertex_t const* indices() const { return indices_; }
    std::optional<weight_t const*> weights() const { return weights_; }

private:
    edge_t const* offsets_{nullptr};
    vertex_t const* indices_{nullptr};
    std::optional<weight_t const*> weights_{std::nullopt};
    edge_t nnz_{0};
};


template <typename vertex_t, typename edge_t, typename weight_t>
class csr_device_view_t {
public:
    csr_device_view_t(edge_t const* offsets,
               vertex_t const* indices,
               std::optional<weight_t const*> weights,
               edge_t nnz)
            : offsets_(offsets),
              indices_(indices),
              weights_(weights ? thrust::optional<weight_t const*>(*weights) : thrust::nullopt), nnz_(nnz) {}

    __host__ __device__ edge_t nnz() const { return nnz_; }
    __host__ __device__ edge_t const* offsets() const { return offsets_; }
    __host__ __device__ vertex_t const* indices() const { return indices_; }
    __host__ __device__ thrust::optional<weight_t const*> weights() const { return weights_; }

    // major_idx == major offset if CSR/CSC, major_offset != major_idx if DCSR/DCSC
    __device__ thrust::tuple<vertex_t const*, thrust::optional<weight_t const*>, edge_t> local_edges(
            vertex_t major_idx) const noexcept
    {
        auto edge_offset  = *(offsets_ + major_idx);
        auto local_degree = *(offsets_ + (major_idx + 1)) - edge_offset;
        auto indices      = indices_ + edge_offset;
        auto weights =
        weights_ ? thrust::optional<weight_t const*>{*weights_ + edge_offset} : thrust::nullopt;
        return thrust::make_tuple(indices, weights, local_degree);
    }

    // major_idx == major offset if CSR/CSC, major_offset != major_idx if DCSR/DCSC
    __device__ edge_t local_degree(vertex_t major_idx) const noexcept
    {
        return *(offsets_ + (major_idx + 1)) - *(offsets_ + major_idx);
    }

    // major_idx == major offset if CSR/CSC, major_offset != major_idx if DCSR/DCSC
    __device__ edge_t local_offset(vertex_t major_idx) const noexcept
    {
        return *(offsets_ + major_idx);
    }

private:
    // should be trivially copyable to device
    edge_t const* offsets_{nullptr};
    vertex_t const* indices_{nullptr};
    thrust::optional<weight_t const*> weights_{thrust::nullopt};
    edge_t nnz_{0};
};


};  // namespace sparse
};  // namespace raft
