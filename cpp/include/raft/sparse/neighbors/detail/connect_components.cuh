/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

#include "raft/core/logger-macros.hpp"
#include <cstdint>
#include <cub/cub.cuh>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/distance/masked_nn.cuh>
#include <raft/label/classlabels.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/matrix/scatter.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/linalg/symmetrize.cuh>
#include <raft/sparse/op/reduce.cuh>

#include <raft/util/cudart_utils.hpp>
#include <raft/util/fast_int_div.cuh>
#include <rmm/device_uvector.hpp>

#include <raft/core/kvp.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <cub/cub.cuh>

#include <limits>

namespace raft::sparse::neighbors::detail {

/**
 * Functor with reduction ops for performing masked 1-nn
 * computation. this change introduces a breaking change to
 * the public API because colors are no longer a part of this
 * op. The connect_components function internally ensures that
 * only cross-component nearest neighbors are found.
 * @tparam value_idx
 * @tparam value_t
 */
template <typename value_idx, typename value_t>
struct FixConnectivitiesRedOp {
  value_idx m;

  // default constructor for cutlass
  DI FixConnectivitiesRedOp() : m(0) {}

  FixConnectivitiesRedOp(value_idx m_) : m(m_){};

  typedef typename raft::KeyValuePair<value_idx, value_t> KVP;
  DI void operator()(value_idx rit, KVP* out, const KVP& other) const
  {
    if (rit < m && other.value < out->value) {
      out->key   = other.key;
      out->value = other.value;
    }
  }

  DI KVP operator()(value_idx rit, const KVP& a, const KVP& b)
  {
    if (rit < m && a.value < b.value) {
      return a;
    } else
      return b;
  }

  DI void init(value_t* out, value_t maxVal) const { *out = maxVal; }
  DI void init(KVP* out, value_t maxVal) const
  {
    out->key   = -1;
    out->value = maxVal;
  }

  DI void init_key(value_t& out, value_idx idx) const { return; }
  DI void init_key(KVP& out, value_idx idx) const { out.key = idx; }

  DI value_t get_value(KVP& out) const { return out.value; }

  DI value_t get_value(value_t& out) const { return out; }

  void gather(raft::resources const& handle, value_idx* map) {}

  void scatter(raft::resources const& handle, value_idx* map) {}
};

/**
 * Assumes 3-iterator tuple containing COO rows, cols, and
 * a cub keyvalue pair object. Sorts the 3 arrays in
 * ascending order: row->col->keyvaluepair
 */
struct TupleComp {
  template <typename one, typename two>
  __host__ __device__ bool operator()(const one& t1, const two& t2)
  {
    // sort first by each sample's color,
    if (thrust::get<0>(t1) < thrust::get<0>(t2)) return true;
    if (thrust::get<0>(t1) > thrust::get<0>(t2)) return false;

    // then by the color of each sample's closest neighbor,
    if (thrust::get<1>(t1) < thrust::get<1>(t2)) return true;
    if (thrust::get<1>(t1) > thrust::get<1>(t2)) return false;

    // then sort by value in descending order
    return thrust::get<2>(t1).value < thrust::get<2>(t2).value;
  }
};

template <typename LabelT, typename DataT>
struct CubKVPMinReduce {
  typedef raft::KeyValuePair<LabelT, DataT> KVP;

  DI KVP

  operator()(LabelT rit, const KVP& a, const KVP& b)
  {
    return b.value < a.value ? b : a;
  }

  DI KVP

  operator()(const KVP& a, const KVP& b)
  {
    return b.value < a.value ? b : a;
  }

};  // KVPMinReduce

/**
 * Gets the number of unique components from array of
 * colors or labels. This does not assume the components are
 * drawn from a monotonically increasing set.
 * @tparam value_idx
 * @param[in] colors array of components
 * @param[in] n_rows size of components array
 * @param[in] stream cuda stream for which to order cuda operations
 * @return total number of components
 */
template <typename value_idx>
value_idx get_n_components(value_idx* colors, size_t n_rows, cudaStream_t stream)
{
  rmm::device_uvector<value_idx> map_ids(0, stream);
  int num_clusters = raft::label::getUniquelabels(map_ids, colors, n_rows, stream);
  return num_clusters;
}

/**
 * Functor to look up a component for a vertex
 * @tparam value_idx
 * @tparam value_t
 */
template <typename value_idx, typename value_t>
struct LookupColorOp {
  value_idx* colors;

  LookupColorOp(value_idx* colors_) : colors(colors_) {}

  DI value_idx

  operator()(const raft::KeyValuePair<value_idx, value_t>& kvp)
  {
    return colors[kvp.key];
  }
};

/**
 * Compute the cross-component 1-nearest neighbors for each row in X using
 * the given array of components
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle
 * @param[out] kvp mapping of closest neighbor vertex and distance for each vertex in the given
 * array of components
 * @param[out] nn_colors components of nearest neighbors for each vertex
 * @param[in] colors components of each vertex
 * @param[in] X original dense data
 * @param[in] n_rows number of rows in original dense data
 * @param[in] n_cols number of columns in original dense data
 * @param[in] batch_size batch size for computing nearest neighbors
 * @param[in] reduction_op reduction operation for computing nearest neighbors
 */
template <typename value_idx, typename value_t, typename red_op>
void perform_1nn(raft::resources const& handle,
                 raft::KeyValuePair<value_idx, value_t>* kvp,
                 value_idx* nn_colors,
                 value_idx* colors,
                 const value_t* X,
                 size_t n_rows,
                 size_t n_cols,
                 size_t batch_size,
                 red_op reduction_op)
{
  RAFT_LOG_INFO("perform_1nn start");
  auto stream      = resource::get_cuda_stream(handle);
  auto exec_policy = resource::get_thrust_policy(handle);

  value_idx n_components = get_n_components(colors, n_rows, stream);

  // colors_group_idxs is an array containing the *end* indices of each color
  // component in colors. That is, the value of colors_group_idxs[j] indicates
  // the start of color j + 1, i.e., it is the inclusive scan of the sizes of
  // the color components.
  auto colors_group_idxs = raft::make_device_vector<value_idx, value_idx>(handle, n_components + 1);
  raft::sparse::convert::sorted_coo_to_csr(
    colors, n_rows, colors_group_idxs.data_handle(), n_components + 1, stream);
  
  auto group_idxs_view = raft::make_device_vector_view(colors_group_idxs.data_handle() + 1, n_components);

  auto x_norm = raft::make_device_vector<value_t, value_idx>(handle, (value_idx)n_rows);
  raft::linalg::rowNorm(
    x_norm.data_handle(), X, n_cols, n_rows, raft::linalg::L2Norm, true, stream);
  
  RAFT_LOG_INFO("X norm computed");

  auto adj     = raft::make_device_matrix<bool, value_idx>(handle, batch_size, n_components);
  using OutT   = raft::KeyValuePair<value_idx, value_t>;
  using ParamT = raft::distance::masked_l2_nn_params<red_op, red_op>;

  bool apply_sqrt      = true;
  bool init_out_buffer = true;
  ParamT params{reduction_op, reduction_op, apply_sqrt, init_out_buffer};

  auto X_full_view = raft::make_device_matrix_view<const value_t, value_idx>(
      X, n_rows, n_cols);

  size_t n_batches = raft::ceildiv(n_rows, batch_size);
  RAFT_LOG_INFO("n_batches %zu", n_batches);
  for (size_t bid = 0; bid < n_batches; bid++) {
    RAFT_LOG_INFO("current batch bid %zu", bid);
    size_t batch_offset   = bid * batch_size;
    size_t rows_per_batch = min(batch_size, n_rows - batch_offset);
    RAFT_LOG_INFO("rows_per_batch %zu", rows_per_batch);

    auto X_batch_view = raft::make_device_matrix_view<const value_t, value_idx>(
      X + batch_offset * n_cols, rows_per_batch, n_cols);
    
    RAFT_LOG_INFO("X_batch_view created");

    auto x_norm_batch_view = raft::make_device_vector_view<value_t, value_idx>(
      x_norm.data_handle() + batch_offset, rows_per_batch);
    
    RAFT_LOG_INFO("X_norm_batch_view created");
    auto mask_op = [colors,
                    n_components = raft::util::FastIntDiv(n_components),
                    batch_offset] __device__(value_idx idx) {
      value_idx row = idx / n_components;
      value_idx col = idx % n_components;
      return colors[batch_offset + row] != col;
    };
    auto adj_view = raft::make_device_matrix_view<bool, value_idx>(
      adj.data_handle(), rows_per_batch, n_components);
    
    RAFT_LOG_INFO("adj view created");
    raft::linalg::map_offset(handle, adj_view, mask_op);

    RAFT_LOG_INFO("adj map_offset done");
    auto kvp_view =
    raft::make_device_vector_view<raft::KeyValuePair<value_idx, value_t>, value_idx>(kvp + batch_offset, rows_per_batch);

    RAFT_LOG_INFO("kvp view created");
    cudaDeviceSynchronize();
    RAFT_LOG_INFO("bid %d Done until start of masked_nn", bid);
    raft::distance::masked_l2_nn<value_t, OutT, value_idx, red_op, red_op>(
      handle,
      params,
      X_batch_view,
      X_full_view,
      x_norm_batch_view,
      x_norm.view(),
      adj_view,
      group_idxs_view,
      kvp_view);
    cudaDeviceSynchronize();
    RAFT_LOG_INFO("bid %d Done until end of masked_nn", bid);
  }
  LookupColorOp<value_idx, value_t> extract_colors_op(colors);
  thrust::transform(exec_policy, kvp, kvp + n_rows, nn_colors, extract_colors_op);
}

/**
 * Sort nearest neighboring components wrt component of source vertices
 * @tparam value_idx
 * @tparam value_t
 * @param[inout] colors components array of source vertices
 * @param[inout] nn_colors nearest neighbors components array
 * @param[inout] kvp nearest neighbor source vertex / distance array
 * @param[inout] src_indices array of source vertex indices which will become arg_sort
 *               indices
 * @param n_rows number of components in `colors`
 * @param stream stream for which to order CUDA operations
 */
template <typename value_idx, typename value_t>
void sort_by_color(raft::resources const& handle,
                   value_idx* colors,
                   value_idx* nn_colors,
                   raft::KeyValuePair<value_idx, value_t>* kvp,
                   value_idx* src_indices,
                   size_t n_rows)
{
  auto exec_policy = resource::get_thrust_policy(handle);
  thrust::counting_iterator<value_idx> arg_sort_iter(0);
  thrust::copy(exec_policy, arg_sort_iter, arg_sort_iter + n_rows, src_indices);

  auto keys = thrust::make_zip_iterator(
    thrust::make_tuple(colors, nn_colors, (KeyValuePair<value_idx, value_t>*)kvp));
  auto vals = thrust::make_zip_iterator(thrust::make_tuple(src_indices));
  // get all the colors in contiguous locations so we can map them to warps.
  thrust::sort_by_key(exec_policy, keys, keys + n_rows, vals, TupleComp());
}

template <typename value_idx, typename value_t>
__global__ void min_components_by_color_kernel(value_idx* out_rows,
                                               value_idx* out_cols,
                                               value_t* out_vals,
                                               const value_idx* out_index,
                                               const value_idx* indices,
                                               const value_idx* sort_plan,
                                               const raft::KeyValuePair<value_idx, value_t>* kvp,
                                               size_t nnz)
{
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= nnz) return;

  int idx = out_index[tid];

  if ((tid == 0 || (out_index[tid - 1] != idx))) {
    out_rows[idx] = sort_plan[indices[tid]];
    out_cols[idx] = sort_plan[kvp[tid].key];
    out_vals[idx] = kvp[tid].value;
  }
}

/**
 * Computes the min set of unique components that neighbor the
 * components of each source vertex.
 * @tparam value_idx
 * @tparam value_t
 * @param[out] coo output edge list
 * @param[in] out_index output indptr for ordering edge list
 * @param[in] indices indices of source vertices for each component
 * @param[in] kvp indices and distances of each destination vertex for each component
 * @param[in] n_colors number of components
 * @param[in] stream cuda stream for which to order cuda operations
 */
template <typename value_idx, typename value_t>
void min_components_by_color(raft::sparse::COO<value_t, value_idx>& coo,
                             const value_idx* out_index,
                             const value_idx* indices,
                             const value_idx* sort_plan,
                             const raft::KeyValuePair<value_idx, value_t>* kvp,
                             size_t nnz,
                             cudaStream_t stream)
{
  /**
   * Arrays should be ordered by: colors_indptr->colors_n->kvp.value
   * so the last element of each column in the input CSR should be
   * the min.
   */
  min_components_by_color_kernel<<<raft::ceildiv(nnz, (size_t)256), 256, 0, stream>>>(
    coo.rows(), coo.cols(), coo.vals(), out_index, indices, sort_plan, kvp, nnz);
}

/**
 * Connects the components of an otherwise unconnected knn graph
 * by computing a 1-nn to neighboring components of each data point
 * (e.g. component(nn) != component(self)) and reducing the results to
 * include the set of smallest destination components for each source
 * component. The result will not necessarily contain
 * n_components^2 - n_components number of elements because many components
 * will likely not be contained in the neighborhoods of 1-nns.
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle
 * @param[out] out output edge list containing nearest cross-component
 *             edges.
 * @param[in] X original (row-major) dense matrix for which knn graph should be constructed.
 * @param[in] orig_colors array containing component number for each row of X
 * @param[in] n_rows number of rows in X
 * @param[in] n_cols number of cols in X
 * @param[in] reduction_op reduction operation for computing nearest neighbors. The reduction
 * operation must have `gather` and `scatter` functions defined. For single linkage clustering,
 * these functions are no-ops. For HDBSCAN, they sort and 'unsort' the core distances based on color
 * @param[in] row_batch_size the batch size for computing nearest neighbors. This parameter controls
 * the number of samples for which the nearest neighbors are computed at once. Therefore, it affects
 * the memory consumption mainly by reducing the size of the adjacency matrix for masked nearest
 * neighbors computation. default 0 indicates that no batching is done
 * @param[in] col_batch_size the input data is sorted and 'unsorted' based on color. An additional
 * scratch space buffer of shape (n_rows, col_batch_size) is created for this. Usually, this
 * parameter affects the memory consumption more drastically than the col_batch_size with a marginal
 * increase in compute time as the col_batch_size is reduced. default 0 indicates that no batching
 * is done
 */
template <typename value_idx, typename value_t, typename red_op>
void connect_components(raft::resources const& handle,
                        raft::sparse::COO<value_t, value_idx>& out,
                        const value_t* X,
                        const value_idx* orig_colors,
                        size_t n_rows,
                        size_t n_cols,
                        red_op reduction_op,
                        size_t row_batch_size,
                        size_t col_batch_size)
{
  RAFT_LOG_INFO("connect_components_start");
  RAFT_EXPECTS(col_batch_size <= n_cols, "col_batch_size should be >= 0 and <= n_cols");
  RAFT_EXPECTS(row_batch_size <= n_rows, "row_batch_size should be >= 0 and <= n_rows");
  if (row_batch_size == 0) { row_batch_size = n_rows; }
  if (col_batch_size == 0) { col_batch_size = n_cols; }
  auto stream = resource::get_cuda_stream(handle);

  rmm::device_uvector<value_idx> colors(n_rows, stream);

  // Normalize colors so they are drawn from a monotonically increasing set
  bool zero_based = true;
  raft::label::make_monotonic(
    colors.data(), const_cast<value_idx*>(orig_colors), n_rows, stream, zero_based);

  auto sort_plan = raft::make_device_vector<value_idx>(handle, (value_idx)n_rows);
  raft::linalg::map_offset(handle, sort_plan.view(), [] __device__(value_idx idx) { return idx; });

  thrust::sort_by_key(resource::get_thrust_policy(handle),
                      colors.data(),
                      colors.data() + n_rows,
                      sort_plan.data_handle());
  
  RAFT_LOG_INFO("sort plan created");
  // Modify the reduction operation based on the sort plan. This is particularly needed for HDBSCAN
  reduction_op.gather(handle, sort_plan.data_handle());

  auto X_mutable_view =
    raft::make_device_matrix_view<value_t, value_idx>(const_cast<value_t*>(X), n_rows, n_cols);
  auto sort_plan_const_view =
    raft::make_device_vector_view<const value_idx, value_idx>(sort_plan.data_handle(), n_rows);
  raft::matrix::gather(handle, X_mutable_view, sort_plan_const_view, (value_idx)col_batch_size);

  RAFT_LOG_INFO("X mutable view created");
  /**
   * First compute 1-nn for all colors where the color of each data point
   * is guaranteed to be != color of its nearest neighbor.
   */
  rmm::device_uvector<value_idx> nn_colors(n_rows, stream);
  rmm::device_uvector<raft::KeyValuePair<value_idx, value_t>> temp_inds_dists(n_rows, stream);
  rmm::device_uvector<value_idx> src_indices(n_rows, stream);

  perform_1nn(handle,
              temp_inds_dists.data(),
              nn_colors.data(),
              colors.data(),
              X,
              n_rows,
              n_cols,
              row_batch_size,
              reduction_op);

  /**
   * Sort data points by color (neighbors are not sorted)
   */
  // max_color + 1 = number of connected components
  // sort nn_colors by key w/ original colors
  sort_by_color(
    handle, colors.data(), nn_colors.data(), temp_inds_dists.data(), src_indices.data(), n_rows);

  RAFT_LOG_INFO("sort_by_colors done");
  /**
   * Take the min for any duplicate colors
   */
  // Compute mask of duplicates
  rmm::device_uvector<value_idx> out_index(n_rows + 1, stream);
  raft::sparse::op::compute_duplicates_mask(
    out_index.data(), colors.data(), nn_colors.data(), n_rows, stream);

  thrust::exclusive_scan(resource::get_thrust_policy(handle),
                         out_index.data(),
                         out_index.data() + out_index.size(),
                         out_index.data());

  // compute final size
  value_idx size = 0;
  raft::update_host(&size, out_index.data() + (out_index.size() - 1), 1, stream);
  resource::sync_stream(handle, stream);

  size++;

  raft::sparse::COO<value_t, value_idx> min_edges(stream);
  min_edges.allocate(size, n_rows, n_rows, true, stream);

  min_components_by_color(min_edges,
                          out_index.data(),
                          src_indices.data(),
                          sort_plan.data_handle(),
                          temp_inds_dists.data(),
                          n_rows,
                          stream);

  raft::matrix::scatter(handle, X_mutable_view, sort_plan_const_view, (value_idx)col_batch_size);
  reduction_op.scatter(handle, sort_plan.data_handle());

  /**
   * Symmetrize resulting edge list
   */
  raft::sparse::linalg::symmetrize(
    handle, min_edges.rows(), min_edges.cols(), min_edges.vals(), n_rows, n_rows, size, out);
}

};  // end namespace raft::sparse::neighbors::detail
