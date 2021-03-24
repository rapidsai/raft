/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include <cub/cub.cuh>

#include <raft/linalg/norm.cuh>
#include <raft/mr/device/buffer.hpp>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/coo.cuh>
#include <raft/sparse/linalg/symmetrize.cuh>

#include <raft/cudart_utils.h>
#include <raft/distance/fused_l2_nn.cuh>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <limits>

#include <cub/cub.cuh>

namespace raft {
namespace linkage {

/**
 * \brief A key identifier paired with a corresponding value
 */
template <typename _Key, typename _Value>
struct KeyValuePair {
  typedef _Key Key;      ///< Key data type
  typedef _Value Value;  ///< Value data type

  Key key;      ///< Item key
  Value value;  ///< Item value

  /// Constructor
  __host__ __device__ __forceinline__ KeyValuePair() {}

  /// Copy Constructor
  __host__ __device__ __forceinline__
  KeyValuePair(cub::KeyValuePair<_Key, _Value> kvp)
    : key(kvp.key), value(kvp.value) {}

  /// Constructor
  __host__ __device__ __forceinline__ KeyValuePair(Key const &key,
                                                   Value const &value)
    : key(key), value(value) {}

  /// Inequality operator
  __host__ __device__ __forceinline__ bool operator!=(const KeyValuePair &b) {
    return (value != b.value) || (key != b.key);
  }
};

/**
 * Functor with reduction ops for performing fused 1-nn
 * computation and guaranteeing only cross-component
 * neighbors are considered.
 * @tparam value_idx
 * @tparam value_t
 */
template <typename value_idx, typename value_t>
struct FixConnectivitiesRedOp {
  value_idx *colors;
  value_idx m;

  FixConnectivitiesRedOp(value_idx *colors_, value_idx m_)
    : colors(colors_), m(m_){};

  typedef typename cub::KeyValuePair<value_idx, value_t> KVP;
  DI void operator()(value_idx rit, KVP *out, const KVP &other) {
    if (rit < m && other.value < out->value &&
        colors[rit] != colors[other.key]) {
      out->key = other.key;
      out->value = other.value;
    }
  }

  DI KVP operator()(value_idx rit, const KVP &a, const KVP &b) {
    if (rit < m && a.value < b.value && colors[rit] != colors[a.key]) {
      return a;
    } else
      return b;
  }

  DI void init(value_t *out, value_t maxVal) { *out = maxVal; }
  DI void init(KVP *out, value_t maxVal) {
    out->key = -1;
    out->value = maxVal;
  }
};

struct TupleComp {
  template <typename one, typename two>
  __host__ __device__ bool operator()(const one &t1, const two &t2) {
    // sort first by each sample's color,
    if (thrust::get<0>(t1) < thrust::get<0>(t2)) return true;
    if (thrust::get<0>(t1) > thrust::get<0>(t2)) return false;

    // then by the color of each sample's closest neighbor,
    if (thrust::get<1>(t1) < thrust::get<1>(t2)) return true;
    if (thrust::get<1>(t1) > thrust::get<1>(t2)) return false;

    // then sort by value in descending order
    return thrust::get<2>(t1).value > thrust::get<2>(t2).value;
  }
};

/**
 * Count the unique vertices adjacent to each component.
 * This is essentially a count_unique_by_key.
 */
template <typename value_idx>
__global__ void count_components_by_color_kernel(value_idx *out_indptr,
                                                 const value_idx *colors_indptr,
                                                 const value_idx *colors_nn,
                                                 value_idx n_colors) {
  value_idx tid = threadIdx.x;
  value_idx row = blockIdx.x;

  value_idx start_offset = colors_indptr[row];
  value_idx stop_offset = colors_indptr[row + 1];

  value_idx agg = 0;

  typedef cub::BlockReduce<value_idx, 256> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  value_idx v = 0;
  for (value_idx i = tid; i < (stop_offset - start_offset) - 1;
       i += blockDim.x) {
    // Columns are sorted by row so only columns that are != next column
    // should get counted
    v += (colors_nn[start_offset + i + 1] != colors_nn[start_offset + i])
         // last thread should always write something
         || i == (stop_offset - start_offset - 2);
  }

  agg = BlockReduce(temp_storage).Reduce(v, cub::Sum());

  if (threadIdx.x == 0) out_indptr[row] = agg;
}

template <typename value_idx, typename value_t>
__global__ void min_components_by_color_kernel(
  value_idx *out_cols, value_t *out_vals, value_idx *out_rows,
  const value_idx *out_indptr, const value_idx *colors_indptr,
  const value_idx *colors_nn, const value_idx *indices,
  const cub::KeyValuePair<value_idx, value_t> *kvp, value_idx n_colors) {
  __shared__ value_idx smem[2];

  value_idx *cur_offset = smem;
  value_idx *mutex = smem + 1;

  if (threadIdx.x == 0) {
    mutex[0] = 0;
    cur_offset[0] = 0;
  }

  __syncthreads();

  value_idx tid = threadIdx.x;
  value_idx row = blockIdx.x;

  value_idx out_offset = out_indptr[blockIdx.x];

  value_idx start_offset = colors_indptr[row];
  value_idx stop_offset = colors_indptr[row + 1];

  for (value_idx i = tid; i < (stop_offset - start_offset) - 1;
       i += blockDim.x) {
    // Columns are sorted by row so ony columns that are != previous column
    // should get counted
    value_idx cur_color = colors_nn[start_offset + i];
    if (colors_nn[start_offset + i + 1] != cur_color
        // last thread should always write something
        || (i == stop_offset - start_offset - 2)) {
      while (atomicCAS(mutex, 0, 1) == 1)
        ;
      __threadfence();

      value_idx local_offset = cur_offset[0];
      out_rows[out_offset + local_offset] = indices[start_offset + i];
      out_cols[out_offset + local_offset] = kvp[start_offset + i].key;
      out_vals[out_offset + local_offset] = kvp[start_offset + i].value;
      cur_offset[0] += 1;
      __threadfence();
      atomicCAS(mutex, 1, 0);
    }
  }
}

/**
 * Compute indptr for the min set of unique components that neighbor the components
 * of each source vertex
 * @tparam value_idx
 * @param[out] out_indptr output indptr
 * @param[in] colors_indptr indptr of components for each source vertex
 * @param[in] colors_nn array of components for the 1-nn around each source vertex
 * @param[in] n_colors number of components
 * @param[in] stream cuda stream for which to order cuda operations
 */
template <typename value_idx>
void count_components_by_color(value_idx *out_indptr,
                               const value_idx *colors_indptr,
                               const value_idx *colors_nn, value_idx n_colors,
                               cudaStream_t stream) {
  count_components_by_color_kernel<<<n_colors, 256, 0, stream>>>(
    out_indptr, colors_indptr, colors_nn, n_colors);
}

template <typename LabelT, typename DataT>
struct CubKVPMinReduce {
  typedef cub::KeyValuePair<LabelT, DataT> KVP;

  DI KVP operator()(LabelT rit, const KVP &a, const KVP &b) {
    return b.value < a.value ? b : a;
  }

  DI KVP operator()(const KVP &a, const KVP &b) {
    return b.value < a.value ? b : a;
  }

};  // KVPMinReduce

/**
 * Computes the min set of unique components that neighbor the
 * components of each source vertex.
 * @tparam value_idx
 * @tparam value_t
 * @param[out] coo output edge list
 * @param[in] out_indptr output indptr for ordering edge list
 * @param[in] colors_indptr indptr of source components
 * @param[in] colors_nn components of nearest neighbors to each source component
 * @param[in] indices indices of source vertices for each component
 * @param[in] kvp indices and distances of each destination vertex for each component
 * @param[in] n_colors number of components
 * @param[in] stream cuda stream for which to order cuda operations
 */
template <typename value_idx, typename value_t>
void min_components_by_color(raft::sparse::COO<value_t, value_idx> &coo,
                             const value_idx *out_indptr,
                             const value_idx *colors_indptr,
                             const value_idx *colors_nn,
                             const value_idx *indices,
                             const cub::KeyValuePair<value_idx, value_t> *kvp,
                             value_idx n_colors, cudaStream_t stream) {
  /**
   * Arrays should be ordered by: colors_indptr->colors_n->kvp.value
   * so the last element of each column in the input CSR should be
   * the min.
   */
  min_components_by_color_kernel<<<n_colors, 256, 0, stream>>>(
    coo.cols(), coo.vals(), coo.rows(), out_indptr, colors_indptr, colors_nn,
    indices, kvp, n_colors);
}

/**
 * Gets max maximum value (max number of components) from array of
 * components. Note that this does not assume the components are
 * drawn from a monotonically increasing set.
 * @tparam value_idx
 * @param[in] colors array of components
 * @param[in] n_rows size of components array
 * @param[in] stream cuda stream for which to order cuda operations
 * @return total number of components
 */
template <typename value_idx>
value_idx get_n_components(value_idx *colors, size_t n_rows,
                           cudaStream_t stream) {
  thrust::device_ptr<value_idx> t_colors = thrust::device_pointer_cast(colors);
  return *(thrust::max_element(thrust::cuda::par.on(stream), t_colors,
                               t_colors + n_rows)) +
         1;
}

/**
 * Build CSR indptr array for sorted edge list mapping components of source
 * vertices to the components of their nearest neighbor vertices
 * @tparam value_idx
 * @param[out] degrees output indptr array
 * @param[in] components_indptr indptr of original CSR array of components
 * @param[in] nn_components indptr of nearest neighbors CSR array of components
 * @param[in] n_components size of nn_components
 * @param[in] stream cuda stream for which to order cuda operations
 */
template <typename value_idx>
void build_output_colors_indptr(value_idx *degrees,
                                const value_idx *components_indptr,
                                const value_idx *nn_components,
                                value_idx n_components, cudaStream_t stream) {
  CUDA_CHECK(cudaMemsetAsync(degrees, 0, (n_components + 1) * sizeof(value_idx),
                             stream));

  /**
   * Create COO array by first computing CSR indptr w/ degrees of each
   * color followed by COO row/col/val arrays.
   */
  // map each component to a separate warp, perform warp reduce by key to find
  // number of unique components in output.

  count_components_by_color(degrees, components_indptr, nn_components,
                            n_components, stream);

  thrust::device_ptr<value_idx> t_degrees =
    thrust::device_pointer_cast(degrees);
  thrust::exclusive_scan(thrust::cuda::par.on(stream), t_degrees,
                         t_degrees + n_components + 1, t_degrees);
}

/**
 * Functor to look up a component for a vertex
 * @tparam value_idx
 * @tparam value_t
 */
template <typename value_idx, typename value_t>
struct LookupColorOp {
  value_idx *colors;

  LookupColorOp(value_idx *colors_) : colors(colors_) {}

  DI value_idx operator()(const cub::KeyValuePair<value_idx, value_t> &kvp) {
    return colors[kvp.key];
  }
};

/**
 * Compute the cross-component 1-nearest neighbors for each row in X using
 * the given array of components
 * @tparam value_idx
 * @tparam value_t
 * @param[out] kvp mapping of closest neighbor vertex and distance for each vertex in the given array of components
 * @param[out] nn_colors components of nearest neighbors for each vertex
 * @param[in] colors components of each vertex
 * @param[in] X original dense data
 * @param[in] n_rows number of rows in original dense data
 * @param[in] n_cols number of columns in original dense data
 * @param[in] d_alloc device allocator to use
 * @param[in] stream cuda stream for which to order cuda operations
 */
template <typename value_idx, typename value_t>
void perform_1nn(cub::KeyValuePair<value_idx, value_t> *kvp,
                 value_idx *nn_colors, value_idx *colors, const value_t *X,
                 size_t n_rows, size_t n_cols,
                 std::shared_ptr<raft::mr::device::allocator> d_alloc,
                 cudaStream_t stream) {
  rmm::device_uvector<int> workspace(n_rows, stream);
  rmm::device_uvector<value_t> x_norm(n_rows, stream);

  raft::linalg::rowNorm(x_norm.data(), X, n_cols, n_rows, raft::linalg::L2Norm,
                        true, stream);

  FixConnectivitiesRedOp<value_idx, value_t> red_op(colors, n_rows);
  raft::distance::fusedL2NN<value_t, cub::KeyValuePair<value_idx, value_t>,
                            value_idx>(kvp, X, X, x_norm.data(), x_norm.data(),
                                       n_rows, n_rows, n_cols, workspace.data(),
                                       red_op, red_op, true, true, stream);

  LookupColorOp<value_idx, value_t> extract_colors_op(colors);
  thrust::transform(thrust::cuda::par.on(stream), kvp, kvp + n_rows, nn_colors,
                    extract_colors_op);
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
void sort_by_color(value_idx *colors, value_idx *nn_colors,
                   cub::KeyValuePair<value_idx, value_t> *kvp,
                   value_idx *src_indices, size_t n_rows, cudaStream_t stream) {
  thrust::counting_iterator<value_idx> arg_sort_iter(0);
  thrust::copy(thrust::cuda::par.on(stream), arg_sort_iter,
               arg_sort_iter + n_rows, src_indices);

  auto keys = thrust::make_zip_iterator(thrust::make_tuple(
    colors, nn_colors, (raft::linkage::KeyValuePair<value_idx, value_t> *)kvp));
  auto vals = thrust::make_zip_iterator(thrust::make_tuple(src_indices));

  // get all the colors in contiguous locations so we can map them to warps.
  thrust::sort_by_key(thrust::cuda::par.on(stream), keys, keys + n_rows, vals,
                      TupleComp());
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
 * @param[in] colors array containing component number for each row of X
 * @param n_rows number of rows in X
 * @param n_cols number of cols in X
 */
template <typename value_idx, typename value_t>
void connect_components(const raft::handle_t &handle,
                        raft::sparse::COO<value_t, value_idx> &out,
                        const value_t *X, value_idx *colors, size_t n_rows,
                        size_t n_cols) {
  auto d_alloc = handle.get_device_allocator();
  auto stream = handle.get_stream();

  value_idx n_components = get_n_components(colors, n_rows, stream);

  /**
   * First compute 1-nn for all colors where the color of each data point
   * is guaranteed to be != color of its nearest neighbor.
   */
  rmm::device_uvector<value_idx> nn_colors(n_rows, stream);
  rmm::device_uvector<cub::KeyValuePair<value_idx, value_t>> temp_inds_dists(
    n_rows, stream);
  rmm::device_uvector<value_idx> src_indices(n_rows, stream);
  rmm::device_uvector<value_idx> color_neigh_degrees(n_components + 1, stream);
  rmm::device_uvector<value_idx> colors_indptr(n_components + 1, stream);

  printf("Performing 1nn\n");

  perform_1nn(temp_inds_dists.data(), nn_colors.data(), colors, X, n_rows,
              n_cols, d_alloc, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  printf("Sorting by color\n");

  /**
   * Sort data points by color (neighbors are not sorted)
   */
  // max_color + 1 = number of connected components
  // sort nn_colors by key w/ original colors
  sort_by_color(colors, nn_colors.data(), temp_inds_dists.data(),
                src_indices.data(), n_rows, stream);

  // create an indptr array for newly sorted colors
  raft::sparse::convert::sorted_coo_to_csr(colors, n_rows, colors_indptr.data(),
                                           n_components + 1, d_alloc, stream);

  printf("Building output colors indptr\n");
  // create output degree array for closest components per row
  build_output_colors_indptr(color_neigh_degrees.data(), colors_indptr.data(),
                             nn_colors.data(), n_components, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  value_idx nnz;
  raft::update_host(&nnz, color_neigh_degrees.data() + n_components, 1, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  printf("Min components by color\n");
  raft::sparse::COO<value_t, value_idx> min_edges(d_alloc, stream, nnz);
  min_components_by_color(min_edges, color_neigh_degrees.data(),
                          colors_indptr.data(), nn_colors.data(),
                          src_indices.data(), temp_inds_dists.data(),
                          n_components, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  // symmetrize

  printf("Symmetrize\n");
  raft::sparse::linalg::symmetrize(handle, min_edges.rows(), min_edges.cols(),
                                   min_edges.vals(), n_rows, n_rows, nnz, out);

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

};  // end namespace linkage
};  // end namespace raft