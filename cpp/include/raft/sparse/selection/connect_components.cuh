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

#include <raft/distance/fused_l2_nn.cuh>
#include <raft/label/classlabels.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/mr/device/buffer.hpp>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/coo.cuh>
#include <raft/sparse/linalg/symmetrize.cuh>
#include <raft/sparse/op/reduce.cuh>

#include <raft/cudart_utils.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>

#include <limits>

namespace raft {
namespace linkage {

/**
 * \brief A key identifier paired with a corresponding value
 *
 * NOTE: This is being included close to where it's being used
 * because it's meant to be temporary. There is a conflict
 * between the cub and thrust_cub namespaces with older CUDA
 * versions so we're using our own as a workaround.
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

/**
 * Assumes 3-iterator tuple containing COO rows, cols, and
 * a cub keyvalue pair object. Sorts the 3 arrays in
 * ascending order: row->col->keyvaluepair
 */
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
    return thrust::get<2>(t1).value < thrust::get<2>(t2).value;
  }
};

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
value_idx get_n_components(value_idx *colors, size_t n_rows,
                           std::shared_ptr<raft::mr::device::allocator> d_alloc,
                           cudaStream_t stream) {
  value_idx *map_ids;
  int num_clusters;
  raft::label::getUniquelabels(colors, n_rows, &map_ids, &num_clusters, stream,
                               d_alloc);
  d_alloc->deallocate(map_ids, num_clusters * sizeof(value_idx), stream);

  return num_clusters;
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

  CUDA_CHECK(cudaStreamSynchronize(stream));

  printf("Finished fusedl2nn");

  LookupColorOp<value_idx, value_t> extract_colors_op(colors);
  thrust::transform(thrust::cuda::par.on(stream), kvp, kvp + n_rows, nn_colors,
                    extract_colors_op);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  printf("Finished thrust transform");
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

template <typename value_idx, typename value_t>
__global__ void min_components_by_color_kernel(
  value_idx *out_rows, value_idx *out_cols, value_t *out_vals,
  const value_idx *out_index, const value_idx *indices,
  const cub::KeyValuePair<value_idx, value_t> *kvp, size_t nnz) {
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= nnz) return;

  int idx = out_index[tid];

  if ((tid == 0 || (out_index[tid - 1] != idx))) {
    out_rows[idx] = indices[tid];
    out_cols[idx] = kvp[tid].key;
    out_vals[idx] = kvp[tid].value;
  }
}

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
                             const value_idx *out_index,
                             const value_idx *indices,
                             const cub::KeyValuePair<value_idx, value_t> *kvp,
                             size_t nnz, cudaStream_t stream) {
  /**
   * Arrays should be ordered by: colors_indptr->colors_n->kvp.value
   * so the last element of each column in the input CSR should be
   * the min.
   */
  min_components_by_color_kernel<<<raft::ceildiv(nnz, (size_t)256), 256, 0,
                                   stream>>>(coo.rows(), coo.cols(), coo.vals(),
                                             out_index, indices, kvp, nnz);
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
 * @param[in] n_rows number of rows in X
 * @param[in] n_cols number of cols in X
 */
template <typename value_idx, typename value_t>
void connect_components(const raft::handle_t &handle,
                        raft::sparse::COO<value_t, value_idx> &out,
                        const value_t *X, const value_idx *orig_colors,
                        size_t n_rows, size_t n_cols,
                        raft::distance::DistanceType metric =
                          raft::distance::DistanceType::L2SqrtExpanded) {
  auto d_alloc = handle.get_device_allocator();
  auto stream = handle.get_stream();

  RAFT_EXPECTS(metric == raft::distance::DistanceType::L2SqrtExpanded,
               "Fixing connectivities for an unconnected k-NN graph only "
               "supports L2SqrtExpanded currently.");

  rmm::device_uvector<value_idx> colors(n_rows, stream);
  raft::copy_async(colors.data(), orig_colors, n_rows, stream);

  // Normalize colors so they are drawn from a monotonically increasing set
  raft::label::make_monotonic(colors.data(), colors.data(), n_rows, stream,
                              d_alloc, true);

  value_idx n_components =
    get_n_components(colors.data(), n_rows, d_alloc, stream);

  /**
   * First compute 1-nn for all colors where the color of each data point
   * is guaranteed to be != color of its nearest neighbor.
   */
  rmm::device_uvector<value_idx> nn_colors(n_rows, stream);
  rmm::device_uvector<cub::KeyValuePair<value_idx, value_t>> temp_inds_dists(
    n_rows, stream);
  rmm::device_uvector<value_idx> src_indices(n_rows, stream);

  perform_1nn(temp_inds_dists.data(), nn_colors.data(), colors.data(), X,
              n_rows, n_cols, d_alloc, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  printf("Finished 1nn");

  /**
   * Sort data points by color (neighbors are not sorted)
   */
  // max_color + 1 = number of connected components
  // sort nn_colors by key w/ original colors
  sort_by_color(colors.data(), nn_colors.data(), temp_inds_dists.data(),
                src_indices.data(), n_rows, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  printf("Finished sort by color");

  /**
   * Take the min for any duplicate colors
   */
  // Compute mask of duplicates
  rmm::device_uvector<value_idx> out_index(n_rows + 1, stream);
  raft::sparse::op::compute_duplicates_mask(out_index.data(), colors.data(),
                                            nn_colors.data(), n_rows, stream);

  thrust::exclusive_scan(thrust::cuda::par.on(stream), out_index.data(),
                         out_index.data() + out_index.size(), out_index.data());

  // compute final size
  value_idx size = 0;
  raft::update_host(&size, out_index.data() + (out_index.size() - 1), 1,
                    stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  size++;

  raft::sparse::COO<value_t, value_idx> min_edges(d_alloc, stream);
  min_edges.allocate(size, n_rows, n_rows, true, stream);

  min_components_by_color(min_edges, out_index.data(), src_indices.data(),
                          temp_inds_dists.data(), n_rows, stream);

  /**
   * Symmetrize resulting edge list
   */
  raft::sparse::linalg::symmetrize(handle, min_edges.rows(), min_edges.cols(),
                                   min_edges.vals(), n_rows, n_rows, size, out);
}

};  // end namespace linkage
};  // end namespace raft