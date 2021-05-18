/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>
#include <raft/handle.hpp>
#include <raft/mr/device/buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace raft {

namespace hierarchy {
namespace detail {

template <typename value_idx, typename value_t>
class UnionFind {
 public:
  value_idx next_label;
  std::vector<value_idx> parent;
  std::vector<value_idx> size;

  value_idx n_indices;

  UnionFind(value_idx N_)
    : n_indices(2 * N_ - 1),
      parent(2 * N_ - 1, -1),
      size(2 * N_ - 1, 1),
      next_label(N_) {
    memset(size.data() + N_, 0, (size.size() - N_) * sizeof(value_idx));
  }

  value_idx find(value_idx n) {
    value_idx p;
    p = n;

    while (parent[n] != -1) n = parent[n];

    // path compression
    while (parent[p] != n) {
      p = parent[p == -1 ? n_indices - 1 : p];
      parent[p == -1 ? n_indices - 1 : p] = n;
    }
    return n;
  }

  void perform_union(value_idx m, value_idx n) {
    size[next_label] = size[m] + size[n];
    parent[m] = next_label;
    parent[n] = next_label;

    next_label += 1;
  }
};

/**
 * Standard single-threaded agglomerative labeling on host. This should work
 * well for smaller sizes of m. This is a C++ port of the original reference
 * implementation of HDBSCAN.
 *
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle the raft handle
 * @param[in] rows src edges of the sorted MST
 * @param[in] cols dst edges of the sorted MST
 * @param[in] nnz the number of edges in the sorted MST
 * @param[out] out_src parents of output
 * @param[out] out_dst children of output
 * @param[out] out_delta distances of output
 * @param[out] out_size cluster sizes of output
 */
template <typename value_idx, typename value_t>
void build_dendrogram_host(const handle_t &handle, const value_idx *rows,
                           const value_idx *cols, const value_t *data,
                           size_t nnz, value_idx *children,
                           rmm::device_uvector<value_t> &out_delta,
                           rmm::device_uvector<value_idx> &out_size) {
  auto d_alloc = handle.get_device_allocator();
  auto stream = handle.get_stream();

  value_idx n_edges = nnz;

  std::vector<value_idx> mst_src_h(n_edges);
  std::vector<value_idx> mst_dst_h(n_edges);
  std::vector<value_t> mst_weights_h(n_edges);

  printf("n_edges: %d\n", n_edges);

  update_host(mst_src_h.data(), rows, n_edges, stream);
  update_host(mst_dst_h.data(), cols, n_edges, stream);
  update_host(mst_weights_h.data(), data, n_edges, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<value_idx> children_h(n_edges * 2);
  std::vector<value_idx> out_size_h(n_edges);

  UnionFind<value_idx, value_t> U(nnz + 1);

  for (value_idx i = 0; i < nnz; i++) {
    value_idx a = mst_src_h[i];
    value_idx b = mst_dst_h[i];

    value_idx aa = U.find(a);
    value_idx bb = U.find(b);

    value_idx children_idx = i * 2;

    children_h[children_idx] = aa;
    children_h[children_idx + 1] = bb;
    out_size_h[i] = U.size[aa] + U.size[bb];

    U.perform_union(aa, bb);
  }

  out_size.resize(n_edges, stream);

  printf("Finished dendrogram\n");

  raft::update_device(children, children_h.data(), n_edges * 2, stream);
  raft::update_device(out_size.data(), out_size_h.data(), n_edges, stream);
}

/**
 * Parallel agglomerative labeling. This amounts to a parallel Kruskal's
 * MST algorithm, which breaks apart the sorted MST results into overlapping
 * subsets and independently runs Kruskal's algorithm on each subset,
 * merging them back together into a single hierarchy when complete.
 *
 * This outputs the same format as the reference HDBSCAN, but as 4 separate
 * arrays, rather than a single 2D array.
 *
 * Reference: http://cucis.ece.northwestern.edu/publications/pdf/HenPat12.pdf
 *
 * TODO: Investigate potential for the following end-to-end single-hierarchy batching:
 *    For each of k (independent) batches over the input:
 *    - Sample n elements from X
 *    - Compute mutual reachability graph of batch
 *    - Construct labels from batch
 *
 * The sampled datasets should have some overlap across batches. This will
 * allow for the cluster hierarchies to be merged. Being able to batch
 * will reduce the memory cost so that the full n^2 pairwise distances
 * don't need to be materialized in memory all at once.
 *
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle the raft handle
 * @param[in] rows src edges of the sorted MST
 * @param[in] cols dst edges of the sorted MST
 * @param[in] nnz the number of edges in the sorted MST
 * @param[out] out_src parents of output
 * @param[out] out_dst children of output
 * @param[out] out_delta distances of output
 * @param[out] out_size cluster sizes of output
 * @param[in] k_folds number of folds for parallelizing label step
 */
template <typename value_idx, typename value_t>
void build_dendrogram_device(const handle_t &handle, const value_idx *rows,
                             const value_idx *cols, const value_t *data,
                             value_idx nnz, value_idx *children,
                             value_t *out_delta, value_idx *out_size,
                             value_idx k_folds) {
  ASSERT(k_folds < nnz / 2, "k_folds must be < n_edges / 2");
  /**
   * divide (sorted) mst coo into overlapping subsets. Easiest way to do this is to
   * break it into k-folds and iterate through two folds at a time.
   */

  // 1. Generate ranges for the overlapping subsets

  // 2. Run union-find in parallel for each pair of folds

  // 3. Sort individual label hierarchies

  // 4. Merge label hierarchies together
}

template <typename value_idx>
__global__ void write_levels_kernel(const value_idx *children,
                                    value_idx *parents, value_idx n_vertices) {
  value_idx tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n_vertices) {
    value_idx level = tid / 2;
    value_idx child = children[tid];
    parents[child] = level;
  }
}

/**
 * Instead of propagating a label from roots to children,
 * the children each iterate up the tree until they find
 * the label of their parent. This increases the potential
 * parallelism.
 * @tparam value_idx
 * @param children
 * @param parents
 * @param n_leaves
 * @param labels
 */
template <typename value_idx>
__global__ void inherit_labels(const value_idx *children,
                               const value_idx *levels, size_t n_leaves,
                               value_idx *labels, int cut_level,
                               value_idx n_vertices) {
  value_idx tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < n_vertices) {
    value_idx node = children[tid];
    value_idx cur_level = tid / 2;

    /**
     * Any roots above the cut level should be ignored.
     * Any leaves at the cut level should already be labeled
     */
    if (cur_level > cut_level) return;

    value_idx cur_parent = node;
    value_idx label = labels[cur_parent];

    while (label == -1) {
      cur_parent = cur_level + n_leaves;
      cur_level = levels[cur_parent];
      label = labels[cur_parent];
    }

    labels[node] = label;
  }
}

template <typename value_idx>
struct init_label_roots {
  init_label_roots(value_idx *labels_) : labels(labels_) {}

  template <typename Tuple>
  __host__ __device__ void operator()(Tuple t) {
    labels[thrust::get<1>(t)] = thrust::get<0>(t);
  }

 private:
  value_idx *labels;
};

/**
 * Cuts the dendrogram at a particular level where the number of nodes
 * is equal to n_clusters, then propagates the resulting labels
 * to all the children.
 *
 * @tparam value_idx
 * @param handle
 * @param labels
 * @param children
 * @param n_clusters
 * @param n_leaves
 */
template <typename value_idx, int tpb = 256>
void extract_flattened_clusters(const raft::handle_t &handle, value_idx *labels,
                                const value_idx *children, size_t n_clusters,
                                size_t n_leaves) {
  auto d_alloc = handle.get_device_allocator();
  auto stream = handle.get_stream();
  auto thrust_policy = rmm::exec_policy(rmm::cuda_stream_view{stream});

  // Handle special case where n_clusters == 1
  if (n_clusters == 1) {
    thrust::fill(thrust_policy, labels, labels + n_leaves, 0);
  } else {
    /**
     * Compute levels for each node
     *
     *     1. Initialize "levels" array of size n_leaves * 2
     *
     *     2. For each entry in children, write parent
     *        out for each of the children
     */

    size_t n_edges = (n_leaves - 1) * 2;

    thrust::device_ptr<const value_idx> d_ptr =
      thrust::device_pointer_cast(children);
    value_idx n_vertices =
      *(thrust::max_element(thrust_policy, d_ptr, d_ptr + n_edges)) + 1;

    // Prevent potential infinite loop from labeling disconnected
    // connectivities graph.
    RAFT_EXPECTS(n_vertices == (n_leaves - 1) * 2,
                 "Multiple components found in MST or MST is invalid. "
                 "Cannot find single-linkage solution.");

    rmm::device_uvector<value_idx> levels(n_vertices, stream);

    value_idx n_blocks = ceildiv(n_vertices, (value_idx)tpb);
    write_levels_kernel<<<n_blocks, tpb, 0, stream>>>(children, levels.data(),
                                                      n_vertices);
    /**
     * Step 1: Find label roots:
     *
     *     1. Copying children[children.size()-(n_clusters-1):] entries to
     *        separate arrayo
     *     2. sort array
     *     3. take first n_clusters entries
     */

    value_idx child_size = (n_clusters - 1) * 2;
    rmm::device_uvector<value_idx> label_roots(child_size, stream);

    value_idx children_cpy_start = n_edges - child_size;
    raft::copy_async(label_roots.data(), children + children_cpy_start,
                     child_size, stream);

    thrust::sort(thrust_policy, label_roots.data(),
                 label_roots.data() + (child_size),
                 thrust::greater<value_idx>());

    rmm::device_uvector<value_idx> tmp_labels(n_vertices, stream);

    // Init labels to -1
    thrust::fill(thrust_policy, tmp_labels.data(),
                 tmp_labels.data() + n_vertices, -1);

    // Write labels for cluster roots to "labels"
    thrust::counting_iterator<uint> first(0);

    auto z_iter = thrust::make_zip_iterator(thrust::make_tuple(
      first, label_roots.data() + (label_roots.size() - n_clusters)));

    thrust::for_each(thrust_policy, z_iter, z_iter + n_clusters,
                     init_label_roots<value_idx>(tmp_labels.data()));

    /**
     * Step 2: Propagate labels by having children iterate through their parents
     *     1. Initialize labels to -1
     *     2. For each element in levels array, propagate until parent's
     *        label is !=-1
     */
    value_idx cut_level = (n_edges / 2) - (n_clusters - 1);

    inherit_labels<<<n_blocks, tpb, 0, stream>>>(children, levels.data(),
                                                 n_leaves, tmp_labels.data(),
                                                 cut_level, n_vertices);

    // copy tmp labels to actual labels
    raft::copy_async(labels, tmp_labels.data(), n_leaves, stream);
  }
}

};  // namespace detail
};  // namespace hierarchy
};  // namespace raft
