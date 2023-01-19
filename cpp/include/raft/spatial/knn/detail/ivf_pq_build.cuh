/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include "ann_kmeans_balanced.cuh"
#include "ann_serialization.h"
#include "ann_utils.cuh"

#include <raft/neighbors/ivf_pq_types.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/operators.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/detail/qr.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/random/rng.cuh>
#include <raft/stats/histogram.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/device_atomics.cuh>
#include <raft/util/integer_utils.hpp>
#include <raft/util/pow2_utils.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <thrust/binary_search.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

#include <variant>

namespace raft::spatial::knn::ivf_pq::detail {

using namespace raft::spatial::knn::detail;  // NOLINT

using raft::neighbors::ivf_pq::codebook_gen;
using raft::neighbors::ivf_pq::index;
using raft::neighbors::ivf_pq::index_params;
using raft::neighbors::ivf_pq::kIndexGroupSize;
using raft::neighbors::ivf_pq::kIndexGroupVecLen;

using pq_vec_t        = TxN_t<uint8_t, kIndexGroupVecLen>::io_t;
using pq_new_vec_exts = extents<size_t, dynamic_extent, dynamic_extent>;
using pq_int_vec_exts = extents<size_t, dynamic_extent, dynamic_extent, kIndexGroupSize>;

namespace {

/**
 * This type mimics the `uint8_t&` for the indexing operator of `bitfield_view_t`.
 *
 * @tparam Bits number of bits comprising the value.
 */
template <uint32_t Bits>
struct bitfield_ref_t {
  static_assert(Bits <= 8 && Bits > 0, "Bit code must fit one byte");
  constexpr static uint8_t kMask = static_cast<uint8_t>((1u << Bits) - 1u);
  uint8_t* ptr;
  uint32_t offset;

  constexpr operator uint8_t()  // NOLINT
  {
    auto pair = static_cast<uint16_t>(ptr[0]);
    if (offset + Bits > 8) { pair |= static_cast<uint16_t>(ptr[1]) << 8; }
    return static_cast<uint8_t>((pair >> offset) & kMask);
  }

  constexpr auto operator=(uint8_t code) -> bitfield_ref_t&
  {
    if (offset + Bits > 8) {
      auto pair = static_cast<uint16_t>(ptr[0]);
      pair |= static_cast<uint16_t>(ptr[1]) << 8;
      pair &= ~(static_cast<uint16_t>(kMask) << offset);
      pair |= static_cast<uint16_t>(code) << offset;
      ptr[0] = static_cast<uint8_t>(Pow2<256>::mod(pair));
      ptr[1] = static_cast<uint8_t>(Pow2<256>::div(pair));
    } else {
      ptr[0] = (ptr[0] & ~(kMask << offset)) | (code << offset);
    }
    return *this;
  }
};

/**
 * View a byte array as an array of unsigned integers of custom small bit size.
 *
 * @tparam Bits number of bits comprising a single element of the array.
 */
template <uint32_t Bits>
struct bitfield_view_t {
  static_assert(Bits <= 8 && Bits > 0, "Bit code must fit one byte");
  uint8_t* raw;

  constexpr auto operator[](uint32_t i) -> bitfield_ref_t<Bits>
  {
    uint32_t bit_offset = i * Bits;
    return bitfield_ref_t<Bits>{raw + Pow2<8>::div(bit_offset), Pow2<8>::mod(bit_offset)};
  }
};

template <uint32_t BlockDim, typename T, typename S>
__launch_bounds__(BlockDim) __global__ void copy_warped_kernel(
  T* out, uint32_t ld_out, const S* in, uint32_t ld_in, uint32_t n_cols, size_t n_rows)
{
  using warp    = Pow2<WarpSize>;
  size_t row_ix = warp::div(size_t(threadIdx.x) + size_t(BlockDim) * size_t(blockIdx.x));
  uint32_t i    = warp::mod(threadIdx.x);
  if (row_ix >= n_rows) return;
  out += row_ix * ld_out;
  in += row_ix * ld_in;
  auto f = utils::mapping<T>{};
  for (uint32_t col_ix = i; col_ix < n_cols; col_ix += warp::Value) {
    auto x = f(in[col_ix]);
    __syncwarp();
    out[col_ix] = x;
  }
}

/**
 * Copy the data one warp-per-row:
 *
 *  1. load the data per-warp
 *  2. apply the `utils::mapping<T>{}`
 *  3. sync within warp
 *  4. store the data.
 *
 * Assuming sizeof(T) >= sizeof(S) and the data is properly aligned (see the usage in `build`), this
 * allows to re-structure the data within rows in-place.
 */
template <typename T, typename S>
void copy_warped(T* out,
                 uint32_t ld_out,
                 const S* in,
                 uint32_t ld_in,
                 uint32_t n_cols,
                 size_t n_rows,
                 rmm::cuda_stream_view stream)
{
  constexpr uint32_t kBlockDim = 128;
  dim3 threads(kBlockDim, 1, 1);
  dim3 blocks(div_rounding_up_safe<size_t>(n_rows, kBlockDim / WarpSize), 1, 1);
  copy_warped_kernel<kBlockDim, T, S>
    <<<blocks, threads, 0, stream>>>(out, ld_out, in, ld_in, n_cols, n_rows);
}

}  // namespace

/**
 * @brief Fill-in a random orthogonal transformation matrix.
 *
 * @param handle
 * @param force_random_rotation
 * @param n_rows
 * @param n_cols
 * @param[out] rotation_matrix device pointer to a row-major matrix of size [n_rows, n_cols].
 * @param rng random number generator state
 */
inline void make_rotation_matrix(const handle_t& handle,
                                 bool force_random_rotation,
                                 uint32_t n_rows,
                                 uint32_t n_cols,
                                 float* rotation_matrix,
                                 raft::random::Rng rng = raft::random::Rng(7ULL))
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_pq::make_rotation_matrix(%u * %u)", n_rows, n_cols);
  auto stream  = handle.get_stream();
  bool inplace = n_rows == n_cols;
  uint32_t n   = std::max(n_rows, n_cols);
  if (force_random_rotation || !inplace) {
    rmm::device_uvector<float> buf(inplace ? 0 : n * n, stream);
    float* mat = inplace ? rotation_matrix : buf.data();
    rng.normal(mat, n * n, 0.0f, 1.0f, stream);
    linalg::detail::qrGetQ_inplace(handle, mat, n, n, stream);
    if (!inplace) {
      RAFT_CUDA_TRY(cudaMemcpy2DAsync(rotation_matrix,
                                      sizeof(float) * n_cols,
                                      mat,
                                      sizeof(float) * n,
                                      sizeof(float) * n_cols,
                                      n_rows,
                                      cudaMemcpyDefault,
                                      stream));
    }
  } else {
    uint32_t stride = n + 1;
    auto rotation_matrix_view =
      raft::make_device_vector_view<float, uint32_t>(rotation_matrix, n * n);
    linalg::index_unary_op(handle, rotation_matrix_view, [stride] __device__(uint32_t i) {
      return static_cast<float>(i % stride == 0u);
    });
  }
}

/**
 * @brief Compute residual vectors from the source dataset given by selected indices.
 *
 * The residual has the form `rotation_matrix %* (dataset[row_ids, :] - center)`
 *
 */
template <typename T, typename IdxT>
void select_residuals(const handle_t& handle,
                      float* residuals,
                      IdxT n_rows,
                      uint32_t dim,
                      uint32_t rot_dim,
                      const float* rotation_matrix,  // [rot_dim, dim]
                      const float* center,           // [dim]
                      const T* dataset,              // [.., dim]
                      const IdxT* row_ids,           // [n_rows]
                      rmm::mr::device_memory_resource* device_memory

)
{
  auto stream = handle.get_stream();
  rmm::device_uvector<float> tmp(size_t(n_rows) * size_t(dim), stream, device_memory);
  utils::copy_selected<float, T>(
    n_rows, (IdxT)dim, dataset, row_ids, (IdxT)dim, tmp.data(), (IdxT)dim, stream);

  raft::matrix::linewiseOp(
    tmp.data(), tmp.data(), IdxT(dim), n_rows, true, raft::sub_op{}, stream, center);

  float alpha = 1.0;
  float beta  = 0.0;
  linalg::gemm(handle,
               true,
               false,
               rot_dim,
               n_rows,
               dim,
               &alpha,
               rotation_matrix,
               dim,
               tmp.data(),
               dim,
               &beta,
               residuals,
               rot_dim,
               stream);
}

/**
 * @brief Compute residual vectors from the source dataset given by selected indices.
 *
 * The residual has the form
 *  `rotation_matrix %* (dataset[:, :] - centers[labels[:], 0:dim])`
 *
 */
template <typename T, typename IdxT>
void flat_compute_residuals(
  const handle_t& handle,
  float* residuals,  // [n_rows, rot_dim]
  IdxT n_rows,
  device_mdspan<const float, extent_2d<uint32_t>, row_major> rotation_matrix,  // [rot_dim, dim]
  device_mdspan<const float, extent_2d<uint32_t>, row_major> centers,          // [n_lists, dim_ext]
  const T* dataset,                                                            // [n_rows, dim]
  const uint32_t* labels,                                                      // [n_rows]
  rmm::mr::device_memory_resource* device_memory)
{
  auto stream  = handle.get_stream();
  auto dim     = rotation_matrix.extent(1);
  auto rot_dim = rotation_matrix.extent(0);
  rmm::device_uvector<float> tmp(n_rows * dim, stream, device_memory);
  auto tmp_view = raft::make_device_vector_view<float, IdxT>(tmp.data(), tmp.size());
  linalg::index_unary_op(handle, tmp_view, [centers, dataset, labels, dim] __device__(size_t i) {
    auto row_ix = i / dim;
    auto el_ix  = i % dim;
    auto label  = labels[row_ix];
    return utils::mapping<float>{}(dataset[i]) - centers(label, el_ix);
  });

  float alpha = 1.0f;
  float beta  = 0.0f;
  linalg::gemm(handle,
               true,
               false,
               rot_dim,
               n_rows,
               dim,
               &alpha,
               rotation_matrix.data_handle(),
               dim,
               tmp.data(),
               dim,
               &beta,
               residuals,
               rot_dim,
               stream);
}

template <uint32_t BlockDim, typename IdxT>
__launch_bounds__(BlockDim) __global__ void fill_indices_kernel(IdxT n_rows,
                                                                IdxT* data_indices,
                                                                IdxT* data_offsets,
                                                                const uint32_t* labels)
{
  const auto i = IdxT(BlockDim) * IdxT(blockIdx.x) + IdxT(threadIdx.x);
  if (i >= n_rows) { return; }
  data_indices[atomicAdd<IdxT>(data_offsets + labels[i], 1)] = i;
}

/**
 * @brief Calculate cluster offsets and arrange data indices into clusters.
 *
 * @param n_rows
 * @param n_lists
 * @param[in] labels output of k-means prediction [n_rows]
 * @param[in] cluster_sizes [n_lists]
 * @param[out] cluster_offsets [n_lists+1]
 * @param[out] data_indices [n_rows]
 *
 * @return size of the largest cluster
 */
template <typename IdxT>
auto calculate_offsets_and_indices(IdxT n_rows,
                                   uint32_t n_lists,
                                   const uint32_t* labels,
                                   const uint32_t* cluster_sizes,
                                   IdxT* cluster_offsets,
                                   IdxT* data_indices,
                                   rmm::cuda_stream_view stream) -> uint32_t
{
  auto exec_policy = rmm::exec_policy(stream);
  // Calculate the offsets
  IdxT cumsum = 0;
  update_device(cluster_offsets, &cumsum, 1, stream);
  thrust::inclusive_scan(
    exec_policy, cluster_sizes, cluster_sizes + n_lists, cluster_offsets + 1, add_op{});
  update_host(&cumsum, cluster_offsets + n_lists, 1, stream);
  uint32_t max_cluster_size =
    *thrust::max_element(exec_policy, cluster_sizes, cluster_sizes + n_lists);
  stream.synchronize();
  RAFT_EXPECTS(cumsum == n_rows, "cluster sizes do not add up.");
  RAFT_LOG_DEBUG("Max cluster size %d", max_cluster_size);
  rmm::device_uvector<IdxT> data_offsets_buf(n_lists, stream);
  auto data_offsets = data_offsets_buf.data();
  copy(data_offsets, cluster_offsets, n_lists, stream);
  constexpr uint32_t n_threads = 128;  // NOLINT
  const IdxT n_blocks          = raft::div_rounding_up_unsafe(n_rows, n_threads);
  fill_indices_kernel<n_threads>
    <<<n_blocks, n_threads, 0, stream>>>(n_rows, data_indices, data_offsets, labels);
  return max_cluster_size;
}

template <typename IdxT>
void transpose_pq_centers(const handle_t& handle,
                          index<IdxT>& index,
                          const float* pq_centers_source)
{
  auto stream  = handle.get_stream();
  auto extents = index.pq_centers().extents();
  static_assert(extents.rank() == 3);
  auto extents_source =
    make_extents<uint32_t>(extents.extent(0), extents.extent(2), extents.extent(1));
  auto span_source =
    make_mdspan<const float, uint32_t, row_major, false, true>(pq_centers_source, extents_source);
  auto pq_centers_view = raft::make_device_vector_view<float, IdxT>(
    index.pq_centers().data_handle(), index.pq_centers().size());
  linalg::index_unary_op(handle, pq_centers_view, [span_source, extents] __device__(size_t i) {
    uint32_t ii[3];
    for (int r = 2; r > 0; r--) {
      ii[r] = i % extents.extent(r);
      i /= extents.extent(r);
    }
    ii[0] = i;
    return span_source(ii[0], ii[2], ii[1]);
  });
}

template <typename IdxT>
void train_per_subset(const handle_t& handle,
                      index<IdxT>& index,
                      size_t n_rows,
                      const float* trainset,   // [n_rows, dim]
                      const uint32_t* labels,  // [n_rows]
                      uint32_t kmeans_n_iters,
                      rmm::mr::device_memory_resource* managed_memory,
                      rmm::mr::device_memory_resource* device_memory)
{
  auto stream = handle.get_stream();

  rmm::device_uvector<float> pq_centers_tmp(index.pq_centers().size(), stream, device_memory);
  rmm::device_uvector<float> sub_trainset(n_rows * size_t(index.pq_len()), stream, device_memory);
  rmm::device_uvector<uint32_t> sub_labels(n_rows, stream, device_memory);

  rmm::device_uvector<uint32_t> pq_cluster_sizes(index.pq_book_size(), stream, device_memory);

  for (uint32_t j = 0; j < index.pq_dim(); j++) {
    common::nvtx::range<common::nvtx::domain::raft> pq_per_subspace_scope(
      "ivf_pq::build::per_subspace[%u]", j);

    // Get the rotated cluster centers for each training vector.
    // This will be subtracted from the input vectors afterwards.
    utils::copy_selected<float, float, size_t, uint32_t>(
      n_rows,
      index.pq_len(),
      index.centers_rot().data_handle() + index.pq_len() * j,
      labels,
      index.rot_dim(),
      sub_trainset.data(),
      index.pq_len(),
      stream);

    // sub_trainset is the slice of: rotate(trainset) - centers_rot
    float alpha = 1.0;
    float beta  = -1.0;
    linalg::gemm(handle,
                 true,
                 false,
                 index.pq_len(),
                 n_rows,
                 index.dim(),
                 &alpha,
                 index.rotation_matrix().data_handle() + index.dim() * index.pq_len() * j,
                 index.dim(),
                 trainset,
                 index.dim(),
                 &beta,
                 sub_trainset.data(),
                 index.pq_len(),
                 stream);

    // train PQ codebook for this subspace
    kmeans::build_clusters(handle,
                           kmeans_n_iters,
                           index.pq_len(),
                           sub_trainset.data(),
                           n_rows,
                           index.pq_book_size(),
                           pq_centers_tmp.data() + index.pq_book_size() * index.pq_len() * j,
                           sub_labels.data(),
                           pq_cluster_sizes.data(),
                           raft::distance::DistanceType::L2Expanded,
                           stream,
                           device_memory);
  }
  transpose_pq_centers(handle, index, pq_centers_tmp.data());
}

template <typename IdxT>
void train_per_cluster(const handle_t& handle,
                       index<IdxT>& index,
                       size_t n_rows,
                       const float* trainset,   // [n_rows, dim]
                       const uint32_t* labels,  // [n_rows]
                       uint32_t kmeans_n_iters,
                       rmm::mr::device_memory_resource* managed_memory,
                       rmm::mr::device_memory_resource* device_memory)
{
  auto stream = handle.get_stream();

  rmm::device_uvector<float> pq_centers_tmp(index.pq_centers().size(), stream, device_memory);
  rmm::device_uvector<uint32_t> cluster_sizes(index.n_lists(), stream, managed_memory);
  rmm::device_uvector<IdxT> indices_buf(n_rows, stream, device_memory);
  rmm::device_uvector<IdxT> offsets_buf(index.list_offsets().size(), stream, managed_memory);

  raft::stats::histogram<uint32_t, size_t>(raft::stats::HistTypeAuto,
                                           reinterpret_cast<int32_t*>(cluster_sizes.data()),
                                           index.n_lists(),
                                           labels,
                                           n_rows,
                                           1,
                                           stream);

  auto cluster_offsets      = offsets_buf.data();
  auto indices              = indices_buf.data();
  uint32_t max_cluster_size = calculate_offsets_and_indices(
    IdxT(n_rows), index.n_lists(), labels, cluster_sizes.data(), cluster_offsets, indices, stream);

  rmm::device_uvector<uint32_t> pq_labels(
    size_t(max_cluster_size) * size_t(index.pq_dim()), stream, device_memory);
  rmm::device_uvector<uint32_t> pq_cluster_sizes(index.pq_book_size(), stream, device_memory);
  rmm::device_uvector<float> rot_vectors(
    size_t(max_cluster_size) * size_t(index.rot_dim()), stream, device_memory);

  handle.sync_stream();  // make sure cluster offsets are up-to-date
  for (uint32_t l = 0; l < index.n_lists(); l++) {
    auto cluster_size = cluster_sizes.data()[l];
    if (cluster_size == 0) continue;
    common::nvtx::range<common::nvtx::domain::raft> pq_per_cluster_scope(
      "ivf_pq::build::per_cluster[%u](size = %u)", l, cluster_size);

    select_residuals(handle,
                     rot_vectors.data(),
                     IdxT(cluster_size),
                     index.dim(),
                     index.rot_dim(),
                     index.rotation_matrix().data_handle(),
                     index.centers().data_handle() + size_t(l) * size_t(index.dim_ext()),
                     trainset,
                     indices + cluster_offsets[l],
                     device_memory);

    // limit the cluster size to bound the training time.
    // [sic] we interpret the data as pq_len-dimensional
    size_t big_enough     = 256ul * std::max<size_t>(index.pq_book_size(), index.pq_dim());
    size_t available_rows = size_t(cluster_size) * size_t(index.pq_dim());
    auto pq_n_rows        = uint32_t(std::min(big_enough, available_rows));
    // train PQ codebook for this cluster
    kmeans::build_clusters(
      handle,
      kmeans_n_iters,
      index.pq_len(),
      rot_vectors.data(),
      pq_n_rows,
      index.pq_book_size(),
      pq_centers_tmp.data() + size_t(index.pq_book_size()) * size_t(index.pq_len()) * size_t(l),
      pq_labels.data(),
      pq_cluster_sizes.data(),
      raft::distance::DistanceType::L2Expanded,
      stream,
      device_memory);
  }
  transpose_pq_centers(handle, index, pq_centers_tmp.data());
}

/**
 * Sort cluster by their size (descending).
 *
 * @return Number of non-empty clusters
 */
inline auto reorder_clusters_by_size_desc(const handle_t& handle,
                                          uint32_t* ordering,
                                          uint32_t* cluster_sizes_out,
                                          const uint32_t* cluster_sizes_in,
                                          uint32_t n_clusters,
                                          rmm::mr::device_memory_resource* device_memory)
  -> uint32_t
{
  auto stream = handle.get_stream();
  rmm::device_uvector<uint32_t> cluster_ordering_in(n_clusters, stream, device_memory);
  thrust::sequence(handle.get_thrust_policy(),
                   cluster_ordering_in.data(),
                   cluster_ordering_in.data() + n_clusters);

  int begin_bit             = 0;
  int end_bit               = sizeof(uint32_t) * 8;
  size_t cub_workspace_size = 0;
  cub::DeviceRadixSort::SortPairsDescending(nullptr,
                                            cub_workspace_size,
                                            cluster_sizes_in,
                                            cluster_sizes_out,
                                            cluster_ordering_in.data(),
                                            ordering,
                                            n_clusters,
                                            begin_bit,
                                            end_bit,
                                            stream);
  rmm::device_buffer cub_workspace(cub_workspace_size, stream, device_memory);
  cub::DeviceRadixSort::SortPairsDescending(cub_workspace.data(),
                                            cub_workspace_size,
                                            cluster_sizes_in,
                                            cluster_sizes_out,
                                            cluster_ordering_in.data(),
                                            ordering,
                                            n_clusters,
                                            begin_bit,
                                            end_bit,
                                            stream);

  return thrust::lower_bound(handle.get_thrust_policy(),
                             cluster_sizes_out,
                             cluster_sizes_out + n_clusters,
                             0,
                             thrust::greater<uint32_t>()) -
         cluster_sizes_out;
}

/**
 * Compute the code: find the closest cluster in each pq_dim-subspace.
 *
 * @tparam SubWarpSize
 *   how many threads work on a single vector;
 *   bouded by either WarpSize or pq_book_size.
 *
 * @param pq_centers
 *   - codebook_gen::PER_SUBSPACE: [pq_dim , pq_len, pq_book_size]
 *   - codebook_gen::PER_CLUSTER:  [n_lists, pq_len, pq_book_size]
 * @param new_vector a single input of length rot_dim, reinterpreted as [pq_dim, pq_len].
 *   the input must be already transformed to floats, rotated, and the level 1 cluster
 *   center must be already substructed (i.e. this is the residual of a single input vector).
 * @param codebook_kind
 * @param j index along pq_dim "dimension"
 * @param cluster_ix is used for PER_CLUSTER codebooks.
 */
template <uint32_t SubWarpSize>
__device__ auto compute_pq_code(
  device_mdspan<const float, extent_3d<uint32_t>, row_major> pq_centers,
  device_mdspan<const float, extent_2d<uint32_t>, row_major> new_vector,
  codebook_gen codebook_kind,
  uint32_t j,
  uint32_t cluster_ix) -> uint8_t
{
  using subwarp_align = Pow2<SubWarpSize>;
  uint32_t lane_id    = subwarp_align::mod(laneId());
  uint32_t partition_ix;
  switch (codebook_kind) {
    case codebook_gen::PER_CLUSTER: {
      partition_ix = cluster_ix;
    } break;
    case codebook_gen::PER_SUBSPACE: {
      partition_ix = j;
    } break;
    default: __builtin_unreachable();
  }

  const uint32_t pq_book_size = pq_centers.extent(2);
  const uint32_t pq_len       = pq_centers.extent(1);
  float min_dist              = std::numeric_limits<float>::infinity();
  uint8_t code                = 0;
  // calculate the distance for each PQ cluster, find the minimum for each thread
  for (uint32_t i = lane_id; i < pq_book_size; i += subwarp_align::Value) {
    // NB: the L2 quantifiers on residuals are always trained on L2 metric.
    float d = 0.0f;
    for (uint32_t k = 0; k < pq_len; k++) {
      auto t = new_vector(j, k) - pq_centers(partition_ix, k, i);
      d += t * t;
    }
    if (d < min_dist) {
      min_dist = d;
      code     = uint8_t(i);
    }
  }
  // reduce among threads
#pragma unroll
  for (uint32_t stride = SubWarpSize >> 1; stride > 0; stride >>= 1) {
    const auto other_dist = shfl_xor(min_dist, stride, SubWarpSize);
    const auto other_code = shfl_xor(code, stride, SubWarpSize);
    if (other_dist < min_dist) {
      min_dist = other_dist;
      code     = other_code;
    }
  }
  return code;
}

template <uint32_t BlockSize, uint32_t PqBits, typename IdxT>
__launch_bounds__(BlockSize) __global__ void process_and_fill_codes_kernel(
  device_mdspan<const float, extent_2d<IdxT>, row_major> new_vectors,
  std::variant<IdxT, const IdxT*> src_offset_or_indices,
  const uint32_t* new_labels,
  device_mdspan<uint32_t, extent_1d<uint32_t>, row_major> list_sizes,
  device_mdspan<const IdxT, extent_1d<uint32_t>, row_major> list_offsets,
  device_mdspan<IdxT, extent_1d<IdxT>, row_major> pq_indices,
  device_mdspan<pq_vec_t, pq_int_vec_exts, row_major> pq_dataset,
  device_mdspan<const float, extent_3d<uint32_t>, row_major> pq_centers,
  codebook_gen codebook_kind)
{
  constexpr uint32_t kSubWarpSize = std::min<uint32_t>(WarpSize, 1u << PqBits);
  using subwarp_align             = Pow2<kSubWarpSize>;
  const uint32_t lane_id          = subwarp_align::mod(threadIdx.x);
  const IdxT row_ix = subwarp_align::div(IdxT{threadIdx.x} + IdxT{blockDim.x} * IdxT{blockIdx.x});
  if (row_ix >= new_vectors.extent(0)) { return; }

  const uint32_t cluster_ix = new_labels[row_ix];
  uint32_t out_incluster_ix;
  if (lane_id == 0) { out_incluster_ix = atomicAdd(&list_sizes(cluster_ix), 1); }
  out_incluster_ix  = shfl(out_incluster_ix, 0, kSubWarpSize);
  const IdxT out_ix = list_offsets(cluster_ix) + out_incluster_ix;

  // write the label
  if (lane_id == 0) {
    if (std::holds_alternative<IdxT>(src_offset_or_indices)) {
      pq_indices(out_ix) = std::get<IdxT>(src_offset_or_indices) + row_ix;
    } else {
      pq_indices(out_ix) = std::get<const IdxT*>(src_offset_or_indices)[row_ix];
    }
  }

  // write the codes
  using group_align         = Pow2<kIndexGroupSize>;
  const uint32_t group_ix   = group_align::div(out_ix);
  const uint32_t ingroup_ix = group_align::mod(out_ix);
  const uint32_t pq_len     = pq_centers.extent(1);
  const uint32_t pq_dim     = new_vectors.extent(1) / pq_len;

  __shared__ pq_vec_t codes[subwarp_align::div(BlockSize)];
  pq_vec_t& code = codes[subwarp_align::div(threadIdx.x)];
  bitfield_view_t<PqBits> out{reinterpret_cast<uint8_t*>(&code)};
  constexpr uint32_t kChunkSize = (sizeof(pq_vec_t) * 8u) / PqBits;
  for (uint32_t j = 0, i = 0; j < pq_dim; i++) {
    // clear the chunk for writing
    if (lane_id == 0) { code = pq_vec_t{}; }
    // fill-in the values, one/pq_dim at a time
#pragma unroll
    for (uint32_t k = 0; k < kChunkSize && j < pq_dim; k++, j++) {
      // find the label
      using layout_t   = typename decltype(new_vectors)::layout_type;
      using accessor_t = typename decltype(new_vectors)::accessor_type;
      auto one_vector  = mdspan<const float, extent_2d<uint32_t>, layout_t, accessor_t>(
        &new_vectors(row_ix, 0), extent_2d<uint32_t>{pq_dim, pq_len});
      auto l = compute_pq_code<kSubWarpSize>(pq_centers, one_vector, codebook_kind, j, cluster_ix);
      if (lane_id == 0) { out[k] = l; }
    }
    // write the chunk into the dataset
    if (lane_id == 0) { pq_dataset(group_ix, i, ingroup_ix) = code; }
  }
}

/**
 * Assuming the index already has some data and allocated the space for more, write more data in it.
 * There must be enough free space in `pq_dataset()` and `indices()`, as computed using
 * `list_offsets()` and `list_sizes()`.
 *
 * NB: Since the pq_dataset is stored in the interleaved blocked format (see ivf_pq_types.hpp), one
 * cannot just concatenate the old and the new codes; the positions for the codes are determined the
 * same way as in the ivfpq_compute_similarity_kernel (see ivf_pq_search.cuh).
 *
 * @tparam T
 * @tparam IdxT
 *
 * @param handle
 * @param index
 * @param[in] new_vectors
 *    a pointer to a row-major device array [index.dim(), n_rows];
 * @param[in] src_offset_or_indices
 *    references for the new data:
 *      either a starting index for the auto-indexing
 *      or a pointer to a device array of explicit indices [n_rows];
 * @param[in] new_labels
 *    cluster ids (first-level quantization) - a device array [n_rows];
 * @param n_rows
 *    the number of records to write in.
 * @param mr
 *    a memory resource to use for device allocations
 */
template <typename T, typename IdxT>
void process_and_fill_codes(const handle_t& handle,
                            index<IdxT>& index,
                            const T* new_vectors,
                            std::variant<IdxT, const IdxT*> src_offset_or_indices,
                            const uint32_t* new_labels,
                            IdxT n_rows,
                            rmm::mr::device_memory_resource* mr)
{
  pq_int_vec_exts pq_extents = make_extents<size_t>(index.pq_dataset().extent(0),
                                                    index.pq_dataset().extent(1),
                                                    index.pq_dataset().static_extent(2));
  auto pq_dataset            = make_mdspan<pq_vec_t, size_t, row_major, false, true>(
    reinterpret_cast<pq_vec_t*>(index.pq_dataset().data_handle()), pq_extents);

  auto new_vectors_residual =
    make_device_mdarray<float>(handle, mr, make_extents<IdxT>(n_rows, index.rot_dim()));

  flat_compute_residuals(handle,
                         new_vectors_residual.data_handle(),
                         n_rows,
                         index.rotation_matrix(),
                         index.centers(),
                         new_vectors,
                         new_labels,
                         mr);

  constexpr uint32_t kBlockSize  = 256;
  const uint32_t threads_per_vec = std::min<uint32_t>(WarpSize, index.pq_book_size());
  dim3 blocks(div_rounding_up_safe<IdxT>(n_rows, kBlockSize / threads_per_vec), 1, 1);
  dim3 threads(kBlockSize, 1, 1);
  auto kernel = [](uint32_t pq_bits) {
    switch (pq_bits) {
      case 4: return process_and_fill_codes_kernel<kBlockSize, 4, IdxT>;
      case 5: return process_and_fill_codes_kernel<kBlockSize, 5, IdxT>;
      case 6: return process_and_fill_codes_kernel<kBlockSize, 6, IdxT>;
      case 7: return process_and_fill_codes_kernel<kBlockSize, 7, IdxT>;
      case 8: return process_and_fill_codes_kernel<kBlockSize, 8, IdxT>;
      default: RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 8]", pq_bits);
    }
  }(index.pq_bits());
  kernel<<<blocks, threads, 0, handle.get_stream()>>>(new_vectors_residual.view(),
                                                      src_offset_or_indices,
                                                      new_labels,
                                                      index.list_sizes(),
                                                      index.list_offsets(),
                                                      index.indices(),
                                                      pq_dataset,
                                                      index.pq_centers(),
                                                      index.codebook_kind());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * Fill the `target` index with the data from the `source`, except `list_offsets`.
 * The `target` index must have the same settings and valid `list_offsets`, and must have been
 * pre-allocated to fit the whole `source` data.
 * As a result, the `target` index is in a valid state; it's identical to the `source`, except
 * has more unused space in `pq_dataset`.
 *
 * @param target the index to be filled-in
 * @param source the index to get data from
 * @param cluster_ordering
 *   a pointer to the managed data [n_clusters];
 *   the mapping `source_label = cluster_ordering[target_label]`
 * @param stream
 */
template <typename IdxT>
void copy_index_data(index<IdxT>& target,
                     const index<IdxT>& source,
                     const uint32_t* cluster_ordering,
                     rmm::cuda_stream_view stream)
{
  auto n_clusters = target.n_lists();
  RAFT_EXPECTS(target.size() >= source.size(),
               "The target index must be not smaller than the source index.");
  RAFT_EXPECTS(n_clusters >= source.n_lists(),
               "The target and the source are not compatible (different numbers of clusters).");

  // Copy the unchanged parts
  copy(target.rotation_matrix().data_handle(),
       source.rotation_matrix().data_handle(),
       source.rotation_matrix().size(),
       stream);

  // copy cluster-ordering-dependent data
  utils::copy_selected(n_clusters,
                       uint32_t{1},
                       source.list_sizes().data_handle(),
                       cluster_ordering,
                       uint32_t{1},
                       target.list_sizes().data_handle(),
                       uint32_t{1},
                       stream);
  utils::copy_selected(n_clusters,
                       target.dim_ext(),
                       source.centers().data_handle(),
                       cluster_ordering,
                       source.dim_ext(),
                       target.centers().data_handle(),
                       target.dim_ext(),
                       stream);
  utils::copy_selected(n_clusters,
                       target.rot_dim(),
                       source.centers_rot().data_handle(),
                       cluster_ordering,
                       source.rot_dim(),
                       target.centers_rot().data_handle(),
                       target.rot_dim(),
                       stream);
  switch (source.codebook_kind()) {
    case codebook_gen::PER_SUBSPACE: {
      copy(target.pq_centers().data_handle(),
           source.pq_centers().data_handle(),
           source.pq_centers().size(),
           stream);
    } break;
    case codebook_gen::PER_CLUSTER: {
      auto d = source.pq_book_size() * source.pq_len();
      utils::copy_selected(n_clusters,
                           d,
                           source.pq_centers().data_handle(),
                           cluster_ordering,
                           d,
                           target.pq_centers().data_handle(),
                           d,
                           stream);
    } break;
    default: RAFT_FAIL("Unreachable code");
  }

  // Fill the data with the old clusters.
  if (source.size() > 0) {
    std::vector<IdxT> target_cluster_offsets(n_clusters + 1);
    std::vector<IdxT> source_cluster_offsets(n_clusters + 1);
    std::vector<uint32_t> source_cluster_sizes(n_clusters);
    copy(target_cluster_offsets.data(),
         target.list_offsets().data_handle(),
         target.list_offsets().size(),
         stream);
    copy(source_cluster_offsets.data(),
         source.list_offsets().data_handle(),
         source.list_offsets().size(),
         stream);
    copy(source_cluster_sizes.data(),
         source.list_sizes().data_handle(),
         source.list_sizes().size(),
         stream);
    stream.synchronize();
    auto data_exts = target.pq_dataset().extents();
    auto data_unit = size_t(data_exts.extent(3)) * size_t(data_exts.extent(1));
    auto data_mod  = size_t(data_exts.extent(2));
    for (uint32_t l = 0; l < target.n_lists(); l++) {
      auto k                   = cluster_ordering[l];
      auto source_cluster_size = source_cluster_sizes[k];
      if (source_cluster_size > 0) {
        copy(target.indices().data_handle() + target_cluster_offsets[l],
             source.indices().data_handle() + source_cluster_offsets[k],
             source_cluster_size,
             stream);
        copy(target.pq_dataset().data_handle() + target_cluster_offsets[l] * data_unit,
             source.pq_dataset().data_handle() + source_cluster_offsets[k] * data_unit,
             round_up_safe<size_t>(source_cluster_size, data_mod) * data_unit,
             stream);
      }
    }
  }
}

/** See raft::spatial::knn::ivf_pq::extend docs */
template <typename T, typename IdxT>
auto extend(const handle_t& handle,
            const index<IdxT>& orig_index,
            const T* new_vectors,
            const IdxT* new_indices,
            IdxT n_rows) -> index<IdxT>
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_pq::extend(%zu, %u)", size_t(n_rows), orig_index.dim());
  auto stream           = handle.get_stream();
  const auto n_clusters = orig_index.n_lists();

  RAFT_EXPECTS(new_indices != nullptr || orig_index.size() == 0,
               "You must pass data indices when the index is non-empty.");

  static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
                "Unsupported data type");

  rmm::mr::device_memory_resource* device_memory = nullptr;
  auto pool_guard = raft::get_pool_memory_resource(device_memory, 1024 * 1024);
  if (pool_guard) {
    RAFT_LOG_DEBUG("ivf_pq::extend: using pool memory resource with initial size %zu bytes",
                   pool_guard->pool_size());
  }

  rmm::mr::managed_memory_resource managed_memory_upstream;
  rmm::mr::pool_memory_resource<rmm::mr::managed_memory_resource> managed_memory(
    &managed_memory_upstream, 1024 * 1024);

  // Try to allocate an index with the same parameters and the projected new size
  // (which can be slightly larger than index.size() + n_rows, due to padding).
  // If this fails, the index would be too big to fit in the device anyway.
  std::optional<index<IdxT>> placeholder_index(std::in_place_t{},
                                               handle,
                                               orig_index.metric(),
                                               orig_index.codebook_kind(),
                                               n_clusters,
                                               orig_index.dim(),
                                               orig_index.pq_bits(),
                                               orig_index.pq_dim(),
                                               orig_index.n_nonempty_lists());
  placeholder_index->allocate(
    handle,
    orig_index.size() + n_rows + (kIndexGroupSize - 1) * std::min<IdxT>(n_clusters, n_rows));

  // Available device memory
  size_t free_mem, total_mem;
  RAFT_CUDA_TRY(cudaMemGetInfo(&free_mem, &total_mem));

  // Decide on an approximate threshold when we'd better start saving device memory by using
  // managed allocations for large device buffers
  rmm::mr::device_memory_resource* labels_mr  = device_memory;
  rmm::mr::device_memory_resource* batches_mr = device_memory;
  if (n_rows *
        (orig_index.dim() * sizeof(T) + orig_index.pq_dim() + sizeof(IdxT) + sizeof(uint32_t)) >
      free_mem) {
    labels_mr = &managed_memory;
  }
  // Allocate a buffer for the new labels (classifying the new data)
  rmm::device_uvector<uint32_t> new_data_labels(n_rows, stream, labels_mr);
  if (labels_mr == device_memory) { free_mem -= sizeof(uint32_t) * n_rows; }

  // Calculate the batch size for the input data if it's not accessible directly from the device
  constexpr size_t kReasonableMaxBatchSize = 65536;
  size_t max_batch_size                    = std::min<size_t>(n_rows, kReasonableMaxBatchSize);
  {
    size_t size_factor = 0;
    // we'll use two temporary buffers for converted inputs when computing the codes.
    size_factor += (orig_index.dim() + orig_index.rot_dim()) * sizeof(float);
    // ...and another buffer for indices
    size_factor += sizeof(IdxT);
    // if the input data is not accessible on device, we'd need a buffer for it.
    switch (utils::check_pointer_residency(new_vectors)) {
      case utils::pointer_residency::device_only:
      case utils::pointer_residency::host_and_device: break;
      default: size_factor += orig_index.dim() * sizeof(T);
    }
    // the same with indices
    if (new_indices != nullptr) {
      switch (utils::check_pointer_residency(new_indices)) {
        case utils::pointer_residency::device_only:
        case utils::pointer_residency::host_and_device: break;
        default: size_factor += sizeof(IdxT);
      }
    }
    // make the batch size fit into the remaining memory
    while (size_factor * max_batch_size > free_mem && max_batch_size > 128) {
      max_batch_size >>= 1;
    }
    if (size_factor * max_batch_size > free_mem) {
      // if that still doesn't fit, resort to the UVM
      batches_mr     = &managed_memory;
      max_batch_size = kReasonableMaxBatchSize;
    } else {
      // If we're keeping the batches in device memory, update the available mem tracker.
      free_mem -= size_factor * max_batch_size;
    }
  }

  // Predict the cluster labels for the new data, in batches if necessary
  utils::batch_load_iterator<T> vec_batches(
    new_vectors, n_rows, orig_index.dim(), max_batch_size, stream, batches_mr);
  // Release the placeholder memory, because we don't intend to allocate any more long-living
  // temporary buffers before we allocate the ext_index data.
  // This memory could potentially speed up UVM accesses, if any.
  placeholder_index.reset();
  {
    // The cluster centers in the index are stored padded, which is not acceptable by
    // the kmeans::predict. Thus, we need the restructuring copy.
    rmm::device_uvector<float> cluster_centers(
      size_t(n_clusters) * size_t(orig_index.dim()), stream, device_memory);
    RAFT_CUDA_TRY(cudaMemcpy2DAsync(cluster_centers.data(),
                                    sizeof(float) * orig_index.dim(),
                                    orig_index.centers().data_handle(),
                                    sizeof(float) * orig_index.dim_ext(),
                                    sizeof(float) * orig_index.dim(),
                                    n_clusters,
                                    cudaMemcpyDefault,
                                    stream));
    for (const auto& batch : vec_batches) {
      kmeans::predict(handle,
                      cluster_centers.data(),
                      n_clusters,
                      orig_index.dim(),
                      batch.data(),
                      batch.size(),
                      new_data_labels.data() + batch.offset(),
                      orig_index.metric(),
                      stream,
                      device_memory);
    }
  }

  // Get the combined cluster sizes and sort the clusters in decreasing order
  // (this makes it easy to estimate the max number of samples during search).
  rmm::device_uvector<uint32_t> cluster_ordering_buf(n_clusters, stream, &managed_memory);
  rmm::device_uvector<uint32_t> ext_cluster_sizes_buf(n_clusters, stream, device_memory);
  auto cluster_ordering     = cluster_ordering_buf.data();
  auto ext_cluster_sizes    = ext_cluster_sizes_buf.data();
  uint32_t n_nonempty_lists = 0;
  {
    rmm::device_uvector<uint32_t> new_cluster_sizes_buf(n_clusters, stream, device_memory);
    auto new_cluster_sizes = new_cluster_sizes_buf.data();
    raft::stats::histogram<uint32_t, IdxT>(raft::stats::HistTypeAuto,
                                           reinterpret_cast<int32_t*>(new_cluster_sizes),
                                           IdxT(n_clusters),
                                           new_data_labels.data(),
                                           n_rows,
                                           1,
                                           stream);
    linalg::add(new_cluster_sizes,
                new_cluster_sizes,
                orig_index.list_sizes().data_handle(),
                n_clusters,
                stream);
    n_nonempty_lists = reorder_clusters_by_size_desc(
      handle, cluster_ordering, ext_cluster_sizes, new_cluster_sizes, n_clusters, device_memory);
  }

  // Assemble the extended index
  index<IdxT> ext_index(handle,
                        orig_index.metric(),
                        orig_index.codebook_kind(),
                        n_clusters,
                        orig_index.dim(),
                        orig_index.pq_bits(),
                        orig_index.pq_dim(),
                        n_nonempty_lists);
  // calculate extended cluster offsets and allocate the index data
  {
    auto ext_cluster_offsets = ext_index.list_offsets().data_handle();
    using group_align        = Pow2<kIndexGroupSize>;
    IdxT size                = 0;
    update_device(ext_cluster_offsets, &size, 1, stream);
    auto sizes_padded = thrust::make_transform_iterator(
      ext_cluster_sizes, [] __device__ __host__(uint32_t x) -> IdxT {
        return IdxT{Pow2<kIndexGroupSize>::roundUp(x)};
      });
    thrust::inclusive_scan(handle.get_thrust_policy(),
                           sizes_padded,
                           sizes_padded + n_clusters,
                           ext_cluster_offsets + 1,
                           add_op{});
    update_host(&size, ext_cluster_offsets + n_clusters, 1, stream);
    handle.sync_stream();  // syncs `size`, `cluster_ordering`
    ext_index.allocate(handle, size);
  }

  // pre-fill the extended index with the data from the original index
  copy_index_data(ext_index, orig_index, cluster_ordering, stream);

  // update the labels to correspond to the new cluster ordering
  {
    rmm::device_uvector<uint32_t> cluster_ordering_rev_buf(n_clusters, stream, &managed_memory);
    auto cluster_ordering_rev = cluster_ordering_rev_buf.data();
    for (uint32_t i = 0; i < n_clusters; i++) {
      cluster_ordering_rev[cluster_ordering[i]] = i;
    }
    linalg::unaryOp(
      new_data_labels.data(),
      new_data_labels.data(),
      new_data_labels.size(),
      [cluster_ordering_rev] __device__(uint32_t i) { return cluster_ordering_rev[i]; },
      stream);
  }

  // fill the extended index with the new data (possibly, in batches)
  utils::batch_load_iterator<IdxT> idx_batches(
    new_indices, n_rows, 1, max_batch_size, stream, batches_mr);
  for (const auto& vec_batch : vec_batches) {
    const auto& idx_batch = *idx_batches++;
    process_and_fill_codes(handle,
                           ext_index,
                           vec_batch.data(),
                           new_indices != nullptr
                             ? std::variant<IdxT, const IdxT*>(idx_batch.data())
                             : std::variant<IdxT, const IdxT*>(IdxT(idx_batch.offset())),
                           new_data_labels.data() + vec_batch.offset(),
                           IdxT(vec_batch.size()),
                           batches_mr);
  }

  return ext_index;
}

/** See raft::spatial::knn::ivf_pq::build docs */
template <typename T, typename IdxT>
auto build(
  const handle_t& handle, const index_params& params, const T* dataset, IdxT n_rows, uint32_t dim)
  -> index<IdxT>
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_pq::build(%zu, %u)", size_t(n_rows), dim);
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
                "Unsupported data type");

  RAFT_EXPECTS(n_rows > 0 && dim > 0, "empty dataset");

  auto stream = handle.get_stream();

  index<IdxT> index(handle, params, dim);
  utils::memzero(index.list_offsets().data_handle(), index.list_offsets().size(), stream);
  utils::memzero(index.list_sizes().data_handle(), index.list_sizes().size(), stream);

  auto trainset_ratio = std::max<size_t>(
    1,
    size_t(n_rows) / std::max<size_t>(params.kmeans_trainset_fraction * n_rows, index.n_lists()));
  size_t n_rows_train = n_rows / trainset_ratio;

  rmm::mr::device_memory_resource* device_memory = nullptr;
  auto pool_guard = raft::get_pool_memory_resource(device_memory, 1024 * 1024);
  if (pool_guard) {
    RAFT_LOG_DEBUG("ivf_pq::build: using pool memory resource with initial size %zu bytes",
                   pool_guard->pool_size());
  }

  rmm::mr::managed_memory_resource managed_memory_upstream;
  rmm::mr::pool_memory_resource<rmm::mr::managed_memory_resource> managed_memory(
    &managed_memory_upstream, 1024 * 1024);

  // If the trainset is small enough to comfortably fit into device memory, put it there.
  // Otherwise, use the managed memory.
  rmm::mr::device_memory_resource* big_memory_resource = &managed_memory;
  {
    size_t free_mem, total_mem;
    constexpr size_t kTolerableRatio = 4;
    RAFT_CUDA_TRY(cudaMemGetInfo(&free_mem, &total_mem));
    if (sizeof(float) * n_rows_train * index.dim() * kTolerableRatio < free_mem) {
      big_memory_resource = device_memory;
    }
  }

  // Besides just sampling, we transform the input dataset into floats to make it easier
  // to use gemm operations from cublas.
  rmm::device_uvector<float> trainset(n_rows_train * index.dim(), stream, big_memory_resource);
  // TODO: a proper sampling
  if constexpr (std::is_same_v<T, float>) {
    RAFT_CUDA_TRY(cudaMemcpy2DAsync(trainset.data(),
                                    sizeof(T) * index.dim(),
                                    dataset,
                                    sizeof(T) * index.dim() * trainset_ratio,
                                    sizeof(T) * index.dim(),
                                    n_rows_train,
                                    cudaMemcpyDefault,
                                    stream));
  } else {
    size_t dim = index.dim();
    cudaPointerAttributes dataset_attr;
    RAFT_CUDA_TRY(cudaPointerGetAttributes(&dataset_attr, dataset));
    if (dataset_attr.devicePointer != nullptr) {
      // data is available on device: just run the kernel to copy and map the data
      auto p = reinterpret_cast<T*>(dataset_attr.devicePointer);
      auto trainset_view =
        raft::make_device_vector_view<float, IdxT>(trainset.data(), dim * n_rows_train);
      linalg::index_unary_op(handle, trainset_view, [p, trainset_ratio, dim] __device__(size_t i) {
        auto col = i % dim;
        return utils::mapping<float>{}(p[(i - col) * size_t(trainset_ratio) + col]);
      });
    } else {
      // data is not available: first copy, then map inplace
      auto trainset_tmp = reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(trainset.data()) +
                                               (sizeof(float) - sizeof(T)) * index.dim());
      // We copy the data in strides, one row at a time, and place the smaller rows of type T
      // at the end of float rows.
      RAFT_CUDA_TRY(cudaMemcpy2DAsync(trainset_tmp,
                                      sizeof(float) * index.dim(),
                                      dataset,
                                      sizeof(T) * index.dim() * trainset_ratio,
                                      sizeof(T) * index.dim(),
                                      n_rows_train,
                                      cudaMemcpyDefault,
                                      stream));
      // Transform the input `{T -> float}`, one row per warp.
      // The threads in each warp copy the data synchronously; this and the layout of the data
      // (content is aligned to the end of the rows) together allow doing the transform in-place.
      copy_warped(trainset.data(),
                  index.dim(),
                  trainset_tmp,
                  index.dim() * sizeof(float) / sizeof(T),
                  index.dim(),
                  n_rows_train,
                  stream);
    }
  }

  // NB: here cluster_centers is used as if it is [n_clusters, data_dim] not [n_clusters, dim_ext]!
  rmm::device_uvector<float> cluster_centers_buf(
    index.n_lists() * index.dim(), stream, device_memory);
  auto cluster_centers = cluster_centers_buf.data();

  // Train balanced hierarchical kmeans clustering
  kmeans::build_hierarchical(handle,
                             params.kmeans_n_iters,
                             index.dim(),
                             trainset.data(),
                             n_rows_train,
                             cluster_centers,
                             index.n_lists(),
                             index.metric(),
                             stream);

  // Trainset labels are needed for training PQ codebooks
  rmm::device_uvector<uint32_t> labels(n_rows_train, stream, big_memory_resource);
  kmeans::predict(handle,
                  cluster_centers,
                  index.n_lists(),
                  index.dim(),
                  trainset.data(),
                  n_rows_train,
                  labels.data(),
                  index.metric(),
                  stream,
                  device_memory);

  {
    // combine cluster_centers and their norms
    RAFT_CUDA_TRY(cudaMemcpy2DAsync(index.centers().data_handle(),
                                    sizeof(float) * index.dim_ext(),
                                    cluster_centers,
                                    sizeof(float) * index.dim(),
                                    sizeof(float) * index.dim(),
                                    index.n_lists(),
                                    cudaMemcpyDefault,
                                    stream));

    rmm::device_uvector<float> center_norms(index.n_lists(), stream, device_memory);
    raft::linalg::rowNorm(center_norms.data(),
                          cluster_centers,
                          index.dim(),
                          index.n_lists(),
                          raft::linalg::L2Norm,
                          true,
                          stream);
    RAFT_CUDA_TRY(cudaMemcpy2DAsync(index.centers().data_handle() + index.dim(),
                                    sizeof(float) * index.dim_ext(),
                                    center_norms.data(),
                                    sizeof(float),
                                    sizeof(float),
                                    index.n_lists(),
                                    cudaMemcpyDefault,
                                    stream));
  }

  // Make rotation matrix
  make_rotation_matrix(handle,
                       params.force_random_rotation,
                       index.rot_dim(),
                       index.dim(),
                       index.rotation_matrix().data_handle());

  // Rotate cluster_centers
  float alpha = 1.0;
  float beta  = 0.0;
  linalg::gemm(handle,
               true,
               false,
               index.rot_dim(),
               index.n_lists(),
               index.dim(),
               &alpha,
               index.rotation_matrix().data_handle(),
               index.dim(),
               cluster_centers,
               index.dim(),
               &beta,
               index.centers_rot().data_handle(),
               index.rot_dim(),
               stream);

  // Train PQ codebooks
  switch (index.codebook_kind()) {
    case codebook_gen::PER_SUBSPACE:
      train_per_subset(handle,
                       index,
                       n_rows_train,
                       trainset.data(),
                       labels.data(),
                       params.kmeans_n_iters,
                       &managed_memory,
                       device_memory);
      break;
    case codebook_gen::PER_CLUSTER:
      train_per_cluster(handle,
                        index,
                        n_rows_train,
                        trainset.data(),
                        labels.data(),
                        params.kmeans_n_iters,
                        &managed_memory,
                        device_memory);
      break;
    default: RAFT_FAIL("Unreachable code");
  }

  // add the data if necessary
  if (params.add_data_on_build) {
    return detail::extend<T, IdxT>(handle, index, dataset, nullptr, n_rows);
  } else {
    return index;
  }
}

static const int serialization_version = 1;

/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index_ IVF-PQ index
 *
 */
template <typename IdxT>
void save(const handle_t& handle_, const std::string& filename, const index<IdxT>& index_)
{
  std::ofstream of(filename, std::ios::out | std::ios::binary);
  if (!of) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  RAFT_LOG_DEBUG("Size %zu, dim %d, pq_dim %d, pq_bits %d",
                 static_cast<size_t>(index_.size()),
                 static_cast<int>(index_.dim()),
                 static_cast<int>(index_.pq_dim()),
                 static_cast<int>(index_.pq_bits()));

  write_scalar(of, serialization_version);
  write_scalar(of, index_.size());
  write_scalar(of, index_.dim());
  write_scalar(of, index_.pq_bits());
  write_scalar(of, index_.pq_dim());

  write_scalar(of, index_.metric());
  write_scalar(of, index_.codebook_kind());
  write_scalar(of, index_.n_lists());
  write_scalar(of, index_.n_nonempty_lists());

  write_mdspan(handle_, of, index_.pq_centers());
  write_mdspan(handle_, of, index_.pq_dataset());
  write_mdspan(handle_, of, index_.indices());
  write_mdspan(handle_, of, index_.rotation_matrix());
  write_mdspan(handle_, of, index_.list_offsets());
  write_mdspan(handle_, of, index_.list_sizes());
  write_mdspan(handle_, of, index_.centers());
  write_mdspan(handle_, of, index_.centers_rot());

  of.close();
  if (!of) { RAFT_FAIL("Error writing output %s", filename.c_str()); }
  return;
}

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[in] index_ IVF-PQ index
 *
 */
template <typename IdxT>
auto load(const handle_t& handle_, const std::string& filename) -> index<IdxT>
{
  std::ifstream infile(filename, std::ios::in | std::ios::binary);

  if (!infile) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  auto ver = read_scalar<int>(infile);
  if (ver != serialization_version) {
    RAFT_FAIL("serialization version mismatch %d vs. %d", ver, serialization_version);
  }
  auto n_rows  = read_scalar<IdxT>(infile);
  auto dim     = read_scalar<uint32_t>(infile);
  auto pq_bits = read_scalar<uint32_t>(infile);
  auto pq_dim  = read_scalar<uint32_t>(infile);

  auto metric           = read_scalar<raft::distance::DistanceType>(infile);
  auto codebook_kind    = read_scalar<raft::neighbors::ivf_pq::codebook_gen>(infile);
  auto n_lists          = read_scalar<uint32_t>(infile);
  auto n_nonempty_lists = read_scalar<uint32_t>(infile);

  RAFT_LOG_DEBUG("n_rows %zu, dim %d, pq_dim %d, pq_bits %d, n_lists %d",
                 static_cast<size_t>(n_rows),
                 static_cast<int>(dim),
                 static_cast<int>(pq_dim),
                 static_cast<int>(pq_bits),
                 static_cast<int>(n_lists));

  auto index_ = raft::neighbors::ivf_pq::index<IdxT>(
    handle_, metric, codebook_kind, n_lists, dim, pq_bits, pq_dim, n_nonempty_lists);
  index_.allocate(handle_, n_rows);

  read_mdspan(handle_, infile, index_.pq_centers());
  read_mdspan(handle_, infile, index_.pq_dataset());
  read_mdspan(handle_, infile, index_.indices());
  read_mdspan(handle_, infile, index_.rotation_matrix());
  read_mdspan(handle_, infile, index_.list_offsets());
  read_mdspan(handle_, infile, index_.list_sizes());
  read_mdspan(handle_, infile, index_.centers());
  read_mdspan(handle_, infile, index_.centers_rot());

  infile.close();

  return index_;
}

}  // namespace raft::spatial::knn::ivf_pq::detail
