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

#include <raft/spatial/knn/detail/ann_utils.cuh>

#include <raft/neighbors/ivf_list.hpp>
#include <raft/neighbors/ivf_pq_types.hpp>

#include <raft/cluster/kmeans_balanced.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/operators.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/detail/qr.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/matrix/linewise_op.cuh>
#include <raft/random/rng.cuh>
#include <raft/stats/histogram.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/device_atomics.cuh>
#include <raft/util/integer_utils.hpp>
#include <raft/util/pow2_utils.cuh>
#include <raft/util/vectorized.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <thrust/extrema.h>
#include <thrust/scan.h>

#include <memory>
#include <variant>

namespace raft::neighbors::ivf_pq::detail {

using namespace raft::spatial::knn::detail;  // NOLINT

/** A chunk of PQ-encoded vector managed by one CUDA thread. */
using pq_vec_t = TxN_t<uint8_t, kIndexGroupVecLen>::io_t;

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
inline void make_rotation_matrix(raft::device_resources const& handle,
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
    linalg::map_offset(handle, rotation_matrix_view, [stride] __device__(uint32_t i) {
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
void select_residuals(raft::device_resources const& handle,
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
  // Note: the number of rows of the input dataset isn't actually n_rows, but matrix::gather doesn't
  // need to know it, any strictly positive number would work.
  cub::TransformInputIterator<float, utils::mapping<float>, const T*> mapping_itr(
    dataset, utils::mapping<float>{});
  raft::matrix::gather(mapping_itr, (IdxT)dim, n_rows, row_ids, n_rows, tmp.data(), stream);

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
  raft::device_resources const& handle,
  float* residuals,  // [n_rows, rot_dim]
  IdxT n_rows,
  device_matrix_view<const float, uint32_t, row_major> rotation_matrix,  // [rot_dim, dim]
  device_matrix_view<const float, uint32_t, row_major> centers,          // [n_lists, dim_ext]
  const T* dataset,                                                      // [n_rows, dim]
  const uint32_t* labels,                                                // [n_rows]
  rmm::mr::device_memory_resource* device_memory)
{
  auto stream  = handle.get_stream();
  auto dim     = rotation_matrix.extent(1);
  auto rot_dim = rotation_matrix.extent(0);
  rmm::device_uvector<float> tmp(n_rows * dim, stream, device_memory);
  auto tmp_view = raft::make_device_vector_view<float, IdxT>(tmp.data(), tmp.size());
  linalg::map_offset(handle, tmp_view, [centers, dataset, labels, dim] __device__(size_t i) {
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
void transpose_pq_centers(const device_resources& handle,
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
  linalg::map_offset(handle, pq_centers_view, [span_source, extents] __device__(size_t i) {
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
void train_per_subset(raft::device_resources const& handle,
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

    // clone the handle and attached the device memory resource to it
    const device_resources new_handle(handle, device_memory);

    // train PQ codebook for this subspace
    auto sub_trainset_view =
      raft::make_device_matrix_view<const float, IdxT>(sub_trainset.data(), n_rows, index.pq_len());
    auto centers_tmp_view = raft::make_device_matrix_view<float, IdxT>(
      pq_centers_tmp.data() + index.pq_book_size() * index.pq_len() * j,
      index.pq_book_size(),
      index.pq_len());
    auto sub_labels_view = raft::make_device_vector_view<uint32_t, IdxT>(sub_labels.data(), n_rows);
    auto cluster_sizes_view =
      raft::make_device_vector_view<uint32_t, IdxT>(pq_cluster_sizes.data(), index.pq_book_size());
    raft::cluster::kmeans_balanced_params kmeans_params;
    kmeans_params.n_iters = kmeans_n_iters;
    kmeans_params.metric  = raft::distance::DistanceType::L2Expanded;
    raft::cluster::kmeans_balanced::helpers::build_clusters(new_handle,
                                                            kmeans_params,
                                                            sub_trainset_view,
                                                            centers_tmp_view,
                                                            sub_labels_view,
                                                            cluster_sizes_view,
                                                            utils::mapping<float>{});
  }
  transpose_pq_centers(handle, index, pq_centers_tmp.data());
}

template <typename IdxT>
void train_per_cluster(raft::device_resources const& handle,
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
  rmm::device_uvector<IdxT> offsets_buf(index.n_lists() + 1, stream, managed_memory);

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

    // clone the handle and attached the device memory resource to it
    const device_resources new_handle(handle, device_memory);

    // limit the cluster size to bound the training time.
    // [sic] we interpret the data as pq_len-dimensional
    size_t big_enough     = 256ul * std::max<size_t>(index.pq_book_size(), index.pq_dim());
    size_t available_rows = size_t(cluster_size) * size_t(index.pq_dim());
    auto pq_n_rows        = uint32_t(std::min(big_enough, available_rows));
    // train PQ codebook for this cluster
    auto rot_vectors_view = raft::make_device_matrix_view<const float, IdxT>(
      rot_vectors.data(), pq_n_rows, index.pq_len());
    auto centers_tmp_view = raft::make_device_matrix_view<float, IdxT>(
      pq_centers_tmp.data() + static_cast<size_t>(index.pq_book_size()) *
                                static_cast<size_t>(index.pq_len()) * static_cast<size_t>(l),
      index.pq_book_size(),
      index.pq_len());
    auto pq_labels_view =
      raft::make_device_vector_view<uint32_t, IdxT>(pq_labels.data(), pq_n_rows);
    auto pq_cluster_sizes_view =
      raft::make_device_vector_view<uint32_t, IdxT>(pq_cluster_sizes.data(), index.pq_book_size());
    raft::cluster::kmeans_balanced_params kmeans_params;
    kmeans_params.n_iters = kmeans_n_iters;
    kmeans_params.metric  = raft::distance::DistanceType::L2Expanded;
    raft::cluster::kmeans_balanced::helpers::build_clusters(new_handle,
                                                            kmeans_params,
                                                            rot_vectors_view,
                                                            centers_tmp_view,
                                                            pq_labels_view,
                                                            pq_cluster_sizes_view,
                                                            utils::mapping<float>{});
  }
  transpose_pq_centers(handle, index, pq_centers_tmp.data());
}

/**
 * Decode a lvl-2 pq-encoded vector in the given list (cluster).
 * One vector per thread.
 * NB: this function only decodes the PQ (second level) encoding; to get the approximation of the
 * original vector, you need to add the cluster centroid and apply the inverse matrix transform to
 * the result of this function.
 *
 * @tparam PqBits
 *
 * @param[out] out_vector the destination for the decoded vector (one-per-thread).
 * @param[in] in_list_data the encoded cluster data.
 * @param[in] pq_centers the codebook
 * @param[in] codebook_kind
 * @param[in] in_ix in-cluster index of the vector to be decoded (one-per-thread).
 * @param[in] cluster_ix label/id of the cluster (one-per-thread).
 */
template <uint32_t PqBits>
__device__ void reconstruct_vector(
  device_vector_view<float, uint32_t, row_major> out_vector,
  device_mdspan<const uint8_t, list_spec<uint32_t, uint32_t>::list_extents, row_major> in_list_data,
  device_mdspan<const float, extent_3d<uint32_t>, row_major> pq_centers,
  codebook_gen codebook_kind,
  uint32_t in_ix,
  uint32_t cluster_ix)
{
  using group_align         = Pow2<kIndexGroupSize>;
  const uint32_t group_ix   = group_align::div(in_ix);
  const uint32_t ingroup_ix = group_align::mod(in_ix);
  const uint32_t pq_len     = pq_centers.extent(1);
  const uint32_t pq_dim     = out_vector.extent(0) / pq_len;

  using layout_t            = typename decltype(out_vector)::layout_type;
  using accessor_t          = typename decltype(out_vector)::accessor_type;
  auto reinterpreted_vector = mdspan<float, extent_2d<uint32_t>, layout_t, accessor_t>(
    out_vector.data_handle(), extent_2d<uint32_t>{pq_dim, pq_len});

  pq_vec_t code_chunk;
  bitfield_view_t<PqBits> code_view{reinterpret_cast<uint8_t*>(&code_chunk)};
  constexpr uint32_t kChunkSize = (sizeof(pq_vec_t) * 8u) / PqBits;
  for (uint32_t j = 0, i = 0; j < pq_dim; i++) {
    // read the chunk
    code_chunk = *reinterpret_cast<const pq_vec_t*>(&in_list_data(group_ix, i, ingroup_ix, 0));
    // read the codes, one/pq_dim at a time
#pragma unroll
    for (uint32_t k = 0; k < kChunkSize && j < pq_dim; k++, j++) {
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
      uint8_t code = code_view[k];
      // read a piece of the reconstructed vector
      for (uint32_t l = 0; l < pq_len; l++) {
        reinterpreted_vector(j, l) = pq_centers(partition_ix, l, code);
      }
    }
  }
}

template <uint32_t BlockSize, uint32_t PqBits>
__launch_bounds__(BlockSize) __global__ void reconstruct_list_data_kernel(
  device_matrix_view<float, uint32_t, row_major> out_vectors,
  device_vector_view<const uint8_t* const, uint32_t, row_major> data_ptrs,
  device_vector_view<const uint32_t, uint32_t, row_major> list_sizes,
  device_mdspan<const float, extent_3d<uint32_t>, row_major> pq_centers,
  device_matrix_view<const float, uint32_t, row_major> centers_rot,
  codebook_gen codebook_kind,
  uint32_t cluster_ix,
  std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  const auto out_dim = out_vectors.extent(1);
  using layout_t     = typename decltype(out_vectors)::layout_type;
  using accessor_t   = typename decltype(out_vectors)::accessor_type;

  const uint32_t pq_dim = out_dim / pq_centers.extent(1);
  auto pq_extents =
    list_spec<uint32_t, uint32_t>{PqBits, pq_dim, true}.make_list_extents(list_sizes[cluster_ix]);
  auto pq_dataset =
    make_mdspan<const uint8_t, uint32_t, row_major, false, true>(data_ptrs[cluster_ix], pq_extents);

  for (uint32_t ix = threadIdx.x + BlockSize * blockIdx.x; ix < out_vectors.extent(0);
       ix += BlockSize) {
    auto one_vector = mdspan<float, extent_1d<uint32_t>, layout_t, accessor_t>(
      &out_vectors(ix, 0), extent_1d<uint32_t>{out_vectors.extent(1)});
    const uint32_t src_ix = std::holds_alternative<uint32_t>(offset_or_indices)
                              ? std::get<uint32_t>(offset_or_indices) + ix
                              : std::get<const uint32_t*>(offset_or_indices)[ix];
    reconstruct_vector<PqBits>(
      one_vector, pq_dataset, pq_centers, codebook_kind, src_ix, cluster_ix);
    for (uint32_t j = 0; j < out_dim; j++) {
      one_vector(j) += centers_rot(cluster_ix, j);
    }
  }
}

/** Decode the list data; see the public interface for the api and usage. */
template <typename T, typename IdxT>
void reconstruct_list_data(raft::device_resources const& res,
                           const index<IdxT>& index,
                           device_matrix_view<T, uint32_t, row_major> out_vectors,
                           uint32_t label,
                           std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  auto n_rows = out_vectors.extent(0);
  if (n_rows == 0) { return; }
  if (std::holds_alternative<uint32_t>(offset_or_indices)) {
    auto n_skip = std::get<uint32_t>(offset_or_indices);
    // sic! I'm using the upper bound `list.size` instead of exact `list_sizes(label)`
    // to avoid an extra device-host data copy and the stream sync.
    RAFT_EXPECTS(n_skip + n_rows <= index.lists()[label]->size.load(),
                 "offset + output size must be not bigger than the cluster size.");
  }

  auto tmp = make_device_mdarray<float>(
    res, res.get_workspace_resource(), make_extents<uint32_t>(n_rows, index.rot_dim()));

  constexpr uint32_t kBlockSize = 256;
  dim3 blocks(div_rounding_up_safe<uint32_t>(n_rows, kBlockSize), 1, 1);
  dim3 threads(kBlockSize, 1, 1);
  auto kernel = [](uint32_t pq_bits) {
    switch (pq_bits) {
      case 4: return reconstruct_list_data_kernel<kBlockSize, 4>;
      case 5: return reconstruct_list_data_kernel<kBlockSize, 5>;
      case 6: return reconstruct_list_data_kernel<kBlockSize, 6>;
      case 7: return reconstruct_list_data_kernel<kBlockSize, 7>;
      case 8: return reconstruct_list_data_kernel<kBlockSize, 8>;
      default: RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 8]", pq_bits);
    }
  }(index.pq_bits());
  kernel<<<blocks, threads, 0, res.get_stream()>>>(tmp.view(),
                                                   index.data_ptrs(),
                                                   index.list_sizes(),
                                                   index.pq_centers(),
                                                   index.centers_rot(),
                                                   index.codebook_kind(),
                                                   label,
                                                   offset_or_indices);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  float* out_float_ptr = nullptr;
  rmm::device_uvector<float> out_float_buf(0, res.get_stream(), res.get_workspace_resource());
  if constexpr (std::is_same_v<T, float>) {
    out_float_ptr = out_vectors.data_handle();
  } else {
    out_float_buf.resize(size_t{n_rows} * size_t{index.dim()}, res.get_stream());
    out_float_ptr = out_float_buf.data();
  }
  // Rotate the results back to the original space
  float alpha = 1.0;
  float beta  = 0.0;
  linalg::gemm(res,
               false,
               false,
               index.dim(),
               n_rows,
               index.rot_dim(),
               &alpha,
               index.rotation_matrix().data_handle(),
               index.dim(),
               tmp.data_handle(),
               index.rot_dim(),
               &beta,
               out_float_ptr,
               index.dim(),
               res.get_stream());
  // Transform the data to the original type, if necessary
  if constexpr (!std::is_same_v<T, float>) {
    linalg::map(res,
                out_vectors,
                utils::mapping<T>{},
                make_device_matrix_view<const float>(out_float_ptr, n_rows, index.dim()));
  }
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

/**
 * Compute a PQ code for a single input vector per subwarp and write it into the
 * appropriate cluster.
 * Subwarp size here is the minimum between WarpSize and the codebook size.
 *
 * @tparam BlockSize
 * @tparam PqBits
 *
 * @param[out] out_list_data an array of pointers to the database clusers.
 * @param[in] in_vector input unencoded data, one-per-subwarp
 * @param[in] pq_centers codebook
 * @param[in] codebook_kind
 * @param[in] out_ix in-cluster output index (where to write the encoded data), one-per-subwarp.
 * @param[in] cluster_ix label/id of the cluster to fill, one-per-subwarp.
 */
template <uint32_t BlockSize, uint32_t PqBits>
__device__ auto compute_and_write_pq_code(
  device_mdspan<uint8_t, list_spec<uint32_t, uint32_t>::list_extents, row_major> out_list_data,
  device_vector_view<const float, uint32_t, row_major> in_vector,
  device_mdspan<const float, extent_3d<uint32_t>, row_major> pq_centers,
  codebook_gen codebook_kind,
  uint32_t out_ix,
  uint32_t cluster_ix)
{
  constexpr uint32_t kSubWarpSize = std::min<uint32_t>(WarpSize, 1u << PqBits);
  using subwarp_align             = Pow2<kSubWarpSize>;
  const uint32_t lane_id          = subwarp_align::mod(threadIdx.x);

  using group_align         = Pow2<kIndexGroupSize>;
  const uint32_t group_ix   = group_align::div(out_ix);
  const uint32_t ingroup_ix = group_align::mod(out_ix);
  const uint32_t pq_len     = pq_centers.extent(1);
  const uint32_t pq_dim     = in_vector.extent(0) / pq_len;

  using layout_t            = typename decltype(in_vector)::layout_type;
  using accessor_t          = typename decltype(in_vector)::accessor_type;
  auto reinterpreted_vector = mdspan<const float, extent_2d<uint32_t>, layout_t, accessor_t>(
    in_vector.data_handle(), extent_2d<uint32_t>{pq_dim, pq_len});

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
      auto l = compute_pq_code<kSubWarpSize>(
        pq_centers, reinterpreted_vector, codebook_kind, j, cluster_ix);
      if (lane_id == 0) { out[k] = l; }
    }
    // write the chunk into the dataset
    if (lane_id == 0) {
      *reinterpret_cast<pq_vec_t*>(&out_list_data(group_ix, i, ingroup_ix, 0)) = code;
    }
  }
}

template <uint32_t BlockSize, uint32_t PqBits, typename IdxT>
__launch_bounds__(BlockSize) __global__ void process_and_fill_codes_kernel(
  device_matrix_view<const float, IdxT, row_major> new_vectors,
  std::variant<IdxT, const IdxT*> src_offset_or_indices,
  const uint32_t* new_labels,
  device_vector_view<uint32_t, uint32_t, row_major> list_sizes,
  device_vector_view<IdxT*, uint32_t, row_major> inds_ptrs,
  device_vector_view<uint8_t*, uint32_t, row_major> data_ptrs,
  device_mdspan<const float, extent_3d<uint32_t>, row_major> pq_centers,
  codebook_gen codebook_kind)
{
  constexpr uint32_t kSubWarpSize = std::min<uint32_t>(WarpSize, 1u << PqBits);
  using subwarp_align             = Pow2<kSubWarpSize>;
  const uint32_t lane_id          = subwarp_align::mod(threadIdx.x);
  const IdxT row_ix = subwarp_align::div(IdxT{threadIdx.x} + IdxT{BlockSize} * IdxT{blockIdx.x});
  if (row_ix >= new_vectors.extent(0)) { return; }

  const uint32_t cluster_ix = new_labels[row_ix];
  uint32_t out_ix;
  if (lane_id == 0) { out_ix = atomicAdd(&list_sizes(cluster_ix), 1); }
  out_ix = shfl(out_ix, 0, kSubWarpSize);

  // write the label  (one record per subwarp)
  auto pq_indices = inds_ptrs(cluster_ix);
  if (lane_id == 0) {
    if (std::holds_alternative<IdxT>(src_offset_or_indices)) {
      pq_indices[out_ix] = std::get<IdxT>(src_offset_or_indices) + row_ix;
    } else {
      pq_indices[out_ix] = std::get<const IdxT*>(src_offset_or_indices)[row_ix];
    }
  }

  // write the codes (one record per subwarp):
  // 1. select input row
  using layout_t    = typename decltype(new_vectors)::layout_type;
  using accessor_t  = typename decltype(new_vectors)::accessor_type;
  const auto in_dim = new_vectors.extent(1);
  auto one_vector =
    mdspan<const float, extent_1d<uint32_t>, layout_t, accessor_t>(&new_vectors(row_ix, 0), in_dim);
  // 2. select output cluster
  const uint32_t pq_dim = in_dim / pq_centers.extent(1);
  auto pq_extents = list_spec<uint32_t, IdxT>{PqBits, pq_dim, true}.make_list_extents(out_ix + 1);
  auto pq_dataset =
    make_mdspan<uint8_t, uint32_t, row_major, false, true>(data_ptrs[cluster_ix], pq_extents);
  // 3. compute and write the vector
  compute_and_write_pq_code<BlockSize, PqBits>(
    pq_dataset, one_vector, pq_centers, codebook_kind, out_ix, cluster_ix);
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
void process_and_fill_codes(raft::device_resources const& handle,
                            index<IdxT>& index,
                            const T* new_vectors,
                            std::variant<IdxT, const IdxT*> src_offset_or_indices,
                            const uint32_t* new_labels,
                            IdxT n_rows,
                            rmm::mr::device_memory_resource* mr)
{
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
                                                      index.inds_ptrs(),
                                                      index.data_ptrs(),
                                                      index.pq_centers(),
                                                      index.codebook_kind());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/** Update the state of the dependent index members. */
template <typename IdxT>
void recompute_internal_state(const raft::device_resources& res, index<IdxT>& index)
{
  auto stream  = res.get_stream();
  auto tmp_res = res.get_workspace_resource();
  rmm::device_uvector<uint32_t> sorted_sizes(index.n_lists(), stream, tmp_res);

  // Actualize the list pointers
  auto data_ptrs = index.data_ptrs();
  auto inds_ptrs = index.inds_ptrs();
  for (uint32_t label = 0; label < index.n_lists(); label++) {
    auto& list          = index.lists()[label];
    const auto data_ptr = list ? list->data.data_handle() : nullptr;
    const auto inds_ptr = list ? list->indices.data_handle() : nullptr;
    copy(&data_ptrs(label), &data_ptr, 1, stream);
    copy(&inds_ptrs(label), &inds_ptr, 1, stream);
  }

  // Sort the cluster sizes in the descending order.
  int begin_bit             = 0;
  int end_bit               = sizeof(uint32_t) * 8;
  size_t cub_workspace_size = 0;
  cub::DeviceRadixSort::SortKeysDescending(nullptr,
                                           cub_workspace_size,
                                           index.list_sizes().data_handle(),
                                           sorted_sizes.data(),
                                           index.n_lists(),
                                           begin_bit,
                                           end_bit,
                                           stream);
  rmm::device_buffer cub_workspace(cub_workspace_size, stream, tmp_res);
  cub::DeviceRadixSort::SortKeysDescending(cub_workspace.data(),
                                           cub_workspace_size,
                                           index.list_sizes().data_handle(),
                                           sorted_sizes.data(),
                                           index.n_lists(),
                                           begin_bit,
                                           end_bit,
                                           stream);
  // copy the results to CPU
  std::vector<uint32_t> sorted_sizes_host(index.n_lists());
  copy(sorted_sizes_host.data(), sorted_sizes.data(), index.n_lists(), stream);
  res.sync_stream();

  // accumulate the sorted cluster sizes
  auto accum_sorted_sizes = index.accum_sorted_sizes();
  accum_sorted_sizes(0)   = 0;
  for (uint32_t label = 0; label < sorted_sizes_host.size(); label++) {
    accum_sorted_sizes(label + 1) = accum_sorted_sizes(label) + sorted_sizes_host[label];
  }
}

/** Copy the state of an index into a new index, but share the list data among the two. */
template <typename IdxT>
auto clone(const raft::device_resources& res, const index<IdxT>& source) -> index<IdxT>
{
  auto stream = res.get_stream();

  // Allocate the new index
  index<IdxT> target(res,
                     source.metric(),
                     source.codebook_kind(),
                     source.n_lists(),
                     source.dim(),
                     source.pq_bits(),
                     source.pq_dim());

  // Copy the independent parts
  copy(target.list_sizes().data_handle(),
       source.list_sizes().data_handle(),
       source.list_sizes().size(),
       stream);
  copy(target.rotation_matrix().data_handle(),
       source.rotation_matrix().data_handle(),
       source.rotation_matrix().size(),
       stream);
  copy(target.pq_centers().data_handle(),
       source.pq_centers().data_handle(),
       source.pq_centers().size(),
       stream);
  copy(target.centers().data_handle(),
       source.centers().data_handle(),
       source.centers().size(),
       stream);
  copy(target.centers_rot().data_handle(),
       source.centers_rot().data_handle(),
       source.centers_rot().size(),
       stream);

  // Copy shared pointers
  target.lists() = source.lists();

  // Make sure the device pointers point to the new lists
  recompute_internal_state(res, target);

  return target;
}

/**
 * Extend the index in-place.
 * See raft::spatial::knn::ivf_pq::extend docs.
 */
template <typename T, typename IdxT>
void extend(raft::device_resources const& handle,
            index<IdxT>* index,
            const T* new_vectors,
            const IdxT* new_indices,
            IdxT n_rows)
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_pq::extend(%zu, %u)", size_t(n_rows), index->dim());
  auto stream           = handle.get_stream();
  const auto n_clusters = index->n_lists();

  RAFT_EXPECTS(new_indices != nullptr || index->size() == 0,
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

  // The spec defines how the clusters look like
  auto spec = list_spec<uint32_t, IdxT>{
    index->pq_bits(), index->pq_dim(), index->conservative_memory_allocation()};
  // Try to allocate an index with the same parameters and the projected new size
  // (which can be slightly larger than index->size() + n_rows, due to padding).
  // If this fails, the index would be too big to fit in the device anyway.
  std::optional<list_data<IdxT, size_t>> placeholder_list(
    std::in_place_t{},
    handle,
    list_spec<size_t, IdxT>{spec},
    n_rows + (kIndexGroupSize - 1) * std::min<IdxT>(n_clusters, n_rows));

  // Available device memory
  size_t free_mem, total_mem;
  RAFT_CUDA_TRY(cudaMemGetInfo(&free_mem, &total_mem));

  // Decide on an approximate threshold when we'd better start saving device memory by using
  // managed allocations for large device buffers
  rmm::mr::device_memory_resource* labels_mr  = device_memory;
  rmm::mr::device_memory_resource* batches_mr = device_memory;
  if (n_rows * (index->dim() * sizeof(T) + index->pq_dim() + sizeof(IdxT) + sizeof(uint32_t)) >
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
    size_factor += (index->dim() + index->rot_dim()) * sizeof(float);
    // ...and another buffer for indices
    size_factor += sizeof(IdxT);
    // if the input data is not accessible on device, we'd need a buffer for it.
    switch (utils::check_pointer_residency(new_vectors)) {
      case utils::pointer_residency::device_only:
      case utils::pointer_residency::host_and_device: break;
      default: size_factor += index->dim() * sizeof(T);
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
    new_vectors, n_rows, index->dim(), max_batch_size, stream, batches_mr);
  // Release the placeholder memory, because we don't intend to allocate any more long-living
  // temporary buffers before we allocate the index data.
  // This memory could potentially speed up UVM accesses, if any.
  placeholder_list.reset();
  {
    // The cluster centers in the index are stored padded, which is not acceptable by
    // the kmeans_balanced::predict. Thus, we need the restructuring copy.
    rmm::device_uvector<float> cluster_centers(
      size_t(n_clusters) * size_t(index->dim()), stream, device_memory);
    RAFT_CUDA_TRY(cudaMemcpy2DAsync(cluster_centers.data(),
                                    sizeof(float) * index->dim(),
                                    index->centers().data_handle(),
                                    sizeof(float) * index->dim_ext(),
                                    sizeof(float) * index->dim(),
                                    n_clusters,
                                    cudaMemcpyDefault,
                                    stream));
    for (const auto& batch : vec_batches) {
      auto batch_data_view =
        raft::make_device_matrix_view<const T, IdxT>(batch.data(), batch.size(), index->dim());
      auto batch_labels_view = raft::make_device_vector_view<uint32_t, IdxT>(
        new_data_labels.data() + batch.offset(), batch.size());
      auto centers_view = raft::make_device_matrix_view<const float, IdxT>(
        cluster_centers.data(), n_clusters, index->dim());
      raft::cluster::kmeans_balanced_params kmeans_params;
      kmeans_params.metric = index->metric();
      raft::cluster::kmeans_balanced::predict(handle,
                                              kmeans_params,
                                              batch_data_view,
                                              centers_view,
                                              batch_labels_view,
                                              utils::mapping<float>{});
    }
  }

  auto list_sizes = index->list_sizes().data_handle();
  // store the current cluster sizes, because we'll need them later
  rmm::device_uvector<uint32_t> orig_list_sizes(n_clusters, stream, device_memory);
  copy(orig_list_sizes.data(), list_sizes, n_clusters, stream);

  // Get the combined cluster sizes
  raft::stats::histogram<uint32_t, IdxT>(raft::stats::HistTypeAuto,
                                         reinterpret_cast<int32_t*>(list_sizes),
                                         IdxT(n_clusters),
                                         new_data_labels.data(),
                                         n_rows,
                                         1,
                                         stream);
  linalg::add(list_sizes, list_sizes, orig_list_sizes.data(), n_clusters, stream);

  // Allocate the lists to fit the new data
  {
    std::vector<uint32_t> new_cluster_sizes(n_clusters);
    std::vector<uint32_t> old_cluster_sizes(n_clusters);
    copy(new_cluster_sizes.data(), list_sizes, n_clusters, stream);
    copy(old_cluster_sizes.data(), orig_list_sizes.data(), n_clusters, stream);
    handle.sync_stream();
    for (uint32_t label = 0; label < n_clusters; label++) {
      ivf::resize_list(
        handle, index->lists()[label], spec, new_cluster_sizes[label], old_cluster_sizes[label]);
    }
  }

  // Update the pointers and the sizes
  recompute_internal_state(handle, *index);

  // Recover old cluster sizes: they are used as counters in the fill-codes kernel
  copy(list_sizes, orig_list_sizes.data(), n_clusters, stream);

  // By this point, the index state is updated and valid except it doesn't contain the new data
  // Fill the extended index with the new data (possibly, in batches)
  utils::batch_load_iterator<IdxT> idx_batches(
    new_indices, n_rows, 1, max_batch_size, stream, batches_mr);
  for (const auto& vec_batch : vec_batches) {
    const auto& idx_batch = *idx_batches++;
    process_and_fill_codes(handle,
                           *index,
                           vec_batch.data(),
                           new_indices != nullptr
                             ? std::variant<IdxT, const IdxT*>(idx_batch.data())
                             : std::variant<IdxT, const IdxT*>(IdxT(idx_batch.offset())),
                           new_data_labels.data() + vec_batch.offset(),
                           IdxT(vec_batch.size()),
                           batches_mr);
  }
}

/**
 * Create a new index that contains more data.
 * See raft::spatial::knn::ivf_pq::extend docs.
 */
template <typename T, typename IdxT>
auto extend(raft::device_resources const& handle,
            const index<IdxT>& orig_index,
            const T* new_vectors,
            const IdxT* new_indices,
            IdxT n_rows) -> index<IdxT>
{
  auto ext_index = clone(handle, orig_index);
  detail::extend(handle, &ext_index, new_vectors, new_indices, n_rows);
  return ext_index;
}

/** See raft::spatial::knn::ivf_pq::build docs */
template <typename T, typename IdxT>
auto build(raft::device_resources const& handle,
           const index_params& params,
           const T* dataset,
           IdxT n_rows,
           uint32_t dim) -> index<IdxT>
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_pq::build(%zu, %u)", size_t(n_rows), dim);
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
                "Unsupported data type");

  RAFT_EXPECTS(n_rows > 0 && dim > 0, "empty dataset");

  auto stream = handle.get_stream();

  index<IdxT> index(handle, params, dim);
  utils::memzero(
    index.accum_sorted_sizes().data_handle(), index.accum_sorted_sizes().size(), stream);
  utils::memzero(index.list_sizes().data_handle(), index.list_sizes().size(), stream);
  utils::memzero(index.data_ptrs().data_handle(), index.data_ptrs().size(), stream);
  utils::memzero(index.inds_ptrs().data_handle(), index.inds_ptrs().size(), stream);

  {
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
        linalg::map_offset(handle, trainset_view, [p, trainset_ratio, dim] __device__(size_t i) {
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

    // NB: here cluster_centers is used as if it is [n_clusters, data_dim] not [n_clusters,
    // dim_ext]!
    rmm::device_uvector<float> cluster_centers_buf(
      index.n_lists() * index.dim(), stream, device_memory);
    auto cluster_centers = cluster_centers_buf.data();

    // Train balanced hierarchical kmeans clustering
    auto trainset_const_view =
      raft::make_device_matrix_view<const float, IdxT>(trainset.data(), n_rows_train, index.dim());
    auto centers_view =
      raft::make_device_matrix_view<float, IdxT>(cluster_centers, index.n_lists(), index.dim());
    raft::cluster::kmeans_balanced_params kmeans_params;
    kmeans_params.n_iters = params.kmeans_n_iters;
    kmeans_params.metric  = index.metric();
    raft::cluster::kmeans_balanced::fit(
      handle, kmeans_params, trainset_const_view, centers_view, utils::mapping<float>{});

    // Trainset labels are needed for training PQ codebooks
    rmm::device_uvector<uint32_t> labels(n_rows_train, stream, big_memory_resource);
    auto centers_const_view = raft::make_device_matrix_view<const float, IdxT>(
      cluster_centers, index.n_lists(), index.dim());
    auto labels_view = raft::make_device_vector_view<uint32_t, IdxT>(labels.data(), n_rows_train);
    raft::cluster::kmeans_balanced::predict(handle,
                                            kmeans_params,
                                            trainset_const_view,
                                            centers_const_view,
                                            labels_view,
                                            utils::mapping<float>());

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
  }

  // add the data if necessary
  if (params.add_data_on_build) {
    detail::extend<T, IdxT>(handle, &index, dataset, nullptr, n_rows);
  }
  return index;
}
}  // namespace raft::neighbors::ivf_pq::detail
