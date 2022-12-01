/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <raft/util/pow2_utils.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <thrust/binary_search.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

namespace raft::spatial::knn::ivf_pq::detail {

using namespace raft::spatial::knn::detail;  // NOLINT

using raft::neighbors::ivf_pq::codebook_gen;
using raft::neighbors::ivf_pq::index;
using raft::neighbors::ivf_pq::index_params;
using raft::neighbors::ivf_pq::kIndexGroupSize;
using raft::neighbors::ivf_pq::kIndexGroupVecLen;

using pq_codes_exts = extents<size_t, dynamic_extent, dynamic_extent, kIndexGroupVecLen>;

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

/*
  NB: label type is uint32_t although it can only contain values up to `1 << pq_bits`.
      We keep it this way to not force one more overload for kmeans::predict.
 */
template <uint32_t PqBits, size_t VecLen>
__device__ void ivfpq_encode_core(uint32_t n_rows,
                                  uint32_t pq_dim,
                                  const uint32_t* label,
                                  uint8_t* output)
{
  constexpr uint32_t kChunkSize = (VecLen * 8u) / PqBits;
  TxN_t<uint8_t, VecLen> vec;
  for (uint32_t j = 0; j < pq_dim;) {
    vec.fill(0);
    bitfield_view_t<PqBits> out{vec.val.data};
#pragma unroll
    for (uint32_t k = 0; k < kChunkSize && j < pq_dim; k++, j++, label += n_rows) {
      out[k] = static_cast<uint8_t>(*label);
    }
    vec.store(output, 0);
    output += VecLen;
  }
}

template <uint32_t BlockDim, uint32_t PqBits>
__launch_bounds__(BlockDim) __global__
  void ivfpq_encode_kernel(uint32_t pq_dim,
                           const uint32_t* label,  // [pq_dim, n_rows]
                           device_mdspan<uint8_t, pq_codes_exts, row_major> output  // [n_rows, ..]
  )
{
  uint32_t i = threadIdx.x + BlockDim * blockIdx.x;
  if (i >= output.extent(0)) return;
  ivfpq_encode_core<PqBits, output.static_extent(2)>(
    output.extent(0),
    pq_dim,
    label + i,
    output.data_handle() + output.extent(1) * output.extent(2) * i);
}
}  // namespace

/**
 * Compress the cluster labels into an encoding with pq_bits bits, and transform it into a form to
 * facilitate vectorized loads
 */
inline void ivfpq_encode(uint32_t pq_dim,
                         uint32_t pq_bits,       // 4 <= pq_bits <= 8
                         const uint32_t* label,  // [pq_dim, n_rows]
                         device_mdspan<uint8_t, pq_codes_exts, row_major> output,  // [n_rows, ..]
                         rmm::cuda_stream_view stream)
{
  constexpr uint32_t kBlockDim = 128;
  dim3 threads(kBlockDim, 1, 1);
  dim3 blocks(raft::ceildiv<uint32_t>(output.extent(0), kBlockDim), 1, 1);
  switch (pq_bits) {
    case 4:
      return ivfpq_encode_kernel<kBlockDim, 4>
        <<<blocks, threads, 0, stream>>>(pq_dim, label, output);
    case 5:
      return ivfpq_encode_kernel<kBlockDim, 5>
        <<<blocks, threads, 0, stream>>>(pq_dim, label, output);
    case 6:
      return ivfpq_encode_kernel<kBlockDim, 6>
        <<<blocks, threads, 0, stream>>>(pq_dim, label, output);
    case 7:
      return ivfpq_encode_kernel<kBlockDim, 7>
        <<<blocks, threads, 0, stream>>>(pq_dim, label, output);
    case 8:
      return ivfpq_encode_kernel<kBlockDim, 8>
        <<<blocks, threads, 0, stream>>>(pq_dim, label, output);
    default: RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 8]", pq_bits);
  }
}

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
    auto f = [stride] __device__(float* out, uint32_t i) -> void { *out = float(i % stride == 0); };
    linalg::writeOnlyUnaryOp(rotation_matrix, n * n, f, stream);
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
 *
 * @param handle,
 * @param n_rows
 * @param data_dim
 * @param rot_dim
 * @param pq_dim
 * @param pq_len
 * @param pq_bits
 * @param n_clusters
 * @param codebook_kind
 * @param max_cluster_size
 * @param cluster_centers           // [n_clusters, data_dim]
 * @param rotation_matrix     // [rot_dim, data_dim]
 * @param dataset                 // [n_rows]
 * @param data_indices
 *    tells which indices to select in the dataset for each cluster [n_rows];
 *    it should be partitioned by the clusters by now.
 * @param cluster_sizes    // [n_clusters]
 * @param cluster_offsets  // [n_clusters + 1]
 * @param pq_centers  // [...] (see ivf_pq::index::pq_centers() layout)
 * @param pq_dataset
 *   // [n_rows, ceildiv(pq_dim, (kIndexGroupVecLen * 8u) / pq_bits), kIndexGroupVecLen]
 *   NB: in contrast to the final interleaved layout in ivf_pq::index::pq_dataset(), this function
 *       produces a non-interleaved data; it gets interleaved later when adding the data to the
 *       index.
 * @param device_memory
 */
template <typename T, typename IdxT>
void compute_pq_codes(
  const handle_t& handle,
  IdxT n_rows,
  uint32_t data_dim,
  uint32_t rot_dim,
  uint32_t pq_dim,
  uint32_t pq_len,
  uint32_t pq_bits,
  uint32_t n_clusters,
  codebook_gen codebook_kind,
  uint32_t max_cluster_size,
  float* cluster_centers,
  const float* rotation_matrix,
  const T* dataset,
  const IdxT* data_indices,
  const uint32_t* cluster_sizes,
  const IdxT* cluster_offsets,
  device_mdspan<const float, typename index<IdxT>::pq_centers_extents, row_major> pq_centers,
  device_mdspan<uint8_t, pq_codes_exts, row_major> pq_dataset,
  rmm::mr::device_memory_resource* device_memory)
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_pq::compute_pq_codes(n_rows = %zu, data_dim = %u, rot_dim = %u (%u * %u), n_clusters = "
    "%u)",
    size_t(n_rows),
    data_dim,
    rot_dim,
    pq_dim,
    pq_len,
    n_clusters);
  auto stream = handle.get_stream();

  //
  // Compute PQ code
  //

  uint32_t pq_width = 1 << pq_bits;
  rmm::device_uvector<float> pq_centers_tmp(pq_len * pq_width, stream, device_memory);
  rmm::device_uvector<float> rot_vectors(
    size_t(max_cluster_size) * size_t(rot_dim), stream, device_memory);
  rmm::device_uvector<float> sub_vectors(
    size_t(max_cluster_size) * size_t(pq_dim * pq_len), stream, device_memory);
  rmm::device_uvector<uint32_t> sub_vector_labels(
    size_t(max_cluster_size) * size_t(pq_dim), stream, device_memory);

  for (uint32_t l = 0; l < n_clusters; l++) {
    auto cluster_size = cluster_sizes[l];
    common::nvtx::range<common::nvtx::domain::raft> cluster_scope(
      "ivf_pq::compute_pq_codes::cluster[%u](size = %u)", l, cluster_size);
    if (cluster_size == 0) continue;

    select_residuals(handle,
                     rot_vectors.data(),
                     IdxT(cluster_size),
                     data_dim,
                     rot_dim,
                     rotation_matrix,
                     cluster_centers + size_t(l) * size_t(data_dim),
                     dataset,
                     data_indices + cluster_offsets[l],
                     device_memory);

    //
    // Change the order of the vector data to facilitate processing in
    // each vector subspace.
    //   input:  rot_vectors[cluster_size, rot_dim] = [cluster_size, pq_dim, pq_len]
    //   output: sub_vectors[pq_dim, cluster_size, pq_len]
    //
    for (uint32_t i = 0; i < pq_dim; i++) {
      RAFT_CUDA_TRY(
        cudaMemcpy2DAsync(sub_vectors.data() + size_t(i) * size_t(pq_len) * size_t(cluster_size),
                          sizeof(float) * pq_len,
                          rot_vectors.data() + i * pq_len,
                          sizeof(float) * rot_dim,
                          sizeof(float) * pq_len,
                          cluster_size,
                          cudaMemcpyDefault,
                          stream));
    }

    if (codebook_kind == codebook_gen::PER_CLUSTER) {
      linalg::writeOnlyUnaryOp(
        pq_centers_tmp.data(),
        pq_len * pq_width,
        [pq_centers, pq_width, pq_len, l] __device__(float* out, uint32_t i) {
          auto i0 = i / pq_len;
          auto i1 = i % pq_len;
          *out    = pq_centers(l, i1, i0);
        },
        stream);
    }

    //
    // Find a label (cluster ID) for each vector subspace.
    //
    for (uint32_t j = 0; j < pq_dim; j++) {
      if (codebook_kind == codebook_gen::PER_SUBSPACE) {
        linalg::writeOnlyUnaryOp(
          pq_centers_tmp.data(),
          pq_len * pq_width,
          [pq_centers, pq_width, pq_len, j] __device__(float* out, uint32_t i) {
            auto i0 = i / pq_len;
            auto i1 = i % pq_len;
            *out    = pq_centers(j, i1, i0);
          },
          stream);
      }
      kmeans::predict(handle,
                      pq_centers_tmp.data(),
                      pq_width,
                      pq_len,
                      sub_vectors.data() + size_t(j) * size_t(cluster_size) * size_t(pq_len),
                      cluster_size,
                      sub_vector_labels.data() + size_t(j) * size_t(cluster_size),
                      raft::distance::DistanceType::L2Expanded,
                      stream,
                      device_memory);
    }

    //
    // PQ encoding
    //
    ivfpq_encode(
      pq_dim,
      pq_bits,
      sub_vector_labels.data(),
      make_mdspan<uint8_t, IdxT, row_major, false, true>(
        pq_dataset.data_handle() +
          size_t(cluster_offsets[l]) * pq_dataset.extent(1) * pq_dataset.extent(2),
        make_extents<IdxT>(cluster_size, pq_dataset.extent(1), pq_dataset.static_extent(2))),
      stream);
  }
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
    exec_policy, cluster_sizes, cluster_sizes + n_lists, cluster_offsets + 1, thrust::plus<IdxT>{});
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
void transpose_pq_centers(index<IdxT>& index,
                          const float* pq_centers_source,
                          rmm::cuda_stream_view stream)
{
  auto extents = index.pq_centers().extents();
  static_assert(extents.rank() == 3);
  auto extents_source =
    make_extents<uint32_t>(extents.extent(0), extents.extent(2), extents.extent(1));
  auto span_source =
    make_mdspan<const float, uint32_t, row_major, false, true>(pq_centers_source, extents_source);
  linalg::writeOnlyUnaryOp(
    index.pq_centers().data_handle(),
    index.pq_centers().size(),
    [span_source, extents] __device__(float* out, size_t i) {
      uint32_t ii[3];
      for (int r = 2; r > 0; r--) {
        ii[r] = i % extents.extent(r);
        i /= extents.extent(r);
      }
      ii[0] = i;
      *out  = span_source(ii[0], ii[2], ii[1]);
    },
    stream);
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
  transpose_pq_centers(index, pq_centers_tmp.data(), stream);
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
  transpose_pq_centers(index, pq_centers_tmp.data(), stream);
}

/**
 * See raft::spatial::knn::ivf_pq::extend docs.
 *
 * This version requires `new_vectors` and `new_indices` (if non-null) to be on-device.
 */
template <typename T, typename IdxT>
inline auto extend_device(const handle_t& handle,
                          const index<IdxT>& orig_index,
                          const T* new_vectors,
                          const IdxT* new_indices,
                          IdxT n_rows) -> index<IdxT>
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_pq::extend(%zu, %u)", size_t(n_rows), orig_index.dim());
  auto stream = handle.get_stream();

  RAFT_EXPECTS(new_indices != nullptr || orig_index.size() == 0,
               "You must pass data indices when the index is non-empty.");

  static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
                "Unsupported data type");

  switch (new_indices != nullptr ? utils::check_pointer_residency(new_vectors, new_indices)
                                 : utils::check_pointer_residency(new_vectors)) {
    case utils::pointer_residency::device_only:
    case utils::pointer_residency::host_and_device: break;
    default: RAFT_FAIL("[ivf_pq::extend_device] The added data must be available on device.");
  }

  rmm::mr::device_memory_resource* device_memory = nullptr;
  auto pool_guard = raft::get_pool_memory_resource(device_memory, 1024 * 1024);
  if (pool_guard) {
    RAFT_LOG_DEBUG("ivf_pq::extend: using pool memory resource with initial size %zu bytes",
                   pool_guard->pool_size());
  }

  rmm::mr::managed_memory_resource managed_memory_upstream;
  rmm::mr::pool_memory_resource<rmm::mr::managed_memory_resource> managed_memory(
    &managed_memory_upstream, 1024 * 1024);

  //
  // The cluster_centers stored in index contain data other than cluster
  // centroids to speed up the search. Here, only the cluster centroids
  // are extracted.
  //
  const auto n_clusters = orig_index.n_lists();

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

  //
  // Use the existing cluster centroids to find the label (cluster ID)
  // of the vector to be added.
  //

  rmm::device_uvector<uint32_t> new_data_labels(n_rows, stream, device_memory);
  utils::memzero(new_data_labels.data(), n_rows, stream);
  rmm::device_uvector<uint32_t> new_cluster_sizes_buf(n_clusters, stream, &managed_memory);
  auto new_cluster_sizes = new_cluster_sizes_buf.data();
  utils::memzero(new_cluster_sizes, n_clusters, stream);

  kmeans::predict(handle,
                  cluster_centers.data(),
                  n_clusters,
                  orig_index.dim(),
                  new_vectors,
                  n_rows,
                  new_data_labels.data(),
                  orig_index.metric(),
                  stream);
  raft::stats::histogram<uint32_t, IdxT>(raft::stats::HistTypeAuto,
                                         reinterpret_cast<int32_t*>(new_cluster_sizes),
                                         IdxT(n_clusters),
                                         new_data_labels.data(),
                                         n_rows,
                                         1,
                                         stream);

  //
  // Make new_cluster_offsets, new_data_indices
  //
  rmm::device_uvector<IdxT> new_data_indices(n_rows, stream, &managed_memory);
  rmm::device_uvector<IdxT> new_cluster_offsets(n_clusters + 1, stream, &managed_memory);
  uint32_t new_max_cluster_size = calculate_offsets_and_indices(n_rows,
                                                                n_clusters,
                                                                new_data_labels.data(),
                                                                new_cluster_sizes,
                                                                new_cluster_offsets.data(),
                                                                new_data_indices.data(),
                                                                stream);

  //
  // Compute PQ code for new vectors
  //
  pq_codes_exts new_pq_exts = make_extents<size_t>(
    n_rows, orig_index.pq_dataset().extent(1), orig_index.pq_dataset().static_extent(3));
  auto new_pq_codes = make_device_mdarray<uint8_t>(handle, device_memory, new_pq_exts);
  compute_pq_codes<T>(handle,
                      n_rows,
                      orig_index.dim(),
                      orig_index.rot_dim(),
                      orig_index.pq_dim(),
                      orig_index.pq_len(),
                      orig_index.pq_bits(),
                      n_clusters,
                      orig_index.codebook_kind(),
                      new_max_cluster_size,
                      cluster_centers.data(),
                      orig_index.rotation_matrix().data_handle(),
                      new_vectors,
                      new_data_indices.data(),
                      new_cluster_sizes,
                      new_cluster_offsets.data(),
                      orig_index.pq_centers(),
                      new_pq_codes.view(),
                      device_memory);

  // Get the combined cluster sizes and sort the clusters in decreasing order
  // (this makes it easy to estimate the max number of samples during search).
  rmm::device_uvector<uint32_t> old_cluster_sizes_buf(n_clusters, stream, &managed_memory);
  rmm::device_uvector<uint32_t> ext_cluster_sizes_buf(n_clusters, stream, &managed_memory);
  rmm::device_uvector<IdxT> old_cluster_offsets_buf(n_clusters + 1, stream, &managed_memory);
  rmm::device_uvector<IdxT> ext_cluster_offsets_buf(n_clusters + 1, stream, &managed_memory);
  rmm::device_uvector<uint32_t> cluster_ordering_buf(n_clusters, stream, &managed_memory);
  auto old_cluster_sizes   = old_cluster_sizes_buf.data();
  auto ext_cluster_sizes   = ext_cluster_sizes_buf.data();
  auto old_cluster_offsets = old_cluster_offsets_buf.data();
  auto ext_cluster_offsets = ext_cluster_offsets_buf.data();
  auto cluster_ordering    = cluster_ordering_buf.data();
  copy(old_cluster_offsets,
       orig_index.list_offsets().data_handle(),
       orig_index.list_offsets().size(),
       stream);
  copy(old_cluster_sizes,
       orig_index.list_sizes().data_handle(),
       orig_index.list_sizes().size(),
       stream);

  uint32_t n_nonempty_lists = 0;
  {
    rmm::device_uvector<uint32_t> ext_cluster_sizes_buf_in(n_clusters, stream, device_memory);
    rmm::device_uvector<uint32_t> cluster_ordering_in(n_clusters, stream, device_memory);
    auto ext_cluster_sizes_in = ext_cluster_sizes_buf_in.data();
    linalg::add(ext_cluster_sizes_in, old_cluster_sizes, new_cluster_sizes, n_clusters, stream);

    thrust::sequence(handle.get_thrust_policy(),
                     cluster_ordering_in.data(),
                     cluster_ordering_in.data() + n_clusters);

    int begin_bit             = 0;
    int end_bit               = sizeof(uint32_t) * 8;
    size_t cub_workspace_size = 0;
    cub::DeviceRadixSort::SortPairsDescending(nullptr,
                                              cub_workspace_size,
                                              ext_cluster_sizes_in,
                                              ext_cluster_sizes,
                                              cluster_ordering_in.data(),
                                              cluster_ordering,
                                              n_clusters,
                                              begin_bit,
                                              end_bit,
                                              stream);
    rmm::device_buffer cub_workspace(cub_workspace_size, stream, device_memory);
    cub::DeviceRadixSort::SortPairsDescending(cub_workspace.data(),
                                              cub_workspace_size,
                                              ext_cluster_sizes_in,
                                              ext_cluster_sizes,
                                              cluster_ordering_in.data(),
                                              cluster_ordering,
                                              n_clusters,
                                              begin_bit,
                                              end_bit,
                                              stream);

    n_nonempty_lists = thrust::lower_bound(handle.get_thrust_policy(),
                                           ext_cluster_sizes,
                                           ext_cluster_sizes + n_clusters,
                                           0,
                                           thrust::greater<uint32_t>()) -
                       ext_cluster_sizes;
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
  // calculate extended cluster offsets
  {
    using group_align = Pow2<kIndexGroupSize>;
    IdxT size         = 0;
    update_device(ext_cluster_offsets, &size, 1, stream);
    thrust::inclusive_scan(
      handle.get_thrust_policy(),
      ext_cluster_sizes,
      ext_cluster_sizes + n_clusters,
      ext_cluster_offsets + 1,
      [] __device__(IdxT a, IdxT b) { return group_align::roundUp(a) + group_align::roundUp(b); });
    update_host(&size, ext_cluster_offsets + n_clusters, 1, stream);
    handle.sync_stream();
    copy(ext_index.list_offsets().data_handle(),
         ext_cluster_offsets,
         ext_index.list_offsets().size(),
         stream);
    copy(ext_index.list_sizes().data_handle(),
         ext_cluster_sizes,
         ext_index.list_sizes().size(),
         stream);
    ext_index.allocate(handle, size);
  }

  // Copy the unchanged parts
  copy(ext_index.rotation_matrix().data_handle(),
       orig_index.rotation_matrix().data_handle(),
       orig_index.rotation_matrix().size(),
       stream);

  // copy cluster-ordering-dependent data
  utils::copy_selected(n_clusters,
                       ext_index.dim_ext(),
                       orig_index.centers().data_handle(),
                       cluster_ordering,
                       orig_index.dim_ext(),
                       ext_index.centers().data_handle(),
                       ext_index.dim_ext(),
                       stream);
  utils::copy_selected(n_clusters,
                       ext_index.rot_dim(),
                       orig_index.centers_rot().data_handle(),
                       cluster_ordering,
                       orig_index.rot_dim(),
                       ext_index.centers_rot().data_handle(),
                       ext_index.rot_dim(),
                       stream);
  switch (orig_index.codebook_kind()) {
    case codebook_gen::PER_SUBSPACE: {
      copy(ext_index.pq_centers().data_handle(),
           orig_index.pq_centers().data_handle(),
           orig_index.pq_centers().size(),
           stream);
    } break;
    case codebook_gen::PER_CLUSTER: {
      auto d = orig_index.pq_book_size() * orig_index.pq_len();
      utils::copy_selected(n_clusters,
                           d,
                           orig_index.pq_centers().data_handle(),
                           cluster_ordering,
                           d,
                           ext_index.pq_centers().data_handle(),
                           d,
                           stream);
    } break;
    default: RAFT_FAIL("Unreachable code");
  }

  // Make ext_indices
  handle.sync_stream();  // make sure cluster sizes are up-to-date
  auto ext_indices = ext_index.indices().data_handle();
  for (uint32_t l = 0; l < ext_index.n_lists(); l++) {
    auto k                = cluster_ordering[l];
    auto old_cluster_size = old_cluster_sizes[k];
    auto new_cluster_size = new_cluster_sizes[k];
    if (old_cluster_size > 0) {
      copy(ext_indices + ext_cluster_offsets[l],
           orig_index.indices().data_handle() + old_cluster_offsets[k],
           old_cluster_size,
           stream);
    }
    if (new_cluster_size > 0) {
      if (new_indices == nullptr) {
        // implies the orig index is empty
        copy(ext_indices + ext_cluster_offsets[l] + old_cluster_size,
             new_data_indices.data() + new_cluster_offsets.data()[k],
             new_cluster_size,
             stream);
      } else {
        utils::copy_selected((IdxT)new_cluster_size,
                             (IdxT)1,
                             new_indices,
                             new_data_indices.data() + new_cluster_offsets.data()[k],
                             (IdxT)1,
                             ext_indices + ext_cluster_offsets[l] + old_cluster_size,
                             (IdxT)1,
                             stream);
      }
    }
  }

  /* Extend the pq_dataset */
  // For simplicity and performance, we reinterpret the last dimension of the dataset
  // as a single vector element.
  using vec_t = TxN_t<uint8_t, kIndexGroupVecLen>::io_t;

  auto data_unit      = ext_index.pq_dataset().extent(1);
  auto ext_pq_dataset = make_mdspan<vec_t, size_t, row_major, false, true>(
    reinterpret_cast<vec_t*>(ext_index.pq_dataset().data_handle()),
    make_extents<size_t>(
      ext_index.pq_dataset().extent(0), data_unit, ext_index.pq_dataset().extent(2)));

  for (uint32_t l = 0; l < ext_index.n_lists(); l++) {
    // Extend the data cluster-by-cluster;
    // The original/old index stores the data interleaved;
    // the new data produced by `compute_pq_codes` is not interleaved.
    auto k                = cluster_ordering[l];
    auto old_cluster_size = old_cluster_sizes[k];
    auto old_pq_dataset   = make_mdspan<const vec_t, size_t, row_major, false, true>(
      reinterpret_cast<const vec_t*>(orig_index.pq_dataset().data_handle()) +
        data_unit * old_cluster_offsets[k],
      make_extents<size_t>(div_rounding_up_safe(old_cluster_size, kIndexGroupSize),
                           data_unit,
                           ext_pq_dataset.extent(2)));
    auto new_pq_data = make_mdspan<vec_t, size_t, row_major, false, true>(
      reinterpret_cast<vec_t*>(new_pq_codes.data_handle()) +
        data_unit * new_cluster_offsets.data()[k],
      make_extents<size_t>(new_cluster_sizes[k], data_unit));
    // Write all cluster data, vec-by-vec
    linalg::writeOnlyUnaryOp(
      ext_pq_dataset.data_handle() + data_unit * ext_cluster_offsets[l],
      data_unit * size_t(ext_cluster_offsets[l + 1] - ext_cluster_offsets[l]),
      [old_pq_dataset, new_pq_data, old_cluster_size] __device__(vec_t * out, size_t i_flat) {
        // find the proper 3D index from the flat offset
        size_t i[3];
        for (int r = 2; r > 0; r--) {
          i[r] = i_flat % old_pq_dataset.extent(r);
          i_flat /= old_pq_dataset.extent(r);
        }
        i[0]        = i_flat;
        auto row_ix = i[0] * old_pq_dataset.extent(2) + i[2];
        if (row_ix < old_cluster_size) {
          // First, pack the original/old data
          *out = old_pq_dataset(i[0], i[1], i[2]);
        } else {
          // Then add the new data
          row_ix -= old_cluster_size;
          if (row_ix < new_pq_data.extent(0)) {
            *out = new_pq_data(row_ix, i[1]);
          } else {
            *out = vec_t{};
          }
        }
      },
      stream);
  }

  return ext_index;
}

/** See raft::spatial::knn::ivf_pq::extend docs */
template <typename T, typename IdxT>
inline auto extend(const handle_t& handle,
                   const index<IdxT>& orig_index,
                   const T* new_vectors,
                   const IdxT* new_indices,
                   IdxT n_rows) -> index<IdxT>
{
  size_t vec_size = sizeof(T) * size_t(n_rows) * size_t(orig_index.dim());
  size_t ind_size = sizeof(IdxT) * size_t(n_rows);
  return utils::with_mapped_memory_t{
    new_vectors, vec_size, [&](const T* new_vectors_dev) {
      return utils::with_mapped_memory_t{
        new_indices, ind_size, [&](const IdxT* new_indices_dev) {
          return extend_device<T, IdxT>(
            handle, orig_index, new_vectors_dev, new_indices_dev, n_rows);
        }}();
    }}();
}

/**
 * See raft::spatial::knn::ivf_pq::build docs.
 *
 * This version requires `dataset` to be on-device.
 */
template <typename T, typename IdxT>
inline auto build_device(
  const handle_t& handle, const index_params& params, const T* dataset, IdxT n_rows, uint32_t dim)
  -> index<IdxT>
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_pq::build(%zu, %u)", size_t(n_rows), dim);
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
                "Unsupported data type");

  RAFT_EXPECTS(n_rows > 0 && dim > 0, "empty dataset");

  switch (utils::check_pointer_residency(dataset)) {
    case utils::pointer_residency::device_only:
    case utils::pointer_residency::host_and_device: break;
    default: RAFT_FAIL("[ivf_pq::build_device] The dataset pointer must be available on device.");
  }

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
    auto dim = index.dim();
    linalg::writeOnlyUnaryOp(
      trainset.data(),
      size_t(index.dim()) * n_rows_train,
      [dataset, trainset_ratio, dim] __device__(float* out, size_t i) {
        auto col = i % dim;
        *out     = utils::mapping<float>{}(dataset[(i - col) * size_t(trainset_ratio) + col]);
      },
      stream);
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
                          stream,
                          raft::sqrt_op());
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
    return detail::extend_device<T, IdxT>(handle, index, dataset, nullptr, n_rows);
  } else {
    return index;
  }
}

/** See raft::spatial::knn::ivf_pq::build docs */
template <typename T, typename IdxT>
inline auto build(
  const handle_t& handle, const index_params& params, const T* dataset, IdxT n_rows, uint32_t dim)
  -> index<IdxT>
{
  size_t data_size = sizeof(T) * size_t(n_rows) * size_t(dim);
  return utils::with_mapped_memory_t{dataset, data_size, [&](const T* dataset_dev) {
                                       return build_device<T, IdxT>(
                                         handle, params, dataset_dev, n_rows, dim);
                                     }}();
}

}  // namespace raft::spatial::knn::ivf_pq::detail
