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

#include "../ivf_pq_types.hpp"
#include "ann_kmeans_balanced.cuh"
#include "ann_utils.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/stats/histogram.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/pow2_utils.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

namespace raft::spatial::knn::ivf_pq::detail {

using namespace raft::spatial::knn::detail;  // NOLINT

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
template <uint32_t PqBits>
HDI void ivfpq_encode_core(uint32_t n_rows, uint32_t pq_dim, const uint32_t* label, uint8_t* output)
{
  bitfield_view_t<PqBits> out{output};
  for (uint32_t j = 0; j < pq_dim; j++, label += n_rows) {
    out[j] = static_cast<uint8_t>(*label);
  }
}

template <uint32_t PqBits>
__global__ void ivfpq_encode_kernel(uint32_t n_rows,
                                    uint32_t pq_dim,
                                    const uint32_t* label,  // [pq_dim, ldDataset]
                                    uint8_t* output         // [n_rows, pq_dim]
)
{
  uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i >= n_rows) return;
  ivfpq_encode_core<PqBits>(n_rows, pq_dim, label + i, output + (pq_dim * PqBits / 8) * i);
}
}  // namespace

template <uint32_t PqBits>
inline void ivfpq_encode_run(uint32_t n_rows,
                             uint32_t pq_dim,
                             const uint32_t* label,  // [pq_dim, ldDataset]
                             uint8_t* output,        // [n_rows, pq_dim]
                             rmm::cuda_stream_view stream)
{
#if 1
  // GPU
  dim3 threads(128, 1, 1);
  dim3 blocks(raft::ceildiv<uint32_t>(n_rows, threads.x), 1, 1);
  ivfpq_encode_kernel<PqBits><<<blocks, threads, 0, stream>>>(n_rows, pq_dim, label, output);
#else
  // CPU
  stream.synchronize();
  for (uint32_t i = 0; i < n_rows; i++) {
    ivfpq_encode_core<PqBits>(n_rows, pq_dim, label + i, output + (pq_dim * PqBits / 8) * i);
  }
  stream.synchronize();
#endif
}

inline void ivfpq_encode(uint32_t n_rows,
                         uint32_t pq_dim,
                         uint32_t pq_bits,       // 4 <= pq_bits <= 8
                         const uint32_t* label,  // [pq_dim, ldDataset]
                         uint8_t* output,        // [n_rows, pq_dim]
                         rmm::cuda_stream_view stream)
{
  switch (pq_bits) {
    case 4: return ivfpq_encode_run<4>(n_rows, pq_dim, label, output, stream);
    case 5: return ivfpq_encode_run<5>(n_rows, pq_dim, label, output, stream);
    case 6: return ivfpq_encode_run<6>(n_rows, pq_dim, label, output, stream);
    case 7: return ivfpq_encode_run<7>(n_rows, pq_dim, label, output, stream);
    case 8: return ivfpq_encode_run<8>(n_rows, pq_dim, label, output, stream);
    default: RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 8]", pq_bits);
  }
}

template <typename T>
auto descending(const void* a, const void* b) -> int
{
  auto val_a = *reinterpret_cast<const T*>(a);
  auto val_b = *reinterpret_cast<const T*>(b);
  if (val_a > val_b) return -1;
  if (val_a < val_b) return 1;
  return 0;
}

template <typename IdxT>
void _cuann_get_inclusiveSumSortedClusterSize(index<IdxT>& index)
{
  auto cluster_offsets     = index.list_offsets().data_handle();
  auto output              = index.inclusiveSumSortedClusterSize();
  index.numClustersSize0() = 0;
  for (uint32_t i = 0; i < index.n_lists(); i++) {
    output(i) = cluster_offsets[i + 1] - cluster_offsets[i];
    if (output(i) > 0) continue;

    index.numClustersSize0() += 1;
  }
  RAFT_LOG_DEBUG("Number of clusters of size zero: %u", index.numClustersSize0());
  // sort
  qsort(output.data_handle(), index.n_lists(), sizeof(IdxT), descending<IdxT>);
  // scan
  for (uint32_t i = 1; i < index.n_lists(); i++) {
    output(i) += output(i - 1);
  }
  RAFT_EXPECTS(output(index.n_lists() - 1) == index.size(), "cluster sizes do not add up");
}

template <typename T, typename X = T, typename Y = T>
T _cuann_dot(int n, const X* x, int incX, const Y* y, int incY)
{
  T val = 0;
  for (int i = 0; i < n; i++) {
    val += utils::mapping<T>{}(x[incX * i]) * utils::mapping<T>{}(y[incY * i]);
  }
  return val;
}

//
template <typename T>
T _cuann_rand()
{
  return (T)rand() / RAND_MAX;
}

// make rotation matrix
inline void make_rotation_matrix(uint32_t nRows,
                                 uint32_t nCols,
                                 uint32_t pq_len,
                                 bool random_rotation,
                                 float* rotation_matrix  // [nRows, nCols]
)
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_pq::make_rotation_matrix(%u * %u, random = %d, pq_len = %u)",
    nRows,
    nCols,
    random_rotation,
    pq_len);
  RAFT_EXPECTS(nRows >= nCols,
               "number of rows (%u) must be no smaller than number or cols (%u)",
               nRows,
               nCols);
  RAFT_EXPECTS(
    nRows % pq_len == 0, "number of rows (%u) must be a multiple of pq_len (%u)", nRows, pq_len);

  if (random_rotation) {
    RAFT_LOG_DEBUG("Creating a random rotation matrix.");
    double dot, norm;
    std::vector<double> matrix(nRows * nCols, 0.0);
    for (uint32_t i = 0; i < nRows * nCols; i++) {
      matrix[i] = _cuann_rand<double>() - 0.5;
    }
    for (uint32_t j = 0; j < nCols; j++) {
      // normalize the j-th col vector
      norm = sqrt(_cuann_dot<double>(nRows, &matrix[j], nCols, &matrix[j], nCols));
      for (uint32_t i = 0; i < nRows; i++) {
        matrix[j + (nCols * i)] /= norm;
      }
      // orthogonalize the j-th col vector with the previous col vectors
      for (uint32_t k = 0; k < j; k++) {
        dot = _cuann_dot<double>(nRows, &matrix[j], nCols, &matrix[k], nCols);
        for (uint32_t i = 0; i < nRows; i++) {
          matrix[j + (nCols * i)] -= dot * matrix[k + (nCols * i)];
        }
      }
      // normalize the j-th col vector again
      norm = sqrt(_cuann_dot<double>(nRows, &matrix[j], nCols, &matrix[j], nCols));
      for (uint32_t i = 0; i < nRows; i++) {
        matrix[j + (nCols * i)] /= norm;
      }
    }
    for (uint32_t i = 0; i < nRows * nCols; i++) {
      rotation_matrix[i] = (float)matrix[i];
    }
  } else {
    if (nRows == nCols) {
      memset(rotation_matrix, 0, sizeof(float) * nRows * nCols);
      for (uint32_t i = 0; i < nCols; i++) {
        rotation_matrix[i + (nCols * i)] = 1.0;
      }
    } else {
      memset(rotation_matrix, 0, sizeof(float) * nRows * nCols);
      uint32_t i = 0;
      for (uint32_t j = 0; j < nCols; j++) {
        rotation_matrix[j + (nCols * i)] = 1.0;
        i += pq_len;
        if (i >= nRows) { i = (i % nRows) + 1; }
      }
    }
  }
}

//
inline auto calc_pq_trainset_size(uint32_t cluster_size, uint32_t pq_dim, uint32_t pq_bits)
  -> uint32_t
{
  return static_cast<uint32_t>(
    std::min<size_t>(size_t{cluster_size} * size_t{pq_dim},
                     size_t{256} * std::max<size_t>(size_t{1} << pq_bits, size_t{pq_dim})));
}

/**
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
 * @param pq_centers                 // [...]
 * @param pq_dataset  // [n_rows, pq_dim * pq_bits / 8]
 * @param managed_memory
 * @param device_memory
 */
template <typename T, typename IdxT>
void compute_pq_codes(const handle_t& handle,
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
                      float* pq_centers,
                      uint8_t* pq_dataset,
                      rmm::mr::device_memory_resource* managed_memory,
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
  utils::memzero(pq_dataset, n_rows * pq_dim * pq_bits / 8, stream);

  rmm::device_uvector<float> res_vectors(max_cluster_size * data_dim, stream, managed_memory);
  rmm::device_uvector<float> rot_vectors(max_cluster_size * rot_dim, stream, managed_memory);
  rmm::device_uvector<float> sub_vectors(
    max_cluster_size * pq_dim * pq_len, stream, managed_memory);
  rmm::device_uvector<uint32_t> sub_vector_labels(
    max_cluster_size * pq_dim, stream, managed_memory);
  rmm::device_uvector<uint8_t> my_pq_dataset(
    max_cluster_size * pq_dim * pq_bits / 8 /* NB: pq_dim * bitPQ % 8 == 0 */,
    stream,
    managed_memory);
  rmm::device_uvector<uint32_t> rot_vector_labels(0, stream, managed_memory);
  rmm::device_uvector<uint32_t> pq_cluster_size(0, stream, managed_memory);
  rmm::device_uvector<float> my_pq_centers(0, stream, managed_memory);

  for (uint32_t l = 0; l < n_clusters; l++) {
    auto cluster_size = cluster_sizes[l];
    common::nvtx::range<common::nvtx::domain::raft> cluster_scope(
      "ivf_pq::compute_pq_codes::cluster[%u](size = %u)", l, cluster_size);
    if (cluster_size == 0) continue;

    //
    // Compute the residual vector of the new vector with its cluster
    // centroids.
    //   resVectors[..] = new_vectors[..] - cluster_centers[..]
    //
    utils::copy_selected<float, T>(cluster_size,
                                   data_dim,
                                   dataset,
                                   data_indices + cluster_offsets[l],
                                   data_dim,
                                   res_vectors.data(),
                                   data_dim,
                                   stream);

    // substract centers from the vectors in the cluster.
    raft::matrix::linewiseOp(
      res_vectors.data(),
      res_vectors.data(),
      data_dim,
      cluster_size,
      true,
      [] __device__(float a, float b) { return a - b; },
      stream,
      cluster_centers + (uint64_t)l * data_dim);

    //
    // Rotate the residual vectors using a rotation matrix
    //
    float alpha = 1.0;
    float beta  = 0.0;
    linalg::gemm(handle,
                 true,
                 false,
                 rot_dim,
                 cluster_size,
                 data_dim,
                 &alpha,
                 rotation_matrix,
                 data_dim,
                 res_vectors.data(),
                 data_dim,
                 &beta,
                 rot_vectors.data(),
                 rot_dim,
                 stream);

    //
    // Change the order of the vector data to facilitate processing in
    // each vector subspace.
    //   input:  rot_vectors[cluster_size, rot_dim] = [cluster_size, pq_dim, pq_len]
    //   output: sub_vectors[pq_dim, cluster_size, pq_len]
    //
    for (uint32_t i = 0; i < pq_dim; i++) {
      RAFT_CUDA_TRY(cudaMemcpy2DAsync(sub_vectors.data() + i * pq_len * cluster_size,
                                      sizeof(float) * pq_len,
                                      rot_vectors.data() + i * pq_len,
                                      sizeof(float) * rot_dim,
                                      sizeof(float) * pq_len,
                                      cluster_size,
                                      cudaMemcpyDefault,
                                      stream));
    }

    //
    // Find a label (cluster ID) for each vector subspace.
    //
    for (uint32_t j = 0; j < pq_dim; j++) {
      float* sub_pq_centers = nullptr;
      switch (codebook_kind) {
        case codebook_gen::PER_SUBSPACE:
          sub_pq_centers = pq_centers + ((1 << pq_bits) * pq_len) * j;
          break;
        case codebook_gen::PER_CLUSTER:
          sub_pq_centers = pq_centers + ((1 << pq_bits) * pq_len) * l;
          break;
        default: RAFT_FAIL("Unreachable code");
      }
      kmeans::predict(handle,
                      sub_pq_centers,
                      (1 << pq_bits),
                      pq_len,
                      sub_vectors.data() + j * (cluster_size * pq_len),
                      cluster_size,
                      sub_vector_labels.data() + j * cluster_size,
                      raft::distance::DistanceType::L2Expanded,
                      stream,
                      device_memory);
    }

    //
    // PQ encoding
    //
    ivfpq_encode(
      cluster_size, pq_dim, pq_bits, sub_vector_labels.data(), my_pq_dataset.data(), stream);
    copy(pq_dataset + cluster_offsets[l] * uint64_t{pq_dim * pq_bits / 8},
         my_pq_dataset.data(),
         cluster_size * pq_dim * pq_bits / 8,
         stream);
    handle.sync_stream();
  }
}

/** See raft::spatial::knn::ivf_pq::extend docs */
template <typename T, typename IdxT>
inline auto extend(const handle_t& handle,
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

  rmm::device_uvector<float> cluster_centers(
    orig_index.n_lists() * orig_index.dim(), stream, &managed_memory);
  for (uint32_t i = 0; i < orig_index.n_lists(); i++) {
    memcpy(cluster_centers.data() + (uint64_t)i * orig_index.dim(),
           orig_index.centers().data_handle() + (uint64_t)i * orig_index.dim_ext(),
           sizeof(float) * orig_index.dim());
  }

  //
  // Use the existing cluster centroids to find the label (cluster ID)
  // of the vector to be added.
  //

  rmm::device_uvector<uint32_t> new_data_labels(n_rows, stream, &managed_memory);
  utils::memzero(new_data_labels.data(), n_rows, stream);
  rmm::device_uvector<uint32_t> cluster_sizes(orig_index.n_lists(), stream, &managed_memory);
  utils::memzero(cluster_sizes.data(), orig_index.n_lists(), stream);

  kmeans::predict(handle,
                  cluster_centers.data(),
                  orig_index.n_lists(),
                  orig_index.dim(),
                  new_vectors,
                  n_rows,
                  new_data_labels.data(),
                  orig_index.metric(),
                  stream);
  raft::stats::histogram<uint32_t, IdxT>(raft::stats::HistTypeAuto,
                                         reinterpret_cast<int32_t*>(cluster_sizes.data()),
                                         IdxT(orig_index.n_lists()),
                                         new_data_labels.data(),
                                         n_rows,
                                         1,
                                         stream);
  handle.sync_stream();

  //
  // Make cluster_offsets, data_indices
  //
  uint32_t max_cluster_size = 0;
  std::vector<IdxT> cluster_offsets(orig_index.n_lists() + 1, 0);
  rmm::device_uvector<IdxT> data_indices(n_rows, stream, &managed_memory);
  // cluster_offsets
  cluster_offsets[0] = 0;
  for (uint32_t l = 0; l < orig_index.n_lists(); l++) {
    cluster_offsets[l + 1] = cluster_offsets[l] + cluster_sizes.data()[l];
    max_cluster_size       = max(max_cluster_size, cluster_sizes.data()[l]);
  }
  RAFT_EXPECTS(cluster_offsets[orig_index.n_lists()] == n_rows, "cluster sizes do not add up.");
  // data_indices
  for (IdxT i = 0; i < n_rows; i++) {
    uint32_t l                              = new_data_labels.data()[i];
    data_indices.data()[cluster_offsets[l]] = i;
    cluster_offsets[l] += 1;
  }
  // Recover cluster_offsets
  for (uint32_t l = 0; l < orig_index.n_lists(); l++) {
    cluster_offsets[l] -= cluster_sizes.data()[l];
  }

  //
  // Create descriptor for new index
  //
  ivf_pq::index<IdxT> ext_index(handle,
                                orig_index.metric(),
                                orig_index.codebook_kind(),
                                orig_index.n_lists(),
                                orig_index.dim(),
                                orig_index.pq_bits(),
                                orig_index.pq_dim());
  ext_index.allocate(handle, orig_index.size() + n_rows);
  RAFT_LOG_DEBUG("Index size: %u -> %u", orig_index.size(), ext_index.size());

  //
  // Copy the unchanged parts
  //
  copy(ext_index.centers().data_handle(),
       orig_index.centers().data_handle(),
       orig_index.centers().size(),
       stream);
  copy(ext_index.centers_rot().data_handle(),
       orig_index.centers_rot().data_handle(),
       orig_index.centers_rot().size(),
       stream);
  copy(ext_index.pq_centers().data_handle(),
       orig_index.pq_centers().data_handle(),
       orig_index.pq_centers().size(),
       stream);
  copy(ext_index.rotation_matrix().data_handle(),
       orig_index.rotation_matrix().data_handle(),
       orig_index.rotation_matrix().size(),
       stream);

  handle.sync_stream();

  //
  // Compute PQ code for new vectors
  //
  rmm::device_uvector<uint8_t> new_pq_codes(
    n_rows * orig_index.pq_dim() * orig_index.pq_bits() / 8, stream, &managed_memory);
  compute_pq_codes<T>(handle,
                      n_rows,
                      orig_index.dim(),
                      orig_index.rot_dim(),
                      orig_index.pq_dim(),
                      orig_index.pq_len(),
                      orig_index.pq_bits(),
                      orig_index.n_lists(),
                      orig_index.codebook_kind(),
                      max_cluster_size,
                      cluster_centers.data(),
                      ext_index.rotation_matrix().data_handle(),
                      new_vectors,
                      data_indices.data(),
                      cluster_sizes.data(),
                      cluster_offsets.data(),
                      ext_index.pq_centers().data_handle(),
                      new_pq_codes.data(),
                      &managed_memory,
                      device_memory);

  auto ext_indices         = ext_index.indices().data_handle();
  auto ext_cluster_offsets = ext_index.list_offsets().data_handle();
  //
  // Make ext_cluster_offsets
  //
  auto old_cluster_offsets = orig_index.list_offsets().data_handle();
  max_cluster_size         = 0;
  ext_cluster_offsets[0]   = 0;
  for (uint32_t l = 0; l < ext_index.n_lists(); l++) {
    auto old_cluster_size =
      static_cast<uint32_t>(old_cluster_offsets[l + 1] - old_cluster_offsets[l]);
    ext_cluster_offsets[l + 1] =
      ext_cluster_offsets[l] + old_cluster_size + cluster_sizes.data()[l];
    max_cluster_size = max(max_cluster_size, old_cluster_size + cluster_sizes.data()[l]);
  }

  //
  // Make ext_indices
  //
  for (uint32_t l = 0; l < ext_index.n_lists(); l++) {
    auto old_cluster_size =
      static_cast<uint32_t>(old_cluster_offsets[l + 1] - old_cluster_offsets[l]);
    auto new_cluster_size = cluster_sizes.data()[l];
    if (old_cluster_size > 0) {
      copy(ext_indices + ext_cluster_offsets[l],
           orig_index.indices().data_handle() + old_cluster_offsets[l],
           old_cluster_size,
           stream);
    }
    if (new_cluster_size > 0) {
      if (new_indices == nullptr) {
        // implies the orig index is empty
        copy(ext_indices + ext_cluster_offsets[l] + old_cluster_size,
             data_indices.data() + cluster_offsets[l],
             new_cluster_size,
             stream);
      } else {
        utils::copy_selected(new_cluster_size,
                             1,
                             new_indices,
                             data_indices.data() + cluster_offsets[l],
                             1,
                             ext_indices + ext_cluster_offsets[l] + old_cluster_size,
                             1,
                             stream);
      }
    }
  }

  /* Extend the pq_dataset */
  auto ext_pq_dataset    = ext_index.pq_dataset().data_handle();
  size_t pq_dataset_unit = ext_index.pq_dim() * ext_index.pq_bits() / 8;
  for (uint32_t l = 0; l < ext_index.n_lists(); l++) {
    auto old_cluster_size =
      static_cast<uint32_t>(old_cluster_offsets[l + 1] - old_cluster_offsets[l]);
    copy(ext_pq_dataset + pq_dataset_unit * ext_cluster_offsets[l],
         orig_index.pq_dataset().data_handle() + pq_dataset_unit * old_cluster_offsets[l],
         pq_dataset_unit * old_cluster_size,
         stream);
    copy(ext_pq_dataset + pq_dataset_unit * (ext_cluster_offsets[l] + old_cluster_size),
         new_pq_codes.data() + pq_dataset_unit * cluster_offsets[l],
         pq_dataset_unit * cluster_sizes.data()[l],
         stream);
  }
  handle.sync_stream();

  _cuann_get_inclusiveSumSortedClusterSize(ext_index);

  return ext_index;
}

/** See raft::spatial::knn::ivf_pq::build docs */
template <typename T, typename IdxT>
inline auto build(
  const handle_t& handle, const index_params& params, const T* dataset, IdxT n_rows, uint32_t dim)
  -> index<IdxT>
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_pq::build(%zu, %u)", size_t(n_rows), dim);
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
                "Unsupported data type");

  RAFT_EXPECTS(n_rows > 0 && dim > 0, "empty dataset");

  auto stream = handle.get_stream();

  ivf_pq::index<IdxT> index(handle, params, dim);
  utils::memzero(index.list_offsets().data_handle(), index.list_offsets().size(), stream);
  utils::memzero(index.pq_centers().data_handle(), index.pq_centers().size(), stream);

  auto trainset_ratio = std::max<IdxT>(
    1, n_rows / std::max<IdxT>(params.kmeans_trainset_fraction * n_rows, index.n_lists()));
  auto n_rows_train = n_rows / trainset_ratio;

  rmm::mr::device_memory_resource* device_memory = nullptr;
  auto pool_guard = raft::get_pool_memory_resource(device_memory, 1024 * 1024);
  if (pool_guard) {
    RAFT_LOG_DEBUG("ivf_pq::build: using pool memory resource with initial size %zu bytes",
                   pool_guard->pool_size());
  }

  rmm::mr::managed_memory_resource managed_memory_upstream;
  rmm::mr::pool_memory_resource<rmm::mr::managed_memory_resource> managed_memory(
    &managed_memory_upstream, 1024 * 1024);

  // Besides just sampling, we transform the input dataset into floats to make it easier
  // to use gemm operations from cublas.
  rmm::device_uvector<float> trainset(n_rows_train * index.dim(), stream, device_memory);
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
      index.dim() * n_rows_train,
      [dataset, trainset_ratio, dim] __device__(float* out, size_t i) {
        auto col = i % dim;
        *out     = utils::mapping<float>{}(dataset[(i - col) * trainset_ratio + col]);
      },
      stream);
  }

  // NB: here cluster_centers is used as if it is [n_clusters, data_dim] not [n_clusters, dim_ext]!
  auto cluster_centers = index.centers().data_handle();
  auto pq_centers      = index.pq_centers().data_handle();
  auto rotation_matrix = index.rotation_matrix().data_handle();
  auto centers_rot     = index.centers_rot().data_handle();

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

  rmm::device_uvector<uint32_t> dataset_labels(n_rows, stream, &managed_memory);
  rmm::device_uvector<uint32_t> cluster_sizes(index.n_lists(), stream, &managed_memory);
  rmm::device_uvector<IdxT> dataset_indices(n_rows, stream, &managed_memory);
  rmm::device_uvector<IdxT> dataset_cluster_offsets(
    index.list_offsets().size(), stream, &managed_memory);

  /* Predict labels of whole dataset

     The labels are needed for training PQ codebooks
   */
  kmeans::predict(handle,
                  cluster_centers,
                  index.n_lists(),
                  index.dim(),
                  dataset,
                  n_rows,
                  dataset_labels.data(),
                  index.metric(),
                  stream,
                  device_memory);
  kmeans::calc_centers_and_sizes(cluster_centers,
                                 cluster_sizes.data(),
                                 index.n_lists(),
                                 index.dim(),
                                 dataset,
                                 n_rows,
                                 dataset_labels.data(),
                                 true,
                                 stream);
  handle.sync_stream();

  // Make rotation matrix
  RAFT_LOG_DEBUG("# data_dim: %u\n", index.dim());
  RAFT_LOG_DEBUG("# rot_dim: %u\n", index.rot_dim());
  RAFT_LOG_DEBUG("# params.random_rotation: %s\n", params.random_rotation ? "enabled" : "disabled");
  make_rotation_matrix(
    index.rot_dim(), index.dim(), index.pq_len(), params.random_rotation, rotation_matrix);

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
               rotation_matrix,
               index.dim(),
               cluster_centers,
               index.dim(),
               &beta,
               centers_rot,
               index.rot_dim(),
               stream);

  //
  // Make cluster_offsets, data_indices
  //
  auto cluster_offsets      = dataset_cluster_offsets.data();
  auto data_indices         = dataset_indices.data();
  uint32_t max_cluster_size = 0;
  // cluster_offsets
  cluster_offsets[0] = 0;
  for (uint32_t l = 0; l < index.n_lists(); l++) {
    cluster_offsets[l + 1] = cluster_offsets[l] + cluster_sizes.data()[l];
    if (max_cluster_size < cluster_sizes.data()[l]) { max_cluster_size = cluster_sizes.data()[l]; }
  }
  RAFT_EXPECTS(cluster_offsets[index.n_lists()] == n_rows, "Cluster sizes do not add up");

  // data_indices
  for (IdxT i = 0; i < n_rows; i++) {
    uint32_t l                       = dataset_labels.data()[i];
    data_indices[cluster_offsets[l]] = i;
    cluster_offsets[l] += 1;
  }

  // Recover cluster_offsets
  for (uint32_t l = 0; l < index.n_lists(); l++) {
    cluster_offsets[l] -= cluster_sizes.data()[l];
  }

  // Training PQ codebooks
  switch (index.codebook_kind()) {
    case codebook_gen::PER_SUBSPACE: {
      rmm::device_uvector<uint32_t> trainset_labels(n_rows_train, stream, &managed_memory);

      // Predict label of trainset again
      kmeans::predict(handle,
                      cluster_centers,
                      index.n_lists(),
                      index.dim(),
                      trainset.data(),
                      n_rows_train,
                      trainset_labels.data(),
                      index.metric(),
                      stream,
                      device_memory);
      handle.sync_stream();

      rmm::device_uvector<float> sub_trainset(
        n_rows_train * index.pq_len(), stream, &managed_memory);
      rmm::device_uvector<uint32_t> sub_trainset_labels(n_rows_train, stream, &managed_memory);

      rmm::device_uvector<uint32_t> pq_cluster_sizes(index.pq_width(), stream, &managed_memory);

      for (uint32_t j = 0; j < index.pq_dim(); j++) {
        common::nvtx::range<common::nvtx::domain::raft> pq_per_subspace_scope(
          "ivf_pq::build::per_subspace[%u]", j);

        // Get the rotated cluster centers to substract them from the input vectors afterwards.
        utils::copy_selected(n_rows_train,
                             index.pq_len(),
                             centers_rot + index.pq_len() * j,
                             trainset_labels.data(),
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
                     n_rows_train,
                     index.dim(),
                     &alpha,
                     rotation_matrix + index.dim() * index.pq_len() * j,
                     index.dim(),
                     trainset.data(),
                     index.dim(),
                     &beta,
                     sub_trainset.data(),
                     index.pq_len(),
                     stream);

        // Train kmeans for each PQ
        kmeans::build_clusters(handle,
                               params.kmeans_n_iters,
                               index.pq_len(),
                               sub_trainset.data(),
                               n_rows_train,
                               index.pq_width(),
                               pq_centers + (index.pq_width() * index.pq_len()) * j,
                               sub_trainset_labels.data(),
                               pq_cluster_sizes.data(),
                               raft::distance::DistanceType::L2Expanded,
                               stream,
                               device_memory);
        handle.sync_stream();
      }
    } break;
    case codebook_gen::PER_CLUSTER: {
      rmm::device_uvector<uint32_t> rot_vector_labels(
        max_cluster_size * index.pq_dim(), stream, &managed_memory);
      rmm::device_uvector<uint32_t> pq_cluster_size(index.pq_width(), stream, &managed_memory);

      rmm::device_uvector<float> res_vectors(
        max_cluster_size * index.dim(), stream, &managed_memory);
      rmm::device_uvector<float> rot_vectors(
        max_cluster_size * index.rot_dim(), stream, &managed_memory);

      for (uint32_t l = 0; l < index.n_lists(); l++) {
        auto cluster_size = cluster_sizes.data()[l];
        if (cluster_size == 0) continue;
        common::nvtx::range<common::nvtx::domain::raft> pq_per_cluster_scope(
          "ivf_pq::build::per_cluster[%u](size = %u)", l, cluster_size);

        //
        // Compute the residual vector of the new vector with its cluster
        // centroids.
        //   resVectors[..] = new_vectors[..] - cluster_centers[..]
        //
        utils::copy_selected<float, T>(cluster_size,
                                       index.dim(),
                                       dataset,
                                       data_indices + cluster_offsets[l],
                                       index.dim(),
                                       res_vectors.data(),
                                       index.dim(),
                                       stream);

        // substract centers from the vectors in the cluster.
        raft::matrix::linewiseOp(
          res_vectors.data(),
          res_vectors.data(),
          index.dim(),
          cluster_size,
          true,
          [] __device__(float a, float b) { return a - b; },
          stream,
          cluster_centers + (uint64_t)l * index.dim());

        //
        // Rotate the residual vectors using a rotation matrix
        //
        float alpha = 1.0;
        float beta  = 0.0;
        linalg::gemm(handle,
                     true,
                     false,
                     index.rot_dim(),
                     cluster_size,
                     index.dim(),
                     &alpha,
                     rotation_matrix,
                     index.dim(),
                     res_vectors.data(),
                     index.dim(),
                     &beta,
                     rot_vectors.data(),
                     index.rot_dim(),
                     stream);

        uint32_t n_rows_cluster =
          calc_pq_trainset_size(cluster_size, index.pq_dim(), index.pq_bits());
        kmeans::build_clusters(handle,
                               params.kmeans_n_iters,
                               index.pq_len(),
                               rot_vectors.data(),
                               n_rows_cluster,
                               index.pq_width(),
                               pq_centers + index.pq_width() * index.pq_len() * l,
                               rot_vector_labels.data(),
                               pq_cluster_size.data(),
                               raft::distance::DistanceType::L2Expanded,
                               stream,
                               device_memory);

        handle.sync_stream();
      }

    } break;
    default: RAFT_FAIL("Unreachable code");
  }

  auto center_norms = index.center_norms().data_handle();
  utils::dots_along_rows(
    index.n_lists(), index.dim(), cluster_centers, index.center_norms().data_handle(), stream);
  stream.synchronize();

  {
    // combine cluster_centers and their norms
    auto cluster_centers_tmp =
      make_host_mdarray<float>(make_extents<uint32_t>(index.n_lists(), index.dim()));
    for (uint32_t i = 0; i < index.n_lists() * index.dim(); i++) {
      cluster_centers_tmp.data_handle()[i] = cluster_centers[i];
    }
    for (uint32_t i = 0; i < index.n_lists(); i++) {
      for (uint32_t j = 0; j < index.dim(); j++) {
        cluster_centers[j + (index.dim_ext() * i)] = cluster_centers_tmp(i, j);
      }
      cluster_centers[index.dim() + (index.dim_ext() * i)] =
        cluster_sizes.data()[i] == 0 ? 1.0f : center_norms[i];
    }
  }

  // add the data if necessary
  if (params.add_data_on_build) {
    return detail::extend<T, IdxT>(handle, index, dataset, nullptr, n_rows);
  } else {
    return index;
  }
}

}  // namespace raft::spatial::knn::ivf_pq::detail
