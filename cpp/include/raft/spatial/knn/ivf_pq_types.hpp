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

#include "common.hpp"

#include <raft/core/error.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/distance/distance_type.hpp>
#include <raft/integer_utils.h>

namespace raft::spatial::knn::ivf_pq {

/** A type for specifying how PQ codebooks are created. */
enum class codebook_gen {
  PER_SUBSPACE = 0,
  PER_CLUSTER  = 1,
};

struct index_params : knn::index_params {
  /**
   * The number of inverted lists (clusters)
   *
   * Hint: the number of vectors per cluster (`n_rows/n_lists`) should be approximately 1,000 to
   * 10,000.
   */
  uint32_t n_lists = 1024;
  /** The number of iterations searching for kmeans centers (index building). */
  uint32_t kmeans_n_iters = 20;
  /** The fraction of data to use during iterative kmeans building. */
  double kmeans_trainset_fraction = 0.5;
  /**
   * The bit length of the vector element after compression by PQ.
   *
   * Possible values: [4, 5, 6, 7, 8].
   *
   * Hint: the smaller the 'pq_bits', the smaller the index size and the better the search
   * performance, but the lower the recall.
   */
  uint32_t pq_bits = 8;
  /**
   * The dimensionality of the vector after compression by PQ. When zero, an optimal value is
   * selected using a heuristic.
   *
   * NB: `pq_dim * pq_bits` must be a multiple of 8.
   *
   * Hint: a smaller 'pq_dim' results in a smaller index size and better search performance, but
   * lower recall. If 'pq_bits' is 8, 'pq_dim' can be set to any number, but multiple of 8 are
   * desirable for good performance. If 'pq_bits' is not 8, 'pq_dim' should be a multiple of 8.
   * For good performance, multiple 32 is desirable.
   */
  uint32_t pq_dim = 0;
  /**
   * If true, dataset and query vectors are rotated by a random rotation matrix created at indexing
   * time.
   *
   * NB: Currently, the rotation matrix is generated on CPU and may measurably increase the indexing
   * time.
   */
  bool random_rotation = true;
  /** How PQ codebooks are created. */
  codebook_gen codebook_kind = codebook_gen::PER_SUBSPACE;
};

struct search_params : knn::search_params {
  /** The number of clusters to search. */
  uint32_t n_probes = 20;
  /**
   * Data type of LUT to be created dynamically at search time.
   *
   * Possible values: [CUDA_R_32F, CUDA_R_16F, CUDA_R_8U]
   *
   * The use of low-precision types reduces the amount of shared memory required at search time, so
   * fast shared memory kernels can be used even for datasets with large dimansionality. Note that
   * the recall is slightly degraded when low-precision type is selected.
   */
  cudaDataType_t smem_lut_dtype = CUDA_R_32F;
  /**
   * Storage data type for distance/similarity computed at search time.
   *
   * Possible values: [CUDA_R_16F, CUDA_R_32F]
   *
   * If the performance limiter at search time is device memory access, selecting FP16 will improve
   * performance slightly.
   */
  cudaDataType_t internal_distance_dtype = CUDA_R_32F;
  /**
   * Thread block size of the distance calculation kernel at search time.
   * When zero, an optimal block size is selected using a heuristic.
   *
   * Possible values: [0, 256, 512, 1024]
   */
  uint32_t preferred_thread_block_size = 0;
};

static_assert(std::is_aggregate_v<index_params>);
static_assert(std::is_aggregate_v<search_params>);

namespace detail {

/* IvfPq */
struct cuannIvfPqDescriptor {
  uint32_t numClusters;
  uint32_t numDataset;
  uint32_t data_dim;
  uint32_t dimDatasetExt;
  uint32_t rot_dim;
  uint32_t pq_dim;
  uint32_t bitPq;
  distance::DistanceType metric;
  codebook_gen typePqCenter;
  cudaDataType_t internalDistanceDtype;
  cudaDataType_t smemLutDtype;
  uint32_t indexVersion;
  uint32_t maxClusterSize;
  uint32_t lenPq;  // rot_dim / pq_dim
  uint32_t numProbes;
  uint32_t topK;
  uint32_t maxQueries;
  uint32_t maxBatchSize;
  uint32_t maxSamples;
  uint32_t* inclusiveSumSortedClusterSize;  // [numClusters,]
  float* sqsumClusters;                     // [numClusters,]
  size_t sizeCubWorkspace;
  uint32_t _numClustersSize0;  // (*) urgent WA, need to be fixed
  uint32_t preferredThreadBlockSize;
  void* index_ptr;
};
using cuannIvfPqDescriptor_t =
  std::unique_ptr<cuannIvfPqDescriptor, std::function<void(cuannIvfPqDescriptor*)>>;

cuannIvfPqDescriptor_t cuannIvfPqCreateDescriptor()
{
  return cuannIvfPqDescriptor_t{[]() {
                                  auto desc                           = new cuannIvfPqDescriptor{};
                                  desc->numClusters                   = 0;
                                  desc->numDataset                    = 0;
                                  desc->data_dim                      = 0;
                                  desc->dimDatasetExt                 = 0;
                                  desc->rot_dim                       = 0;
                                  desc->pq_dim                        = 0;
                                  desc->bitPq                         = 0;
                                  desc->numProbes                     = 0;
                                  desc->topK                          = 0;
                                  desc->maxQueries                    = 0;
                                  desc->maxBatchSize                  = 0;
                                  desc->maxSamples                    = 0;
                                  desc->inclusiveSumSortedClusterSize = nullptr;
                                  desc->sqsumClusters                 = nullptr;
                                  desc->index_ptr                     = nullptr;
                                  return desc;
                                }(),
                                [](cuannIvfPqDescriptor* desc) {
                                  if (desc->inclusiveSumSortedClusterSize != nullptr) {
                                    free(desc->inclusiveSumSortedClusterSize);
                                  }
                                  if (desc->sqsumClusters != nullptr) {
                                    RAFT_CUDA_TRY_NO_THROW(cudaFree(desc->sqsumClusters));
                                  }
                                  if (desc->index_ptr != nullptr) {
                                    RAFT_CUDA_TRY_NO_THROW(cudaFree(desc->index_ptr));
                                  }
                                  delete desc;
                                }};
}

}  // namespace detail

/**
 * @brief IVF-PQ index.
 *
 * @tparam IdxT type of the indices in the source dataset
 *
 */
template <typename IdxT>
struct index : knn::index {
  static_assert(!raft::is_narrowing_v<uint32_t, IdxT>,
                "IdxT must be able to represent all values of uint32_t");

 public:
  /** Dimensionality of the data. */
  [[nodiscard]] constexpr inline auto dim() const noexcept -> uint32_t { return dim_; }
  /** Bit length of the encoded PQ vector element (see index_parameters).  */
  [[nodiscard]] constexpr inline auto pq_dim() const noexcept -> uint32_t { return pq_dim_; }
  /** Distance metric used for clustering. */
  [[nodiscard]] constexpr inline auto metric() const noexcept -> raft::distance::DistanceType
  {
    return metric_;
  }
  /** Number of clusters/inverted lists. */
  [[nodiscard]] constexpr inline auto n_lists() const noexcept -> uint32_t { return n_lists_; }

  inline auto desc() noexcept -> detail::cuannIvfPqDescriptor_t& { return cuann_desc_; }
  [[nodiscard]] inline auto desc() const noexcept -> const detail::cuannIvfPqDescriptor_t&
  {
    return cuann_desc_;
  }

  // Don't allow copying the index for performance reasons (try avoiding copying data)
  index(const index&) = delete;
  index(index&&)      = default;
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index& = default;
  ~index()                          = default;

  /** Construct an empty index. It needs to be trained and then populated. */
  index(const handle_t& handle,
        raft::distance::DistanceType metric,
        uint32_t n_lists,
        uint32_t dim,
        uint32_t pq_dim = 0)
    : knn::index(),
      n_lists_(n_lists),
      metric_(metric),
      dim_(dim),
      pq_dim_(pq_dim == 0 ? calculate_pq_dim(dim) : pq_dim),
      cuann_desc_{detail::cuannIvfPqCreateDescriptor()}
  {
    check_consistency();
  }

 private:
  raft::distance::DistanceType metric_;
  uint32_t n_lists_;
  uint32_t dim_;
  uint32_t pq_dim_;
  detail::cuannIvfPqDescriptor_t cuann_desc_;

  /** Throw an error if the index content is inconsistent. */
  void check_consistency() {}

  static inline auto calculate_pq_dim(uint32_t dim) -> uint32_t
  {
    // If the dimensionality is large enough, we can reduce it to improve performance
    if (dim >= 128) { dim /= 2; }
    // Round it up to 32 to improve performance.
    uint32_t r = raft::alignDown<uint32_t>(dim, 32);
    if (r > 0) return r;
    // If the dimensionality is really low, round it to the closest power-of-two
    r = 1;
    while ((r << 1) <= dim) {
      r = r << 1;
    }
    return r;
  }
};

}  // namespace raft::spatial::knn::ivf_pq
