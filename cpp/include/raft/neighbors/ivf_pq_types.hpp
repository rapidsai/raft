/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/ann_types.hpp>
#include <raft/neighbors/ivf_list_types.hpp>
#include <raft/util/integer_utils.hpp>

#include <thrust/fill.h>

#include <memory>
#include <type_traits>

namespace raft::neighbors::ivf_pq {

/**
 * @addtogroup ivf_pq
 * @{
 */

/** A type for specifying how PQ codebooks are created. */
enum class codebook_gen {  // NOLINT
  PER_SUBSPACE = 0,        // NOLINT
  PER_CLUSTER  = 1,        // NOLINT
};

struct index_params : ann::index_params {
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
   * For good performance, it is desirable that 'pq_dim' is a multiple of 32. Ideally, 'pq_dim'
   * should be also a divisor of the dataset dim.
   */
  uint32_t pq_dim = 0;
  /** How PQ codebooks are created. */
  codebook_gen codebook_kind = codebook_gen::PER_SUBSPACE;
  /**
   * Apply a random rotation matrix on the input data and queries even if `dim % pq_dim == 0`.
   *
   * Note: if `dim` is not multiple of `pq_dim`, a random rotation is always applied to the input
   * data and queries to transform the working space from `dim` to `rot_dim`, which may be slightly
   * larger than the original space and and is a multiple of `pq_dim` (`rot_dim % pq_dim == 0`).
   * However, this transform is not necessary when `dim` is multiple of `pq_dim`
   *   (`dim == rot_dim`, hence no need in adding "extra" data columns / features).
   *
   * By default, if `dim == rot_dim`, the rotation transform is initialized with the identity
   * matrix. When `force_random_rotation == true`, a random orthogonal transform matrix is generated
   * regardless of the values of `dim` and `pq_dim`.
   */
  bool force_random_rotation = false;
  /**
   * By default, the algorithm allocates more space than necessary for individual clusters
   * (`list_data`). This allows to amortize the cost of memory allocation and reduce the number of
   * data copies during repeated calls to `extend` (extending the database).
   *
   * The alternative is the conservative allocation behavior; when enabled, the algorithm always
   * allocates the minimum amount of memory required to store the given number of records. Set this
   * flag to `true` if you prefer to use as little GPU memory for the database as possible.
   */
  bool conservative_memory_allocation = false;

  /**
   * Creates index_params based on shape of the input dataset.
   * Usage example:
   * @code{.cpp}
   *   using namespace raft::neighbors;
   *   raft::resources res;
   *   // create index_params for a [N. D] dataset and have InnerProduct as the distance metric
   *   auto dataset = raft::make_device_matrix<float, int64_t>(res, N, D);
   *   ivf_pq::index_params index_params =
   *     ivf_pq::index_params::from_dataset(dataset.view(), raft::distance::InnerProduct);
   *   // modify/update index_params as needed
   *   index_params.add_data_on_build = true;
   * @endcode
   */
  template <typename DataT, typename Accessor>
  static index_params from_dataset(
    mdspan<const DataT, matrix_extent<int64_t>, row_major, Accessor> dataset,
    raft::distance::DistanceType metric = raft::distance::L2Expanded)
  {
    index_params params;
    params.n_lists =
      dataset.extent(0) < 4 * 2500 ? 4 : static_cast<uint32_t>(std::sqrt(dataset.extent(0)));
    params.pq_dim =
      round_up_safe(static_cast<uint32_t>(dataset.extent(1) / 4), static_cast<uint32_t>(8));
    params.pq_bits                  = 8;
    params.kmeans_trainset_fraction = dataset.extent(0) < 10000 ? 1 : 0.1;
    params.metric                   = metric;
    return params;
  }
};

struct search_params : ann::search_params {
  /** The number of clusters to search. */
  uint32_t n_probes = 20;
  /**
   * Data type of look up table to be created dynamically at search time.
   *
   * Possible values: [CUDA_R_32F, CUDA_R_16F, CUDA_R_8U]
   *
   * The use of low-precision types reduces the amount of shared memory required at search time, so
   * fast shared memory kernels can be used even for datasets with large dimansionality. Note that
   * the recall is slightly degraded when low-precision type is selected.
   */
  cudaDataType_t lut_dtype = CUDA_R_32F;
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
   * Preferred fraction of SM's unified memory / L1 cache to be used as shared memory.
   *
   * Possible values: [0.0 - 1.0] as a fraction of the `sharedMemPerMultiprocessor`.
   *
   * One wants to increase the carveout to make sure a good GPU occupancy for the main search
   * kernel, but not to keep it too high to leave some memory to be used as L1 cache. Note, this
   * value is interpreted only as a hint. Moreover, a GPU usually allows only a fixed set of cache
   * configurations, so the provided value is rounded up to the nearest configuration. Refer to the
   * NVIDIA tuning guide for the target GPU architecture.
   *
   * Note, this is a low-level tuning parameter that can have drastic negative effects on the search
   * performance if tweaked incorrectly.
   */
  double preferred_shmem_carveout = 1.0;
};

static_assert(std::is_aggregate_v<index_params>);
static_assert(std::is_aggregate_v<search_params>);

/** Size of the interleaved group. */
constexpr static uint32_t kIndexGroupSize = 32;
/** Stride of the interleaved group for vectorized loads. */
constexpr static uint32_t kIndexGroupVecLen = 16;

/**
 * Default value returned by `search` when the `n_probes` is too small and top-k is too large.
 * One may encounter it if the combined size of probed clusters is smaller than the requested
 * number of results per query.
 */
template <typename IdxT>
constexpr static IdxT kOutOfBoundsRecord = std::numeric_limits<IdxT>::max();

template <typename SizeT, typename IdxT>
struct list_spec {
  using value_type = uint8_t;
  using index_type = IdxT;
  /** PQ-encoded data stored in the interleaved format:
   *
   *    [ ceildiv(list_size, kIndexGroupSize)
   *    , ceildiv(pq_dim, (kIndexGroupVecLen * 8u) / pq_bits)
   *    , kIndexGroupSize
   *    , kIndexGroupVecLen
   *    ].
   */
  using list_extents =
    extents<SizeT, dynamic_extent, dynamic_extent, kIndexGroupSize, kIndexGroupVecLen>;

  SizeT align_max;
  SizeT align_min;
  uint32_t pq_bits;
  uint32_t pq_dim;

  constexpr list_spec(uint32_t pq_bits, uint32_t pq_dim, bool conservative_memory_allocation)
    : pq_bits(pq_bits),
      pq_dim(pq_dim),
      align_min(kIndexGroupSize),
      align_max(conservative_memory_allocation ? kIndexGroupSize : 1024)
  {
  }

  // Allow casting between different size-types (for safer size and offset calculations)
  template <typename OtherSizeT>
  constexpr explicit list_spec(const list_spec<OtherSizeT, IdxT>& other_spec)
    : pq_bits{other_spec.pq_bits},
      pq_dim{other_spec.pq_dim},
      align_min{other_spec.align_min},
      align_max{other_spec.align_max}
  {
  }

  /** Determine the extents of an array enough to hold a given amount of data. */
  constexpr auto make_list_extents(SizeT n_rows) const -> list_extents
  {
    // how many elems of pq_dim fit into one kIndexGroupVecLen-byte chunk
    auto pq_chunk = (kIndexGroupVecLen * 8u) / pq_bits;
    return make_extents<SizeT>(div_rounding_up_safe<SizeT>(n_rows, kIndexGroupSize),
                               div_rounding_up_safe<SizeT>(pq_dim, pq_chunk),
                               kIndexGroupSize,
                               kIndexGroupVecLen);
  }
};

template <typename IdxT, typename SizeT = uint32_t>
using list_data = ivf::list<list_spec, SizeT, IdxT>;

/**
 * @brief IVF-PQ index.
 *
 * In the IVF-PQ index, a database vector y is approximated with two level quantization:
 *
 * y = Q_1(y) + Q_2(y - Q_1(y))
 *
 * The first level quantizer (Q_1), maps the vector y to the nearest cluster center. The number of
 * clusters is n_lists.
 *
 * The second quantizer encodes the residual, and it is defined as a product quantizer [1].
 *
 * A product quantizer encodes a `dim` dimensional vector with a `pq_dim` dimensional vector.
 * First we split the input vector into `pq_dim` subvectors (denoted by u), where each u vector
 * contains `pq_len` distinct components of y
 *
 * y_1, y_2, ... y_{pq_len}, y_{pq_len+1}, ... y_{2*pq_len}, ... y_{dim-pq_len+1} ... y_{dim}
 *  \___________________/     \____________________________/      \______________________/
 *         u_1                         u_2                          u_{pq_dim}
 *
 * Then each subvector encoded with a separate quantizer q_i, end the results are concatenated
 *
 * Q_2(y) = q_1(u_1),q_2(u_2),...,q_{pq_dim}(u_pq_dim})
 *
 * Each quantizer q_i outputs a code with pq_bit bits. The second level quantizers are also defined
 * by k-means clustering in the corresponding sub-space: the reproduction values are the centroids,
 * and the set of reproduction values is the codebook.
 *
 * When the data dimensionality `dim` is not multiple of `pq_dim`, the feature space is transformed
 * using a random orthogonal matrix to have `rot_dim = pq_dim * pq_len` dimensions
 * (`rot_dim >= dim`).
 *
 * The second-level quantizers are trained either for each subspace or for each cluster:
 *   (a) codebook_gen::PER_SUBSPACE:
 *         creates `pq_dim` second-level quantizers - one for each slice of the data along features;
 *   (b) codebook_gen::PER_CLUSTER:
 *         creates `n_lists` second-level quantizers - one for each first-level cluster.
 * In either case, the centroids are again found using k-means clustering interpreting the data as
 * having pq_len dimensions.
 *
 * [1] Product quantization for nearest neighbor search Herve Jegou, Matthijs Douze, Cordelia Schmid
 *
 * @tparam IdxT type of the indices in the source dataset
 *
 */
template <typename IdxT>
struct index : ann::index {
  static_assert(!raft::is_narrowing_v<uint32_t, IdxT>,
                "IdxT must be able to represent all values of uint32_t");

 public:
  /** Total length of the index. */
  [[nodiscard]] constexpr inline auto size() const noexcept -> IdxT
  {
    return accum_sorted_sizes_(n_lists());
  }
  /** Dimensionality of the input data. */
  [[nodiscard]] constexpr inline auto dim() const noexcept -> uint32_t { return dim_; }
  /**
   * Dimensionality of the cluster centers:
   * input data dim extended with vector norms and padded to 8 elems.
   */
  [[nodiscard]] constexpr inline auto dim_ext() const noexcept -> uint32_t
  {
    return raft::round_up_safe(dim() + 1, 8u);
  }
  /**
   * Dimensionality of the data after transforming it for PQ processing
   * (rotated and augmented to be muplitple of `pq_dim`).
   */
  [[nodiscard]] constexpr inline auto rot_dim() const noexcept -> uint32_t
  {
    return pq_len() * pq_dim();
  }
  /** The bit length of an encoded vector element after compression by PQ. */
  [[nodiscard]] constexpr inline auto pq_bits() const noexcept -> uint32_t { return pq_bits_; }
  /** The dimensionality of an encoded vector after compression by PQ. */
  [[nodiscard]] constexpr inline auto pq_dim() const noexcept -> uint32_t { return pq_dim_; }
  /** Dimensionality of a subspaces, i.e. the number of vector components mapped to a subspace */
  [[nodiscard]] constexpr inline auto pq_len() const noexcept -> uint32_t
  {
    return raft::div_rounding_up_unsafe(dim(), pq_dim());
  }
  /** The number of vectors in a PQ codebook (`1 << pq_bits`). */
  [[nodiscard]] constexpr inline auto pq_book_size() const noexcept -> uint32_t
  {
    return 1 << pq_bits();
  }
  /** Distance metric used for clustering. */
  [[nodiscard]] constexpr inline auto metric() const noexcept -> raft::distance::DistanceType
  {
    return metric_;
  }
  /** How PQ codebooks are created. */
  [[nodiscard]] constexpr inline auto codebook_kind() const noexcept -> codebook_gen
  {
    return codebook_kind_;
  }
  /** Number of clusters/inverted lists (first level quantization). */
  [[nodiscard]] constexpr inline auto n_lists() const noexcept -> uint32_t { return lists_.size(); }
  /**
   * Whether to use convervative memory allocation when extending the list (cluster) data
   * (see index_params.conservative_memory_allocation).
   */
  [[nodiscard]] constexpr inline auto conservative_memory_allocation() const noexcept -> bool
  {
    return conservative_memory_allocation_;
  }

  // Don't allow copying the index for performance reasons (try avoiding copying data)
  index(const index&)                    = delete;
  index(index&&)                         = default;
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index&      = default;
  ~index()                               = default;

  /** Construct an empty index. It needs to be trained and then populated. */
  index(raft::resources const& handle,
        raft::distance::DistanceType metric,
        codebook_gen codebook_kind,
        uint32_t n_lists,
        uint32_t dim,
        uint32_t pq_bits                    = 8,
        uint32_t pq_dim                     = 0,
        bool conservative_memory_allocation = false)
    : ann::index(),
      metric_(metric),
      codebook_kind_(codebook_kind),
      dim_(dim),
      pq_bits_(pq_bits),
      pq_dim_(pq_dim == 0 ? calculate_pq_dim(dim) : pq_dim),
      conservative_memory_allocation_(conservative_memory_allocation),
      pq_centers_{make_device_mdarray<float>(handle, make_pq_centers_extents())},
      lists_{n_lists},
      rotation_matrix_{make_device_matrix<float, uint32_t>(handle, this->rot_dim(), this->dim())},
      list_sizes_{make_device_vector<uint32_t, uint32_t>(handle, n_lists)},
      centers_{make_device_matrix<float, uint32_t>(handle, n_lists, this->dim_ext())},
      centers_rot_{make_device_matrix<float, uint32_t>(handle, n_lists, this->rot_dim())},
      data_ptrs_{make_device_vector<uint8_t*, uint32_t>(handle, n_lists)},
      inds_ptrs_{make_device_vector<IdxT*, uint32_t>(handle, n_lists)},
      accum_sorted_sizes_{make_host_vector<IdxT, uint32_t>(n_lists + 1)}
  {
    check_consistency();
    accum_sorted_sizes_(n_lists) = 0;
  }

  /** Construct an empty index. It needs to be trained and then populated. */
  index(raft::resources const& handle, const index_params& params, uint32_t dim)
    : index(handle,
            params.metric,
            params.codebook_kind,
            params.n_lists,
            dim,
            params.pq_bits,
            params.pq_dim,
            params.conservative_memory_allocation)
  {
  }

  using pq_centers_extents =
    std::experimental::extents<uint32_t, dynamic_extent, dynamic_extent, dynamic_extent>;
  /**
   * PQ cluster centers
   *
   *   - codebook_gen::PER_SUBSPACE: [pq_dim , pq_len, pq_book_size]
   *   - codebook_gen::PER_CLUSTER:  [n_lists, pq_len, pq_book_size]
   */
  inline auto pq_centers() noexcept -> device_mdspan<float, pq_centers_extents, row_major>
  {
    return pq_centers_.view();
  }
  [[nodiscard]] inline auto pq_centers() const noexcept
    -> device_mdspan<const float, pq_centers_extents, row_major>
  {
    return pq_centers_.view();
  }

  /** Lists' data and indices. */
  inline auto lists() noexcept -> std::vector<std::shared_ptr<list_data<IdxT>>>& { return lists_; }
  [[nodiscard]] inline auto lists() const noexcept
    -> const std::vector<std::shared_ptr<list_data<IdxT>>>&
  {
    return lists_;
  }

  /** Pointers to the inverted lists (clusters) data  [n_lists]. */
  inline auto data_ptrs() noexcept -> device_vector_view<uint8_t*, uint32_t, row_major>
  {
    return data_ptrs_.view();
  }
  [[nodiscard]] inline auto data_ptrs() const noexcept
    -> device_vector_view<const uint8_t* const, uint32_t, row_major>
  {
    return make_mdspan<const uint8_t* const, uint32_t, row_major, false, true>(
      data_ptrs_.data_handle(), data_ptrs_.extents());
  }

  /** Pointers to the inverted lists (clusters) indices  [n_lists]. */
  inline auto inds_ptrs() noexcept -> device_vector_view<IdxT*, uint32_t, row_major>
  {
    return inds_ptrs_.view();
  }
  [[nodiscard]] inline auto inds_ptrs() const noexcept
    -> device_vector_view<const IdxT* const, uint32_t, row_major>
  {
    return make_mdspan<const IdxT* const, uint32_t, row_major, false, true>(
      inds_ptrs_.data_handle(), inds_ptrs_.extents());
  }

  /** The transform matrix (original space -> rotated padded space) [rot_dim, dim] */
  inline auto rotation_matrix() noexcept -> device_matrix_view<float, uint32_t, row_major>
  {
    return rotation_matrix_.view();
  }
  [[nodiscard]] inline auto rotation_matrix() const noexcept
    -> device_matrix_view<const float, uint32_t, row_major>
  {
    return rotation_matrix_.view();
  }

  /**
   * Accumulated list sizes, sorted in descending order [n_lists + 1].
   * The last value contains the total length of the index.
   * The value at index zero is always zero.
   *
   * That is, the content of this span is as if the `list_sizes` was sorted and then accumulated.
   *
   * This span is used during search to estimate the maximum size of the workspace.
   */
  inline auto accum_sorted_sizes() noexcept -> host_vector_view<IdxT, uint32_t, row_major>
  {
    return accum_sorted_sizes_.view();
  }
  [[nodiscard]] inline auto accum_sorted_sizes() const noexcept
    -> host_vector_view<const IdxT, uint32_t, row_major>
  {
    return accum_sorted_sizes_.view();
  }

  /** Sizes of the lists [n_lists]. */
  inline auto list_sizes() noexcept -> device_vector_view<uint32_t, uint32_t, row_major>
  {
    return list_sizes_.view();
  }
  [[nodiscard]] inline auto list_sizes() const noexcept
    -> device_vector_view<const uint32_t, uint32_t, row_major>
  {
    return list_sizes_.view();
  }

  /** Cluster centers corresponding to the lists in the original space [n_lists, dim_ext] */
  inline auto centers() noexcept -> device_matrix_view<float, uint32_t, row_major>
  {
    return centers_.view();
  }
  [[nodiscard]] inline auto centers() const noexcept
    -> device_matrix_view<const float, uint32_t, row_major>
  {
    return centers_.view();
  }

  /** Cluster centers corresponding to the lists in the rotated space [n_lists, rot_dim] */
  inline auto centers_rot() noexcept -> device_matrix_view<float, uint32_t, row_major>
  {
    return centers_rot_.view();
  }
  [[nodiscard]] inline auto centers_rot() const noexcept
    -> device_matrix_view<const float, uint32_t, row_major>
  {
    return centers_rot_.view();
  }

  /** fetch size of a particular IVF list in bytes using the list extents.
   * Usage example:
   * @code{.cpp}
   *   raft::resources res;
   *   // use default index params
   *   ivf_pq::index_params index_params;
   *   // extend the IVF lists while building the index
   *   index_params.add_data_on_build = true;
   *   // create and fill the index from a [N, D] dataset
   *   auto index = raft::neighbors::ivf_pq::build<int64_t>(res, index_params, dataset, N, D);
   *   // Fetch the size of the fourth list
   *   uint32_t size = index.get_list_size_in_bytes(3);
   * @endcode
   *
   * @param[in] label list ID
   */
  inline auto get_list_size_in_bytes(uint32_t label) -> uint32_t
  {
    RAFT_EXPECTS(label < this->n_lists(),
                 "Expected label to be less than number of lists in the index");
    auto list_data = this->lists()[label]->data;
    return list_data.size();
  }

 private:
  raft::distance::DistanceType metric_;
  codebook_gen codebook_kind_;
  uint32_t dim_;
  uint32_t pq_bits_;
  uint32_t pq_dim_;
  bool conservative_memory_allocation_;

  // Primary data members
  std::vector<std::shared_ptr<list_data<IdxT>>> lists_;
  device_vector<uint32_t, uint32_t, row_major> list_sizes_;
  device_mdarray<float, pq_centers_extents, row_major> pq_centers_;
  device_matrix<float, uint32_t, row_major> centers_;
  device_matrix<float, uint32_t, row_major> centers_rot_;
  device_matrix<float, uint32_t, row_major> rotation_matrix_;

  // Computed members for accelerating search.
  device_vector<uint8_t*, uint32_t, row_major> data_ptrs_;
  device_vector<IdxT*, uint32_t, row_major> inds_ptrs_;
  host_vector<IdxT, uint32_t, row_major> accum_sorted_sizes_;

  /** Throw an error if the index content is inconsistent. */
  void check_consistency()
  {
    RAFT_EXPECTS(pq_bits() >= 4 && pq_bits() <= 8,
                 "`pq_bits` must be within closed range [4,8], but got %u.",
                 pq_bits());
    RAFT_EXPECTS((pq_bits() * pq_dim()) % 8 == 0,
                 "`pq_bits * pq_dim` must be a multiple of 8, but got %u * %u = %u.",
                 pq_bits(),
                 pq_dim(),
                 pq_bits() * pq_dim());
  }

  auto make_pq_centers_extents() -> pq_centers_extents
  {
    switch (codebook_kind()) {
      case codebook_gen::PER_SUBSPACE:
        return make_extents<uint32_t>(pq_dim(), pq_len(), pq_book_size());
      case codebook_gen::PER_CLUSTER:
        return make_extents<uint32_t>(n_lists(), pq_len(), pq_book_size());
      default: RAFT_FAIL("Unreachable code");
    }
  }

  static inline auto calculate_pq_dim(uint32_t dim) -> uint32_t
  {
    // If the dimensionality is large enough, we can reduce it to improve performance
    if (dim >= 128) { dim /= 2; }
    // Round it down to 32 to improve performance.
    auto r = raft::round_down_safe<uint32_t>(dim, 32);
    if (r > 0) return r;
    // If the dimensionality is really low, round it to the closest power-of-two
    r = 1;
    while ((r << 1) <= dim) {
      r = r << 1;
    }
    return r;
  }
};

/** @} */

}  // namespace raft::neighbors::ivf_pq
