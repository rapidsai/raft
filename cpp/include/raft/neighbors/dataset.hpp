/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>   // get_device_for_address
#include <raft/util/integer_utils.hpp>  // rounding up

#include <memory>
#include <numeric>
#include <type_traits>

#ifdef __cpp_lib_bitops
#include <bit>
#endif

namespace raft::neighbors {

/** Two-dimensional dataset; maybe owning, maybe compressed, maybe strided. */
template <typename IdxT>
struct dataset {
  using index_type = IdxT;
  /**  Size of the dataset. */
  [[nodiscard]] virtual auto n_rows() const noexcept -> index_type = 0;
  /** Dimensionality of the dataset. */
  [[nodiscard]] virtual auto dim() const noexcept -> uint32_t = 0;
  /** Whether the object owns the data. */
  [[nodiscard]] virtual auto is_owning() const noexcept -> bool = 0;
  virtual ~dataset() noexcept                                   = default;
};

template <typename IdxT>
struct empty_dataset : public dataset<IdxT> {
  using index_type = IdxT;
  uint32_t suggested_dim;
  explicit empty_dataset(uint32_t dim) noexcept : suggested_dim(dim) {}
  [[nodiscard]] auto n_rows() const noexcept -> index_type final { return 0; }
  [[nodiscard]] auto dim() const noexcept -> uint32_t final { return suggested_dim; }
  [[nodiscard]] auto is_owning() const noexcept -> bool final { return true; }
};

template <typename DataT, typename IdxT>
struct strided_dataset : public dataset<IdxT> {
  using index_type = IdxT;
  using value_type = DataT;
  using view_type  = device_matrix_view<const value_type, index_type, layout_stride>;
  [[nodiscard]] auto n_rows() const noexcept -> index_type final { return view().extent(0); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t final
  {
    return static_cast<uint32_t>(view().extent(1));
  }
  /** Leading dimension of the dataset. */
  [[nodiscard]] constexpr auto stride() const noexcept -> uint32_t
  {
    auto v = view();
    return static_cast<uint32_t>(v.stride(0) > 0 ? v.stride(0) : v.extent(1));
  }
  /** Get the view of the data. */
  [[nodiscard]] virtual auto view() const noexcept -> view_type = 0;
};

template <typename DataT, typename IdxT>
struct non_owning_dataset : public strided_dataset<DataT, IdxT> {
  using index_type = IdxT;
  using value_type = DataT;
  using typename strided_dataset<value_type, index_type>::view_type;
  view_type data;
  explicit non_owning_dataset(view_type v) noexcept : data(v) {}
  [[nodiscard]] auto is_owning() const noexcept -> bool final { return false; }
  [[nodiscard]] auto view() const noexcept -> view_type final { return data; };
};

template <typename DataT, typename IdxT, typename LayoutPolicy, typename ContainerPolicy>
struct owning_dataset : public strided_dataset<DataT, IdxT> {
  using index_type = IdxT;
  using value_type = DataT;
  using typename strided_dataset<value_type, index_type>::view_type;
  using storage_type =
    mdarray<value_type, matrix_extent<index_type>, LayoutPolicy, ContainerPolicy>;
  using mapping_type = typename view_type::mapping_type;
  storage_type data;
  mapping_type view_mapping;
  owning_dataset(storage_type&& store, mapping_type view_mapping) noexcept
    : data{std::move(store)}, view_mapping{view_mapping}
  {
  }

  [[nodiscard]] auto is_owning() const noexcept -> bool final { return true; }
  [[nodiscard]] auto view() const noexcept -> view_type final
  {
    return view_type{data.data_handle(), view_mapping};
  };
};

/**
 * @brief Contstruct a strided matrix from any mdarray or mdspan.
 *
 * This function constructs a non-owning view if the input satisfied two conditions:
 *
 *   1) The data is accessible from the current device
 *   2) The memory layout is the same as expected (row-major matrix with the required stride)
 *
 * Otherwise, this function constructs an owning device matrix and copies the data.
 * When the data is copied, padding elements are filled with zeroes.
 *
 * @tparam SrcT the source mdarray or mdspan
 *
 * @param[in] res raft resources handle
 * @param[in] src the source mdarray or mdspan
 * @param[in] required_stride the leading dimension (in elements)
 * @return maybe owning current-device-accessible strided matrix
 */
template <typename SrcT>
auto make_strided_dataset(const raft::resources& res, const SrcT& src, uint32_t required_stride)
  -> std::unique_ptr<strided_dataset<typename SrcT::value_type, typename SrcT::index_type>>
{
  using extents_type = typename SrcT::extents_type;
  using value_type   = typename SrcT::value_type;
  using index_type   = typename SrcT::index_type;
  using layout_type  = typename SrcT::layout_type;
  static_assert(extents_type::rank() == 2, "The input must be a matrix.");
  static_assert(std::is_same_v<layout_type, layout_right> ||
                  std::is_same_v<layout_type, layout_right_padded<value_type>> ||
                  std::is_same_v<layout_type, layout_stride>,
                "The input must be row-major");
  RAFT_EXPECTS(src.extent(1) <= required_stride,
               "The input row length must be not larger than the desired stride.");
  cudaPointerAttributes ptr_attrs;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&ptr_attrs, src.data_handle()));
  auto* device_ptr             = reinterpret_cast<value_type*>(ptr_attrs.devicePointer);
  const uint32_t src_stride    = src.stride(0) > 0 ? src.stride(0) : src.extent(1);
  const bool device_accessible = device_ptr != nullptr;
  const bool row_major         = src.stride(1) <= 1;
  const bool stride_matches    = required_stride == src_stride;

  if (device_accessible && row_major && stride_matches) {
    // Everything matches: make a non-owning dataset
    return std::make_unique<non_owning_dataset<value_type, index_type>>(
      make_device_strided_matrix_view<const value_type, index_type>(
        device_ptr, src.extent(0), src.extent(1), required_stride));
  }
  // Something is wrong: have to make a copy and produce an owning dataset
  auto out_layout =
    make_strided_layout(src.extents(), std::array<index_type, 2>{required_stride, 1});
  auto out_array = make_device_matrix<value_type, index_type>(res, src.extent(0), required_stride);

  using out_mdarray_type          = decltype(out_array);
  using out_layout_type           = typename out_mdarray_type::layout_type;
  using out_container_policy_type = typename out_mdarray_type::container_policy_type;
  using out_owning_type =
    owning_dataset<value_type, index_type, out_layout_type, out_container_policy_type>;

  RAFT_CUDA_TRY(cudaMemsetAsync(out_array.data_handle(),
                                0,
                                out_array.size() * sizeof(value_type),
                                resource::get_cuda_stream(res)));
  RAFT_CUDA_TRY(cudaMemcpy2DAsync(out_array.data_handle(),
                                  sizeof(value_type) * required_stride,
                                  src.data_handle(),
                                  sizeof(value_type) * src_stride,
                                  sizeof(value_type) * src.extent(1),
                                  src.extent(0),
                                  cudaMemcpyDefault,
                                  resource::get_cuda_stream(res)));

  return std::make_unique<out_owning_type>(std::move(out_array), out_layout);
}

/**
 * @brief Contstruct a strided matrix from any mdarray or mdspan.
 *
 * A variant `make_strided_dataset` that allows specifying the byte alignment instead of the
 * explicit stride length.
 *
 * @tparam SrcT the source mdarray or mdspan
 *
 * @param[in] res raft resources handle
 * @param[in] src the source mdarray or mdspan
 * @param[in] align_bytes the required byte alignment for the dataset rows.
 * @return maybe owning current-device-accessible strided matrix
 */
template <typename SrcT>
auto make_aligned_dataset(const raft::resources& res, const SrcT& src, uint32_t align_bytes = 16)
  -> std::unique_ptr<strided_dataset<typename SrcT::value_type, typename SrcT::index_type>>
{
  using value_type       = typename SrcT::value_type;
  constexpr size_t kSize = sizeof(value_type);
  uint32_t required_stride =
    raft::round_up_safe<size_t>(src.extent(1) * kSize, std::lcm(align_bytes, kSize)) / kSize;
  return make_strided_dataset(res, src, required_stride);
}

/** Parameters for VPQ compression. */
struct vpq_params {
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
   * The dimensionality of the vector after compression by PQ.
   * When zero, an optimal value is selected using a heuristic.
   *
   * TODO: at the moment `dim` must be a multiple `pq_dim`.
   */
  uint32_t pq_dim = 0;
  /**
   * Vector Quantization (VQ) codebook size - number of "coarse cluster centers".
   * When zero, an optimal value is selected using a heuristic.
   */
  uint32_t vq_n_centers = 0;
  /** The number of iterations searching for kmeans centers (both VQ & PQ phases). */
  uint32_t kmeans_n_iters = 25;
  /**
   * The fraction of data to use during iterative kmeans building (VQ phase).
   * When zero, an optimal value is selected using a heuristic.
   */
  double vq_kmeans_trainset_fraction = 0;
  /**
   * The fraction of data to use during iterative kmeans building (PQ phase).
   * When zero, an optimal value is selected using a heuristic.
   */
  double pq_kmeans_trainset_fraction = 0;
};

/**
 * @brief VPQ compressed dataset.
 *
 * The dataset is compressed using two level quantization
 *
 *   1. Vector Quantization
 *   2. Product Quantization of residuals
 *
 * @tparam MathT the type of elements in the codebooks
 * @tparam IdxT type of the vector indices (represent dataset.extent(0))
 *
 */
template <typename MathT, typename IdxT>
struct vpq_dataset : public dataset<IdxT> {
  /** Vector Quantization codebook - "coarse cluster centers". */
  device_matrix<MathT, uint32_t, row_major> vq_code_book;
  /** Product Quantization codebook - "fine cluster centers".  */
  device_matrix<MathT, uint32_t, row_major> pq_code_book;
  /** Compressed dataset.  */
  device_matrix<uint8_t, IdxT, row_major> data;

  vpq_dataset(device_matrix<MathT, uint32_t, row_major>&& vq_code_book,
              device_matrix<MathT, uint32_t, row_major>&& pq_code_book,
              device_matrix<uint8_t, IdxT, row_major>&& data)
    : vq_code_book{std::move(vq_code_book)},
      pq_code_book{std::move(pq_code_book)},
      data{std::move(data)}
  {
  }

  [[nodiscard]] auto n_rows() const noexcept -> IdxT final { return data.extent(0); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t final { return vq_code_book.extent(1); }
  [[nodiscard]] auto is_owning() const noexcept -> bool final { return true; }

  /** Row length of the encoded data in bytes. */
  [[nodiscard]] constexpr inline auto encoded_row_length() const noexcept -> uint32_t
  {
    return data.extent(1);
  }
  /** The number of "coarse cluster centers" */
  [[nodiscard]] constexpr inline auto vq_n_centers() const noexcept -> uint32_t
  {
    return vq_code_book.extent(0);
  }
  /** The bit length of an encoded vector element after compression by PQ. */
  [[nodiscard]] constexpr inline auto pq_bits() const noexcept -> uint32_t
  {
    /*
    NOTE: pq_bits and the book size

    Normally, we'd store `pq_bits` as a part of the index.
    However, we know there's an invariant `pq_n_centers = 1 << pq_bits`, i.e. the codebook size is
    the same as the number of possible code values. Hence, we don't store the pq_bits and derive it
    from the array dimensions instead.
     */
    auto pq_width = pq_n_centers();
#ifdef __cpp_lib_bitops
    return std::countr_zero(pq_width);
#else
    uint32_t pq_bits = 0;
    while (pq_width > 1) {
      pq_bits++;
      pq_width >>= 1;
    }
    return pq_bits;
#endif
  }
  /** The dimensionality of an encoded vector after compression by PQ. */
  [[nodiscard]] constexpr inline auto pq_dim() const noexcept -> uint32_t
  {
    return raft::div_rounding_up_unsafe(dim(), pq_len());
  }
  /** Dimensionality of a subspaces, i.e. the number of vector components mapped to a subspace */
  [[nodiscard]] constexpr inline auto pq_len() const noexcept -> uint32_t
  {
    return pq_code_book.extent(1);
  }
  /** The number of vectors in a PQ codebook (`1 << pq_bits`). */
  [[nodiscard]] constexpr inline auto pq_n_centers() const noexcept -> uint32_t
  {
    return pq_code_book.extent(0);
  }
};

}  // namespace raft::neighbors
