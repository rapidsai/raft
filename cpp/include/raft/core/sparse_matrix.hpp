/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <raft/core/device_resources.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/span.hpp>
#include <raft/core/sparse_matrix_types.hpp>

namespace raft {

/**
 * Maintains metadata about the structure and sparsity of a sparse matrix.
 * @tparam RowType
 * @tparam ColType
 * @tparam NZType
 * @tparam is_device
 */
template <typename RowType, typename ColType, typename NZType, int is_device>
class sparse_structure {
 public:
  using row_type = RowType;
  using col_type = ColType;
  using nnz_type = NZType;

  /**
   * Constructor when sparsity is already known
   * @param n_rows total number of rows in matrix
   * @param n_cols total number of columns in matrix
   * @param nnz sparsity of matrix
   */
  sparse_structure(row_type n_rows, col_type n_cols, nnz_type nnz)
    : n_rows_(n_rows), n_cols_(n_cols), nnz_(nnz){};

  /**
   * Constructor when sparsity is not yet known
   * @param n_rows total number of rows in matrix
   * @param n_cols total number of columns in matrix
   */
  sparse_structure(row_type n_rows, col_type n_cols) : n_rows_(n_rows), n_cols_(n_cols), nnz_(0) {}

  /**
   * Return the sparsity of the matrix (this will be 0 when sparsity is not yet known)
   * @return sparsity of matrix
   */
  nnz_type get_nnz() { return nnz_; }

  /**
   * Return the total number of rows in the matrix
   * @return total number of rows in the matriz
   */
  row_type get_n_rows() { return n_rows_; }

  /**
   * Return the total number of columns in the matrix
   * @return total number of columns
   */
  col_type get_n_cols() { return n_cols_; }

  /**
   * Initialize the matrix sparsity when it was not known
   * upon construction.
   * @param nnz
   */
  virtual void initialize_sparsity(nnz_type nnz) { nnz_ = nnz; }

 protected:
  row_type n_rows_;
  col_type n_cols_;
  nnz_type nnz_;
};

template <typename IndptrType, typename IndicesType, typename NZType, int is_device>
class compressed_structure_t : public sparse_structure<IndptrType, IndicesType, NZType, is_device> {
 public:
  /**
   * Constructor when sparsity is already known
   * @param n_rows total number of rows in matrix
   * @param n_cols total number of columns in matrix
   * @param nnz sparsity of matrix
   */
  compressed_structure_t(IndptrType n_rows, IndicesType n_cols, NZType nnz)
    : sparse_structure<IndptrType, IndicesType, NZType, is_device>(n_rows, n_cols, nnz){};

  /**
   * Return span containing underlying indptr array
   * @return span containing underlying indptr array
   */
  virtual span<IndptrType, is_device> get_indptr() = 0;

  /**
   * Return span containing underlying indices array
   * @return span containing underlying indices array
   */
  virtual span<IndicesType, is_device> get_indices() = 0;
};

template <typename RowType, typename ColType, typename NZType, int is_device>
class coordinate_structure_t : public sparse_structure<RowType, ColType, NZType, is_device> {
 public:
  coordinate_structure_t(RowType n_rows, ColType n_cols, NZType nnz)
    : sparse_structure<RowType, ColType, NZType, is_device>(n_rows, n_cols, nnz){};

  /**
   * Return span containing underlying rows array
   * @return span containing underlying rows array
   */
  virtual span<RowType, is_device> get_rows() = 0;

  /**
   * Return span containing underlying cols array
   * @return span containing underlying cols array
   */
  virtual span<ColType, is_device> get_cols() = 0;
};

/**
 * A non-owning view into a compressed sparse structure
 *
 * The structure representation does not have a value/weight
 * component so that its const-ness can be varied from it.
 *
 * @tparam IndptrType
 * @tparam IndicesType
 */
template <typename IndptrType, typename IndicesType, typename NZType, bool is_device>
class compressed_structure_view
  : public compressed_structure_t<IndptrType, IndicesType, NZType, is_device> {
 public:
  static constexpr SparsityType type_enum = SparsityType::PRESERVING;
  using view_type = compressed_structure_view<IndptrType, IndicesType, NZType, is_device>;
  using indptr_type =
    typename sparse_structure<IndptrType, IndicesType, NZType, is_device>::row_type;
  using indices_type =
    typename sparse_structure<IndptrType, IndicesType, NZType, is_device>::col_type;

  constexpr auto get_type_enum() { return type_enum; }
  compressed_structure_view(span<indptr_type, is_device> indptr,
                            span<indices_type, is_device> indices,
                            indices_type n_cols)
    : compressed_structure_t<IndptrType, IndicesType, NZType, is_device>(
        indptr.size() - 1, n_cols, indices.size()),
      indptr_(indptr),
      indices_(indices)
  {
  }

  /**
   * Return span containing underlying indptr array
   * @return span containing underlying indptr array
   */
  span<indptr_type, is_device> get_indptr() override { return indptr_; }

  /**
   * Return span containing underlying indices array
   * @return span containing underlying indices array
   */
  span<indices_type, is_device> get_indices() override { return indices_; }

  /**
   * Create a view from this view. Note that this is for interface compatibility
   * @return
   */
  view_type view() { return view_type(indptr_, indices_, this->n_rows_, this->n_cols_); }

  /**
   * Initialize sparsity when it was not known upon construction.
   *
   * Note: A view is sparsity-preserving so the sparsity cannot be mutated.
   */
  void initialize_sparsity(NZType)
  {
    RAFT_FAIL("The sparsity of structure-preserving sparse formats cannot be changed");
  }

 protected:
  raft::span<indptr_type, is_device> indptr_;
  raft::span<indices_type, is_device> indices_;
};

/**
 * Represents a sparse compressed structure (or adjacency list)
 * which can be used to model both a CSR and CSC matrix.
 *
 * The structure representation does not have a value/weight
 * component so that its const-ness can be varied from it.
 *
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam ContainerPolicy
 */
template <typename IndptrType,
          typename IndicesType,
          typename NZType,
          bool is_device,
          template <typename T>
          typename ContainerPolicy>
class compressed_structure
  : public compressed_structure_t<IndptrType, IndicesType, NZType, is_device> {
 public:
  static constexpr SparsityType type_enum = SparsityType::OWNING;
  using sparse_structure_type = compressed_structure_t<IndptrType, IndicesType, NZType, is_device>;
  using view_type = compressed_structure_view<IndptrType, IndicesType, NZType, is_device>;
  using indptr_container_policy_type  = ContainerPolicy<IndptrType>;
  using indices_container_policy_type = ContainerPolicy<IndicesType>;
  using indptr_container_type         = typename indptr_container_policy_type::container_type;
  using indices_container_type        = typename indices_container_policy_type::container_type;

  constexpr auto get_type_enum() { return type_enum; }
  constexpr compressed_structure(
    raft::device_resources const& handle,
    IndptrType n_rows,
    IndicesType n_cols,
    NZType nnz = 0) noexcept(std::is_nothrow_default_constructible_v<indptr_container_type>)
    : sparse_structure_type{n_rows, n_cols, nnz},
      handle_{handle},
      cp_indptr_{handle.get_stream()},
      cp_indices_{handle.get_stream()},
      c_indptr_{cp_indptr_.create(n_rows + 1)},
      c_indices_{cp_indices_.create(nnz)} {};

  compressed_structure(compressed_structure const&) noexcept(
    std::is_nothrow_copy_constructible_v<indptr_container_type>) = default;
  compressed_structure(compressed_structure&&) noexcept(
    std::is_nothrow_move_constructible<indptr_container_type>::value) = default;

  constexpr auto operator=(compressed_structure const&) noexcept(
    std::is_nothrow_copy_assignable<indptr_container_type>::value)
    -> compressed_structure& = default;
  constexpr auto operator    =(compressed_structure&&) noexcept(
    std::is_nothrow_move_assignable<indptr_container_type>::value)
    -> compressed_structure& = default;

  /**
   * Return span containing underlying indptr array
   * @return span containing underlying indptr array
   */
  span<IndptrType, is_device> get_indptr() override
  {
    return raft::span<IndptrType, is_device>(c_indptr_.data(), this->get_n_rows() + 1);
  }

  /**
   * Return span containing underlying indices array
   * @return span containing underlying indices array
   */
  span<IndicesType, is_device> get_indices() override
  {
    if (this->get_nnz() == 0) {
      RAFT_LOG_WARN("Indices requested for structure that has uninitialized sparsity.");
    }
    return raft::span<IndicesType, is_device>(c_indices_.data(), this->get_nnz());
  }

  ~compressed_structure() noexcept(std::is_nothrow_destructible<indptr_container_type>::value) =
    default;

  /**
   * Return a view of the compressed structure. Structural views are sparsity-preserving
   * so while the structural elements can be updated in a non-const view, the sparsity
   * itself (number of nonzeros) cannot be changed.
   * @return compressed structure view
   */
  view_type view()
  {
    if (this->get_nnz() > 0) {
      RAFT_LOG_WARN(
        "Cannot create compressed_structure.view() because it has not been initialized (sparsity "
        "is 0)");
    }
    auto indptr_span  = raft::span<IndptrType, is_device>(c_indptr_.data(), this->get_n_rows() + 1);
    auto indices_span = raft::span<IndicesType, is_device>(c_indices_.data(), this->get_nnz());
    return view_type(indptr_span, indices_span, this->get_n_cols());
  }

  /**
   * Change the sparsity of the current compressed structure. This will
   * resize the underlying data arrays.
   * @param nnz new sparsity
   */
  void initialize_sparsity(NZType nnz)
  {
    sparse_structure_type::initialize_sparsity(nnz);
    c_indptr_.resize(this->get_n_rows() + 1, handle_.get_stream());
    c_indices_.resize(nnz, handle_.get_stream());
  }

 protected:
  raft::device_resources const& handle_;
  indptr_container_policy_type cp_indptr_;
  indices_container_policy_type cp_indices_;
  indptr_container_type c_indptr_;
  indices_container_type c_indices_;
};

/**
 * A non-owning view into a coordinate structure
 *
 * The structure representation does not have a value/weight
 * component so that its const-ness can be varied from it.
 *
 * @tparam RowType
 * @tparam ColType
 */
template <typename RowType, typename ColType, typename NZType, bool is_device>
class coordinate_structure_view
  : public coordinate_structure_t<RowType, ColType, NZType, is_device> {
 public:
  static constexpr SparsityType type_enum = PRESERVING;
  using view_type = coordinate_structure_view<RowType, ColType, NZType, is_device>;
  using row_type  = typename sparse_structure<RowType, ColType, NZType, is_device>::row_type;
  using col_type  = typename sparse_structure<RowType, ColType, NZType, is_device>::col_type;
  using nnz_type  = typename sparse_structure<RowType, ColType, NZType, is_device>::nnz_type;

  coordinate_structure_view(span<row_type, is_device> rows,
                            span<col_type, is_device> cols,
                            row_type n_rows,
                            col_type n_cols)
    : coordinate_structure_t<RowType, ColType, NZType, is_device>(n_rows, n_cols, rows.size()),
      rows_{rows},
      cols_{cols}
  {
  }

  /**
   * Create a view from this view. Note that this is for interface compatibility
   * @return
   */
  view_type view() { return view_type(rows_, cols_, this->get_n_rows(), this->get_n_cols()); }

  /**
   * Initialize sparsity when it was not known upon construction.
   *
   * Note: A view is sparsity-preserving so the sparsity cannot be mutated.
   */
  void initialize_sparsity(nnz_type)
  {
    RAFT_FAIL("The sparsity of structure-preserving sparse formats cannot be changed");
  }

  /**
   * Return span containing underlying rows array
   * @return span containing underlying rows array
   */
  span<row_type, is_device> get_rows() override { return rows_; }

  /**
   * Return span containing underlying cols array
   * @return span containing underlying cols array
   */
  span<col_type, is_device> get_cols() override { return cols_; }

 protected:
  raft::span<row_type, is_device> rows_;
  raft::span<col_type, is_device> cols_;
};

/**
 * Represents a sparse coordinate structure (or edge list)
 * which can be used to model a COO matrix.
 *
 * The structure representation does not have a value/weight
 * component so that its const-ness can be varied from it.
 *
 * @tparam RowType
 * @tparam ColType
 * @tparam ContainerPolicy
 */
template <typename RowType,
          typename ColType,
          typename NZType,
          bool is_device,
          template <typename T>
          typename ContainerPolicy>
class coordinate_structure : public coordinate_structure_t<RowType, ColType, NZType, is_device> {
 public:
  static constexpr SparsityType type_enum = OWNING;
  using sparse_structure_type = coordinate_structure_t<RowType, ColType, NZType, is_device>;
  using row_type              = typename sparse_structure_type::row_type;
  using col_type              = typename sparse_structure_type::col_type;
  using nnz_type              = typename sparse_structure_type::nnz_type;

  using view_type = coordinate_structure_view<row_type, col_type, nnz_type, is_device>;
  using row_container_policy_type = ContainerPolicy<RowType>;
  using col_container_policy_type = ContainerPolicy<ColType>;
  using row_container_type        = typename row_container_policy_type::container_type;
  using col_container_type        = typename col_container_policy_type::container_type;

  coordinate_structure(
    raft::device_resources const& handle,
    row_type n_rows,
    col_type n_cols,
    nnz_type nnz = 0) noexcept(std::is_nothrow_default_constructible_v<row_container_type>)
    : coordinate_structure_t<RowType, ColType, NZType, is_device>(n_rows, n_cols, nnz),
      handle_{handle},
      cp_rows_{handle.get_stream()},
      cp_cols_{handle.get_stream()},
      c_rows_{cp_rows_.create(0)},
      c_cols_{cp_cols_.create(0)} {};

  coordinate_structure(coordinate_structure const&) noexcept(
    std::is_nothrow_copy_constructible_v<row_container_type>) = default;
  coordinate_structure(coordinate_structure&&) noexcept(
    std::is_nothrow_move_constructible<row_container_type>::value) = default;

  constexpr auto operator=(coordinate_structure const&) noexcept(
    std::is_nothrow_copy_assignable<row_container_type>::value) -> coordinate_structure& = default;
  constexpr auto operator=(coordinate_structure&&) noexcept(
    std::is_nothrow_move_assignable<row_container_type>::value) -> coordinate_structure& = default;

  ~coordinate_structure() noexcept(std::is_nothrow_destructible<row_container_type>::value) =
    default;

  /**
   * Return a view of the coordinate structure. Structural views are sparsity-preserving
   * so while the structural elements can be updated in a non-const view, the sparsity
   * itself (number of nonzeros) cannot be changed.
   * @return coordinate structure view
   */
  view_type view()
  {
    if (this->get_nnz() > 0) {
      RAFT_LOG_WARN(
        "Cannot create coordinate_structure.view() because it has not been initialized "
        "(sparsity is 0)");
    }
    auto row_span = raft::span<row_type, is_device>(c_rows_.data(), this->get_nnz());
    auto col_span = raft::span<col_type, is_device>(c_cols_.data(), this->get_nnz());
    return view_type(row_span, col_span, this->get_n_rows(), this->get_n_cols());
  }

  /**
   * Return span containing underlying rows array
   * @return span containing underlying rows array
   */
  span<row_type, is_device> get_rows() override
  {
    return raft::span<row_type, is_device>(c_rows_.data(), this->get_n_rows());
  }

  /**
   * Return span containing underlying cols array
   * @return span containing underlying cols array
   */
  span<col_type, is_device> get_cols() override
  {
    return raft::span<col_type, is_device>(c_cols_.data(), this->get_n_cols());
  }

  /**
   * Change the sparsity of the current compressed structure. This will
   * resize the underlying data arrays.
   * @param nnz new sparsity
   */
  void initialize_sparsity(nnz_type nnz)
  {
    sparse_structure_type::initialize_sparsity(nnz);
    c_rows_.resize(nnz, handle_.get_stream());
    c_cols_.resize(nnz, handle_.get_stream());
  }

 protected:
  raft::device_resources const& handle_;
  row_container_policy_type cp_rows_;
  col_container_policy_type cp_cols_;
  row_container_type c_rows_;
  col_container_type c_cols_;
};

/**
 * A non-owning view of a sparse matrix, which includes a
 * structure component coupled with its elements/weights
 *
 * @tparam ElementType
 * @tparam sparse_structure
 */
template <typename ElementType, typename StructureType, bool is_device>
class sparse_matrix_view {
 public:
  using element_type        = ElementType;
  using structure_view_type = typename StructureType::view_type;

  sparse_matrix_view(raft::span<ElementType, is_device> element_span,
                     structure_view_type structure_view)
    : element_span_(element_span), structure_view_(structure_view)
  {
    // FIXME: Validate structure sizes match span size.
  }

  /**
   * Return a view of the structure underlying this matrix
   * @return
   */
  structure_view_type get_structure() { return structure_view_; }

  /**
   * Return a span of the nonzero elements of the matrix
   * @return span of the nonzero elements of the matrix
   */
  span<element_type, is_device> get_elements() { return element_span_; }

 protected:
  raft::span<element_type, is_device> element_span_;
  structure_view_type structure_view_;
};

/**
 * TODO: Need to support the following types of configurations:
 * 1. solid: immutable_sparse_matrix_view<const ElementType, const StructureType>
 *      - This is an immutable view type, nothing can change.
 * 2. liquid: sparse_matrix<ElementType, const StructureType>
 *      - sparse_matrix owning container w/ StructureType=immutable view?
 * 3. gas: sparse_matrix<ElementType, StructureType>
 *      - sparse_matrix owning container w/ StructureType owning container?
 */

/**
 * An owning container for a sparse matrix, which includes a
 * structure component coupled with its elements/weights
 * @tparam ElementType
 * @tparam sparse_structure
 * @tparam ContainerPolicy
 */
template <typename ElementType,
          typename StructureType,
          typename ViewType,
          bool is_device,
          template <typename T>
          typename ContainerPolicy>
class sparse_matrix {
 public:
  using view_type      = ViewType;
  using element_type   = typename view_type::element_type;
  using structure_type = StructureType;
  using row_type       = typename structure_type::row_type;
  using col_type       = typename structure_type::col_type;
  using nnz_type       = typename structure_type::nnz_type;

  using structure_view_type   = typename structure_type::view_type;
  using container_policy_type = ContainerPolicy<element_type>;
  using container_type        = typename container_policy_type::container_type;

  // Constructor that owns both the data and the structure
  sparse_matrix(raft::device_resources const& handle,
                row_type n_rows,
                col_type n_cols,
                nnz_type nnz = 0) noexcept(std::is_nothrow_default_constructible_v<container_type>)
    : handle_(handle),
      structure_{std::make_shared<structure_type>(handle, n_rows, n_cols, nnz)},
      cp_{rmm::cuda_stream_default},
      c_elements_{cp_.create(0)} {};

  // Constructor that owns the data but not the structure
  sparse_matrix(raft::device_resources const& handle,
                std::shared_ptr<structure_type>
                  structure) noexcept(std::is_nothrow_default_constructible_v<container_type>)
    : handle_(handle),
      structure_{structure},
      cp_{rmm::cuda_stream_default},
      c_elements_{cp_.create(structure.get().get_nnz())} {};

  constexpr sparse_matrix(sparse_matrix const&) noexcept(
    std::is_nothrow_copy_constructible_v<container_type>) = default;
  constexpr sparse_matrix(sparse_matrix&&) noexcept(
    std::is_nothrow_move_constructible<container_type>::value) = default;

  constexpr auto operator=(sparse_matrix const&) noexcept(
    std::is_nothrow_copy_assignable<container_type>::value) -> sparse_matrix& = default;
  constexpr auto operator=(sparse_matrix&&) noexcept(
    std::is_nothrow_move_assignable<container_type>::value) -> sparse_matrix& = default;

  ~sparse_matrix() noexcept(std::is_nothrow_destructible<container_type>::value) = default;

  void initialize_sparsity(nnz_type nnz) { c_elements_.resize(nnz, this->handle_.get_stream()); };

  raft::span<ElementType, is_device> get_elements()
  {
    return raft::span<ElementType, is_device>(c_elements_.data(), structure_view().get_nnz());
  }

  /**
   * Return a view of the structure underlying this matrix
   * @return
   */
  virtual structure_view_type structure_view() = 0;

  /**
   * Return a sparsity-preserving view of this sparse matrix
   * @return view of this sparse matrix
   */
  view_type view()
  {
    auto struct_view = structure_view();
    auto element_span =
      raft::span<ElementType, is_device>(c_elements_.data(), struct_view.get_nnz());
    return view_type(element_span, struct_view);
  }

 protected:
  raft::device_resources const& handle_;
  std::shared_ptr<structure_type> structure_;
  container_policy_type cp_;
  container_type c_elements_;
};

template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType,
          bool is_device>
class csr_matrix_view
  : public sparse_matrix_view<ElementType,
                              compressed_structure_view<IndptrType, IndicesType, NZType, is_device>,
                              is_device> {
 public:
  csr_matrix_view(
    raft::span<ElementType, is_device> element_span,
    compressed_structure_view<IndptrType, IndicesType, NZType, is_device> structure_view)
    : sparse_matrix_view<ElementType,
                         compressed_structure_view<IndptrType, IndicesType, NZType, is_device>,
                         is_device>(element_span, structure_view){};
};

template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType,
          bool is_device,
          template <typename T>
          typename ContainerPolicy,
          SparsityType type_enum  = OWNING,
          typename structure_type = std::conditional_t<
            type_enum == SparsityType::OWNING,
            compressed_structure<IndptrType, IndicesType, NZType, is_device, ContainerPolicy>,
            compressed_structure_view<IndptrType, IndicesType, NZType, is_device>>>
class csr_matrix
  : public sparse_matrix<ElementType,
                         structure_type,
                         csr_matrix_view<ElementType, IndptrType, IndicesType, NZType, is_device>,
                         is_device,
                         ContainerPolicy> {
 public:
  using structure_view_type = typename structure_type::view_type;

  using sparse_matrix_type =
    sparse_matrix<ElementType,
                  structure_type,
                  csr_matrix_view<ElementType, IndptrType, IndicesType, NZType, is_device>,
                  is_device,
                  ContainerPolicy>;
  using container_type = typename ContainerPolicy<ElementType>::container_type;

  template <typename = typename std::enable_if<type_enum == SparsityType::OWNING>>
  csr_matrix(raft::device_resources const& handle,
             IndptrType n_rows,
             IndicesType n_cols,
             NZType nnz = 0) noexcept(std::is_nothrow_default_constructible_v<container_type>)
    : sparse_matrix_type(handle, n_rows, n_cols, nnz){};

  // Constructor that owns the data but not the structure

  template <typename = typename std::enable_if<type_enum == SparsityType::PRESERVING>>
  csr_matrix(raft::device_resources const& handle,
             std::shared_ptr<structure_type>
               structure) noexcept(std::is_nothrow_default_constructible_v<container_type>)
    : sparse_matrix_type(handle, structure){};

  /**
   * Initialize the sparsity on this instance if it was not known upon construction
   * Please note this will resize the underlying memory buffers
   * @param nnz new sparsity to initialize.
   */
  template <typename = std::enable_if_t<type_enum == SparsityType::OWNING>>
  void initialize_sparsity(NZType nnz)
  {
    sparse_matrix_type::initialize_sparsity(nnz);
    this->structure_.get()->initialize_sparsity(nnz);
  }

  /**
   * Return a view of the structure underlying this matrix
   * @return
   */
  structure_view_type structure_view() { return this->structure_.get()->view(); }
};

template <typename ElementType, typename RowType, typename ColType, typename NZType, bool is_device>
class coo_matrix_view
  : public sparse_matrix_view<ElementType,
                              coordinate_structure_view<RowType, ColType, NZType, is_device>,
                              is_device> {
 public:
  coo_matrix_view(raft::span<ElementType, is_device> element_span,
                  coordinate_structure_view<RowType, ColType, NZType, is_device> structure_view)
    : sparse_matrix_view<ElementType,
                         coordinate_structure_view<RowType, ColType, NZType, is_device>,
                         is_device>(element_span, structure_view)
  {
  }
};

template <typename ElementType,
          typename RowType,
          typename ColType,
          typename NZType,
          bool is_device,
          template <typename T>
          typename ContainerPolicy,
          SparsityType type_enum  = SparsityType::OWNING,
          typename structure_type = std::conditional_t<
            type_enum == SparsityType::OWNING,
            coordinate_structure<RowType, ColType, NZType, is_device, ContainerPolicy>,
            coordinate_structure_view<RowType, ColType, NZType, is_device>>>
class coo_matrix
  : public sparse_matrix<ElementType,
                         structure_type,
                         coo_matrix_view<ElementType, RowType, ColType, NZType, is_device>,
                         is_device,
                         ContainerPolicy> {
 public:
  using structure_view_type = typename structure_type::view_type;
  using container_type      = typename ContainerPolicy<ElementType>::container_type;
  using sparse_matrix_type =
    sparse_matrix<ElementType,
                  structure_type,
                  coo_matrix_view<ElementType, RowType, ColType, NZType, is_device>,
                  is_device,
                  ContainerPolicy>;
  template <typename = typename std::enable_if<type_enum == SparsityType::OWNING>>
  coo_matrix(raft::device_resources const& handle,
             RowType n_rows,
             ColType n_cols,
             NZType nnz = 0) noexcept(std::is_nothrow_default_constructible_v<container_type>)
    : sparse_matrix_type(handle, n_rows, n_cols, nnz){};

  // Constructor that owns the data but not the structure
  template <typename = typename std::enable_if<type_enum == SparsityType::PRESERVING>>
  coo_matrix(raft::device_resources const& handle,
             std::shared_ptr<structure_type>
               structure) noexcept(std::is_nothrow_default_constructible_v<container_type>)
    : sparse_matrix_type(handle, structure){};

  /**
   * Return a view of the structure underlying this matrix
   * @return
   */
  structure_view_type structure_view() { return this->structure_.get()->view(); }

  /**
   * Initialize the sparsity on this instance if it was not known upon construction
   * Please note this will resize the underlying memory buffers
   * @param nnz new sparsity to initialize.
   */
  template <typename = std::enable_if_t<type_enum == SparsityType::OWNING>>
  void initialize_sparsity(NZType nnz)
  {
    sparse_matrix_type::initialize_sparsity(nnz);
    this->structure_.get()->initialize_sparsity(nnz);
  }
};

// template <typename ElementType,
//          typename IndptrType,
//          typename IndicesType,
//          typename NZType,
//          bool is_device,
//          template <typename T>
//          typename ContainerPolicy>
// class sparsity_owning_csr_matrix
//  : public csr_matrix<
//      ElementType,
//      IndptrType,
//      IndicesType,
//      NZType,
//      compressed_structure<IndptrType, IndicesType, NZType, is_device, ContainerPolicy>,
//      is_device,
//      ContainerPolicy> {
// public:
//  using csr_matrix_type =
//    csr_matrix<ElementType,
//               IndptrType,
//               IndicesType,
//               NZType,
//               compressed_structure<IndptrType, IndicesType, NZType, is_device, ContainerPolicy>,
//               is_device,
//               ContainerPolicy>;
//  using container_type = typename ContainerPolicy<ElementType>::container_type;
//  using structure_view_type =
//    typename compressed_structure<IndptrType, IndicesType, NZType, is_device, ContainerPolicy>::
//      view_type;
//
//  // Constructor that owns both data and structure
//  sparsity_owning_csr_matrix(
//    raft::device_resources const& handle,
//    IndptrType n_rows,
//    IndicesType n_cols,
//    NZType nnz = 0) noexcept(std::is_nothrow_default_constructible_v<container_type>)
//    : csr_matrix_type(handle, n_rows, n_cols, nnz){};
//
//  /**
//   * Initialize the sparsity on this instance if it was not known upon construction
//   * Please note this will resize the underlying memory buffers
//   * @param nnz new sparsity to initialize.
//   */
//  void initialize_sparsity(NZType nnz)
//  {
//    csr_matrix_type::initialize_sparsity(nnz);
//    this->structure_.get()->initialize_sparsity(nnz);
//  }
//
//  /**
//   * Return a view of the structure underlying this matrix
//   * @return
//   */
//  structure_view_type structure_view() { return this->structure_.get()->view(); }
//};
//
// template <typename ElementType,
//          typename IndptrType,
//          typename IndicesType,
//          typename NZType,
//          bool is_device,
//          template <typename T>
//          typename ContainerPolicy>
// class sparsity_preserving_csr_matrix
//  : public csr_matrix<ElementType,
//                      IndptrType,
//                      IndicesType,
//                      NZType,
//                      compressed_structure_view<IndptrType, IndicesType, NZType, is_device>,
//                      is_device,
//                      ContainerPolicy> {
// public:
//  using csr_matrix_type =
//    csr_matrix<ElementType,
//               IndptrType,
//               IndicesType,
//               NZType,
//               compressed_structure_view<IndptrType, IndicesType, NZType, is_device>,
//               is_device,
//               ContainerPolicy>;
//  using container_type      = typename ContainerPolicy<ElementType>::container_type;
//  using structure_view_type = compressed_structure_view<IndptrType, IndicesType, NZType,
//  is_device>;
//
//  // Constructor that owns both data and structure
//  sparsity_preserving_csr_matrix(
//    raft::device_resources const& handle,
//    std::shared_ptr<structure_view_type>
//      structure) noexcept(std::is_nothrow_default_constructible_v<container_type>)
//    : csr_matrix_type(handle, structure){};
//
//  /**
//   * Return a view of the structure underlying this matrix
//   * @return
//   */
//  structure_view_type structure_view() { return (*this->structure_.get()); }
//
//  /**
//   * Initialize the sparsity on this instance if it was not known upon construction
//   * Note: Sparsity can not be adjusted in sparsity-preserving matrices
//   */
//  void initialize_sparsity(NZType)
//  {
//    RAFT_FAIL("The sparsity of structure-preserving sparse formats cannot be changed");
//  }
//};
//
// template <typename ElementType,
//          typename RowType,
//          typename ColType,
//          typename NZType,
//          bool is_device,
//          template <typename T>
//          typename ContainerPolicy>
// class sparsity_owning_coo_matrix
//  : public coo_matrix<ElementType,
//                      RowType,
//                      ColType,
//                      NZType,
//                      coordinate_structure<RowType, ColType, NZType, is_device, ContainerPolicy>,
//                      is_device,
//                      ContainerPolicy> {
// public:
//  using coo_matrix_type =
//    coo_matrix<ElementType,
//               RowType,
//               ColType,
//               NZType,
//               coordinate_structure<RowType, ColType, NZType, is_device, ContainerPolicy>,
//               is_device,
//               ContainerPolicy>;
//  using container_type = typename ContainerPolicy<ElementType>::container_type;
//  using structure_view_type =
//    typename coordinate_structure<RowType, ColType, NZType, is_device,
//    ContainerPolicy>::view_type;
//  sparsity_owning_coo_matrix(
//    raft::device_resources const& handle,
//    RowType n_rows,
//    ColType n_cols,
//    NZType nnz = 0) noexcept(std::is_nothrow_default_constructible_v<container_type>)
//    : coo_matrix_type{handle, n_rows, n_cols, nnz} {};
//
//};
//
// template <typename ElementType,
//          typename RowType,
//          typename ColType,
//          typename NZType,
//          bool is_device,
//          template <typename T>
//          typename ContainerPolicy>
// class sparsity_preserving_coo_matrix
//  : public coo_matrix<ElementType,
//                      RowType,
//                      ColType,
//                      NZType,
//                      compressed_structure_view<RowType, ColType, NZType, is_device>,
//                      is_device,
//                      ContainerPolicy> {
// public:
//  using coo_matrix_type     = coo_matrix<ElementType,
//                                     RowType,
//                                     ColType,
//                                     NZType,
//                                     compressed_structure_view<RowType, ColType, NZType,
//                                     is_device>, is_device, ContainerPolicy>;
//  using container_type      = typename ContainerPolicy<ElementType>::container_type;
//  using structure_view_type = compressed_structure_view<RowType, ColType, NZType, is_device>;
//
//  // Constructor that owns both data and structure
//  sparsity_preserving_coo_matrix(
//    raft::device_resources const& handle,
//    std::shared_ptr<structure_view_type>
//      structure) noexcept(std::is_nothrow_default_constructible_v<container_type>)
//    : coo_matrix_type(handle, structure){};
//
//  /**
//   * Initialize the sparsity on this instance if it was not known upon construction
//   * Note: Sparsity can not be adjusted in sparsity-preserving matrices
//   */
//  void initialize_sparsity(NZType)
//  {
//    RAFT_FAIL("The sparsity of structure-preserving sparse formats cannot be changed");
//  }
//
//  /**
//   * Return a view of the structure underlying this matrix
//   * @return
//   */
//  structure_view_type structure_view() override { return (*this->structure_.get()); }
//};

}  // namespace raft