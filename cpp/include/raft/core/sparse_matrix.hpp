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

#include <raft/core/span.hpp>

namespace raft {

enum MutabilityClass { SOLID, LIQUID, GAS };

// TODO: Define container_policy interface

template <typename RowType, typename ColType, typename NZType, int is_device>
class sparse_structure {
 public:
  using row_type = RowType;
  using col_type = ColType;
  using nnz_type = NZType;

  sparse_structure(row_type n_rows, col_type n_cols, nnz_type nnz)
    : n_rows_(n_rows), n_cols_(n_cols), nnz_(nnz){};

  sparse_structure(row_type n_rows, col_type n_cols) : n_rows_(n_rows), n_cols_(n_cols), nnz_(0) {}
  nnz_type get_nnz() { return nnz_; }

  row_type get_n_rows() { return n_rows_; }

  col_type get_n_cols() { return n_cols_; }

  virtual void resize(nnz_type nnz) = 0;

 protected:
  row_type n_rows_;
  col_type n_cols_;
  nnz_type nnz_;
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
  : public sparse_structure<IndptrType, IndicesType, NZType, is_device> {
 public:
  using view_type = compressed_structure_view<IndptrType, IndicesType, NZType, is_device>;
  using indptr_type =
    typename sparse_structure<IndptrType, IndicesType, NZType, is_device>::row_type;
  using indices_type =
    typename sparse_structure<IndptrType, IndicesType, NZType, is_device>::col_type;

  compressed_structure_view(span<indptr_type, is_device> indptr,
                            span<indices_type, is_device> indices,
                            indices_type n_cols)
    : sparse_structure<IndptrType, IndicesType, NZType, is_device>(
        indptr.size() - 1, n_cols, indices.size()),
      indptr_(indptr),
      indices_(indices)
  {
  }

  span<indptr_type, is_device> get_indptr() { return indptr_; }
  span<indices_type, is_device> get_indices() { return indices_; }

  view_type view() { return view_type(indptr_, indices_, this->n_rows_, this->n_cols_); }

  void resize(NZType nnz)
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
class compressed_structure : public sparse_structure<IndptrType, IndicesType, NZType, is_device> {
 public:
  using view_type = compressed_structure_view<IndptrType, IndicesType, NZType, is_device>;
  using indptr_container_policy_type  = ContainerPolicy<IndptrType>;
  using indices_container_policy_type = ContainerPolicy<IndicesType>;
  using indptr_container_type         = typename indptr_container_policy_type::container_type;
  using indices_container_type        = typename indices_container_policy_type::container_type;

  constexpr compressed_structure(
    raft::device_resources const& handle,
    IndptrType n_rows,
    IndicesType n_cols,
    NZType nnz = 0) noexcept(std::is_nothrow_default_constructible_v<indptr_container_type>)
    : handle_{handle},
      cp_indptr_{handle.get_stream()},
      cp_indices_{handle.get_stream()},
      c_indptr_{cp_indptr_.create(0)},
      c_indices_{cp_indices_.create(0), n_rows_(n_rows), n_cols_(n_cols), nnz_(nnz)} {};

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

  ~compressed_structure() noexcept(std::is_nothrow_destructible<indptr_container_type>::value) =
    default;
  view_type view()
  {
    RAFT_EXPECTS(this->get_nnz() > 0,
                 "Cannot create compressed_structure.view() because it has not been initialized "
                 "(sparsity is 0)");
    auto indptr_span  = raft::span<IndptrType, is_device>(c_indptr_.data(), this->get_nnz());
    auto indices_span = raft::span<IndicesType, is_device>(c_indices_.data(), this->get_nnz());
    return view_type(indptr_span, indices_span, this->get_n_cols());
  }

  void resize(NZType nnz)
  {
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
class coordinate_structure_view : public sparse_structure<RowType, ColType, NZType, is_device> {
 public:
  using view_type = coordinate_structure_view<RowType, ColType, NZType, is_device>;
  using row_type  = typename sparse_structure<RowType, ColType, NZType, is_device>::row_type;
  using col_type  = typename sparse_structure<RowType, ColType, NZType, is_device>::col_type;
  using nnz_type  = typename sparse_structure<RowType, ColType, NZType, is_device>::nnz_type;

  coordinate_structure_view(span<row_type, is_device> rows,
                            span<col_type, is_device> cols,
                            row_type n_rows,
                            col_type n_cols)
    : sparse_structure<RowType, ColType, NZType, is_device>(n_rows, n_cols, rows.size()),
      rows_{rows},
      cols_{cols}
  {
  }

  view_type view() { return view_type(rows_, cols_, this->get_n_rows(), this->get_n_cols()); }

  void resize(nnz_type nnz)
  {
    RAFT_FAIL("The sparsity of structure-preserving sparse formats cannot be changed");
  }

  span<row_type, is_device> get_rows() { return rows_; }
  span<col_type, is_device> get_cols() { return cols_; }

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
class coordinate_structure : public sparse_structure<RowType, ColType, NZType, is_device> {
 public:
  using row_type = typename sparse_structure<RowType, ColType, NZType, is_device>::row_type;
  using col_type = typename sparse_structure<RowType, ColType, NZType, is_device>::col_type;
  using nnz_type = typename sparse_structure<RowType, ColType, NZType, is_device>::nnz_type;

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
    : sparse_structure<RowType, ColType, NZType, is_device>(n_rows, n_cols, nnz),
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
  view_type view()
  {
    RAFT_EXPECTS(this->get_nnz() > 0,
                 "Cannot create coordinate_structure.view() because it has not been initialized "
                 "(sparsity is 0)");
    auto row_span = raft::span<row_type, is_device>(c_rows_.data(), this->get_nnz());
    auto col_span = raft::span<col_type, is_device>(c_cols_.data(), this->get_nnz());
    return view_type(row_span, col_span, this->get_n_rows(), this->get_n_cols());
  }

  void resize(nnz_type nnz)
  {
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

  structure_view_type get_structure() { return structure_view_; }

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

  void resize(nnz_type nnz) { c_elements_.resize(nnz); };

  virtual structure_view_type structure_view() = 0;
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
          typename ContainerPolicy>
class sparsity_owning_csr_matrix
  : public sparse_matrix<
      ElementType,
      compressed_structure<IndptrType, IndicesType, NZType, is_device, ContainerPolicy>,
      csr_matrix_view<ElementType, IndptrType, IndicesType, NZType, is_device>,
      is_device,
      ContainerPolicy> {
 public:
  using container_type = typename ContainerPolicy<ElementType>::container_type;
  using structure_view_type =
    typename compressed_structure<IndptrType, IndicesType, NZType, is_device, ContainerPolicy>::
      view_type;

  // Constructor that owns both data and structure
  sparsity_owning_csr_matrix(
    raft::device_resources const& handle,
    IndptrType n_rows,
    IndicesType n_cols,
    NZType nnz = 0) noexcept(std::is_nothrow_default_constructible_v<container_type>)
    : sparse_matrix<
        ElementType,
        compressed_structure<IndptrType, IndicesType, NZType, is_device, ContainerPolicy>,
        csr_matrix_view<ElementType, IndptrType, IndicesType, NZType, is_device>,
        is_device,
        ContainerPolicy>(handle, n_rows, n_cols, nnz){};

  void resize(NZType nnz)
  {
    this->c_elements_.resize(nnz);
    this->structure_.resize(nnz);
  }

  structure_view_type structure_view() { return this->structure_.get()->view(); }
};

template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType,
          bool is_device,
          template <typename T>
          typename ContainerPolicy>
class sparsity_preserving_csr_matrix
  : public sparse_matrix<ElementType,
                         compressed_structure_view<IndptrType, IndicesType, NZType, is_device>,
                         csr_matrix_view<ElementType, IndptrType, IndicesType, NZType, is_device>,
                         is_device,
                         ContainerPolicy> {
 public:
  using container_type      = typename ContainerPolicy<ElementType>::container_type;
  using structure_view_type = compressed_structure_view<IndptrType, IndicesType, NZType, is_device>;

  // Constructor that owns both data and structure
  sparsity_preserving_csr_matrix(
    raft::device_resources const& handle,
    std::shared_ptr<structure_view_type>
      structure) noexcept(std::is_nothrow_default_constructible_v<container_type>)
    : sparse_matrix<
        ElementType,
        compressed_structure<IndptrType, IndicesType, NZType, is_device, ContainerPolicy>,
        csr_matrix_view<ElementType, IndptrType, IndicesType, NZType, is_device>,
        is_device,
        ContainerPolicy>(handle, structure){};

  structure_view_type structure_view() { return (*this->structure_.get()); }

  void resize(NZType)
  {
    RAFT_FAIL("The sparsity of structure-preserving sparse formats cannot be changed");
  }
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
    // FIXME: Validate structure sizes match span size.
  }
};

template <typename ElementType,
          typename RowType,
          typename ColType,
          typename NZType,
          bool is_device,
          template <typename T>
          typename ContainerPolicy>
class sparsity_owning_coo_matrix
  : public sparse_matrix<ElementType,
                         coordinate_structure<RowType, ColType, NZType, is_device, ContainerPolicy>,
                         coo_matrix_view<ElementType, RowType, ColType, NZType, is_device>,
                         is_device,
                         ContainerPolicy> {
 public:
  using container_type = typename ContainerPolicy<ElementType>::container_type;
  using structure_view_type =
    typename coordinate_structure<RowType, ColType, NZType, is_device, ContainerPolicy>::view_type;
  sparsity_owning_coo_matrix(
    raft::device_resources const& handle,
    RowType n_rows,
    ColType n_cols,
    NZType nnz = 0) noexcept(std::is_nothrow_default_constructible_v<container_type>)
    : sparse_matrix<ElementType,
                    coordinate_structure<RowType, ColType, NZType, is_device, ContainerPolicy>,
                    coo_matrix_view<ElementType, RowType, ColType, NZType, is_device>,
                    is_device,
                    ContainerPolicy>{handle, n_rows, n_cols, nnz} {};

  //    sparsity_owning_csr_matrix(
  //            raft::device_resources const& handle,
  //            IndptrType n_rows,
  //            IndicesType n_cols,
  //            NZType nnz=0) noexcept(std::is_nothrow_default_constructible_v<container_type>)
  //            : sparse_matrix<ElementType,
  //                    compressed_structure<IndptrType, IndicesType, NZType, is_device>,
  //                    csr_matrix_view<ElementType, IndptrType, IndicesType, NZType, is_device>,
  //                    is_device,
  //                    ContainerPolicy>(handle, n_rows, n_cols, nnz){};
  //

  structure_view_type structure_view() { return this->structure_.get()->view(); }

  void resize(NZType nnz)
  {
    this->c_elements_.resize(nnz);
    this->structure_.resize(nnz);
  }
};

template <typename ElementType,
          typename RowType,
          typename ColType,
          typename NZType,
          bool is_device,
          template <typename T>
          typename ContainerPolicy>
class sparsity_preserving_coo_matrix
  : public sparse_matrix<ElementType,
                         compressed_structure_view<RowType, ColType, NZType, is_device>,
                         coo_matrix_view<ElementType, RowType, ColType, NZType, is_device>,
                         is_device,
                         ContainerPolicy> {
 public:
  using container_type      = typename ContainerPolicy<ElementType>::container_type;
  using structure_view_type = compressed_structure_view<RowType, ColType, NZType, is_device>;

  // Constructor that owns both data and structure
  sparsity_preserving_coo_matrix(
    raft::device_resources const& handle,
    std::shared_ptr<structure_view_type>
      structure) noexcept(std::is_nothrow_default_constructible_v<container_type>)
    : sparse_matrix<ElementType,
                    coordinate_structure_view<RowType, ColType, NZType, is_device>,
                    coo_matrix_view<ElementType, RowType, ColType, NZType, is_device>,
                    is_device,
                    ContainerPolicy>(handle, structure){};

  void resize(NZType)
  {
    RAFT_FAIL("The sparsity of structure-preserving sparse formats cannot be changed");
  }

  structure_view_type structure_view() override { return (*this->structure_.get()); }
};

}  // namespace raft