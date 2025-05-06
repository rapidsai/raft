/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <raft/sparse/detail/coo.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

namespace raft {
namespace sparse {
namespace detail {

/** @brief A Container object for sparse coordinate stored on the host.
 *
 * This container simplifies code by bundling the three arrays (rows, cols, vals)
 * needed for COO format. It also allows for lazy allocation, where the object
 * can be created and passed as an output argument, with the actual memory
 * allocation happening later within a sparse primitive function.
 *
 * @tparam T: the type of the value array.
 * @tparam Index_Type: the type of index array
 * @tparam nnz_type: the type for the count of non-zero elements
 */
template <typename T, typename Index_Type = int, typename nnz_type = uint64_t>
class host_COO {
 protected:
  std::vector<Index_Type> rows_arr;
  std::vector<Index_Type> cols_arr;
  std::vector<T> vals_arr;

 public:
  using value_t = T;
  using index_t = Index_Type;
  using nnz_t   = nnz_type;

  nnz_type nnz;
  Index_Type n_rows;
  Index_Type n_cols;

  /**
   * @brief Default constructor, initializes empty vectors and zero sizes.
   */
  host_COO() : nnz(0), n_rows(0), n_cols(0) {}

  /**
   * @brief Constructor taking existing vectors (moves them).
   * @param rows: coo rows vector
   * @param cols: coo cols vector
   * @param vals: coo vals vector
   * @param nnz: size of the rows/cols/vals vectors
   * @param n_rows: number of rows
   * @param n_cols: number of cols
   */
  host_COO(std::vector<Index_Type>&& rows,
           std::vector<Index_Type>&& cols,
           std::vector<T>&& vals,
           Index_Type n_rows,
           Index_Type n_cols)
    : rows_arr(std::move(rows)),
      cols_arr(std::move(cols)),
      vals_arr(std::move(vals)),
      nnz(rows.size()),
      n_rows(n_rows),
      n_cols(n_cols)
  {
    ASSERT(validate_mem(), "rows, cols and vals arrays should have similar number of elements");
  }

  /**
   * @brief Constructor allocating vectors of a specific size.
   * @param nnz: size of the rows/cols/vals arrays
   * @param n_rows: number of rows
   * @param n_cols: number of cols
   * @param init: initialize arrays with zeros
   */
  host_COO(nnz_type nnz, Index_Type n_rows = 0, Index_Type n_cols = 0, bool init = true)
    : rows_arr(nnz), cols_arr(nnz), vals_arr(nnz), nnz(nnz), n_rows(n_rows), n_cols(n_cols)
  {
    if (init) { init_arrays(); }
  }

  void init(COO<T, Index_Type, nnz_type>& other, cudaStream_t stream)
  {
    nnz    = other.nnz;
    n_rows = other.n_rows;
    n_cols = other.n_cols;

    rows_arr.resize(nnz);
    cols_arr.resize(nnz);
    vals_arr.resize(nnz);

    raft::copy(rows_arr.data(), other.rows(), nnz, stream);
    raft::copy(cols_arr.data(), other.cols(), nnz, stream);
    raft::copy(vals_arr.data(), other.vals(), nnz, stream);
  }

  void init_arrays()
  {
    std::fill(rows_arr.begin(), rows_arr.end(), 0);
    std::fill(cols_arr.begin(), cols_arr.end(), 0);
    std::fill(vals_arr.begin(), vals_arr.end(), 0);
  }

  ~host_COO() {}

  /**
   * @brief Size should be > 0, with the number of rows
   * and cols being > 0.
   */
  bool validate_size() const
  {
    if (nnz <= 0 || n_rows <= 0 || n_cols <= 0) return false;
    return true;
  }

  /**
   * @brief If the underlying arrays have not been properly sized,
   * return false. Otherwise true.
   */
  bool validate_mem() const
  {
    // Check if vectors have the expected size based on nnz
    if (rows_arr.size() != nnz || cols_arr.size() != nnz || vals_arr.size() != nnz) {
      return false;
    }
    return true;
  }

  /*
   * @brief Returns a pointer to the underlying rows array data.
   */
  Index_Type* rows() { return rows_arr.data(); }
  const Index_Type* rows() const { return rows_arr.data(); }

  /**
   * @brief Returns a pointer to the underlying cols array data.
   */
  Index_Type* cols() { return cols_arr.data(); }
  const Index_Type* cols() const { return cols_arr.data(); }

  /**
   * @brief Returns a pointer to the underlying vals array data.
   */
  T* vals() { return vals_arr.data(); }
  const T* vals() const { return vals_arr.data(); }

  /**
   * @brief Send human-readable state information to output stream
   */
  friend std::ostream& operator<<(std::ostream& out, const host_COO<T, Index_Type, nnz_type>& c)
  {
    if (c.validate_size() && c.validate_mem()) {
      out << "rows: [";
      for (size_t i = 0; i < c.nnz; ++i)
        out << c.rows_arr[i] << (i == c.nnz - 1 ? "" : ", ");
      out << "]" << std::endl;

      out << "cols: [";
      for (size_t i = 0; i < c.nnz; ++i)
        out << c.cols_arr[i] << (i == c.nnz - 1 ? "" : ", ");
      out << "]" << std::endl;

      out << "vals: [";
      for (size_t i = 0; i < c.nnz; ++i)
        out << c.vals_arr[i] << (i == c.nnz - 1 ? "" : ", ");
      out << "]" << std::endl;

      out << "nnz=" << c.nnz << std::endl;
      out << "n_rows=" << c.n_rows << std::endl;
      out << "n_cols=" << c.n_cols << std::endl;
    } else {
      out << "Cannot print host_COO object: Uninitialized or invalid." << std::endl;
    }
    return out;
  }

  /**
   * @brief Set the number of rows and cols
   * @param n_rows: number of rows
   * @param n_cols: number of columns
   */
  void setSize(Index_Type n_rows, Index_Type n_cols)
  {
    this->n_rows = n_rows;
    this->n_cols = n_cols;
  }

  /**
   * @brief Set the number of rows and cols for a square dense matrix
   * @param n: number of rows and cols
   */
  void setSize(Index_Type n)
  {
    this->n_rows = n;
    this->n_cols = n;
  }

  /**
   * @brief Allocate or resize the underlying arrays
   * @param new_nnz: new size of underlying row/col/val arrays
   * @param init: should new values be initialized to 0
   */
  void allocate(nnz_type new_nnz, bool init = true)
  {
    this->allocate(new_nnz, this->n_rows, this->n_cols, init);
  }

  /**
   * @brief Allocate or resize the underlying arrays
   * @param new_nnz: new size of the underlying row/col/val arrays
   * @param size: the number of rows/cols in a square dense matrix
   * @param init: should new values be initialized to 0
   */
  void allocate(nnz_type new_nnz, Index_Type size, bool init = true)
  {
    this->allocate(new_nnz, size, size, init);
  }

  /**
   * @brief Allocate or resize the underlying arrays
   * @param new_nnz: new size of the underlying row/col/val arrays
   * @param n_rows: number of rows
   * @param n_cols: number of columns
   * @param init: should new values be initialized to 0
   */
  void allocate(nnz_type new_nnz, Index_Type n_rows, Index_Type n_cols, bool init = true)
  {
    this->n_rows = n_rows;
    this->n_cols = n_cols;
    this->nnz    = new_nnz;

    this->rows_arr.resize(new_nnz);
    this->cols_arr.resize(new_nnz);
    this->vals_arr.resize(new_nnz);

    if (init) { init_arrays(); }
  }
};

};  // namespace detail
};  // namespace sparse
};  // namespace raft
