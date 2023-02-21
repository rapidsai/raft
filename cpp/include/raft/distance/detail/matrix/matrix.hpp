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

#include <raft/core/error.hpp>

namespace raft::distance::matrix::detail {

template <typename math_t>
class DenseMatrix;
template <typename math_t>
class CsrMatrix;

/*
 * Thin matrix wrapper to allow single API for different matrix representations
 */
template <typename math_t>
class Matrix {
 public:
  Matrix(int rows, int cols) : n_rows(rows), n_cols(cols){};
  virtual bool isDense() const = 0;
  virtual ~Matrix(){};

  DenseMatrix<math_t>* asDense()
  {
    DenseMatrix<math_t>* cast = dynamic_cast<DenseMatrix<math_t>*>(this);
    ASSERT(cast != nullptr, "Invalid cast! Please check for isDense() before casting.");
    return cast;
  };

  CsrMatrix<math_t>* asCsr()
  {
    CsrMatrix<math_t>* cast = dynamic_cast<CsrMatrix<math_t>*>(this);
    ASSERT(cast != nullptr, "Invalid cast! Please check for isDense() before casting.");
    return cast;
  };

  const DenseMatrix<math_t>* asDense() const
  {
    const DenseMatrix<math_t>* cast = dynamic_cast<const DenseMatrix<math_t>*>(this);
    ASSERT(cast != nullptr, "Invalid cast! Please check for isDense() before casting.");
    return cast;
  };

  const CsrMatrix<math_t>* asCsr() const
  {
    const CsrMatrix<math_t>* cast = dynamic_cast<const CsrMatrix<math_t>*>(this);
    ASSERT(cast != nullptr, "Invalid cast! Please check for isDense() before casting.");
    return cast;
  };

  int n_rows;
  int n_cols;
};

template <typename math_t>
class DenseMatrix : public Matrix<math_t> {
 public:
  DenseMatrix(math_t* data, int rows, int cols, bool row_major = false, int ld_in = 0)
    : Matrix<math_t>(rows, cols), data(data), is_row_major(row_major), ld(ld_in)
  {
    if (ld <= 0) ld = is_row_major ? cols : rows;
  }
  bool isDense() const { return true; }
  math_t* data;
  bool is_row_major;
  int ld;
};

template <typename math_t>
class CsrMatrix : public Matrix<math_t> {
 public:
  CsrMatrix(int* indptr, int* indices, math_t* data, int nnz, int rows, int cols)
    : Matrix<math_t>(rows, cols), indptr(indptr), indices(indices), data(data), nnz(nnz)
  {
  }
  bool isDense() const { return false; }

  int nnz;
  int* indptr;
  int* indices;
  math_t* data;
};

}  // namespace raft::distance::matrix::detail