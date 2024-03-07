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

#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance.cuh>
#include <raft/distance/distance_types.hpp>
// #include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/sparse/distance/distance.cuh>
#include <raft/sparse/linalg/spmm.hpp>

namespace raft::distance::kernels::detail {

template <typename math_t>
using dense_input_matrix_view_t = raft::device_matrix_view<const math_t, int, layout_stride>;
template <typename math_t>
using dense_output_matrix_view_t = raft::device_matrix_view<math_t, int, layout_stride>;
template <typename math_t>
using csr_input_matrix_view_t = raft::device_csr_matrix_view<const math_t, int, int, int>;

/**
 * Base class for general Gram matrices
 * A Gram matrix is the Hermitian matrix of inner probucts G_ik = <x_i, x_k>
 * Here, the  inner product is evaluated for all elements from vectors sets X1,
 * and X2.
 *
 * To be more precise, on exit the output buffer will store:
 * - if is_row_major == true: out[j+k*n1] = <x1_j, x2_k>,
 * - if is_row_major == false: out[j*n2 + k] = <x1_j, x2_k>,
 * where x1_j is the j-th vector from the x1 set and x2_k is the k-th vector
 * from the x2 set.
 */
template <typename math_t>
class GramMatrixBase {
 protected:
  cublasHandle_t cublas_handle;
  bool legacy_interface;

 public:
  GramMatrixBase() : legacy_interface(false){};
  [[deprecated]] GramMatrixBase(cublasHandle_t cublas_handle)
    : cublas_handle(cublas_handle), legacy_interface(true){};

  virtual ~GramMatrixBase(){};

  /** Convenience function to evaluate the Gram matrix for two vector sets.
   *  Vector sets are provided in Matrix format
   *
   * @param [in] handle raft handle
   * @param [in] x1 dense device matrix view, size [n1*n_cols]
   * @param [in] x2 dense device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 optional L2-norm of x1's rows for computation within RBF.
   * @param norm_x2 optional L2-norm of x2's rows for computation within RBF.
   */
  void operator()(raft::resources const& handle,
                  dense_input_matrix_view_t<math_t> x1,
                  dense_input_matrix_view_t<math_t> x2,
                  dense_output_matrix_view_t<math_t> out,
                  math_t* norm_x1 = nullptr,
                  math_t* norm_x2 = nullptr)
  {
    evaluate(handle, x1, x2, out, norm_x1, norm_x2);
  }

  /** Convenience function to evaluate the Gram matrix for two vector sets.
   *  Vector sets are provided in Matrix format
   *
   * @param [in] handle raft handle
   * @param [in] x1 csr device matrix view, size [n1*n_cols]
   * @param [in] x2 dense device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 optional L2-norm of x1's rows for computation within RBF.
   * @param norm_x2 optional L2-norm of x2's rows for computation within RBF.
   */
  void operator()(raft::resources const& handle,
                  csr_input_matrix_view_t<math_t> x1,
                  dense_input_matrix_view_t<math_t> x2,
                  dense_output_matrix_view_t<math_t> out,
                  math_t* norm_x1 = nullptr,
                  math_t* norm_x2 = nullptr)
  {
    evaluate(handle, x1, x2, out, norm_x1, norm_x2);
  }

  /** Convenience function to evaluate the Gram matrix for two vector sets.
   *  Vector sets are provided in Matrix format
   *
   * @param [in] handle raft handle
   * @param [in] x1 csr device matrix view, size [n1*n_cols]
   * @param [in] x2 csr device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 optional L2-norm of x1's rows for computation within RBF.
   * @param norm_x2 optional L2-norm of x2's rows for computation within RBF.
   */
  void operator()(raft::resources const& handle,
                  csr_input_matrix_view_t<math_t> x1,
                  csr_input_matrix_view_t<math_t> x2,
                  dense_output_matrix_view_t<math_t> out,
                  math_t* norm_x1 = nullptr,
                  math_t* norm_x2 = nullptr)
  {
    evaluate(handle, x1, x2, out, norm_x1, norm_x2);
  }

  // unfortunately, 'evaluate' cannot be templatized as it needs to be virtual

  /** Evaluate the Gram matrix for two vector sets using simple dot product.
   *
   * @param [in] handle raft handle
   * @param [in] x1 dense device matrix view, size [n1*n_cols]
   * @param [in] x2 dense device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 unused.
   * @param norm_x2 unused.
   */
  virtual void evaluate(raft::resources const& handle,
                        dense_input_matrix_view_t<math_t> x1,
                        dense_input_matrix_view_t<math_t> x2,
                        dense_output_matrix_view_t<math_t> out,
                        math_t* norm_x1,
                        math_t* norm_x2)
  {
    linear(handle, x1, x2, out);
  }
  /** Evaluate the Gram matrix for two vector sets using simple dot product.
   *
   * @param [in] handle raft handle
   * @param [in] x1 csr device matrix view, size [n1*n_cols]
   * @param [in] x2 dense device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 unused.
   * @param norm_x2 unused.
   */
  virtual void evaluate(raft::resources const& handle,
                        csr_input_matrix_view_t<math_t> x1,
                        dense_input_matrix_view_t<math_t> x2,
                        dense_output_matrix_view_t<math_t> out,
                        math_t* norm_x1,
                        math_t* norm_x2)
  {
    linear(handle, x1, x2, out);
  }
  /** Evaluate the Gram matrix for two vector sets using simple dot product.
   *
   * @param [in] handle raft handle
   * @param [in] x1 csr device matrix view, size [n1*n_cols]
   * @param [in] x2 csr device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 unused.
   * @param norm_x2 unused.
   */
  virtual void evaluate(raft::resources const& handle,
                        csr_input_matrix_view_t<math_t> x1,
                        csr_input_matrix_view_t<math_t> x2,
                        dense_output_matrix_view_t<math_t> out,
                        math_t* norm_x1,
                        math_t* norm_x2)
  {
    linear(handle, x1, x2, out);
  }

  /** Evaluate the Gram matrix for two vector sets using simple dot product.
   *
   * @param [in] x1 device array of vectors, size [n1*n_cols]
   * @param [in] n1 number vectors in x1
   * @param [in] n_cols number of columns (features) in x1 and x2
   * @param [in] x2 device array of vectors, size [n2*n_cols]
   * @param [in] n2 number vectors in x2
   * @param [out] out device buffer to store the Gram matrix, size [n1*n2]
   * @param [in] is_row_major whether the input and output matrices are in row
   *        major format
   * @param [in] stream cuda stream
   * @param ld1 leading dimension of x1 (usually it is n1)
   * @param ld2 leading dimension of x2 (usually it is n2)
   * @param ld_out leading dimension of out (usually it is n1)
   */
  [[deprecated]] virtual void evaluate(const math_t* x1,
                                       int n1,
                                       int n_cols,
                                       const math_t* x2,
                                       int n2,
                                       math_t* out,
                                       bool is_row_major,
                                       cudaStream_t stream,
                                       int ld1,
                                       int ld2,
                                       int ld_out)
  {
    linear(x1, n1, n_cols, x2, n2, out, is_row_major, stream, ld1, ld2, ld_out);
  }

  /** Convenience function to evaluate the Gram matrix for two vector sets.
   *
   * @param [in] x1 device array of vectors, size [n1*n_cols]
   * @param [in] n1 number vectors in x1
   * @param [in] n_cols number of columns (features) in x1 and x2
   * @param [in] x2 device array of vectors, size [n2*n_cols]
   * @param [in] n2 number vectors in x2
   * @param [out] out device buffer to store the Gram matrix, size [n1*n2]
   * @param [in] is_row_major whether the input and output matrices are in row
   *        major format
   * @param [in] stream cuda stream
   * @param ld1 leading dimension of x1
   * @param ld2 leading dimension of x2
   * @param ld_out leading dimension of out
   */
  [[deprecated]] void operator()(const math_t* x1,
                                 int n1,
                                 int n_cols,
                                 const math_t* x2,
                                 int n2,
                                 math_t* out,
                                 bool is_row_major,
                                 cudaStream_t stream,
                                 int ld1    = 0,
                                 int ld2    = 0,
                                 int ld_out = 0)
  {
    ASSERT(legacy_interface, "Legacy interface can only be used with legacy ctor.");
    if (ld1 <= 0) { ld1 = is_row_major ? n_cols : n1; }
    if (ld2 <= 0) { ld2 = is_row_major ? n_cols : n2; }
    if (ld_out <= 0) { ld_out = is_row_major ? n2 : n1; }
    evaluate(x1, n1, n_cols, x2, n2, out, is_row_major, stream, ld1, ld2, ld_out);
  }

 protected:
  /** Calculates the Gram matrix using simple dot product between vector sets.
   *
   * out = x1 * x2
   *
   * Can be used as a building block for more complex kernel functions.
   *
   * @param [in] x1 device array of vectors, size [n1*n_cols]
   * @param [in] n1 number vectors in x1
   * @param [in] n_cols number of columns (features) in x1 and x2
   * @param [in] x2 device array of vectors, size [n2*n_cols]
   * @param [in] n2 number vectors in x2
   * @param [out] out device buffer to store the Gram matrix, size [n1*n2]
   * @param [in] is_row_major whether the input and output matrices are in row
   *        major format
   * @param [in] stream cuda stream
   * @param ld1 leading dimension of x1
   * @param ld2 leading dimension of x2
   * @param ld_out leading dimension of out
   */
  [[deprecated]] void linear(const math_t* x1,
                             int n1,
                             int n_cols,
                             const math_t* x2,
                             int n2,
                             math_t* out,
                             bool is_row_major,
                             cudaStream_t stream,
                             int ld1,
                             int ld2,
                             int ld_out)
  {
    math_t alpha = 1.0;
    math_t beta  = 0.0;
    if (is_row_major) {
      // #TODO: Call from public API when ready
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemm(cublas_handle,
                                                       CUBLAS_OP_T,
                                                       CUBLAS_OP_N,
                                                       n2,
                                                       n1,
                                                       n_cols,
                                                       &alpha,
                                                       x2,
                                                       ld2,
                                                       x1,
                                                       ld1,
                                                       &beta,
                                                       out,
                                                       ld_out,
                                                       stream));
    } else {
      // #TODO: Call from public API when ready
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemm(cublas_handle,
                                                       CUBLAS_OP_N,
                                                       CUBLAS_OP_T,
                                                       n1,
                                                       n2,
                                                       n_cols,
                                                       &alpha,
                                                       x1,
                                                       ld1,
                                                       x2,
                                                       ld2,
                                                       &beta,
                                                       out,
                                                       ld_out,
                                                       stream));
    }
  }

 protected:
  bool get_is_row_major(dense_output_matrix_view_t<math_t> matrix)
  {
    return (matrix.stride(1) == 1);
  }

  bool get_is_row_major(dense_input_matrix_view_t<math_t> matrix)
  {
    return (matrix.stride(1) == 1);
  }

  bool get_is_col_major(dense_output_matrix_view_t<math_t> matrix)
  {
    return (matrix.stride(0) == 1);
  }

  bool get_is_col_major(dense_input_matrix_view_t<math_t> matrix)
  {
    return (matrix.stride(0) == 1);
  }

  /** Calculates the Gram matrix using simple dot product between vector sets.
   *
   * out = x1 * x2
   *
   * Can be used as a building block for more complex kernel functions.
   *
   * @param [in] handle raft handle
   * @param [in] x1 dense device matrix view, size [n1*n_cols]
   * @param [in] x2 dense device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   */
  void linear(raft::resources const& handle,
              dense_input_matrix_view_t<math_t> x1,
              dense_input_matrix_view_t<math_t> x2,
              dense_output_matrix_view_t<math_t> out)
  {
    // check is_row_major consistency
    bool is_row_major = get_is_row_major(x1) && get_is_row_major(x2) && get_is_row_major(out);
    bool is_col_major = get_is_col_major(x1) && get_is_col_major(x2) && get_is_col_major(out);
    ASSERT(is_row_major || is_col_major,
           "GramMatrix leading dimensions for x1, x2 and out do not match");

    // check dimensions
    int n1     = out.extent(0);
    int n2     = out.extent(1);
    int n_cols = x1.extent(1);
    ASSERT(x1.extent(0) == n1, "GramMatrix input matrix dimensions for x1 and out do not match");
    ASSERT(x2.extent(0) == n2, "GramMatrix input matrix dimensions for x2 and out do not match");
    ASSERT(x2.extent(1) == n_cols, "GramMatrix input matrix dimensions for x1 and x2 do not match");

    // extract major stride
    int ld1    = is_row_major ? x1.stride(0) : x1.stride(1);
    int ld2    = is_row_major ? x2.stride(0) : x2.stride(1);
    int ld_out = is_row_major ? out.stride(0) : out.stride(1);

    math_t alpha = 1.0;
    math_t beta  = 0.0;
    if (is_row_major) {
      // #TODO: Use mdspan-based API when stride-capable
      // https://github.com/rapidsai/raft/issues/875
      raft::linalg::gemm(handle,
                         true,
                         false,
                         n2,
                         n1,
                         n_cols,
                         &alpha,
                         x2.data_handle(),
                         ld2,
                         x1.data_handle(),
                         ld1,
                         &beta,
                         out.data_handle(),
                         ld_out,
                         resource::get_cuda_stream(handle));
    } else {
      // #TODO: Use mdspan-based API when stride-capable
      // https://github.com/rapidsai/raft/issues/875
      raft::linalg::gemm(handle,
                         false,
                         true,
                         n1,
                         n2,
                         n_cols,
                         &alpha,
                         x1.data_handle(),
                         ld1,
                         x2.data_handle(),
                         ld2,
                         &beta,
                         out.data_handle(),
                         ld_out,
                         resource::get_cuda_stream(handle));
    }
  }

  /** Calculates the Gram matrix using simple dot product between vector sets.
   *
   * out = x1 * x2
   *
   * Can be used as a building block for more complex kernel functions.
   *
   * @param [in] handle raft handle
   * @param [in] x1 csr device matrix view, size [n1*n_cols]
   * @param [in] x2 dense device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   */
  void linear(raft::resources const& handle,
              csr_input_matrix_view_t<math_t> x1,
              dense_input_matrix_view_t<math_t> x2,
              dense_output_matrix_view_t<math_t> out)
  {
    // check is_row_major consistency
    bool is_row_major = get_is_row_major(x2) && get_is_row_major(out);
    bool is_col_major = get_is_col_major(x2) && get_is_col_major(out);
    ASSERT(is_row_major || is_col_major,
           "GramMatrix leading dimensions for x2 and out do not match");

    // check dimensions
    auto x1_structure = x1.structure_view();
    ASSERT(x1_structure.get_n_rows() == out.extent(0),
           "GramMatrix input matrix dimensions for x1 and out do not match");
    ASSERT(x2.extent(0) == out.extent(1),
           "GramMatrix input matrix dimensions for x2 and out do not match");
    ASSERT(x2.extent(1) == x1_structure.get_n_cols(),
           "GramMatrix input matrix dimensions for x1 and x2 do not match");

    math_t alpha = 1.0;
    math_t beta  = 0.0;

    raft::sparse::linalg::spmm(handle, false, true, &alpha, x1, x2, &beta, out);
  }

  /** Calculates the Gram matrix using simple dot product between vector sets.
   *
   * out = x1 * x2
   *
   * Can be used as a building block for more complex kernel functions.
   *
   * @param [in] handle raft handle
   * @param [in] x1 csr device matrix view, size [n1*n_cols]
   * @param [in] x2 csr device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   */
  void linear(raft::resources const& handle,
              csr_input_matrix_view_t<math_t> x1,
              csr_input_matrix_view_t<math_t> x2,
              dense_output_matrix_view_t<math_t> out)
  {
    // check layout consistency (w.r.t. strides a matrix might be both row & col major)
    bool is_row_major_nopad = get_is_row_major(out) && out.stride(0) == out.extent(1);
    bool is_col_major_nopad = get_is_col_major(out) && out.stride(1) == out.extent(0);

    ASSERT(is_row_major_nopad || is_col_major_nopad,
           "Sparse linear Kernel distance does not support ld_out parameter");

    // switch a,b based on is_row_major
    if (is_col_major_nopad) {
      auto out_row_major = raft::make_device_matrix_view<math_t, int, raft::row_major>(
        out.data_handle(), out.extent(1), out.extent(0));
      raft::sparse::distance::pairwise_distance(
        handle, x2, x1, out_row_major, raft::distance::DistanceType::InnerProduct, 0.0);
    } else {
      auto out_row_major = raft::make_device_matrix_view<math_t, int, raft::row_major>(
        out.data_handle(), out.extent(0), out.extent(1));
      raft::sparse::distance::pairwise_distance(
        handle, x1, x2, out_row_major, raft::distance::DistanceType::InnerProduct, 0.0);
    }
  }
};

};  // end namespace raft::distance::kernels::detail
