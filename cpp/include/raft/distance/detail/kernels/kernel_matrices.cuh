/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "gram_matrix.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/distance/detail/kernels/rbf_fin_op.cuh>
#include <raft/distance/distance.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/sparse/linalg/norm.cuh>
#include <raft/util/cuda_utils.cuh>

namespace raft::distance::kernels::detail {

/** Epiloge function for polynomial kernel without padding.
 * Calculates output = (gain*in + offset)^exponent
 * @param inout device vector in column major format, size [len]
 * @param len array length
 * @param exponent
 * @param gain
 * @param offset
 */
template <typename math_t, typename exp_t>
RAFT_KERNEL polynomial_kernel_nopad(
  math_t* inout, size_t len, exp_t exponent, math_t gain, math_t offset)
{
  for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < len;
       tid += blockDim.x * gridDim.x) {
    inout[tid] = pow(gain * inout[tid] + offset, exponent);
  }
}

/** Epiloge function for polynomial kernel with padding.
 * Calculates output = (gain*input + offset)^exponent
 * @param inout device vector in column major format, size [ld * cols]
 * @param ld leading dimension of the inout buffer
 * @param rows number of rows (rows <= ld)
 * @param cols number of columns
 * @param exponent
 * @param gain
 * @param offset
 */
template <typename math_t, typename exp_t>
RAFT_KERNEL polynomial_kernel(
  math_t* inout, int ld, int rows, int cols, exp_t exponent, math_t gain, math_t offset)
{
  for (size_t tidy = threadIdx.y + blockIdx.y * blockDim.y; tidy < cols;
       tidy += blockDim.y * gridDim.y)
    for (size_t tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < rows;
         tidx += blockDim.x * gridDim.x) {
      inout[tidx + tidy * ld] = pow(gain * inout[tidx + tidy * ld] + offset, exponent);
    }
}

/** Epiloge function for tanh kernel without padding.
 * Calculates output = tanh(gain*input + offset)
 * @param inout device vector, size [len]
 * @param len length of the input vector
 * @param gain
 * @param offset
 */
template <typename math_t>
RAFT_KERNEL tanh_kernel_nopad(math_t* inout, size_t len, math_t gain, math_t offset)
{
  for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < len;
       tid += blockDim.x * gridDim.x) {
    inout[tid] = tanh(gain * inout[tid] + offset);
  }
}

/** Epiloge function for tanh kernel without padding.
 * Calculates output = tanh(gain*input + offset)
 * @param inout device vector in column major format, size [ld * cols]
 * @param ld leading dimension of the inout buffer
 * @param rows number of rows (rows <= ld)
 * @param cols number of columns
 * @param gain
 * @param offset
 */
template <typename math_t>
RAFT_KERNEL tanh_kernel(math_t* inout, int ld, int rows, int cols, math_t gain, math_t offset)
{
  for (size_t tidy = threadIdx.y + blockIdx.y * blockDim.y; tidy < cols;
       tidy += blockDim.y * gridDim.y)
    for (size_t tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < rows;
         tidx += blockDim.x * gridDim.x) {
      inout[tidx + tidy * ld] = tanh(gain * inout[tidx + tidy * ld] + offset);
    }
}

/** Epiloge function for rbf kernel using expansion.
 *
 * Calculates output_ij = exp(-gain * (norm_x_i + norm_y_j - 2*input_ij));
 *
 * Intended usage
 *   - input is the product of two matrices X and Y input_ij = sum_k X_ik * Y_jk
 *   - norm_x_i = l2_norm(x_i), where x_i is the i-th row of matrix X
 *   - norm_y_j = l2_norm(y_j), where y_j is the j-th row of matrix Y
 *
 * @param inout device vector in column major format, size [ld * cols]
 * @param ld leading dimension of the inout buffer
 * @param rows number of rows (rows <= ld)
 * @param cols number of columns
 * @param norm_x l2-norm of X's rows
 * @param norm_y l2-norm of Y's rows
 * @param gain
 */
template <typename math_t>
RAFT_KERNEL rbf_kernel_expanded(
  math_t* inout, int ld, int rows, int cols, math_t* norm_x, math_t* norm_y, math_t gain)
{
  for (size_t tidy = threadIdx.y + blockIdx.y * blockDim.y; tidy < cols;
       tidy += blockDim.y * gridDim.y) {
    math_t norm_y_val = norm_y[tidy];
    for (size_t tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < rows;
         tidx += blockDim.x * gridDim.x) {
      inout[tidx + tidy * ld] =
        exp(-1.0 * gain * (norm_x[tidx] + norm_y_val - inout[tidx + tidy * ld] * 2));
    }
  }
}

namespace {
std::tuple<dim3, dim3> generateLaunchConfig2dElementwiseOp(int n1, int n2)
{
  dim3 block_shape       = dim3(32, 4);
  const int num_blocks_x = raft::ceildiv(n1, 32);
  const int num_blocks_y = std::min(raft::ceildiv(n2, 32), (1 << 16) - 1);
  dim3 grid_shape        = dim3(num_blocks_x, num_blocks_y);
  return std::make_tuple(grid_shape, block_shape);
}
}  // namespace

/**
 * Create a kernel matrix using polynomial kernel function.
 */
template <typename math_t, typename exp_t>
class PolynomialKernel : public GramMatrixBase<math_t> {
  exp_t exponent;
  math_t gain;
  math_t offset;

  void applyKernel(
    math_t* inout, int ld, int rows, int cols, bool is_row_major, cudaStream_t stream)
  {
    const int n_minor = is_row_major ? cols : rows;
    if (ld == n_minor) {
      polynomial_kernel_nopad<<<raft::ceildiv<size_t>((size_t)rows * cols, 128), 128, 0, stream>>>(
        inout, rows * cols, exponent, gain, offset);
    } else {
      int n1                         = is_row_major ? cols : rows;
      int n2                         = is_row_major ? rows : cols;
      auto [grid_shape, block_shape] = generateLaunchConfig2dElementwiseOp(n1, n2);
      polynomial_kernel<<<grid_shape, block_shape, 0, stream>>>(
        inout, ld, n1, n2, exponent, gain, offset);
    }
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

 public:
  /**
   * Constructs a polynomial kernel object.
   * It evaluates the kernel matrix using the following formula:
   * K_ij = (gain*<x1_i, x2_k> + offset)^exponent
   *
   * @tparam math_t floating point type
   * @tparam exp_t type of exponent
   * @param exponent
   * @param gain
   * @param offset
   */
  PolynomialKernel(exp_t exponent, math_t gain, math_t offset)
    : GramMatrixBase<math_t>(), exponent(exponent), gain(gain), offset(offset)
  {
  }

  [[deprecated]] PolynomialKernel(exp_t exponent, math_t gain, math_t offset, cublasHandle_t handle)
    : GramMatrixBase<math_t>(handle), exponent(exponent), gain(gain), offset(offset)
  {
  }

  /** Evaluate kernel matrix using polynomial kernel.
   *
   * output[i,k] = (gain*<x1_i, x2_k> + offset)^exponent,
   * where x1_i is the i-th vector from the x1 set, and x2_k is k-th vector
   * in the x2 set, and < , > denotes dot product.
   *
   * @param [in] handle raft handle
   * @param [in] x1 dense device matrix view, size [n1*n_cols]
   * @param [in] x2 dense device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 unused.
   * @param norm_x2 unused.
   */
  void evaluate(raft::resources const& handle,
                dense_input_matrix_view_t<math_t> x1,
                dense_input_matrix_view_t<math_t> x2,
                dense_output_matrix_view_t<math_t> out,
                math_t* norm_x1,
                math_t* norm_x2)
  {
    bool is_row_major = GramMatrixBase<math_t>::get_is_row_major(out);
    int ld_out        = is_row_major ? out.stride(0) : out.stride(1);
    GramMatrixBase<math_t>::linear(handle, x1, x2, out);
    applyKernel(out.data_handle(),
                ld_out,
                out.extent(0),
                out.extent(1),
                is_row_major,
                resource::get_cuda_stream(handle));
  }

  /** Evaluate kernel matrix using polynomial kernel.
   *
   * output[i,k] = (gain*<x1_i, x2_k> + offset)^exponent,
   * where x1_i is the i-th vector from the x1 set, and x2_k is k-th vector
   * in the x2 set, and < , > denotes dot product.
   *
   * @param [in] handle raft handle
   * @param [in] x1 csr device matrix view, size [n1*n_cols]
   * @param [in] x2 dense device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 unused.
   * @param norm_x2 unused.
   */
  void evaluate(raft::resources const& handle,
                csr_input_matrix_view_t<math_t> x1,
                dense_input_matrix_view_t<math_t> x2,
                dense_output_matrix_view_t<math_t> out,
                math_t* norm_x1,
                math_t* norm_x2)
  {
    bool is_row_major = GramMatrixBase<math_t>::get_is_row_major(out);
    int ld_out        = is_row_major ? out.stride(0) : out.stride(1);
    GramMatrixBase<math_t>::linear(handle, x1, x2, out);
    applyKernel(out.data_handle(),
                ld_out,
                out.extent(0),
                out.extent(1),
                is_row_major,
                resource::get_cuda_stream(handle));
  }

  /** Evaluate kernel matrix using polynomial kernel.
   *
   * output[i,k] = (gain*<x1_i, x2_k> + offset)^exponent,
   * where x1_i is the i-th vector from the x1 set, and x2_k is k-th vector
   * in the x2 set, and < , > denotes dot product.
   *
   * @param [in] handle raft handle
   * @param [in] x1 csr device matrix view, size [n1*n_cols]
   * @param [in] x2 csr device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 unused.
   * @param norm_x2 unused.
   */
  void evaluate(raft::resources const& handle,
                csr_input_matrix_view_t<math_t> x1,
                csr_input_matrix_view_t<math_t> x2,
                dense_output_matrix_view_t<math_t> out,
                math_t* norm_x1,
                math_t* norm_x2)
  {
    bool is_row_major = GramMatrixBase<math_t>::get_is_row_major(out);
    int ld_out        = is_row_major ? out.stride(0) : out.stride(1);
    GramMatrixBase<math_t>::linear(handle, x1, x2, out);
    applyKernel(out.data_handle(),
                ld_out,
                out.extent(0),
                out.extent(1),
                is_row_major,
                resource::get_cuda_stream(handle));
  }

  /** Evaluate the Gram matrix using the legacy interface.
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
  [[deprecated]] void evaluate(const math_t* x1,
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
    ASSERT(GramMatrixBase<math_t>::legacy_interface,
           "Legacy interface can only be used with legacy ctor.");
    GramMatrixBase<math_t>::linear(
      x1, n1, n_cols, x2, n2, out, is_row_major, stream, ld1, ld2, ld_out);
    applyKernel(out, ld_out, n1, n2, is_row_major, stream);
  }
};

/**
 * Create a kernel matrix using tanh kernel function.
 */
template <typename math_t>
class TanhKernel : public GramMatrixBase<math_t> {
  math_t gain, offset;

  void applyKernel(
    math_t* inout, int ld, int rows, int cols, bool is_row_major, cudaStream_t stream)
  {
    const int n_minor = is_row_major ? cols : rows;
    if (ld == n_minor) {
      tanh_kernel_nopad<<<raft::ceildiv<size_t>((size_t)rows * cols, 128), 128, 0, stream>>>(
        inout, rows * cols, gain, offset);
    } else {
      int n1                         = is_row_major ? cols : rows;
      int n2                         = is_row_major ? rows : cols;
      auto [grid_shape, block_shape] = generateLaunchConfig2dElementwiseOp(n1, n2);
      tanh_kernel<<<grid_shape, block_shape, 0, stream>>>(inout, ld, n1, n2, gain, offset);
    }
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

 public:
  /**
   * Constructs a tanh kernel object.
   * It evaluates the kernel matrix using the following formula:
   * K_ij = tanh(gain*<x1_i, x2_k> + offset)
   *
   * @tparam math_t floating point type
   * @param gain
   * @param offset
   */
  TanhKernel(math_t gain, math_t offset) : GramMatrixBase<math_t>(), gain(gain), offset(offset) {}

  [[deprecated]] TanhKernel(math_t gain, math_t offset, cublasHandle_t handle)
    : GramMatrixBase<math_t>(handle), gain(gain), offset(offset)
  {
  }

  /** Evaluate kernel matrix using tanh kernel.
   *
   * output_[i + k*n1] = (gain*<x1_i, x2_k> + offset)^exponent,
   * where x1_i is the i-th vector from the x1 set, and x2_k is k-th vector
   * in the x2 set, and < , > denotes dot product.
   *
   * @param [in] handle raft handle
   * @param [in] x1 dense device matrix view, size [n1*n_cols]
   * @param [in] x2 dense device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 unused.
   * @param norm_x2 unused.
   */
  void evaluate(raft::resources const& handle,
                dense_input_matrix_view_t<math_t> x1,
                dense_input_matrix_view_t<math_t> x2,
                dense_output_matrix_view_t<math_t> out,
                math_t* norm_x1,
                math_t* norm_x2)
  {
    bool is_row_major = GramMatrixBase<math_t>::get_is_row_major(out);
    int ld_out        = is_row_major ? out.stride(0) : out.stride(1);
    GramMatrixBase<math_t>::linear(handle, x1, x2, out);
    applyKernel(out.data_handle(),
                ld_out,
                out.extent(0),
                out.extent(1),
                is_row_major,
                resource::get_cuda_stream(handle));
  }

  /** Evaluate kernel matrix using tanh kernel.
   *
   * output_[i + k*n1] = (gain*<x1_i, x2_k> + offset)^exponent,
   * where x1_i is the i-th vector from the x1 set, and x2_k is k-th vector
   * in the x2 set, and < , > denotes dot product.
   *
   * @param [in] handle raft handle
   * @param [in] x1 csr device matrix view, size [n1*n_cols]
   * @param [in] x2 dense device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 unused.
   * @param norm_x2 unused.
   */
  void evaluate(raft::resources const& handle,
                csr_input_matrix_view_t<math_t> x1,
                dense_input_matrix_view_t<math_t> x2,
                dense_output_matrix_view_t<math_t> out,
                math_t* norm_x1,
                math_t* norm_x2)
  {
    bool is_row_major = GramMatrixBase<math_t>::get_is_row_major(out);
    int ld_out        = is_row_major ? out.stride(0) : out.stride(1);
    GramMatrixBase<math_t>::linear(handle, x1, x2, out);
    applyKernel(out.data_handle(),
                ld_out,
                out.extent(0),
                out.extent(1),
                is_row_major,
                resource::get_cuda_stream(handle));
  }

  /** Evaluate kernel matrix using tanh kernel.
   *
   * output_[i + k*n1] = (gain*<x1_i, x2_k> + offset)^exponent,
   * where x1_i is the i-th vector from the x1 set, and x2_k is k-th vector
   * in the x2 set, and < , > denotes dot product.
   *
   * @param [in] handle raft handle
   * @param [in] x1 csr device matrix view, size [n1*n_cols]
   * @param [in] x2 csr device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 unused.
   * @param norm_x2 unused.
   */
  void evaluate(raft::resources const& handle,
                csr_input_matrix_view_t<math_t> x1,
                csr_input_matrix_view_t<math_t> x2,
                dense_output_matrix_view_t<math_t> out,
                math_t* norm_x1,
                math_t* norm_x2)
  {
    bool is_row_major = GramMatrixBase<math_t>::get_is_row_major(out);
    int ld_out        = is_row_major ? out.stride(0) : out.stride(1);
    GramMatrixBase<math_t>::linear(handle, x1, x2, out);
    applyKernel(out.data_handle(),
                ld_out,
                out.extent(0),
                out.extent(1),
                is_row_major,
                resource::get_cuda_stream(handle));
  }

  /** Evaluate the Gram matrix using the legacy interface.
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
  [[deprecated]] void evaluate(const math_t* x1,
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
    ASSERT(GramMatrixBase<math_t>::legacy_interface,
           "Legacy interface can only be used with legacy ctor.");
    GramMatrixBase<math_t>::linear(
      x1, n1, n_cols, x2, n2, out, is_row_major, stream, ld1, ld2, ld_out);
    applyKernel(out, ld_out, n1, n2, is_row_major, stream);
  }
};

/**
 * Create a kernel matrix using RBF kernel function.
 */
template <typename math_t>
class RBFKernel : public GramMatrixBase<math_t> {
  math_t gain;

  void applyKernel(math_t* inout,
                   int ld,
                   int rows,
                   int cols,
                   math_t* norm_x1,
                   math_t* norm_x2,
                   bool is_row_major,
                   cudaStream_t stream)
  {
    int n1                         = is_row_major ? cols : rows;
    int n2                         = is_row_major ? rows : cols;
    math_t* norm_n1                = is_row_major ? norm_x2 : norm_x1;
    math_t* norm_n2                = is_row_major ? norm_x1 : norm_x2;
    auto [grid_shape, block_shape] = generateLaunchConfig2dElementwiseOp(n1, n2);
    rbf_kernel_expanded<<<grid_shape, block_shape, 0, stream>>>(
      inout, ld, n1, n2, norm_n1, norm_n2, gain);
  }

 public:
  /**
   * Constructs a RBF kernel object.
   * It evaluates the kernel matrix using the following formula:
   * K_ij = exp(-gain*|x1_i- x2_k|^2)
   *
   * @tparam math_t floating point type
   * @param gain
   */
  RBFKernel(math_t gain) : GramMatrixBase<math_t>(), gain(gain) {}

  [[deprecated]] RBFKernel(math_t gain, cublasHandle_t handle)
    : GramMatrixBase<math_t>(handle), gain(gain)
  {
  }

  void matrixRowNormL2(raft::resources const& handle,
                       dense_input_matrix_view_t<math_t> matrix,
                       math_t* target)
  {
    bool is_row_major = GramMatrixBase<math_t>::get_is_row_major(matrix);
    int minor         = is_row_major ? matrix.extent(1) : matrix.extent(0);
    int ld            = is_row_major ? matrix.stride(0) : matrix.stride(1);
    ASSERT(ld == minor, "RBF Kernel lazy rowNorm compute does not support ld parameter");
    raft::linalg::rowNorm(target,
                          matrix.data_handle(),
                          matrix.extent(1),
                          matrix.extent(0),
                          raft::linalg::NormType::L2Norm,
                          is_row_major,
                          resource::get_cuda_stream(handle));
  }

  void matrixRowNormL2(raft::resources const& handle,
                       csr_input_matrix_view_t<math_t> matrix,
                       math_t* target)
  {
    auto matrix_structure = matrix.structure_view();
    raft::sparse::linalg::rowNormCsr(handle,
                                     matrix_structure.get_indptr().data(),
                                     matrix.get_elements().data(),
                                     matrix_structure.get_nnz(),
                                     matrix_structure.get_n_rows(),
                                     target,
                                     raft::linalg::NormType::L2Norm);
  }

  /** Evaluate kernel matrix using RBF kernel.
   *
   * output_[i + k*n1] = exp(-gain*|x1_i - x2_k|^2),
   * where x1_i is the i-th vector from the x1 set, and x2_k is k-th vector
   * in the x2 set, and | | euclidean distance.
   *
   * @param [in] handle raft handle
   * @param [in] x1 dense device matrix view, size [n1*n_cols]
   * @param [in] x2 dense device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 optional L2-norm of x1's rows for computation within RBF.
   * @param norm_x2 optional L2-norm of x2's rows for computation within RBF.
   */
  void evaluate(raft::resources const& handle,
                dense_input_matrix_view_t<math_t> x1,
                dense_input_matrix_view_t<math_t> x2,
                dense_output_matrix_view_t<math_t> out,
                math_t* norm_x1,
                math_t* norm_x2)
  {
    cudaStream_t stream = resource::get_cuda_stream(handle);
    // lazy compute norms if not given
    rmm::device_uvector<math_t> tmp_norm_x1(0, stream);
    rmm::device_uvector<math_t> tmp_norm_x2(0, stream);
    if (norm_x1 == nullptr) {
      tmp_norm_x1.reserve(x1.extent(0), stream);
      norm_x1 = tmp_norm_x1.data();
      matrixRowNormL2(handle, x1, norm_x1);
    }
    if (norm_x2 == nullptr) {
      tmp_norm_x2.reserve(x2.extent(0), stream);
      norm_x2 = tmp_norm_x2.data();
      matrixRowNormL2(handle, x2, norm_x2);
    }

    // compute L2expanded
    bool is_row_major = GramMatrixBase<math_t>::get_is_row_major(out);
    int ld_out        = is_row_major ? out.stride(0) : out.stride(1);
    GramMatrixBase<math_t>::linear(handle, x1, x2, out);
    applyKernel(out.data_handle(),
                ld_out,
                out.extent(0),
                out.extent(1),
                norm_x1,
                norm_x2,
                is_row_major,
                resource::get_cuda_stream(handle));
  }

  /** Evaluate kernel matrix using RBF kernel.
   *
   * output_[i + k*n1] = exp(-gain*|x1_i - x2_k|^2),
   * where x1_i is the i-th vector from the x1 set, and x2_k is k-th vector
   * in the x2 set, and | | euclidean distance.
   *
   * @param [in] handle raft handle
   * @param [in] x1 csr device matrix view, size [n1*n_cols]
   * @param [in] x2 dense device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 optional L2-norm of x1's rows for computation within RBF.
   * @param norm_x2 optional L2-norm of x2's rows for computation within RBF.
   */
  void evaluate(raft::resources const& handle,
                csr_input_matrix_view_t<math_t> x1,
                dense_input_matrix_view_t<math_t> x2,
                dense_output_matrix_view_t<math_t> out,
                math_t* norm_x1,
                math_t* norm_x2)
  {
    cudaStream_t stream = resource::get_cuda_stream(handle);

    // lazy compute norms if not given
    rmm::device_uvector<math_t> tmp_norm_x1(0, stream);
    rmm::device_uvector<math_t> tmp_norm_x2(0, stream);
    if (norm_x1 == nullptr) {
      tmp_norm_x1.reserve(x1.structure_view().get_n_rows(), stream);
      norm_x1 = tmp_norm_x1.data();
      matrixRowNormL2(handle, x1, norm_x1);
    }
    if (norm_x2 == nullptr) {
      tmp_norm_x2.reserve(x2.extent(0), stream);
      norm_x2 = tmp_norm_x2.data();
      matrixRowNormL2(handle, x2, norm_x2);
    }

    // compute L2expanded
    bool is_row_major = GramMatrixBase<math_t>::get_is_row_major(out);
    int ld_out        = is_row_major ? out.stride(0) : out.stride(1);
    GramMatrixBase<math_t>::linear(handle, x1, x2, out);
    applyKernel(out.data_handle(),
                ld_out,
                out.extent(0),
                out.extent(1),
                norm_x1,
                norm_x2,
                is_row_major,
                resource::get_cuda_stream(handle));
  }

  /** Evaluate kernel matrix using RBF kernel.
   *
   * output_[i + k*n1] = exp(-gain*|x1_i - x2_k|^2),
   * where x1_i is the i-th vector from the x1 set, and x2_k is k-th vector
   * in the x2 set, and | | euclidean distance.
   *
   * @param [in] handle raft handle
   * @param [in] x1 csr device matrix view, size [n1*n_cols]
   * @param [in] x2 csr device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 optional L2-norm of x1's rows for computation within RBF.
   * @param norm_x2 optional L2-norm of x2's rows for computation within RBF.
   */
  void evaluate(raft::resources const& handle,
                csr_input_matrix_view_t<math_t> x1,
                csr_input_matrix_view_t<math_t> x2,
                dense_output_matrix_view_t<math_t> out,
                math_t* norm_x1,
                math_t* norm_x2)
  {
    cudaStream_t stream = resource::get_cuda_stream(handle);

    // lazy compute norms if not given
    rmm::device_uvector<math_t> tmp_norm_x1(0, stream);
    rmm::device_uvector<math_t> tmp_norm_x2(0, stream);
    if (norm_x1 == nullptr) {
      tmp_norm_x1.reserve(x1.structure_view().get_n_rows(), stream);
      norm_x1 = tmp_norm_x1.data();
      matrixRowNormL2(handle, x1, norm_x1);
    }
    if (norm_x2 == nullptr) {
      tmp_norm_x2.reserve(x2.structure_view().get_n_rows(), stream);
      norm_x2 = tmp_norm_x2.data();
      matrixRowNormL2(handle, x2, norm_x2);
    }

    // compute L2expanded
    bool is_row_major = GramMatrixBase<math_t>::get_is_row_major(out);
    int ld_out        = is_row_major ? out.stride(0) : out.stride(1);
    GramMatrixBase<math_t>::linear(handle, x1, x2, out);
    applyKernel(out.data_handle(),
                ld_out,
                out.extent(0),
                out.extent(1),
                norm_x1,
                norm_x2,
                is_row_major,
                resource::get_cuda_stream(handle));
  }

  /** Evaluate the Gram matrix using the legacy interface.
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
  [[deprecated]] void evaluate(const math_t* x1,
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
    ASSERT(GramMatrixBase<math_t>::legacy_interface,
           "Legacy interface can only be used with legacy ctor.");
    int minor1    = is_row_major ? n_cols : n1;
    int minor2    = is_row_major ? n_cols : n2;
    int minor_out = is_row_major ? n2 : n1;
    ASSERT(ld1 == minor1, "RBF Kernel distance does not support ld1 parameter");
    ASSERT(ld2 == minor2, "RBF Kernel distance does not support ld2 parameter");
    ASSERT(ld_out == minor_out, "RBF Kernel distance does not support ld_out parameter");

    math_t gain   = this->gain;
    using index_t = int64_t;

    rbf_fin_op fin_op{gain};

    raft::resources handle;
    resource::set_cuda_stream(handle, stream);

    raft::distance::distance<raft::distance::DistanceType::L2Unexpanded,
                             math_t,
                             math_t,
                             math_t,
                             decltype(fin_op),
                             index_t>(handle,
                                      const_cast<math_t*>(x1),
                                      const_cast<math_t*>(x2),
                                      out,
                                      n1,
                                      n2,
                                      n_cols,
                                      NULL,
                                      0,
                                      fin_op,
                                      is_row_major);
  }
};

};  // end namespace raft::distance::kernels::detail
