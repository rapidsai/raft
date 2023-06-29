/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#if defined RAFT_DISTANCE_COMPILED
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/distance/specializations.cuh>
#endif

#include "../distance/gram_base.cuh"
#include "../test_utils.cuh"
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <raft/distance/distance_types.hpp>
#include <raft/distance/kernels.cuh>
#include <raft/random/rng.cuh>
#include <raft/sparse/convert/dense.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/itertools.hpp>
#include <rmm/device_uvector.hpp>

namespace raft::distance::kernels {

/**
 * Structure to describe structure of the input matrices:
 *  - DENSE: dense, dense
 *  - MIX: CSR, dense
 *  - CSR: CSR, CSR
 */
enum SparseType { DENSE, MIX, CSR };

struct GramMatrixInputs {
  int n1;      // feature vectors in matrix 1
  int n2;      // featuer vectors in matrix 2
  int n_cols;  // number of elements in a feature vector
  bool is_row_major;
  SparseType sparse_input;
  KernelParams kernel;
  int ld1;
  int ld2;
  int ld_out;
  // We will generate random input using the dimensions given here.
  // The reference output is calculated by a custom kernel.
};

std::ostream& operator<<(std::ostream& os, const GramMatrixInputs& p)
{
  std::vector<std::string> kernel_names{"linear", "poly", "rbf", "tanh"};
  os << "/" << p.n1 << "x" << p.n2 << "x" << p.n_cols << "/"
     << (p.is_row_major ? "RowMajor/" : "ColMajor/")
     << (p.sparse_input == SparseType::DENSE
           ? "DenseDense/"
           : (p.sparse_input == SparseType::MIX ? "CsrDense/" : "CsrCsr/"))
     << kernel_names[p.kernel.kernel] << "/ld_" << p.ld1 << "x" << p.ld2 << "x" << p.ld_out;
  return os;
}

/*struct KernelParams {
  // Kernel function parameters
  KernelType kernel;  //!< Type of the kernel function
  int degree;         //!< Degree of polynomial kernel (ignored by others)
  double gamma;       //!< multiplier in the
  double coef0;       //!< additive constant in poly and tanh kernels
};*/

// const KernelParams linear_kernel_params{.kernel=KernelType::LINEAR};

// {KernelType::POLYNOMIAL, 2, 0.5, 2.4}, {KernelType::TANH, 0, 0.5, 2.4}, {KernelType::RBF, 0, 0.5}
const std::vector<GramMatrixInputs> inputs = raft::util::itertools::product<GramMatrixInputs>(
  {42},
  {137},
  {2},
  {true, false},
  {SparseType::DENSE, SparseType::MIX, SparseType::CSR},
  {KernelParams{KernelType::LINEAR},
   KernelParams{KernelType::POLYNOMIAL, 2, 0.5, 2.4},
   KernelParams{KernelType::TANH, 0, 0.5, 2.4},
   KernelParams{KernelType::RBF, 0, 0.5}});

// (ld_1, ld_2, ld_out) not supported by RBF and CSR
const std::vector<GramMatrixInputs> inputs_ld = raft::util::itertools::product<GramMatrixInputs>(
  {137},
  {42},
  {2},
  {true, false},
  {SparseType::DENSE, SparseType::MIX},
  {KernelParams{KernelType::LINEAR},
   KernelParams{KernelType::POLYNOMIAL, 2, 0.5, 2.4},
   KernelParams{KernelType::TANH, 0, 0.5, 2.4}},
  {159},
  {73},
  {144});

// (ld_1, ld_2) are supported by CSR
const std::vector<GramMatrixInputs> inputs_ld_csr =
  raft::util::itertools::product<GramMatrixInputs>(
    {42},
    {137},
    {2},
    {true, false},
    {SparseType::CSR, SparseType::MIX},
    {KernelParams{KernelType::LINEAR},
     KernelParams{KernelType::POLYNOMIAL, 2, 0.5, 2.4},
     KernelParams{KernelType::TANH, 0, 0.5, 2.4}},
    {64},
    {155},
    {0});

template <typename math_t>
class GramMatrixTest : public ::testing::TestWithParam<GramMatrixInputs> {
 protected:
  GramMatrixTest()
    : params(GetParam()),
      stream(0),
      x1(0, stream),
      x2(0, stream),
      x1_csr_indptr(0, stream),
      x1_csr_indices(0, stream),
      x1_csr_data(0, stream),
      x2_csr_indptr(0, stream),
      x2_csr_indices(0, stream),
      x2_csr_data(0, stream),
      gram(0, stream),
      gram_host(0)
  {
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));

    if (params.ld1 == 0) { params.ld1 = params.is_row_major ? params.n_cols : params.n1; }
    if (params.ld2 == 0) { params.ld2 = params.is_row_major ? params.n_cols : params.n2; }
    if (params.ld_out == 0) { params.ld_out = params.is_row_major ? params.n2 : params.n1; }
    // Derive the size of the output from the offset of the last element.
    size_t size = get_offset(params.n1 - 1, params.n_cols - 1, params.ld1, params.is_row_major) + 1;
    x1.resize(size, stream);
    size = get_offset(params.n2 - 1, params.n_cols - 1, params.ld2, params.is_row_major) + 1;
    x2.resize(size, stream);
    size = get_offset(params.n1 - 1, params.n2 - 1, params.ld_out, params.is_row_major) + 1;

    gram.resize(size, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(gram.data(), 0, gram.size() * sizeof(math_t), stream));
    gram_host.resize(gram.size());
    std::fill(gram_host.begin(), gram_host.end(), 0);

    raft::random::Rng r(42137ULL);
    r.uniform(x1.data(), x1.size(), math_t(0), math_t(1), stream);
    r.uniform(x2.data(), x2.size(), math_t(0), math_t(1), stream);

    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  ~GramMatrixTest() override { RAFT_CUDA_TRY_NO_THROW(cudaStreamDestroy(stream)); }

  int prepareCsr(math_t* dense, int n_rows, int ld, int* indptr, int* indices, math_t* data)
  {
    int nnz           = 0;
    double eps        = 1e-6;
    int n_cols        = params.n_cols;
    bool is_row_major = params.is_row_major;
    size_t dense_size = get_offset(n_rows - 1, n_cols - 1, ld, is_row_major) + 1;

    std::vector<math_t> dense_host(dense_size);
    raft::update_host(dense_host.data(), dense, dense_size, stream);
    resource::sync_stream(handle, stream);

    std::vector<int> indptr_host(n_rows + 1);
    std::vector<int> indices_host(n_rows * n_cols);
    std::vector<math_t> data_host(n_rows * n_cols);

    // create csr matrix from dense (with threshold)
    for (int i = 0; i < n_rows; ++i) {
      indptr_host[i] = nnz;
      for (int j = 0; j < n_cols; ++j) {
        math_t value = dense_host[get_offset(i, j, ld, is_row_major)];
        if (value > eps) {
          indices_host[nnz] = j;
          data_host[nnz]    = value;
          nnz++;
        }
      }
    }
    indptr_host[n_rows] = nnz;

    // fill back dense matrix from CSR
    std::fill(dense_host.data(), dense_host.data() + dense_size, 0);
    for (int i = 0; i < n_rows; ++i) {
      for (int idx = indptr_host[i]; idx < indptr_host[i + 1]; ++idx) {
        dense_host[get_offset(i, indices_host[idx], ld, is_row_major)] = data_host[idx];
      }
    }

    raft::update_device(dense, dense_host.data(), dense_size, stream);
    raft::update_device(indptr, indptr_host.data(), n_rows + 1, stream);
    raft::update_device(indices, indices_host.data(), nnz, stream);
    raft::update_device(data, data_host.data(), nnz, stream);
    resource::sync_stream(handle, stream);
    return nnz;
  }

  void runTest()
  {
    std::unique_ptr<GramMatrixBase<math_t>> kernel =
      std::unique_ptr<GramMatrixBase<math_t>>(KernelFactory<math_t>::create(params.kernel));

    auto x1_span =
      params.is_row_major
        ? raft::make_device_strided_matrix_view<const math_t, int, raft::layout_c_contiguous>(
            x1.data(), params.n1, params.n_cols, params.ld1)
        : raft::make_device_strided_matrix_view<const math_t, int, raft::layout_f_contiguous>(
            x1.data(), params.n1, params.n_cols, params.ld1);
    auto x2_span =
      params.is_row_major
        ? raft::make_device_strided_matrix_view<const math_t, int, raft::layout_c_contiguous>(
            x2.data(), params.n2, params.n_cols, params.ld2)
        : raft::make_device_strided_matrix_view<const math_t, int, raft::layout_f_contiguous>(
            x2.data(), params.n2, params.n_cols, params.ld2);
    auto out_span =
      params.is_row_major
        ? raft::make_device_strided_matrix_view<math_t, int, raft::layout_c_contiguous>(
            gram.data(), params.n1, params.n2, params.ld_out)
        : raft::make_device_strided_matrix_view<math_t, int, raft::layout_f_contiguous>(
            gram.data(), params.n1, params.n2, params.ld_out);

    if (params.sparse_input == SparseType::DENSE) {
      (*kernel)(handle, x1_span, x2_span, out_span);
    } else {
      x1_csr_indptr.reserve(params.n1 + 1, stream);
      x1_csr_indices.reserve(params.n1 * params.n_cols, stream);
      x1_csr_data.reserve(params.n1 * params.n_cols, stream);
      int x1_nnz = prepareCsr(x1.data(),
                              params.n1,
                              params.ld1,
                              x1_csr_indptr.data(),
                              x1_csr_indices.data(),
                              x1_csr_data.data());

      auto x1_csr_structure = raft::make_device_compressed_structure_view<int, int, int>(
        x1_csr_indptr.data(), x1_csr_indices.data(), params.n1, params.n_cols, x1_nnz);
      auto x1_csr = raft::device_csr_matrix_view<const math_t, int, int, int>(
        raft::device_span<const math_t>(x1_csr_data.data(), x1_csr_structure.get_nnz()),
        x1_csr_structure);

      if (params.sparse_input == SparseType::MIX) {
        (*kernel)(handle, x1_csr, x2_span, out_span);
      } else {
        x2_csr_indptr.reserve(params.n2 + 1, stream);
        x2_csr_indices.reserve(params.n2 * params.n_cols, stream);
        x2_csr_data.reserve(params.n2 * params.n_cols, stream);
        int x2_nnz = prepareCsr(x2.data(),
                                params.n2,
                                params.ld2,
                                x2_csr_indptr.data(),
                                x2_csr_indices.data(),
                                x2_csr_data.data());

        auto x2_csr_structure = raft::make_device_compressed_structure_view<int, int, int>(
          x2_csr_indptr.data(), x2_csr_indices.data(), params.n2, params.n_cols, x2_nnz);
        auto x2_csr = raft::device_csr_matrix_view<const math_t, int, int, int>(
          raft::device_span<const math_t>(x2_csr_data.data(), x2_csr_structure.get_nnz()),
          x2_csr_structure);

        (*kernel)(handle, x1_csr, x2_csr, out_span);
      }
    }
    // Something in gram is executing not on the 'stream' and therefore
    // a full device sync is required
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    naiveGramMatrixKernel(params.n1,
                          params.n2,
                          params.n_cols,
                          x1,
                          x2,
                          gram_host.data(),
                          params.ld1,
                          params.ld2,
                          params.ld_out,
                          params.is_row_major,
                          params.kernel,
                          stream,
                          handle);
    resource::sync_stream(handle, stream);

    ASSERT_TRUE(raft::devArrMatchHost(
      gram_host.data(), gram.data(), gram.size(), raft::CompareApprox<math_t>(1e-6f), stream));
  }

  raft::resources handle;
  cudaStream_t stream = 0;
  GramMatrixInputs params;

  rmm::device_uvector<math_t> x1;
  rmm::device_uvector<math_t> x2;

  rmm::device_uvector<int> x1_csr_indptr;
  rmm::device_uvector<int> x1_csr_indices;
  rmm::device_uvector<math_t> x1_csr_data;
  rmm::device_uvector<int> x2_csr_indptr;
  rmm::device_uvector<int> x2_csr_indices;
  rmm::device_uvector<math_t> x2_csr_data;

  rmm::device_uvector<math_t> gram;
  std::vector<math_t> gram_host;
};

typedef GramMatrixTest<float> GramMatrixTestFloatStandard;
typedef GramMatrixTest<float> GramMatrixTestFloatLd;
typedef GramMatrixTest<float> GramMatrixTestFloatLdCsr;

TEST_P(GramMatrixTestFloatStandard, Gram) { runTest(); }
TEST_P(GramMatrixTestFloatLd, Gram) { runTest(); }
TEST_P(GramMatrixTestFloatLdCsr, Gram) { runTest(); }

INSTANTIATE_TEST_SUITE_P(GramMatrixTests, GramMatrixTestFloatStandard, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_SUITE_P(GramMatrixTests, GramMatrixTestFloatLd, ::testing::ValuesIn(inputs_ld));
INSTANTIATE_TEST_SUITE_P(GramMatrixTests,
                         GramMatrixTestFloatLdCsr,
                         ::testing::ValuesIn(inputs_ld_csr));
};  // end namespace raft::distance::kernels
