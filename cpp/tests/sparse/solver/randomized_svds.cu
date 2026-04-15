/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../test_utils.cuh"

#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/svd.cuh>
#include <raft/sparse/solver/detail/csr_linear_operator.cuh>
#include <raft/sparse/solver/randomized_svds.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace raft::sparse::solver {

// ============================================================================
// Golden data: 20x15 sparse matrix (nnz=120), generated with cupy seed=42
// Expected singular values computed via cupy.linalg.svd
// ============================================================================

template <typename IndexType>
std::vector<IndexType> golden_indptr()
{
  return {0, 4, 12, 17, 21, 27, 35, 41, 48, 55, 61, 67, 74, 79, 84, 90, 97, 102, 110, 114, 120};
}

template <typename IndexType>
std::vector<IndexType> golden_indices()
{
  return {0,  4,  11, 14, 0,  2,  4,  5,  6,  12, 13, 14, 1,  6,  7,  8,  10, 0,  1,  6,
          11, 0,  3,  4,  5,  13, 14, 0,  1,  2,  3,  8,  10, 12, 14, 1,  4,  8,  9,  11,
          12, 0,  3,  4,  7,  10, 11, 13, 0,  2,  4,  8,  9,  11, 13, 0,  4,  8,  9,  10,
          11, 3,  4,  5,  12, 13, 14, 0,  1,  2,  5,  7,  10, 13, 0,  2,  7,  13, 14, 3,
          8,  10, 12, 14, 0,  3,  4,  5,  9,  11, 0,  3,  7,  9,  11, 12, 14, 0,  4,  9,
          10, 12, 0,  1,  5,  6,  8,  9,  10, 13, 4,  6,  9,  11, 2,  4,  6,  7,  8,  14};
}

template <typename ValueType>
std::vector<ValueType> golden_values()
{
  return {0.1887315, 0.1368716, 0.1613618, 0.4839568, 0.4151455, 0.9012728, 0.7806592, 0.6713938,
          0.4281978, 0.5782541, 0.4538292, 0.0404218, 0.1264595, 0.3271811, 0.0725956, 0.0031886,
          0.6529871, 0.7025284, 0.1509823, 0.2051058, 0.6743041, 0.7965932, 0.5618279, 0.3343601,
          0.8330396, 0.6796557, 0.5684967, 0.2353606, 0.8159586, 0.3035201, 0.7145222, 0.7080131,
          0.4448774, 0.8879446, 0.2896976, 0.5502138, 0.3758983, 0.6988429, 0.6113330, 0.3722922,
          0.4468638, 0.8808504, 0.0408660, 0.6595733, 0.5429990, 0.7350115, 0.3736090, 0.9308268,
          0.8861471, 0.3089533, 0.3642297, 0.4673888, 0.8658971, 0.8499948, 0.6036414, 0.0184879,
          0.6970348, 0.1920955, 0.0650081, 0.7614104, 0.1568318, 0.3703384, 0.5869733, 0.1282817,
          0.2451350, 0.9711478, 0.5114062, 0.5030190, 0.5759066, 0.8097631, 0.3580396, 0.4049093,
          0.1618905, 0.2125666, 0.1163929, 0.4818056, 0.7425613, 0.7851025, 0.7386931, 0.7113866,
          0.5566008, 0.5112362, 0.5538779, 0.0846058, 0.3478690, 0.2713018, 0.5058042, 0.4784714,
          0.6136972, 0.5514663, 0.6401569, 0.1399612, 0.3880760, 0.6163719, 0.5653080, 0.4741685,
          0.3447469, 0.8718122, 0.8952977, 0.7043414, 0.7425023, 0.4490941, 0.3434426, 0.4157445,
          0.2318376, 0.6219735, 0.0641180, 0.8132253, 0.7734252, 0.0820893, 0.1556361, 0.1998085,
          0.2101485, 0.0405817, 0.2176879, 0.9811354, 0.2048657, 0.4163201, 0.3513670, 0.2649395};
}

template <typename ValueType>
std::vector<ValueType> golden_singular_values()
{
  return {3.819689, 2.097294, 1.926829};
}

// ============================================================================
// Test: randomized SVD on golden data
// ============================================================================

template <typename IndexType, typename ValueType>
class RandomizedSvdsTest : public ::testing::Test {
 public:
  RandomizedSvdsTest()
    : stream(resource::get_cuda_stream(handle)),
      m(20),
      n(15),
      k(3),
      nnz(120),
      d_indptr(raft::make_device_vector<IndexType, uint32_t>(handle, m + 1)),
      d_indices(raft::make_device_vector<IndexType, uint32_t>(handle, nnz)),
      d_values(raft::make_device_vector<ValueType, uint32_t>(handle, nnz)),
      expected_S(raft::make_device_vector<ValueType, uint32_t>(handle, k))
  {
  }

 protected:
  void SetUp() override
  {
    auto h_indptr  = golden_indptr<IndexType>();
    auto h_indices = golden_indices<IndexType>();
    auto h_values  = golden_values<ValueType>();
    auto h_S       = golden_singular_values<ValueType>();

    raft::update_device(d_indptr.data_handle(), h_indptr.data(), m + 1, stream);
    raft::update_device(d_indices.data_handle(), h_indices.data(), nnz, stream);
    raft::update_device(d_values.data_handle(), h_values.data(), nnz, stream);
    raft::update_device(expected_S.data_handle(), h_S.data(), k, stream);
  }

  void Run()
  {
    auto csr_structure =
      raft::make_device_compressed_structure_view<IndexType, IndexType, IndexType>(
        d_indptr.data_handle(), d_indices.data_handle(), m, n, nnz);
    auto csr_matrix = raft::make_device_csr_matrix_view<ValueType, IndexType, IndexType, IndexType>(
      d_values.data_handle(), csr_structure);

    sparse_svd_config<ValueType> config;
    config.n_components  = k;
    config.n_oversamples = 10;
    config.n_power_iters = 4;
    config.seed          = 42;

    auto S  = raft::make_device_vector<ValueType, uint32_t>(handle, k);
    auto U  = raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(handle, m, k);
    auto Vt = raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(handle, k, n);

    sparse_randomized_svd(handle, config, csr_matrix, S.view(), U.view(), Vt.view());

    // Singular values must match golden ground truth
    ASSERT_TRUE(raft::devArrMatch<ValueType>(
      S.data_handle(), expected_S.data_handle(), k, raft::CompareApprox<ValueType>(0.05), stream));

    // U must be orthogonal: U^T U ~ I_k
    auto UtU      = raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(handle, k, k);
    ValueType one = 1, zero = 0;
    raft::linalg::gemm(handle,
                       U.data_handle(),
                       m,
                       k,
                       U.data_handle(),
                       UtU.data_handle(),
                       k,
                       k,
                       CUBLAS_OP_T,
                       CUBLAS_OP_N,
                       one,
                       zero,
                       stream);

    std::vector<ValueType> I_k(k * k, 0);
    for (int i = 0; i < k; i++)
      I_k[i * k + i] = 1;
    auto I_k_dev = raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(handle, k, k);
    raft::update_device(I_k_dev.data_handle(), I_k.data(), k * k, stream);

    ASSERT_TRUE(raft::devArrMatch<ValueType>(UtU.data_handle(),
                                             I_k_dev.data_handle(),
                                             k * k,
                                             raft::CompareApprox<ValueType>(1e-4),
                                             stream));

    // Vt must have orthonormal rows: Vt Vt^T ~ I_k
    auto VVt = raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(handle, k, k);
    raft::linalg::gemm(handle,
                       Vt.data_handle(),
                       k,
                       n,
                       Vt.data_handle(),
                       VVt.data_handle(),
                       k,
                       k,
                       CUBLAS_OP_N,
                       CUBLAS_OP_T,
                       one,
                       zero,
                       stream);

    ASSERT_TRUE(raft::devArrMatch<ValueType>(VVt.data_handle(),
                                             I_k_dev.data_handle(),
                                             k * k,
                                             raft::CompareApprox<ValueType>(1e-4),
                                             stream));
  }

  raft::resources handle;
  cudaStream_t stream;
  int m, n, k, nnz;
  raft::device_vector<IndexType, uint32_t> d_indptr;
  raft::device_vector<IndexType, uint32_t> d_indices;
  raft::device_vector<ValueType, uint32_t> d_values;
  raft::device_vector<ValueType, uint32_t> expected_S;
};

using RandomizedSvdsTestF = RandomizedSvdsTest<int, float>;
TEST_F(RandomizedSvdsTestF, GoldenData) { Run(); }

using RandomizedSvdsTestD = RandomizedSvdsTest<int, double>;
TEST_F(RandomizedSvdsTestD, GoldenData) { Run(); }

// ============================================================================
// Test: reconstruction error ||A - U diag(S) Vt||_F
// ============================================================================

struct ReconstructionErrorTest : public ::testing::Test {
  raft::resources handle;
  cudaStream_t stream;
  ReconstructionErrorTest() : stream(resource::get_cuda_stream(handle)) {}

  void Run()
  {
    using ValueType = float;
    int m = 20, n = 15, k = 3, nnz = 120;

    auto h_indptr  = golden_indptr<int>();
    auto h_indices = golden_indices<int>();
    auto h_values  = golden_values<ValueType>();

    auto d_indptr  = raft::make_device_vector<int, uint32_t>(handle, m + 1);
    auto d_indices = raft::make_device_vector<int, uint32_t>(handle, nnz);
    auto d_values  = raft::make_device_vector<ValueType, uint32_t>(handle, nnz);
    raft::update_device(d_indptr.data_handle(), h_indptr.data(), m + 1, stream);
    raft::update_device(d_indices.data_handle(), h_indices.data(), nnz, stream);
    raft::update_device(d_values.data_handle(), h_values.data(), nnz, stream);

    auto csr_structure = raft::make_device_compressed_structure_view<int, int, int>(
      d_indptr.data_handle(), d_indices.data_handle(), m, n, nnz);
    auto csr_matrix = raft::make_device_csr_matrix_view<ValueType, int, int, int>(
      d_values.data_handle(), csr_structure);

    sparse_svd_config<ValueType> config;
    config.n_components  = k;
    config.n_oversamples = 10;
    config.n_power_iters = 4;
    config.seed          = 42;

    auto S  = raft::make_device_vector<ValueType, uint32_t>(handle, k);
    auto U  = raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(handle, m, k);
    auto Vt = raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(handle, k, n);
    sparse_randomized_svd(handle, config, csr_matrix, S.view(), U.view(), Vt.view());

    // Reconstruct: recon = U @ diag(S) @ Vt
    // First: US = U * S (scale columns of U by S)
    auto US = raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(handle, m, k);
    raft::copy(US.data_handle(), U.data_handle(), m * k, stream);
    std::vector<ValueType> h_S(k);
    raft::update_host(h_S.data(), S.data_handle(), k, stream);
    resource::sync_stream(handle);
    for (int j = 0; j < k; j++) {
      auto cublas_h = raft::resource::get_cublas_handle(handle);
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasscal(
        cublas_h, m, &h_S[j], US.data_handle() + j * m, 1, stream));
    }

    // recon = US @ Vt
    auto recon    = raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(handle, m, n);
    ValueType one = 1, zero = 0;
    raft::linalg::gemm(handle,
                       US.data_handle(),
                       m,
                       k,
                       Vt.data_handle(),
                       recon.data_handle(),
                       m,
                       n,
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       one,
                       zero,
                       stream);

    // Build dense A on host
    std::vector<ValueType> h_dense(m * n, 0);
    int vi = 0;
    for (int i = 0; i < m; i++)
      for (int jj = h_indptr[i]; jj < h_indptr[i + 1]; jj++)
        h_dense[h_indices[jj] * m + i] = h_values[vi++];

    auto dense_A = raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(handle, m, n);
    raft::update_device(dense_A.data_handle(), h_dense.data(), m * n, stream);

    // error = A - recon
    // Compute ||A - recon||_F / ||A||_F on host
    std::vector<ValueType> h_recon(m * n);
    raft::update_host(h_recon.data(), recon.data_handle(), m * n, stream);
    resource::sync_stream(handle);

    double err_sq = 0, norm_sq = 0;
    for (int i = 0; i < m * n; i++) {
      double diff = h_dense[i] - h_recon[i];
      err_sq += diff * diff;
      norm_sq += (double)h_dense[i] * h_dense[i];
    }
    double rel_err = std::sqrt(err_sq / norm_sq);

    // With k=3 out of min(20,15)=15 components, relative error should be < 1.0
    ASSERT_LT(rel_err, 1.0) << "Reconstruction relative error too large: " << rel_err;
  }
};

TEST_F(ReconstructionErrorTest, RelativeError) { Run(); }

// ============================================================================
// Test: mean-centered linear operator
// Ground truth: dense SVD of explicitly centered matrix
// ============================================================================

template <typename ValueType, typename NNZType>
struct mean_centered_operator {
  detail::csr_linear_operator<ValueType, NNZType> base_op_;
  ValueType* col_means_;
  int m_, n_;

  mean_centered_operator(raft::device_csr_matrix_view<ValueType, int, int, NNZType> A,
                         ValueType* col_means,
                         int m,
                         int n)
    : base_op_(A), col_means_(col_means), m_(m), n_(n)
  {
  }

  int rows() const { return m_; }
  int cols() const { return n_; }
  auto csr_view() const { return base_op_.csr_view(); }

  // Y = (A - 1*mean^T) @ X = A@X - ones * (mean^T @ X)
  void apply(raft::resources const& handle,
             raft::device_matrix_view<const ValueType, uint32_t, raft::col_major> X,
             raft::device_matrix_view<ValueType, uint32_t, raft::col_major> Y) const
  {
    auto stream = raft::resource::get_cuda_stream(handle);
    auto cublas = raft::resource::get_cublas_handle(handle);
    int bk      = X.extent(1);
    base_op_.apply(handle, X, Y);
    rmm::device_uvector<ValueType> corr(bk, stream);
    rmm::device_uvector<ValueType> ones(m_, stream);
    std::vector<ValueType> h(m_, 1);
    raft::update_device(ones.data(), h.data(), m_, stream);
    ValueType a1 = 1, a0 = 0, am1 = -1;
    RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemv(cublas,
                                                     CUBLAS_OP_T,
                                                     n_,
                                                     bk,
                                                     &a1,
                                                     X.data_handle(),
                                                     n_,
                                                     col_means_,
                                                     1,
                                                     &a0,
                                                     corr.data(),
                                                     1,
                                                     stream));
    RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemm(cublas,
                                                     CUBLAS_OP_N,
                                                     CUBLAS_OP_N,
                                                     m_,
                                                     bk,
                                                     1,
                                                     &am1,
                                                     ones.data(),
                                                     m_,
                                                     corr.data(),
                                                     1,
                                                     &a1,
                                                     Y.data_handle(),
                                                     m_,
                                                     stream));
  }

  // Z = (A - 1*mean^T)^T @ X = A^T@X - mean * (1^T @ X)
  void apply_transpose(raft::resources const& handle,
                       raft::device_matrix_view<const ValueType, uint32_t, raft::col_major> X,
                       raft::device_matrix_view<ValueType, uint32_t, raft::col_major> Z) const
  {
    auto stream = raft::resource::get_cuda_stream(handle);
    auto cublas = raft::resource::get_cublas_handle(handle);
    int bk      = X.extent(1);
    base_op_.apply_transpose(handle, X, Z);
    rmm::device_uvector<ValueType> sums(bk, stream);
    rmm::device_uvector<ValueType> ones(m_, stream);
    std::vector<ValueType> h(m_, 1);
    raft::update_device(ones.data(), h.data(), m_, stream);
    ValueType a1 = 1, a0 = 0, am1 = -1;
    RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemv(cublas,
                                                     CUBLAS_OP_T,
                                                     m_,
                                                     bk,
                                                     &a1,
                                                     X.data_handle(),
                                                     m_,
                                                     ones.data(),
                                                     1,
                                                     &a0,
                                                     sums.data(),
                                                     1,
                                                     stream));
    RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemm(cublas,
                                                     CUBLAS_OP_N,
                                                     CUBLAS_OP_N,
                                                     n_,
                                                     bk,
                                                     1,
                                                     &am1,
                                                     col_means_,
                                                     n_,
                                                     sums.data(),
                                                     1,
                                                     &a1,
                                                     Z.data_handle(),
                                                     n_,
                                                     stream));
  }
};

class MeanCenteredOperatorTest : public ::testing::Test {
 public:
  MeanCenteredOperatorTest() : stream(resource::get_cuda_stream(handle)) {}

 protected:
  void Run()
  {
    using ValueType = float;
    using IndexType = int;

    int m = 20, n = 15, k = 3, nnz = 120;

    auto h_indptr  = golden_indptr<IndexType>();
    auto h_indices = golden_indices<IndexType>();
    auto h_values  = golden_values<ValueType>();

    auto d_indptr  = raft::make_device_vector<IndexType, uint32_t>(handle, m + 1);
    auto d_indices = raft::make_device_vector<IndexType, uint32_t>(handle, nnz);
    auto d_values  = raft::make_device_vector<ValueType, uint32_t>(handle, nnz);
    raft::update_device(d_indptr.data_handle(), h_indptr.data(), m + 1, stream);
    raft::update_device(d_indices.data_handle(), h_indices.data(), nnz, stream);
    raft::update_device(d_values.data_handle(), h_values.data(), nnz, stream);

    auto csr_structure =
      raft::make_device_compressed_structure_view<IndexType, IndexType, IndexType>(
        d_indptr.data_handle(), d_indices.data_handle(), m, n, nnz);
    auto csr_matrix = raft::make_device_csr_matrix_view<ValueType, IndexType, IndexType, IndexType>(
      d_values.data_handle(), csr_structure);

    // Reconstruct dense matrix on host to compute column means
    std::vector<ValueType> dense(m * n, 0);
    int vi = 0;
    for (int i = 0; i < m; i++) {
      for (int jj = h_indptr[i]; jj < h_indptr[i + 1]; jj++) {
        dense[h_indices[jj] * m + i] = h_values[vi++];
      }
    }

    std::vector<ValueType> h_means(n, 0);
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < m; i++)
        h_means[j] += dense[j * m + i];
      h_means[j] /= m;
    }
    auto d_means = raft::make_device_vector<ValueType, uint32_t>(handle, n);
    raft::update_device(d_means.data_handle(), h_means.data(), n, stream);

    // Ground truth: dense SVD of (A - 1*mean^T)
    std::vector<ValueType> centered(m * n);
    for (int j = 0; j < n; j++)
      for (int i = 0; i < m; i++)
        centered[j * m + i] = dense[j * m + i] - h_means[j];

    auto centered_dev =
      raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(handle, m, n);
    raft::update_device(centered_dev.data_handle(), centered.data(), m * n, stream);
    auto ref_S  = raft::make_device_vector<ValueType, uint32_t>(handle, n);
    auto ref_U  = raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(handle, m, n);
    auto ref_Vt = raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(handle, n, n);
    raft::linalg::svdQR(handle,
                        centered_dev.data_handle(),
                        m,
                        n,
                        ref_S.data_handle(),
                        ref_U.data_handle(),
                        ref_Vt.data_handle(),
                        false,
                        true,
                        true,
                        stream);

    // Operator-based SVD
    mean_centered_operator<ValueType, IndexType> op(csr_matrix, d_means.data_handle(), m, n);

    sparse_svd_config<ValueType> config;
    config.n_components  = k;
    config.n_oversamples = 10;
    config.n_power_iters = 4;
    config.seed          = 42;

    auto S  = raft::make_device_vector<ValueType, uint32_t>(handle, k);
    auto U  = raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(handle, m, k);
    auto Vt = raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(handle, k, n);
    sparse_randomized_svd(handle, config, op, S.view(), U.view(), Vt.view());

    // Singular values must match dense centered ground truth
    ASSERT_TRUE(raft::devArrMatch<ValueType>(
      S.data_handle(), ref_S.data_handle(), k, raft::CompareApprox<ValueType>(0.1), stream));
  }

  raft::resources handle;
  cudaStream_t stream;
};

TEST_F(MeanCenteredOperatorTest, OperatorVsDenseGroundTruth) { Run(); }

}  // namespace raft::sparse::solver
