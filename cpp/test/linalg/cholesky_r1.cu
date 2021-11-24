/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>
#include <raft/cudart_utils.h>
#include <raft/linalg/cusolver_wrappers.h>
#include <raft/handle.hpp>
#include <raft/linalg/cholesky_r1_update.cuh>
#include <raft/mr/device/allocator.hpp>
#include <raft/mr/device/buffer.hpp>
#include <sstream>
#include <vector>
#include "../test_utils.h"
namespace raft {
namespace linalg {

template <typename math_t>
class CholeskyR1Test : public ::testing::Test {
 protected:
  CholeskyR1Test()
    : allocator(handle.get_device_allocator()),
      G(allocator, handle.get_stream(), n_rows * n_rows),
      L(allocator, handle.get_stream(), n_rows * n_rows),
      L_exp(allocator, handle.get_stream(), n_rows * n_rows),
      devInfo(allocator, handle.get_stream(), 1),
      workspace(allocator, handle.get_stream()) {
    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.set_stream(stream);
    raft::update_device(G.data(), G_host, n_rows * n_rows, stream);

    // Allocate workspace
    solver_handle = handle.get_cusolver_dn_handle();
    CUSOLVER_CHECK(raft::linalg::cusolverDnpotrf_bufferSize(
      solver_handle, CUBLAS_FILL_MODE_LOWER, n_rows, L.data(), n_rows, &Lwork));
    int n_bytes = 0;
    // Initializing in CUBLAS_FILL_MODE_LOWER, because that has larger workspace
    // requirements.
    raft::linalg::choleskyRank1Update(handle, L.data(), n_rows, n_rows, nullptr,
                                      &n_bytes, CUBLAS_FILL_MODE_LOWER, stream);
    Lwork = std::max(Lwork * sizeof(math_t), (size_t)n_bytes);
    workspace.resize(Lwork, stream);
  }

  void TearDown() override { CUDA_CHECK(cudaStreamDestroy(stream)); }

  void testR1Update() {
    int n = n_rows * n_rows;
    std::vector<cublasFillMode_t> fillmode{CUBLAS_FILL_MODE_LOWER,
                                           CUBLAS_FILL_MODE_UPPER};
    for (auto uplo : fillmode) {
      raft::copy(L.data(), G.data(), n, stream);
      for (int rank = 1; rank <= n_rows; rank++) {
        std::stringstream ss;
        ss << "Rank " << rank
           << ((uplo == CUBLAS_FILL_MODE_LOWER) ? ", lower" : ", upper");
        SCOPED_TRACE(ss.str());

        // Expected solution using Cholesky factorization from scratch
        raft::copy(L_exp.data(), G.data(), n, stream);
        CUSOLVER_CHECK(raft::linalg::cusolverDnpotrf(
          solver_handle, uplo, rank, L_exp.data(), n_rows,
          (math_t*)workspace.data(), Lwork, devInfo.data(), stream));

        // Incremental Cholesky factorization using rank one updates.
        raft::linalg::choleskyRank1Update(handle, L.data(), rank, n_rows,
                                          workspace.data(), &Lwork, uplo,
                                          stream);

        ASSERT_TRUE(raft::devArrMatch(L_exp.data(), L.data(), n_rows * rank,
                                      raft::CompareApprox<math_t>(3e-3)));
      }
    }
  }

  void testR1Error() {
    raft::update_device(G.data(), G2_host, 4, stream);
    std::vector<cublasFillMode_t> fillmode{CUBLAS_FILL_MODE_LOWER,
                                           CUBLAS_FILL_MODE_UPPER};
    for (auto uplo : fillmode) {
      raft::copy(L.data(), G.data(), 4, stream);
      ASSERT_NO_THROW(raft::linalg::choleskyRank1Update(
        handle, L.data(), 1, 2, workspace.data(), &Lwork, uplo, stream));
      ASSERT_THROW(
        raft::linalg::choleskyRank1Update(
          handle, L.data(), 2, 2, workspace.data(), &Lwork, uplo, stream),
        raft::exception);

      math_t eps = std::numeric_limits<math_t>::epsilon();
      ASSERT_NO_THROW(raft::linalg::choleskyRank1Update(
        handle, L.data(), 2, 2, workspace.data(), &Lwork, uplo, stream, eps));
    }
  }

  raft::handle_t handle;
  std::shared_ptr<raft::mr::device::allocator> allocator;
  cusolverDnHandle_t solver_handle;
  cudaStream_t stream;

  int n_rows = 4;
  int Lwork;
  math_t G_host[16] =  // clang-format off
    {107.,  1393.,  1141.,  91.,
     1393., 21132., 15689., 9539.,
     1141., 15689., 13103., 2889.,
     91.,   9539.,  2889.,  23649.};
                       // clang-format on

  math_t G2_host[4] = {3, 4, 2, 1};

  raft::mr::device::buffer<int> devInfo;
  raft::mr::device::buffer<math_t> G;
  raft::mr::device::buffer<math_t> L_exp;
  raft::mr::device::buffer<math_t> L;
  raft::mr::device::buffer<char> workspace;
};

typedef ::testing::Types<float, double> FloatTypes;

TYPED_TEST_CASE(CholeskyR1Test, FloatTypes);

TYPED_TEST(CholeskyR1Test, update) { this->testR1Update(); }
TYPED_TEST(CholeskyR1Test, throwError) { this->testR1Error(); }

};  // namespace linalg
};  // namespace raft
