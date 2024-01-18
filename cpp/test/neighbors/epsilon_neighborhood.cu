/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include "../test_utils.cuh"
#include <gtest/gtest.h>
#include <memory>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/neighbors/ball_cover.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/spatial/knn/epsilon_neighborhood.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>

namespace raft {
namespace spatial {
namespace knn {
template <typename T, typename IdxT>
struct EpsInputs {
  IdxT n_row, n_col, n_centers, n_batches;
  T eps;
};

template <typename T, typename IdxT>
::std::ostream& operator<<(::std::ostream& os, const EpsInputs<T, IdxT>& p)
{
  return os;
}

template <typename T, typename IdxT>
class EpsNeighTest : public ::testing::TestWithParam<EpsInputs<T, IdxT>> {
 protected:
  EpsNeighTest()
    : data(0, resource::get_cuda_stream(handle)),
      adj(0, resource::get_cuda_stream(handle)),
      labels(0, resource::get_cuda_stream(handle)),
      vd(0, resource::get_cuda_stream(handle))
  {
  }

  void SetUp() override
  {
    auto stream = resource::get_cuda_stream(handle);
    param       = ::testing::TestWithParam<EpsInputs<T, IdxT>>::GetParam();
    data.resize(param.n_row * param.n_col, stream);
    labels.resize(param.n_row, stream);
    batchSize = param.n_row / param.n_batches;
    adj.resize(param.n_row * batchSize, stream);
    vd.resize(batchSize + 1, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(vd.data(), 0, vd.size() * sizeof(IdxT), stream));
    random::make_blobs<T, IdxT>(data.data(),
                                labels.data(),
                                param.n_row,
                                param.n_col,
                                param.n_centers,
                                stream,
                                true,
                                nullptr,
                                nullptr,
                                T(0.01),
                                false);
  }

  const raft::resources handle;
  EpsInputs<T, IdxT> param;
  cudaStream_t stream = 0;
  rmm::device_uvector<T> data;
  rmm::device_uvector<bool> adj;
  rmm::device_uvector<IdxT> labels, vd;
  IdxT batchSize;
};  // class EpsNeighTest

const std::vector<EpsInputs<float, int64_t>> inputsfi = {
  {100, 16, 5, 2, 2.f},
  {1500, 16, 5, 3, 2.f},
  {15000, 16, 5, 1, 2.f},
  {15000, 3, 5, 1, 2.f},
  {14000, 16, 5, 1, 2.f},
  {15000, 17, 5, 1, 2.f},
  {14000, 17, 5, 1, 2.f},
  {15000, 18, 5, 1, 2.f},
  {14000, 18, 5, 1, 2.f},
  {15000, 32, 5, 1, 2.f},
  {14000, 32, 5, 1, 2.f},
  {14000, 32, 5, 10, 2.f},
  {20000, 10000, 10, 1, 2.f},
  {20000, 10000, 10, 2, 2.f},
};

typedef EpsNeighTest<float, int64_t> EpsNeighTestFI;

TEST_P(EpsNeighTestFI, ResultBruteForce)
{
  for (int i = 0; i < param.n_batches; ++i) {
    RAFT_CUDA_TRY(cudaMemsetAsync(adj.data(), 0, sizeof(bool) * param.n_row * batchSize, stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(vd.data(), 0, sizeof(int64_t) * (batchSize + 1), stream));

    auto adj_view = make_device_matrix_view<bool, int64_t>(adj.data(), batchSize, param.n_row);
    auto vd_view  = make_device_vector_view<int64_t, int64_t>(vd.data(), batchSize + 1);
    auto x_view   = make_device_matrix_view<float, int64_t>(
      data.data() + (i * batchSize * param.n_col), batchSize, param.n_col);
    auto y_view = make_device_matrix_view<float, int64_t>(data.data(), param.n_row, param.n_col);

    eps_neighbors_l2sq<float, int64_t, int64_t>(
      handle, x_view, y_view, adj_view, vd_view, param.eps * param.eps);

    ASSERT_TRUE(raft::devArrMatch(
      param.n_row / param.n_centers, vd.data(), batchSize, raft::Compare<int64_t>(), stream));
  }
}

INSTANTIATE_TEST_CASE_P(EpsNeighTests, EpsNeighTestFI, ::testing::ValuesIn(inputsfi));

// rbc examples take fewer points as correctness checks are very costly
const std::vector<EpsInputs<float, int64_t>> inputsfi_rbc = {
  {100, 16, 5, 2, 2.f},
  {1500, 16, 5, 3, 2.f},
  {1500, 16, 5, 1, 2.f},
  {1500, 3, 5, 1, 2.f},
  {1400, 16, 5, 1, 2.f},
  {1500, 17, 5, 1, 2.f},
  {1400, 17, 5, 1, 2.f},
  {1500, 18, 5, 1, 2.f},
  {1400, 18, 5, 1, 2.f},
  {1500, 32, 5, 1, 2.f},
  {1400, 32, 5, 1, 2.f},
  {1400, 32, 5, 10, 2.f},
  {2000, 1000, 10, 1, 2.f},
  {2000, 1000, 10, 2, 2.f},
};

typedef EpsNeighTest<float, int64_t> EpsNeighRbcTestFI;

TEST_P(EpsNeighRbcTestFI, DenseRbc)
{
  rmm::device_uvector<bool> adj_baseline(param.n_row * batchSize,
                                         resource::get_cuda_stream(handle));

  raft::neighbors::ball_cover::BallCoverIndex<int64_t, float, int64_t, int64_t> rbc_index(
    handle, data.data(), param.n_row, param.n_col, raft::distance::DistanceType::L2SqrtUnexpanded);
  raft::neighbors::ball_cover::build_index(handle, rbc_index);

  for (int i = 0; i < param.n_batches; ++i) {
    // invalidate
    RAFT_CUDA_TRY(cudaMemsetAsync(adj.data(), 1, sizeof(bool) * param.n_row * batchSize, stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(vd.data(), 1, sizeof(int64_t) * (batchSize + 1), stream));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(adj_baseline.data(), 1, sizeof(bool) * param.n_row * batchSize, stream));

    float* query = data.data() + (i * batchSize * param.n_col);

    raft::neighbors::ball_cover::eps_nn<int64_t, float, int64_t, int64_t>(
      handle,
      rbc_index,
      make_device_matrix_view<bool, int64_t>(adj.data(), batchSize, param.n_row),
      make_device_vector_view<int64_t, int64_t>(vd.data(), batchSize + 1),
      make_device_matrix_view<float, int64_t>(query, batchSize, param.n_col),
      param.eps * param.eps);

    ASSERT_TRUE(raft::devArrMatch(
      param.n_row / param.n_centers, vd.data(), batchSize, raft::Compare<int64_t>(), stream));

    // compute baseline via brute force + compare
    epsUnexpL2SqNeighborhood<float, int64_t>(adj_baseline.data(),
                                             nullptr,
                                             query,
                                             data.data(),
                                             batchSize,
                                             param.n_row,
                                             param.n_col,
                                             param.eps * param.eps,
                                             stream);

    ASSERT_TRUE(raft::devArrMatch(
      adj_baseline.data(), adj.data(), batchSize, param.n_row, raft::Compare<bool>(), stream));
  }
}

template <typename T>
testing::AssertionResult assertCsrEqualUnordered(
  T* ia_exp, T* ja_exp, T* ia_act, T* ja_act, size_t rows, size_t cols, cudaStream_t stream)
{
  std::unique_ptr<T[]> ia_exp_h(new T[rows + 1]);
  std::unique_ptr<T[]> ia_act_h(new T[rows + 1]);
  raft::update_host<T>(ia_exp_h.get(), ia_exp, rows + 1, stream);
  raft::update_host<T>(ia_act_h.get(), ia_act, rows + 1, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  size_t nnz = ia_exp_h.get()[rows];
  std::unique_ptr<T[]> ja_exp_h(new T[nnz]);
  std::unique_ptr<T[]> ja_act_h(new T[nnz]);
  raft::update_host<T>(ja_exp_h.get(), ja_exp, nnz, stream);
  raft::update_host<T>(ja_act_h.get(), ja_act, nnz, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  for (size_t i(0); i < rows; ++i) {
    auto row_start = ia_exp_h.get()[i];
    auto row_end   = ia_exp_h.get()[i + 1];

    // sort ja's
    std::sort(ja_exp_h.get() + row_start, ja_exp_h.get() + row_end);
    std::sort(ja_act_h.get() + row_start, ja_act_h.get() + row_end);

    for (size_t idx(row_start); idx < (size_t)row_end; ++idx) {
      auto exp = ja_exp_h.get()[idx];
      auto act = ja_act_h.get()[idx];
      if (exp != act) {
        return testing::AssertionFailure()
               << "actual=" << act << " != expected=" << exp << " @" << i << "," << idx;
      }
    }
  }
  return testing::AssertionSuccess();
}

TEST_P(EpsNeighRbcTestFI, SparseRbc)
{
  rmm::device_uvector<int64_t> adj_ia(batchSize + 1, resource::get_cuda_stream(handle));
  rmm::device_uvector<int64_t> adj_ja(param.n_row * batchSize, resource::get_cuda_stream(handle));

  rmm::device_uvector<int64_t> vd_baseline(batchSize + 1, resource::get_cuda_stream(handle));
  rmm::device_uvector<int64_t> adj_ia_baseline(batchSize + 1, resource::get_cuda_stream(handle));
  rmm::device_uvector<int64_t> adj_ja_baseline(param.n_row * batchSize,
                                               resource::get_cuda_stream(handle));

  raft::neighbors::ball_cover::BallCoverIndex<int64_t, float, int64_t, int64_t> rbc_index(
    handle, data.data(), param.n_row, param.n_col, raft::distance::DistanceType::L2SqrtUnexpanded);
  raft::neighbors::ball_cover::build_index(handle, rbc_index);

  for (int i = 0; i < param.n_batches; ++i) {
    // reset full array -- that way we can compare the full size
    RAFT_CUDA_TRY(
      cudaMemsetAsync(adj_ja.data(), 0, sizeof(int64_t) * param.n_row * batchSize, stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(
      adj_ja_baseline.data(), 0, sizeof(int64_t) * param.n_row * batchSize, stream));

    float* query = data.data() + (i * batchSize * param.n_col);

    // compute dense baseline and convert adj to csr
    {
      raft::neighbors::ball_cover::eps_nn<int64_t, float, int64_t, int64_t>(
        handle,
        rbc_index,
        make_device_matrix_view<bool, int64_t>(adj.data(), batchSize, param.n_row),
        make_device_vector_view<int64_t, int64_t>(vd_baseline.data(), batchSize + 1),
        make_device_matrix_view<float, int64_t>(query, batchSize, param.n_col),
        param.eps * param.eps);
      thrust::exclusive_scan(resource::get_thrust_policy(handle),
                             vd_baseline.data(),
                             vd_baseline.data() + batchSize + 1,
                             adj_ia_baseline.data());
      raft::sparse::convert::adj_to_csr(handle,
                                        adj.data(),
                                        adj_ia_baseline.data(),
                                        batchSize,
                                        param.n_row,
                                        labels.data(),
                                        adj_ja_baseline.data());
    }

    // exact computation with 2 passes
    {
      raft::neighbors::ball_cover::eps_nn<int64_t, float, int64_t, int64_t>(
        handle,
        rbc_index,
        make_device_vector_view<int64_t, int64_t>(adj_ia.data(), batchSize + 1),
        make_device_vector_view<int64_t, int64_t>(nullptr, 0),
        make_device_vector_view<int64_t, int64_t>(vd.data(), batchSize + 1),
        make_device_matrix_view<float, int64_t>(query, batchSize, param.n_col),
        param.eps * param.eps);
      raft::neighbors::ball_cover::eps_nn<int64_t, float, int64_t, int64_t>(
        handle,
        rbc_index,
        make_device_vector_view<int64_t, int64_t>(adj_ia.data(), batchSize + 1),
        make_device_vector_view<int64_t, int64_t>(adj_ja.data(), batchSize * param.n_row),
        make_device_vector_view<int64_t, int64_t>(vd.data(), batchSize + 1),
        make_device_matrix_view<float, int64_t>(query, batchSize, param.n_col),
        param.eps * param.eps);
      ASSERT_TRUE(raft::devArrMatch(
        adj_ia_baseline.data(), adj_ia.data(), batchSize + 1, raft::Compare<int64_t>(), stream));
      ASSERT_TRUE(assertCsrEqualUnordered(adj_ia_baseline.data(),
                                          adj_ja_baseline.data(),
                                          adj_ia.data(),
                                          adj_ja.data(),
                                          batchSize,
                                          param.n_row,
                                          stream));
    }
  }
}

TEST_P(EpsNeighRbcTestFI, SparseRbcMaxK)
{
  rmm::device_uvector<int64_t> adj_ia(batchSize + 1, resource::get_cuda_stream(handle));
  rmm::device_uvector<int64_t> adj_ja(param.n_row * batchSize, resource::get_cuda_stream(handle));

  rmm::device_uvector<int64_t> vd_baseline(batchSize + 1, resource::get_cuda_stream(handle));
  rmm::device_uvector<int64_t> adj_ia_baseline(batchSize + 1, resource::get_cuda_stream(handle));
  rmm::device_uvector<int64_t> adj_ja_baseline(param.n_row * batchSize,
                                               resource::get_cuda_stream(handle));

  raft::neighbors::ball_cover::BallCoverIndex<int64_t, float, int64_t, int64_t> rbc_index(
    handle, data.data(), param.n_row, param.n_col, raft::distance::DistanceType::L2SqrtUnexpanded);
  raft::neighbors::ball_cover::build_index(handle, rbc_index);

  int64_t expected_max_k = param.n_row / param.n_centers;

  for (int i = 0; i < param.n_batches; ++i) {
    // reset full array -- that way we can compare the full size
    RAFT_CUDA_TRY(
      cudaMemsetAsync(adj_ja.data(), 0, sizeof(int64_t) * param.n_row * batchSize, stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(
      adj_ja_baseline.data(), 0, sizeof(int64_t) * param.n_row * batchSize, stream));

    float* query = data.data() + (i * batchSize * param.n_col);

    // compute dense baseline and convert adj to csr
    {
      raft::neighbors::ball_cover::eps_nn<int64_t, float, int64_t, int64_t>(
        handle,
        rbc_index,
        make_device_matrix_view<bool, int64_t>(adj.data(), batchSize, param.n_row),
        make_device_vector_view<int64_t, int64_t>(vd_baseline.data(), batchSize + 1),
        make_device_matrix_view<float, int64_t>(query, batchSize, param.n_col),
        param.eps * param.eps);
      thrust::exclusive_scan(resource::get_thrust_policy(handle),
                             vd_baseline.data(),
                             vd_baseline.data() + batchSize + 1,
                             adj_ia_baseline.data());
      raft::sparse::convert::adj_to_csr(handle,
                                        adj.data(),
                                        adj_ia_baseline.data(),
                                        batchSize,
                                        param.n_row,
                                        labels.data(),
                                        adj_ja_baseline.data());
    }

    // exact computation with 1 pass
    {
      int64_t max_k = expected_max_k;
      raft::neighbors::ball_cover::eps_nn<int64_t, float, int64_t, int64_t>(
        handle,
        rbc_index,
        make_device_vector_view<int64_t, int64_t>(adj_ia.data(), batchSize + 1),
        make_device_vector_view<int64_t, int64_t>(adj_ja.data(), batchSize * param.n_row),
        make_device_vector_view<int64_t, int64_t>(vd.data(), batchSize + 1),
        make_device_matrix_view<float, int64_t>(query, batchSize, param.n_col),
        param.eps * param.eps,
        make_host_scalar_view<int64_t, int64_t>(&max_k));
      ASSERT_TRUE(raft::devArrMatch(
        adj_ia_baseline.data(), adj_ia.data(), batchSize + 1, raft::Compare<int64_t>(), stream));
      ASSERT_TRUE(assertCsrEqualUnordered(adj_ia_baseline.data(),
                                          adj_ja_baseline.data(),
                                          adj_ia.data(),
                                          adj_ja.data(),
                                          batchSize,
                                          param.n_row,
                                          stream));
      ASSERT_TRUE(raft::devArrMatch(
        vd_baseline.data(), vd.data(), batchSize + 1, raft::Compare<int64_t>(), stream));
      ASSERT_TRUE(max_k == expected_max_k);
    }

    // k-limited computation with 1 pass
    {
      int64_t max_k = expected_max_k / 2;
      raft::neighbors::ball_cover::eps_nn<int64_t, float, int64_t, int64_t>(
        handle,
        rbc_index,
        make_device_vector_view<int64_t, int64_t>(adj_ia.data(), batchSize + 1),
        make_device_vector_view<int64_t, int64_t>(adj_ja.data(), batchSize * param.n_row),
        make_device_vector_view<int64_t, int64_t>(vd.data(), batchSize + 1),
        make_device_matrix_view<float, int64_t>(query, batchSize, param.n_col),
        param.eps * param.eps,
        make_host_scalar_view<int64_t, int64_t>(&max_k));
      ASSERT_TRUE(max_k == expected_max_k);
      ASSERT_TRUE(raft::devArrMatch(
        expected_max_k / 2, vd.data(), batchSize, raft::Compare<int64_t>(), stream));
      ASSERT_TRUE(raft::devArrMatch(expected_max_k / 2 * batchSize,
                                    vd.data() + batchSize,
                                    1,
                                    raft::Compare<int64_t>(),
                                    stream));
    }
  }
}

INSTANTIATE_TEST_CASE_P(EpsNeighTests, EpsNeighRbcTestFI, ::testing::ValuesIn(inputsfi_rbc));

};  // namespace knn
};  // namespace spatial
};  // namespace raft
