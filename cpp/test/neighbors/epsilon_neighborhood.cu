/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/make_blobs.cuh>
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

const std::vector<EpsInputs<float, int>> inputsfi = {
  {15000, 16, 5, 1, 2.f},
  {14000, 16, 5, 1, 2.f},
  {15000, 17, 5, 1, 2.f},
  {14000, 17, 5, 1, 2.f},
  {15000, 18, 5, 1, 2.f},
  {14000, 18, 5, 1, 2.f},
  {15000, 32, 5, 1, 2.f},
  {14000, 32, 5, 1, 2.f},
  {20000, 10000, 10, 1, 2.f},
  {20000, 10000, 10, 2, 2.f},
};
typedef EpsNeighTest<float, int> EpsNeighTestFI;
TEST_P(EpsNeighTestFI, Result)
{
  for (int i = 0; i < param.n_batches; ++i) {
    RAFT_CUDA_TRY(cudaMemsetAsync(adj.data(), 0, sizeof(bool) * param.n_row * batchSize, stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(vd.data(), 0, sizeof(int) * (batchSize + 1), stream));

    auto adj_view = make_device_matrix_view<bool, int>(adj.data(), param.n_row, batchSize);
    auto vd_view  = make_device_vector_view<int, int>(vd.data(), batchSize + 1);
    auto x_view   = make_device_matrix_view<float, int>(data.data(), param.n_row, param.n_col);
    auto y_view   = make_device_matrix_view<float, int>(
      data.data() + (i * batchSize * param.n_col), batchSize, param.n_col);

    eps_neighbors_l2sq<float, int, int>(
      handle, x_view, y_view, adj_view, vd_view, param.eps * param.eps);

    ASSERT_TRUE(raft::devArrMatch(
      param.n_row / param.n_centers, vd.data(), batchSize, raft::Compare<int>(), stream));
  }
}
INSTANTIATE_TEST_CASE_P(EpsNeighTests, EpsNeighTestFI, ::testing::ValuesIn(inputsfi));

};  // namespace knn
};  // namespace spatial
};  // namespace raft
