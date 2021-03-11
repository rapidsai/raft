/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <raft/random/rng.cuh>
#include "../test_utils.h"
#include <rmm/device_uvector.hpp>

#include <raft/sparse/coo.cuh>
#include <raft/sparse/selection/knn_graph.cuh>

#include <iostream>

namespace raft {
namespace sparse {

template <typename value_idx, typename value_t>
__global__ void assert_symmetry(value_idx *rows, value_idx *cols, value_t *vals,
                                value_idx nnz, value_idx *sum) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= nnz) return;

  atomicAdd(sum, rows[tid]);
  atomicAdd(sum, -1 * cols[tid]);
}

template <typename value_idx, typename value_t>
struct KNNGraphInputs {
  value_idx m;
  value_idx n;

  std::vector<value_t> X;

  int k = 2;
};

template <typename value_idx, typename value_t>
::std::ostream &operator<<(::std::ostream &os,
                           const KNNGraphInputs<value_idx, value_t> &dims) {
  return os;
}

template <typename value_idx, typename value_t>
class KNNGraphTest
  : public ::testing::TestWithParam<KNNGraphInputs<value_idx, value_t>> {
  void SetUp() override {
    params =
      ::testing::TestWithParam<KNNGraphInputs<value_idx, value_t>>::GetParam();

    raft::handle_t handle;

    auto alloc = handle.get_device_allocator();
    stream = handle.get_stream();

    out = new raft::sparse::COO<value_t, value_idx>(alloc, stream);

    allocate(X, params.X.size());

    update_device(X, params.X.data(), params.X.size(), stream);

    raft::sparse::selection::knn_graph(
      handle, X, params.m, params.n, raft::distance::DistanceType::L2Unexpanded,
      *out);

    rmm::device_uvector<value_idx> sum(1, stream);

    CUDA_CHECK(cudaMemsetAsync(sum.data(), 0, 1 * sizeof(value_idx), stream));

    /**
     * Assert the knn graph is symmetric
     */
    assert_symmetry<<<raft::ceildiv(out->nnz, 256), 256, 0, stream>>>(
      out->rows(), out->cols(), out->vals(), out->nnz, sum.data());

    raft::update_host(&sum_h, sum.data(), 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(X));

    delete out;
  }

 protected:
  cudaStream_t stream;

  // input data
  raft::sparse::COO<value_t, value_idx> *out;

  value_t *X;

  value_idx sum_h;

  KNNGraphInputs<value_idx, value_t> params;
};

const std::vector<KNNGraphInputs<int, float>> knn_graph_inputs_fint = {
  // Test n_clusters == n_points
  {4, 2, {0, 100, 0.01, 0.02, 5000, 10000, -5, -2}, 2}};

typedef KNNGraphTest<int, float> KNNGraphTestF_int;
TEST_P(KNNGraphTestF_int, Result) {
  // nnz should not be larger than twice m * k
  ASSERT_TRUE(out->nnz <= (params.m * params.k * 2));
  ASSERT_TRUE(sum_h == 0);
}

INSTANTIATE_TEST_CASE_P(KNNGraphTest, KNNGraphTestF_int,
                        ::testing::ValuesIn(knn_graph_inputs_fint));

}  // namespace sparse
}  // namespace raft
