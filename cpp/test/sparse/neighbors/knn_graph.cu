/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include "../../test_utils.cuh"
#include <gtest/gtest.h>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <raft/sparse/coo.hpp>
#include <raft/sparse/neighbors/knn_graph.cuh>

#include <iostream>

namespace raft {
namespace sparse {

template <typename value_idx, typename value_t>
RAFT_KERNEL assert_symmetry(
  value_idx* rows, value_idx* cols, value_t* vals, value_idx nnz, value_idx* sum)
{
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
::std::ostream& operator<<(::std::ostream& os, const KNNGraphInputs<value_idx, value_t>& dims)
{
  return os;
}

template <typename value_idx, typename value_t>
class KNNGraphTest : public ::testing::TestWithParam<KNNGraphInputs<value_idx, value_t>> {
 public:
  KNNGraphTest()
    : params(::testing::TestWithParam<KNNGraphInputs<value_idx, value_t>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      X(0, stream)
  {
    X.resize(params.X.size(), stream);
  }

 protected:
  void SetUp() override
  {
    out = new raft::sparse::COO<value_t, value_idx>(stream);

    update_device(X.data(), params.X.data(), params.X.size(), stream);

    raft::sparse::neighbors::knn_graph(
      handle, X.data(), params.m, params.n, raft::distance::DistanceType::L2Unexpanded, *out);

    rmm::device_scalar<value_idx> sum(stream);
    sum.set_value_to_zero_async(stream);

    /**
     * Assert the knn graph is symmetric
     */
    assert_symmetry<<<raft::ceildiv(out->nnz, 256), 256, 0, stream>>>(
      out->rows(), out->cols(), out->vals(), out->nnz, sum.data());

    sum_h = sum.value(stream);
    resource::sync_stream(handle, stream);
  }

  void TearDown() override { delete out; }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  // input data
  raft::sparse::COO<value_t, value_idx>* out;

  rmm::device_uvector<value_t> X;

  value_idx sum_h;

  KNNGraphInputs<value_idx, value_t> params;
};

const std::vector<KNNGraphInputs<int, float>> knn_graph_inputs_fint = {
  // Test n_clusters == n_points
  {4, 2, {0, 100, 0.01, 0.02, 5000, 10000, -5, -2}, 2}};

typedef KNNGraphTest<int, float> KNNGraphTestF_int;
TEST_P(KNNGraphTestF_int, Result)
{
  // nnz should not be larger than twice m * k
  ASSERT_TRUE(out->nnz <= (params.m * params.k * 2));
  ASSERT_TRUE(sum_h == 0);
}

INSTANTIATE_TEST_CASE_P(KNNGraphTest,
                        KNNGraphTestF_int,
                        ::testing::ValuesIn(knn_graph_inputs_fint));

}  // namespace sparse
}  // namespace raft
