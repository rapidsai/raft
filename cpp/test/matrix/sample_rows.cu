/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/sample_rows.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace matrix {

struct inputs {
  int N;
  int dim;
  int n_samples;
};

::std::ostream& operator<<(::std::ostream& os, const inputs p)
{
  os << p.N << "#" << p.dim << "#" << p.n_samples;
  return os;
}

template <typename T>
class SampleRowsTest : public ::testing::TestWithParam<inputs> {
 public:
  SampleRowsTest()
    : params(::testing::TestWithParam<inputs>::GetParam()),
      state{137ULL},
      in(make_device_matrix<T, int64_t>(res, params.N, params.dim)),
      out(make_device_matrix<T, int64_t>(res, 0, 0))

  {
    raft::random::uniform(res, state, in.data_handle(), in.size(), T(-1.0), T(1.0));
  }

  void check()
  {
    out = raft::matrix::sample_rows<T, int64_t>(
      res, state, make_const_mdspan(in.view()), params.n_samples);
    ASSERT_TRUE(out.extent(0) == params.n_samples);
    ASSERT_TRUE(out.extent(1) == params.dim);
    // TODO(tfeher): check sampled values
    // TODO(tfeher): check host / device input
  }

 protected:
  inputs params;
  raft::resources res;
  cudaStream_t stream;
  random::RngState state;
  device_matrix<T, int64_t> out, in;
};

const std::vector<inputs> input1 = {
  {10, 1, 1}, {10, 4, 1}, {10, 4, 10}, {10, 10}, {137, 42, 59}, {10000, 128, 893}};

using SampleRowsTestInt64 = SampleRowsTest<float>;
TEST_P(SampleRowsTestInt64, SamplingTest) { check(); }
INSTANTIATE_TEST_SUITE_P(SampleRowsTests, SampleRowsTestInt64, ::testing::ValuesIn(input1));

}  // namespace matrix
}  // namespace raft
