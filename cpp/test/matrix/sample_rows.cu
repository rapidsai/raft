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
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/sample_rows.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/itertools.hpp>

#include <gtest/gtest.h>

#include <unordered_set>

namespace raft {
namespace matrix {

struct inputs {
  int N;
  int dim;
  int n_samples;
  bool host;
};

::std::ostream& operator<<(::std::ostream& os, const inputs p)
{
  os << p.N << "#" << p.dim << "#" << p.n_samples << (p.host ? "#host" : "#device");
  return os;
}

template <typename T>
class SampleRowsTest : public ::testing::TestWithParam<inputs> {
 public:
  SampleRowsTest()
    : params(::testing::TestWithParam<inputs>::GetParam()),
      stream(resource::get_cuda_stream(res)),
      state{137ULL},
      in(make_device_matrix<T, int64_t>(res, params.N, params.dim)),
      out(make_device_matrix<T, int64_t>(res, 0, 0)),
      in_h(make_host_matrix<T, int64_t>(res, params.N, params.dim)),
      out_h(make_host_matrix<T, int64_t>(res, params.n_samples, params.dim))
  {
    raft::random::uniform(res, state, in.data_handle(), in.size(), T(-1.0), T(1.0));
    for (int64_t i = 0; i < params.N; i++) {
      for (int64_t k = 0; k < params.dim; k++)
        in_h(i, k) = i * 1000 + k;
    }
    raft::copy(in.data_handle(), in_h.data_handle(), in_h.size(), stream);
  }

  void check()
  {
    if (params.host) {
      out = raft::matrix::sample_rows<T, int64_t>(
        res, state, make_const_mdspan(in_h.view()), (int64_t)params.n_samples);
    } else {
      out = raft::matrix::sample_rows<T, int64_t>(
        res, state, make_const_mdspan(in.view()), (int64_t)params.n_samples);
    }

    raft::copy(out_h.data_handle(), out.data_handle(), out.size(), stream);
    resource::sync_stream(res, stream);

    ASSERT_TRUE(out.extent(0) == params.n_samples);
    ASSERT_TRUE(out.extent(1) == params.dim);

    std::unordered_set<int> occurrence;

    for (int64_t i = 0; i < params.n_samples; ++i) {
      T val = out_h(i, 0) / 1000;
      ASSERT_TRUE(0 <= val && val < params.N)
        << "out-of-range index @i=" << i << " val=" << val << " params=" << params;
      EXPECT_TRUE(occurrence.find(val) == occurrence.end())
        << "repeated index @i=" << i << " idx=" << val << " params=" << params;
      occurrence.insert(val);
      for (int64_t k = 0; k < params.dim; k++) {
        ASSERT_TRUE(raft::match(out_h(i, k), val * 1000 + k, raft::CompareApprox<T>(1e-6)));
      }
    }
  }

 protected:
  inputs params;
  raft::resources res;
  cudaStream_t stream;
  random::RngState state;
  device_matrix<T, int64_t> in, out;
  host_matrix<T, int64_t> in_h, out_h;
};

inline std::vector<inputs> generate_inputs()
{
  std::vector<inputs> input1 =
    raft::util::itertools::product<inputs>({10}, {1, 17, 96}, {1, 6, 9, 10}, {false});

  std::vector<inputs> input2 =
    raft::util::itertools::product<inputs>({137}, {1, 17, 128}, {1, 10, 100, 137}, {false});
  input1.insert(input1.end(), input2.begin(), input2.end());

  input2 = raft::util::itertools::product<inputs>(
    {100000}, {1, 42}, {1, 137, 1000, 10000, 50000, 62000, 100000}, {false});

  input1.insert(input1.end(), input2.begin(), input2.end());

  int n = input1.size();
  // Add same tests for host data
  for (int i = 0; i < n; i++) {
    inputs x = input1[i];
    x.host   = true;
    input1.push_back(x);
  }
  return input1;
}

const std::vector<inputs> inputs1 = generate_inputs();

using SampleRowsTestInt64 = SampleRowsTest<float>;
TEST_P(SampleRowsTestInt64, SamplingTest) { check(); }
INSTANTIATE_TEST_SUITE_P(SampleRowsTests, SampleRowsTestInt64, ::testing::ValuesIn(inputs1));

}  // namespace matrix
}  // namespace raft
