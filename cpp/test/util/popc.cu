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
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/popc.cuh>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <queue>
#include <random>
#include <unordered_set>
#include <vector>

namespace raft {
template <typename index_t>
struct PopcInputs {
  index_t n_rows;
  index_t n_cols;
  float sparsity;
  bool owning;
};

template <typename index_t, typename bits_t = uint32_t>
class PopcTest : public ::testing::TestWithParam<PopcInputs<index_t>> {
 public:
  PopcTest()
    : stream(resource::get_cuda_stream(handle)),
      params(::testing::TestWithParam<PopcInputs<index_t>>::GetParam()),
      bits_d(0, stream)
  {
  }

 protected:
  index_t create_bitmap(index_t m, index_t n, float sparsity, std::vector<bits_t>& bitmap)
  {
    index_t total    = static_cast<index_t>(m * n);
    index_t num_ones = static_cast<index_t>((total * 1.0f) * sparsity);
    index_t res      = num_ones;

    for (auto& item : bitmap) {
      item = static_cast<bits_t>(0);
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<index_t> dis(0, total - 1);

    while (num_ones > 0) {
      index_t index = dis(gen);

      bits_t& element      = bitmap[index / (8 * sizeof(bits_t))];
      index_t bit_position = index % (8 * sizeof(bits_t));

      if (((element >> bit_position) & 1) == 0) {
        element |= (static_cast<index_t>(1) << bit_position);
        num_ones--;
      }
    }
    return res;
  }

  void SetUp() override
  {
    index_t element = raft::ceildiv(params.n_rows * params.n_cols, index_t(sizeof(bits_t) * 8));
    std::vector<bits_t> bits_h(element);

    nnz_expected = create_bitmap(params.n_rows, params.n_cols, params.sparsity, bits_h);
    bits_d.resize(bits_h.size(), stream);
    update_device(bits_d.data(), bits_h.data(), bits_h.size(), stream);

    resource::sync_stream(handle);
  }

  void Run()
  {
    auto bits_view =
      raft::make_device_vector_view<const bits_t, index_t>(bits_d.data(), bits_d.size());

    index_t max_len   = params.n_rows * params.n_cols;
    auto max_len_view = raft::make_host_scalar_view<index_t>(&max_len);

    index_t nnz_actual_h = 0;
    rmm::device_scalar<index_t> nnz_actual_d(0, stream);
    auto nnz_actual_view = raft::make_device_scalar_view<index_t>(nnz_actual_d.data());

    raft::popc(handle, bits_view, max_len_view, nnz_actual_view);
    raft::copy(&nnz_actual_h, nnz_actual_d.data(), 1, stream);
    resource::sync_stream(handle);

    ASSERT_EQ(nnz_expected, nnz_actual_h);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  PopcInputs<index_t> params;
  rmm::device_uvector<bits_t> bits_d;
  index_t nnz_expected;
};

using PopcTestI32 = PopcTest<int32_t>;
TEST_P(PopcTestI32, Result) { Run(); }

template <typename index_t>
const std::vector<PopcInputs<index_t>> popc_inputs = {
  {0, 0, 0.2},
  {10, 32, 0.4},
  {10, 3, 0.2},
  {32, 1024, 0.4},
  {1024, 1048576, 0.01},
  {1024, 1024, 0.4},
  {64 * 1024 + 10, 2, 0.3},
  {16, 16, 0.3},
  {17, 16, 0.3},
  {18, 16, 0.3},
  {32 + 9, 33, 0.2},
  {2, 33, 0.2},
  {0, 0, 0.2},
  {10, 32, 0.4},
  {10, 3, 0.2},
  {32, 1024, 0.4},
  {1024, 1048576, 0.01},
  {1024, 1024, 0.4},
  {64 * 1024 + 10, 2, 0.3},
  {16, 16, 0.3},
  {17, 16, 0.3},
  {18, 16, 0.3},
  {32 + 9, 33, 0.2},
  {2, 33, 0.2},
};

INSTANTIATE_TEST_CASE_P(PopcTest, PopcTestI32, ::testing::ValuesIn(popc_inputs<int32_t>));

}  // namespace raft
