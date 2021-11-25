/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include "../test_utils.h"

#include <raft/sparse/utils.h>
#include <raft/spatial/knn/knn.hpp>

namespace raft {
namespace spatial {
namespace selection {

using namespace raft;
using namespace raft::sparse;

template <typename value_idx, typename value_t>
struct SparseSelectionInputs {
  value_idx n_rows;
  value_idx n_cols;

  std::vector<value_t> dists_h;

  std::vector<value_t> out_dists_ref_h;
  std::vector<value_idx> out_indices_ref_h;

  int k;

  bool select_min;
};

template <typename value_idx, typename value_t>
::std::ostream& operator<<(::std::ostream& os,
                           const SparseSelectionInputs<value_idx, value_t>& dims)
{
  return os;
}

template <typename value_idx, typename value_t>
class SparseSelectionTest
  : public ::testing::TestWithParam<SparseSelectionInputs<value_idx, value_t>> {
 public:
  SparseSelectionTest()
    : params(::testing::TestWithParam<SparseSelectionInputs<value_idx, value_t>>::GetParam()),
      stream(handle.get_stream()),
      dists(0, stream),
      inds(0, stream),
      out_indices_ref(0, stream),
      out_dists_ref(0, stream),
      out_dists(0, stream),
      out_indices(0, stream)
  {
  }

 protected:
  void make_data()
  {
    std::vector<value_t> dists_h = params.dists_h;

    dists.resize(n_rows * n_cols, stream);
    inds.resize(n_rows * n_cols, stream);
    out_dists.resize(n_rows * k, stream);
    out_indices.resize(n_rows * k, stream);

    update_device(dists.data(), dists_h.data(), dists_h.size(), stream);
    iota_fill(inds.data(), n_rows, n_cols, stream);

    std::vector<value_t> out_dists_ref_h     = params.out_dists_ref_h;
    std::vector<value_idx> out_indices_ref_h = params.out_indices_ref_h;
    out_indices_ref.resize(out_indices_ref_h.size(), stream);
    out_dists_ref.resize(out_dists_ref_h.size(), stream);

    update_device(
      out_indices_ref.data(), out_indices_ref_h.data(), out_indices_ref_h.size(), stream);
    update_device(out_dists_ref.data(), out_dists_ref_h.data(), out_dists_ref_h.size(), stream);
  }

  void SetUp() override
  {
    n_rows = params.n_rows;
    n_cols = params.n_cols;
    k      = params.k;

    make_data();

    raft::spatial::knn::select_k(dists.data(),
                                 inds.data(),
                                 n_rows,
                                 n_cols,
                                 out_dists.data(),
                                 out_indices.data(),
                                 params.select_min,
                                 k,
                                 stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void compare()
  {
    ASSERT_TRUE(
      devArrMatch(out_dists_ref.data(), out_dists.data(), n_rows * k, Compare<value_t>()));
    ASSERT_TRUE(
      devArrMatch(out_indices_ref.data(), out_indices.data(), n_rows * k, Compare<value_idx>()));
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;

  int n_rows, n_cols, k;

  // input data
  rmm::device_uvector<value_t> dists;
  rmm::device_uvector<value_idx> inds;

  // output data
  rmm::device_uvector<value_idx> out_indices;
  rmm::device_uvector<value_t> out_dists;

  rmm::device_uvector<value_idx> out_indices_ref;
  rmm::device_uvector<value_t> out_dists_ref;

  SparseSelectionInputs<value_idx, value_t> params;
};

const std::vector<SparseSelectionInputs<int, float>> inputs_i32_f = {
  {5,
   5,
   {5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 5.0,
    1.0, 4.0, 5.0, 3.0, 2.0, 4.0, 1.0, 1.0, 3.0, 2.0, 5.0, 4.0},
   {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0,
    4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0},
   {4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 3, 0, 1, 4, 2, 4, 2, 1, 3, 0, 0, 2, 1, 4, 3},
   5,
   true}};
typedef SparseSelectionTest<int, float> SparseSelectionTestF;
TEST_P(SparseSelectionTestF, Result) { compare(); }
INSTANTIATE_TEST_CASE_P(SparseSelectionTest,
                        SparseSelectionTestF,
                        ::testing::ValuesIn(inputs_i32_f));

};  // end namespace selection
};  // end namespace spatial
};  // end namespace raft
