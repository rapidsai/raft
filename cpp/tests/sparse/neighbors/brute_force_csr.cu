/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/sparse/neighbors/knn.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cusparse_v2.h>
#include <gtest/gtest.h>

namespace raft {
namespace sparse {
namespace selection {

using namespace raft;
using namespace raft::sparse;

template <typename value_idx, typename value_t>
struct SparseKNNInputs {
  value_idx n_cols;

  std::vector<value_idx> indptr_h;
  std::vector<value_idx> indices_h;
  std::vector<value_t> data_h;

  std::vector<value_t> out_dists_ref_h;
  std::vector<value_idx> out_indices_ref_h;

  int k;

  int batch_size_index = 2;
  int batch_size_query = 2;

  raft::distance::DistanceType metric = raft::distance::DistanceType::L2SqrtExpanded;
};

template <typename value_idx, typename value_t>
::std::ostream& operator<<(::std::ostream& os, const SparseKNNInputs<value_idx, value_t>& dims)
{
  return os;
}

template <typename value_idx, typename value_t>
class SparseKNNCSRTest : public ::testing::TestWithParam<SparseKNNInputs<value_idx, value_t>> {
 public:
  SparseKNNCSRTest()
    : params(::testing::TestWithParam<SparseKNNInputs<value_idx, value_t>>::GetParam()),
      indptr(0, resource::get_cuda_stream(handle)),
      indices(0, resource::get_cuda_stream(handle)),
      data(0, resource::get_cuda_stream(handle)),
      out_indices(0, resource::get_cuda_stream(handle)),
      out_dists(0, resource::get_cuda_stream(handle)),
      out_indices_ref(0, resource::get_cuda_stream(handle)),
      out_dists_ref(0, resource::get_cuda_stream(handle))
  {
  }

 protected:
  void SetUp() override
  {
    n_rows               = params.indptr_h.size() - 1;
    nnz                  = params.indices_h.size();
    k                    = params.k;
    auto out_indices_dev = raft::make_device_vector<value_idx, int64_t>(handle, n_rows * k);
    auto out_dists_dev   = raft::make_device_vector<value_t, int64_t>(handle, n_rows * k);

    cudaStream_t stream = raft::resource::get_cuda_stream(handle);

    make_data();
    auto csr_struct_view = raft::make_device_compressed_structure_view(
      indptr.data(), indices.data(), n_rows, params.n_cols, int(data.size()));
    auto c_matrix = raft::make_device_csr_matrix<float, int, int, int>(handle, csr_struct_view);

    raft::update_device<float>(
      c_matrix.view().get_elements().data(), data.data(), data.size(), stream);

    raft::sparse::neighbors::brute_force_knn<int, float>(c_matrix,
                                                         c_matrix,
                                                         out_indices_dev.view(),
                                                         out_dists_dev.view(),
                                                         k,
                                                         handle,
                                                         params.batch_size_index,
                                                         params.batch_size_query,
                                                         params.metric);

    raft::copy(out_indices.data(), out_indices_dev.data_handle(), out_indices_dev.size(), stream);
    raft::copy(out_dists.data(), out_dists_dev.data_handle(), out_dists_dev.size(), stream);
    std::cout << "finished copy" << std::endl;

    RAFT_CUDA_TRY(cudaStreamSynchronize(resource::get_cuda_stream(handle)));
  }

  void compare()
  {
    ASSERT_TRUE(devArrMatch(
      out_dists_ref.data(), out_dists.data(), n_rows * k, CompareApprox<value_t>(1e-4)));
    ASSERT_TRUE(
      devArrMatch(out_indices_ref.data(), out_indices.data(), n_rows * k, Compare<value_idx>()));
  }

 protected:
  void make_data()
  {
    std::vector<value_idx> indptr_h  = params.indptr_h;
    std::vector<value_idx> indices_h = params.indices_h;
    std::vector<value_t> data_h      = params.data_h;

    auto stream = resource::get_cuda_stream(handle);
    indptr.resize(indptr_h.size(), stream);
    indices.resize(indices_h.size(), stream);
    data.resize(data_h.size(), stream);

    update_device(indptr.data(), indptr_h.data(), indptr_h.size(), stream);
    update_device(indices.data(), indices_h.data(), indices_h.size(), stream);
    update_device(data.data(), data_h.data(), data_h.size(), stream);

    std::vector<value_t> out_dists_ref_h     = params.out_dists_ref_h;
    std::vector<value_idx> out_indices_ref_h = params.out_indices_ref_h;

    out_indices_ref.resize(out_indices_ref_h.size(), stream);
    out_dists_ref.resize(out_dists_ref_h.size(), stream);

    update_device(
      out_indices_ref.data(), out_indices_ref_h.data(), out_indices_ref_h.size(), stream);
    update_device(out_dists_ref.data(), out_dists_ref_h.data(), out_dists_ref_h.size(), stream);

    out_dists.resize(n_rows * k, stream);
    out_indices.resize(n_rows * k, stream);
  }

  raft::resources handle;

  int n_rows, nnz, k;

  // input data
  rmm::device_uvector<value_idx> indptr, indices;
  rmm::device_uvector<value_t> data;

  // output data
  rmm::device_uvector<value_idx> out_indices;
  rmm::device_uvector<value_t> out_dists;

  rmm::device_uvector<value_idx> out_indices_ref;
  rmm::device_uvector<value_t> out_dists_ref;

  SparseKNNInputs<value_idx, value_t> params;
};

const std::vector<SparseKNNInputs<int, float>> inputs_i32_f = {
  {9,                                                 // ncols
   {0, 2, 4, 6, 8},                                   // indptr
   {0, 4, 0, 3, 0, 2, 0, 8},                          // indices
   {0.0f, 1.0f, 5.0f, 6.0f, 5.0f, 6.0f, 0.0f, 1.0f},  // data
   {0, 1.41421, 0, 7.87401, 0, 7.87401, 0, 1.41421},  // dists
   {0, 3, 1, 0, 2, 0, 3, 0},                          // inds
   2,
   2,
   2,
   raft::distance::DistanceType::L2SqrtExpanded}};
typedef SparseKNNCSRTest<int, float> SparseKNNCSRTestF;
TEST_P(SparseKNNCSRTestF, Result) { compare(); }
INSTANTIATE_TEST_CASE_P(SparseKNNCSRTest, SparseKNNCSRTestF, ::testing::ValuesIn(inputs_i32_f));

};  // end namespace selection
};  // end namespace sparse
};  // end namespace raft
