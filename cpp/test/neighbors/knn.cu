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

#include "../test_utils.cuh"
#include <raft/core/resource/cuda_stream.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/brute_force.cuh>

#include <rmm/device_buffer.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <iostream>
#include <vector>

namespace raft::neighbors::brute_force {
struct KNNInputs {
  std::vector<std::vector<float>> input;
  int k;
  std::vector<int> labels;
};

template <typename IdxT>
RAFT_KERNEL build_actual_output(
  int* output, int n_rows, int k, const int* idx_labels, const IdxT* indices)
{
  int element = threadIdx.x + blockDim.x * blockIdx.x;
  if (element >= n_rows * k) return;

  output[element] = idx_labels[indices[element]];
}

RAFT_KERNEL build_expected_output(int* output, int n_rows, int k, const int* labels)
{
  int row = threadIdx.x + blockDim.x * blockIdx.x;
  if (row >= n_rows) return;

  int cur_label = labels[row];
  for (int i = 0; i < k; i++) {
    output[row * k + i] = cur_label;
  }
}

template <typename T, typename IdxT>
class KNNTest : public ::testing::TestWithParam<KNNInputs> {
 public:
  KNNTest()
    : params_(::testing::TestWithParam<KNNInputs>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      actual_labels_(0, stream),
      expected_labels_(0, stream),
      input_(0, stream),
      search_data_(0, stream),
      indices_(0, stream),
      distances_(0, stream),
      search_labels_(0, stream)
  {
  }

 protected:
  void testBruteForce()
  {
    // #if (RAFT_ACTIVE_LEVEL >= RAFT_LEVEL_DEBUG)
    raft::print_device_vector("Input array: ", input_.data(), rows_ * cols_, std::cout);
    std::cout << "K: " << k_ << std::endl;
    raft::print_device_vector("Labels array: ", search_labels_.data(), rows_, std::cout);
    // #endif

    std::vector<device_matrix_view<const T, IdxT, row_major>> index = {
      make_device_matrix_view((const T*)(input_.data()), rows_, cols_)};
    auto search = raft::make_device_matrix_view<const T, IdxT, row_major>(
      (const T*)(search_data_.data()), rows_, cols_);

    auto indices = raft::make_device_matrix_view<IdxT, IdxT, row_major>(indices_.data(), rows_, k_);
    auto distances =
      raft::make_device_matrix_view<T, IdxT, row_major>(distances_.data(), rows_, k_);

    auto metric = raft::distance::DistanceType::L2Unexpanded;
    knn(handle, index, search, indices, distances, metric, std::make_optional<IdxT>(0));

    build_actual_output<<<raft::ceildiv(rows_ * k_, 32), 32, 0, stream>>>(
      actual_labels_.data(), rows_, k_, search_labels_.data(), indices_.data());

    build_expected_output<<<raft::ceildiv(rows_ * k_, 32), 32, 0, stream>>>(
      expected_labels_.data(), rows_, k_, search_labels_.data());

    ASSERT_TRUE(devArrMatch(
      expected_labels_.data(), actual_labels_.data(), rows_ * k_, raft::Compare<int>(), stream));
  }

  void SetUp() override
  {
    rows_ = params_.input.size();
    cols_ = params_.input[0].size();
    k_    = params_.k;

    actual_labels_.resize(rows_ * k_, stream);
    expected_labels_.resize(rows_ * k_, stream);
    input_.resize(rows_ * cols_, stream);
    search_data_.resize(rows_ * cols_, stream);
    indices_.resize(rows_ * k_, stream);
    distances_.resize(rows_ * k_, stream);
    search_labels_.resize(rows_, stream);

    RAFT_CUDA_TRY(
      cudaMemsetAsync(actual_labels_.data(), 0, actual_labels_.size() * sizeof(int), stream));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(expected_labels_.data(), 0, expected_labels_.size() * sizeof(int), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(input_.data(), 0, input_.size() * sizeof(float), stream));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(search_data_.data(), 0, search_data_.size() * sizeof(float), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(indices_.data(), 0, indices_.size() * sizeof(IdxT), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(distances_.data(), 0, distances_.size() * sizeof(float), stream));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(search_labels_.data(), 0, search_labels_.size() * sizeof(int), stream));

    std::vector<float> row_major_input;
    for (std::size_t i = 0; i < params_.input.size(); ++i) {
      for (std::size_t j = 0; j < params_.input[i].size(); ++j) {
        row_major_input.push_back(params_.input[i][j]);
      }
    }
    rmm::device_buffer input_d =
      rmm::device_buffer(row_major_input.data(), row_major_input.size() * sizeof(float), stream);
    float* input_ptr = static_cast<float*>(input_d.data());

    rmm::device_buffer labels_d =
      rmm::device_buffer(params_.labels.data(), params_.labels.size() * sizeof(int), stream);
    int* labels_ptr = static_cast<int*>(labels_d.data());

    raft::copy(input_.data(), input_ptr, rows_ * cols_, stream);
    raft::copy(search_data_.data(), input_ptr, rows_ * cols_, stream);
    raft::copy(search_labels_.data(), labels_ptr, rows_, stream);
    resource::sync_stream(handle, stream);
  }

 private:
  raft::resources handle;
  cudaStream_t stream;

  KNNInputs params_;
  int rows_;
  int cols_;
  rmm::device_uvector<float> input_;
  rmm::device_uvector<float> search_data_;
  rmm::device_uvector<IdxT> indices_;
  rmm::device_uvector<float> distances_;
  int k_;

  rmm::device_uvector<int> search_labels_;
  rmm::device_uvector<int> actual_labels_;
  rmm::device_uvector<int> expected_labels_;
};

const std::vector<KNNInputs> inputs = {
  // 2D
  {{
     {2.7810836, 2.550537003},
     {1.465489372, 2.362125076},
     {3.396561688, 4.400293529},
     {1.38807019, 1.850220317},
     {3.06407232, 3.005305973},
     {7.627531214, 2.759262235},
     {5.332441248, 2.088626775},
     {6.922596716, 1.77106367},
     {8.675418651, -0.242068655},
     {7.673756466, 3.508563011},
   },
   2,
   {0, 0, 0, 0, 0, 1, 1, 1, 1, 1}}};

typedef KNNTest<float, int> KNNTestFint32_t;
TEST_P(KNNTestFint32_t, BruteForce) { this->testBruteForce(); }
typedef KNNTest<float, uint32_t> KNNTestFuint32_t;
TEST_P(KNNTestFuint32_t, BruteForce) { this->testBruteForce(); }

INSTANTIATE_TEST_CASE_P(KNNTest, KNNTestFint32_t, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(KNNTest, KNNTestFuint32_t, ::testing::ValuesIn(inputs));

}  // namespace raft::neighbors::brute_force
