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

#include "../test_utils.h"

#include <raft/linalg/distance_type.h>
#include <raft/spatial/knn/knn.hpp>

#include <rmm/device_buffer.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <iostream>
#include <vector>

namespace raft {
namespace spatial {
namespace knn {
struct KNNInputs {
  std::vector<std::vector<float>> input;
  int k;
  std::vector<int> labels;
};

__global__ void build_actual_output(int *output, int n_rows, int k,
                                    const int *idx_labels,
                                    const int64_t *indices) {
  int element = threadIdx.x + blockDim.x * blockIdx.x;
  if (element >= n_rows * k) return;

  output[element] = idx_labels[indices[element]];
}

__global__ void build_expected_output(int *output, int n_rows, int k,
                                      const int *labels) {
  int row = threadIdx.x + blockDim.x * blockIdx.x;
  if (row >= n_rows) return;

  int cur_label = labels[row];
  for (int i = 0; i < k; i++) {
    output[row * k + i] = cur_label;
  }
}

template <typename T>
class KNNTest : public ::testing::TestWithParam<KNNInputs> {
 protected:
  void testBruteForce() {
    raft::print_device_vector("Input array: ", input_, rows_ * cols_,
                              std::cout);
    std::cout << "K: " << k_ << "\n";
    raft::print_device_vector("Labels array: ", search_labels_, rows_,
                              std::cout);

    auto stream = handle_.get_stream();

    raft::allocate(actual_labels_, rows_ * k_, stream, true);
    raft::allocate(expected_labels_, rows_ * k_, stream, true);

    std::vector<float *> input_vec;
    std::vector<int> sizes_vec;
    input_vec.push_back(input_);
    sizes_vec.push_back(rows_);

    brute_force_knn(handle_, input_vec, sizes_vec, cols_, search_data_, rows_,
                    indices_, distances_, k_, true, true);

    build_actual_output<<<raft::ceildiv(rows_ * k_, 32), 32, 0, stream>>>(
      actual_labels_, rows_, k_, search_labels_, indices_);

    build_expected_output<<<raft::ceildiv(rows_ * k_, 32), 32, 0, stream>>>(
      expected_labels_, rows_, k_, search_labels_);

    ASSERT_TRUE(devArrMatch(expected_labels_, actual_labels_, rows_ * k_,
                            raft::Compare<int>()));
  }

  void SetUp() override {
    params_ = ::testing::TestWithParam<KNNInputs>::GetParam();
    rows_ = params_.input.size();
    cols_ = params_.input[0].size();
    k_ = params_.k;

    cudaStream_t stream = handle_.get_stream();

    std::vector<float> row_major_input;
    for (std::size_t i = 0; i < params_.input.size(); ++i) {
      for (std::size_t j = 0; j < params_.input[i].size(); ++j) {
        row_major_input.push_back(params_.input[i][j]);
      }
    }
    rmm::device_buffer input_d = rmm::device_buffer(
      row_major_input.data(), row_major_input.size() * sizeof(float), stream);
    float *input_ptr = static_cast<float *>(input_d.data());

    rmm::device_buffer labels_d = rmm::device_buffer(
      params_.labels.data(), params_.labels.size() * sizeof(int), stream);
    int *labels_ptr = static_cast<int *>(labels_d.data());

    raft::allocate(input_, rows_ * cols_, stream, true);
    raft::allocate(search_data_, rows_ * cols_, stream, true);
    raft::allocate(indices_, rows_ * k_, stream, true);
    raft::allocate(distances_, rows_ * k_, stream, true);
    raft::allocate(search_labels_, rows_, stream, true);

    raft::copy(input_, input_ptr, rows_ * cols_, stream);
    raft::copy(search_data_, input_ptr, rows_ * cols_, stream);
    raft::copy(search_labels_, labels_ptr, rows_, stream);
  }

  void TearDown() override {
    cudaStream_t stream = handle_.get_stream();
    raft::deallocate_all(stream);
  }

 private:
  raft::handle_t handle_;
  KNNInputs params_;
  int rows_;
  int cols_;
  float *input_;
  float *search_data_;
  int64_t *indices_;
  float *distances_;
  int k_;

  int *search_labels_;
  int *actual_labels_;
  int *expected_labels_;
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

typedef KNNTest<float> KNNTestF;
TEST_P(KNNTestF, BruteForce) { this->testBruteForce(); }

INSTANTIATE_TEST_CASE_P(KNNTest, KNNTestF, ::testing::ValuesIn(inputs));

}  // namespace knn
}  // namespace spatial
}  // namespace raft
