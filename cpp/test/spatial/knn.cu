/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <iostream>
#include <vector>
#include <raft/spatial/knn/knn.hpp>

namespace raft {

  struct KNNInputs {
    std::vector<std::vector<float>> input;
    int k;
  };

template <typename T>
class KNNTest : public ::testing::TestWithParam<KNNInputs> {
 protected:
  void testBruteForce() {

		brute_force_knn(handle_,
										input_,
                    sizes_,
                    cols_,
                    search_data_,
                    rows_,
                    indices_,
                    distances_,
                    k_,
                    true,
                    true);
  }

  void SetUp() override {
    params = ::testing::TestWithParam<KNNInputs>::GetParam();
    rows_ = params.input.size();
    cols_ = params.input[0].size();
    k_ = params.k;

    float *input_d = rmm::device_buffer(params.input.data(),
                                        params.input.size() * sizeof(float));

    input_.push_back(input_d);
    sizes_.push_back(rows_);

    raft::allocate(search_data_, row_ * cols_, true);
    raft::allocate(indices_,
                   rows_ * cols_,
                   true);
    raft::allocate(distances_,
                   rows_ * cols_,
                   true);
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(search_data));
    CUDA_CHECK(cudaFree(search_labels));
    CUDA_CHECK(cudaFree(output_dists));
    CUDA_CHECK(cudaFree(output_indices));
    CUDA_CHECK(cudaFree(actual_labels));
    CUDA_CHECK(cudaFree(expected_labels));
  }

 private:
  raft::handle_t handle_;
  KNNInputs params_;
  int rows_;
  int cols_;
  std::vector<float *> input_;
  std::vector<int> sizes_;
  float *search_data_;
  int64_t indices_;
  float* distances_;
  int k_;
};


const std::vector<KNNInputs> inputs = {
  // 2D
  {
    {
      { 7.89611  ,  -6.3093657 },
      { 8.198494 ,  -6.6102095 },
      {-1.067701 ,   0.2757877 },
      { 5.5629272,  -4.0279684 },
      { 8.466168 ,  -6.3818727 },
      { 7.373038 ,  -3.2476108 },
      { 7.3618903,  -6.311329  },
      { 3.5585778,   2.3175476 },
      { 8.722544 ,  -6.184722  },
      { 5.9165254,  -4.0085735 },
      {-2.4502695,   1.8806121 },
      { 1.250205 ,   1.6940732 },
      { 7.702861 ,  -5.5382366 },
      {-0.32521492,  1.0503006 },
      { 7.203165 ,  -6.1078873 },
      { 0.7067232,  -0.02844107},
      {-0.6195269,   1.6659582 },
      { 7.3585844,  -6.5425425 },
      { 0.2946735,   0.7920021 },
      { 5.9978905,  -4.235259  }},
    2},
};

typedef KNNTest<float> KNNTestF;
TEST_P(KNNTestF, BruteForce) { this->testBruteForce(); }

INSTANTIATE_TEST_CASE_P(KNNTest, KNNTestF, ::testing::ValuesIn(inputs));

} // namespace raft
