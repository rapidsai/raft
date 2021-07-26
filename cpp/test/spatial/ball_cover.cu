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
#include <raft/linalg/distance_type.h>
#include <iostream>
#include <raft/cudart_utils.h>
#include <raft/spatial/knn/detail/knn_brute_force_faiss.cuh>
#include <raft/spatial/knn/detail/ball_cover.cuh>
#include <rmm/device_uvector.hpp>
#include <vector>
#include "../test_utils.h"

#include <thrust/transform.h>
#include <rmm/exec_policy.hpp>

namespace raft {
namespace spatial {
namespace knn {

using namespace std;

std::vector<std::string> split(std::string str, std::string delim) {

  std::vector<std::string> tokens;
  std::string token;
  auto start = 0U;
  auto end = str.find(delim);
  while(end != std::string::npos) {
    token = str.substr(start, end-start);
    tokens.push_back(token);
    start = end + delim.length();
    end = str.find(delim, start);
  }
  tokens.push_back(str.substr(start, end));

  return tokens;
}

inline std::vector<float> read_csv2(std::string filename, int lines_to_read=200000){
  std::vector<float> result;
  std::ifstream myFile(filename);
  if(!myFile.is_open()) throw std::runtime_error("Could not open file");

  std::string line;

  int n_lines = 0;
  if(myFile.good()) {
    while (std::getline(myFile, line) && n_lines < lines_to_read) {
      if(n_lines > 0) {
          std::vector<std::string> tokens = split(line, ",");
          for(int i = 0; i < tokens.size(); i++) {
            float val = stof(tokens[i]);
            result.push_back(val);
          }
      }
      n_lines++;
    }
  }

  printf("lines read: %d\n", n_lines);
  myFile.close();
  return result;
}

struct ToRadians {

  __device__ __host__ float operator()(float a) {
    return a * (CUDART_PI_F / 180.0);
  }
};

template <typename value_idx, typename value_t>
class BallCoverKNNTest : public ::testing::Test {

 protected:
  void basicTest() {
    raft::handle_t handle;

    auto exec_policy = rmm::exec_policy(handle.get_stream());

    cout << "Reading CSV" << endl;

    // make testdata on host
    std::vector<value_t> h_train_inputs = read_csv2(
      "/share/workspace/reproducers/miguel_haversine_knn/OSM_KNN.csv");

    cout << "Done" << endl;

    int n = h_train_inputs.size() / d;

    cout << "n_inputs " << n << endl;

    // Allocate input
    rmm::device_uvector<value_t> d_train_inputs(n * d, handle.get_stream());
    raft::update_device(d_train_inputs.data(), h_train_inputs.data(), n * d, handle.get_stream());

    rmm::device_uvector<value_idx> d_ref_I(n * k, handle.get_stream());
    rmm::device_uvector<value_t>   d_ref_D(n * k, handle.get_stream());

    std::vector<float *> input_vec = {d_train_inputs.data()};
    std::vector<int> sizes_vec = {n};

    cudaStream_t *int_streams = nullptr;
    std::vector<int64_t> *translations = nullptr;

    thrust::transform(exec_policy, d_train_inputs.data(),
                      d_train_inputs.data()+d_train_inputs.size(),
                      d_train_inputs.data(), ToRadians());

    cout << "Calling brute force knn " << endl;

    auto bfknn_start = curTimeMillis();
    // Perform bfknn for comparison
    raft::spatial::knn::detail::brute_force_knn_impl(
      input_vec, sizes_vec,
      d, d_train_inputs.data(), n,
      d_ref_I.data(), d_ref_D.data(), k,
      handle.get_device_allocator(),
      handle.get_stream(), int_streams, 0,
      true, true, translations,
      raft::distance::DistanceType::Haversine);

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
    cout << "Done in: " << curTimeMillis() - bfknn_start << "ms." << endl;

    // Allocate predicted arrays
    rmm::device_uvector<value_idx> d_pred_I(n * k, handle.get_stream());
    rmm::device_uvector<value_t> d_pred_D(n * k, handle.get_stream());

    cout << "Calling ball cover" << endl;
    auto rbc_start = curTimeMillis();

    raft::spatial::knn::detail::random_ball_cover(handle, d_train_inputs.data(), n, d, k,
                                          d_pred_I.data(), d_pred_D.data());

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    cout << "Done in: " << curTimeMillis() -  rbc_start << "ms." << endl;

    printf("Done.\n");

    raft::print_device_vector("inds", d_pred_I.data(), 4 * k, std::cout);
    raft::print_device_vector("dists", d_pred_D.data(), 4 * k, std::cout);

    raft::print_device_vector("actual inds", d_ref_I.data(), 4 * k, std::cout);
    raft::print_device_vector("actual dists", d_ref_D.data(), 4 * k, std::cout);

    // What we really want are for the distances to match exactly. The
    // indices may or may not match exactly, depending upon the ordering which
    // can be nondeterministic.
    ASSERT_TRUE(raft::devArrMatch(d_ref_D.data(), d_pred_D.data(), n * k,
                                  raft::CompareApprox<float>(1e-4)));
//    ASSERT_TRUE(
//      raft::devArrMatch(d_ref_I.data(), d_pred_I.data(), n * k, raft::Compare<int>()));
  }

  void SetUp() override { }

  void TearDown() override {}

 protected:
  int d = 2;
  int k = 7;
};

typedef BallCoverKNNTest<int64_t, float> BallCoverKNNTestF;

TEST_F(BallCoverKNNTestF, Fit) { basicTest();}

}  // namespace knn
}  // namespace spatial
}  // namespace raft
