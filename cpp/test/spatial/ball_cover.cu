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
#include <raft/cudart_utils.h>
#include <raft/linalg/distance_type.h>
#include <iostream>
#include <raft/spatial/knn/detail/ball_cover.cuh>
#include <raft/spatial/knn/detail/haversine_distance.cuh>
#include <raft/spatial/knn/detail/knn_brute_force_faiss.cuh>
#include <rmm/device_uvector.hpp>
#include <vector>
#include "../test_utils.h"

#include <thrust/transform.h>
#include <rmm/exec_policy.hpp>

namespace raft {
namespace spatial {
namespace knn {

using namespace std;

template <typename value_t>
void write_array_to_csv(value_t *host_arr, int m, int n, std::string filename,
                        int precision = 10) {
  std::ofstream file;
  file.open(filename);
  file.setf(std::iostream::fixed);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      file << setprecision(10) << host_arr[i * n + j];
      if (n - j > 1) file << ", ";
    }
    file << endl;
  }
  file.close();
}

std::vector<std::string> split(std::string str, std::string delim) {
  std::vector<std::string> tokens;
  std::string token;
  auto start = 0U;
  auto end = str.find(delim);
  while (end != std::string::npos) {
    token = str.substr(start, end - start);
    tokens.push_back(token);
    start = end + delim.length();
    end = str.find(delim, start);
  }
  tokens.push_back(str.substr(start, end));

  return tokens;
}

inline std::vector<float> read_csv2(std::string filename,
                                    int lines_to_read = 3000000,
                                    int lines_to_skip = 1,
                                    int dupe_cols = 1) {
  std::vector<float> result;
  std::ifstream myFile(filename);
  if (!myFile.is_open()) throw std::runtime_error("Could not open file");

  std::string line;

  int n_lines = 0;
  if (myFile.good()) {
    while (std::getline(myFile, line) && n_lines < lines_to_read) {
      if (n_lines > lines_to_skip - 1) {
        for (int k = 0; k < dupe_cols; k++) {
          std::vector<std::string> tokens = split(line, ",");
          for (int i = 0; i < tokens.size(); i++) {
            float val = stof(tokens[i]);
            result.push_back(val);
          }
        }
      }
      n_lines++;
    }
  }

  printf("lines read: %d\n", n_lines);
  myFile.close();
  return result;
}

inline std::vector<int64_t> read_csv2_i(std::string filename,
                                        int lines_to_read = 3000000,
                                        int lines_to_skip = 1) {
  std::vector<int64_t> result;
  std::ifstream myFile(filename);
  if (!myFile.is_open()) throw std::runtime_error("Could not open file");

  std::string line;

  int n_lines = 0;
  if (myFile.good()) {
    while (std::getline(myFile, line) && n_lines < lines_to_read) {
      if (n_lines > lines_to_skip - 1) {
        std::vector<std::string> tokens = split(line, ",");
        for (int i = 0; i < tokens.size(); i++) {
          int64_t val = stoi(tokens[i]);
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

template <typename value_idx, typename value_t>
__global__ void count_discrepancies_kernel(value_idx *actual_idx,
                                           value_idx *expected_idx,
                                           value_t *actual, value_t *expected,
                                           int m, int n, int *out,
                                           float thres = 1e-1) {
  int row = blockDim.x * blockIdx.x + threadIdx.x;

  int n_diffs = 0;
  if (row < m) {
    for (int i = 0; i < n; i++) {
      value_t d = actual[row * n + i] - expected[row * n + i];
      bool matches = fabsf(d) <= thres;
      n_diffs += !matches;
      if(!matches)
        printf("Diff in idx=%d, expected=%ld, actual=%ld, dist1=%f, dist2=%f\n",
               row, expected_idx[row*n+i], actual_idx[row*n+i], expected[row*n+i], actual[row*n+i]);
      out[row] = n_diffs;
    }
  }
}

struct is_nonzero {
  __host__ __device__ bool operator()(int &i) { return i > 0; }
};

template <typename value_idx, typename value_t>
int count_discrepancies(value_idx *actual_idx, value_idx *expected_idx,
                        value_t *actual, value_t *expected, int m, int n,
                        int *out, cudaStream_t stream) {
  count_discrepancies_kernel<<<raft::ceildiv(m, 256),
                               256, 0, stream>>>(
    actual_idx, expected_idx, actual, expected, m, n, out);

  auto exec_policy = rmm::exec_policy(stream);

  int result = thrust::count_if(exec_policy, out, out + m, is_nonzero());
  return result;
}

struct ToRadians {
  __device__ __host__ float operator()(float a) {
    return a * (CUDART_PI_F / 180.0);
  }
};

struct HaversineFunc {
  template <typename value_t>
  __device__ __host__ __forceinline__ value_t operator()(const value_t *a,
                                                         const value_t *b,
                                                         const int n_dims) {
    return raft::spatial::knn::detail::compute_haversine(a[0], b[0], a[1],
                                                         b[1]);
  }
};

struct EuclideanFunc {
  template <typename value_t>
  __device__ __host__ __forceinline__ value_t operator()(const value_t *a,
                                                         const value_t *b,
                                                         const int n_dims) {
    value_t sum_sq = 0;
    for (int i = 0; i < n_dims; i++) {
      value_t diff = a[i] - b[i];
      sum_sq += diff * diff;
    }

    return sum_sq;
  }
};

template <typename value_idx, typename value_t>
class BallCoverKNNTest : public ::testing::Test {
 protected:
  void basicTest() {
    raft::handle_t handle;

    auto exec_policy = rmm::exec_policy(handle.get_stream());

    cout << "Reading CSV" << endl;

    int dim_mult = 1;
    d = d * dim_mult;

    // make testdata on host
    std::vector<value_t> h_train_inputs =
//      read_csv2("/share/workspace/reproducers/miguel_haversine_knn/OSM_KNN.csv",
//                500, 1, dim_mult);
      read_csv2("/share/workspace/blobs.csv",500000, 1, dim_mult);

    /**
     * Load reference data from CSV files
     */
    //
    //        std::vector<value_t> h_bfknn_dists = read_csv2("/share/workspace/brute_force_dists.csv",
    //                                                       2000000, 0);
    //        std::vector<value_idx> h_bfknn_inds = read_csv2_i("/share/workspace/brute_force_inds.csv",
    //                                                          2000000, 0);

    ////
    //        raft::copy(d_ref_I.data(), h_bfknn_inds.data(), n * k, handle.get_stream());
    //        raft::copy(d_ref_D.data(), h_bfknn_dists.data(), n * k, handle.get_stream());

    cout << "Done" << endl;

    int n = h_train_inputs.size() / d;

    cout << "n_inputs " << n << endl;

    // Allocate input
    rmm::device_uvector<value_t> d_train_inputs(n * d, handle.get_stream());
    raft::update_device(d_train_inputs.data(), h_train_inputs.data(),
                        n * d, handle.get_stream());

    rmm::device_uvector<value_idx> d_ref_I(n * k, handle.get_stream());
    rmm::device_uvector<value_t> d_ref_D(n * k, handle.get_stream());


    /**
     * FOr haversine, convert degrees to radians
     */
    //    thrust::transform(exec_policy, d_train_inputs.data(),
    //                      d_train_inputs.data() + d_train_inputs.size(),
    //                      d_train_inputs.data(), ToRadians());

    cout << "Calling brute force knn " << endl;

    /**
     * Or execute brute-force knn
     */

    cudaStream_t *int_streams = nullptr;
    std::vector<int64_t> *translations = nullptr;
    auto bfknn_start = curTimeMillis();

    std::vector<float *> input_vec = {d_train_inputs.data()};
    std::vector<int> sizes_vec = {n};

    // Perform bfknn for comparison
    raft::spatial::knn::detail::brute_force_knn_impl<int, int64_t>(
      input_vec, sizes_vec, d, d_train_inputs.data(), n,
      d_ref_I.data(), d_ref_D.data(), k, handle.get_device_allocator(),
      handle.get_stream(), int_streams, 0, true, true, translations,
      raft::distance::DistanceType::L2Expanded);

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
    cout << "Done in: " << curTimeMillis() - bfknn_start << "ms." << endl;

//    raft::print_device_vector("actual inds", d_ref_I.data() + (k * 39077),
//                              k * 100, std::cout);
//    raft::print_device_vector("actual dists", d_ref_D.data() + (k * 39077),
//                              k * 100, std::cout);

    // Allocate predicted arrays
    rmm::device_uvector<value_idx> d_pred_I(n * k, handle.get_stream());
    rmm::device_uvector<value_t> d_pred_D(n * k, handle.get_stream());

    cout << "Calling ball cover" << endl;
    auto rbc_start = curTimeMillis();

    BallCoverIndex<value_idx, value_t> index(
      handle, d_train_inputs.data(), n, d,
      raft::distance::DistanceType::L2Expanded);

    printf("n_landmarks=%d\n", index.n_landmarks);

    float weight = 1.0;
    raft::spatial::knn::detail::rbc_all_knn_query(
      handle, index, k, d_pred_I.data(), d_pred_D.data(),
      EuclideanFunc(), true, weight);

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    cout << "Done in: " << curTimeMillis() - rbc_start << "ms." << endl;

    printf("Done.\n");

    //Diff in idx=326622, expected=326720, actual=326623
//    raft::print_device_vector("inds", d_pred_I.data() + (k * 39077),
//                              k * 100, std::cout);
//    raft::print_device_vector("dists", d_pred_D.data() + (k * 39077),
//                              k * 100, std::cout);
//
//    raft::print_device_vector("landmark 39077", index.get_X() + (index.n * 39077),
//                              index.n, std::cout);

    //
    //

    /**
     * Write brute-force results to CSV files
     */
    //    std::vector<value_t> host_D(d_ref_D.size());
    //    std::vector<value_idx> host_I(d_ref_I.size());
    //
    //    raft::copy(host_D.data(), d_ref_D.data(), d_ref_D.size(), handle.get_stream());
    //    raft::copy(host_I.data(), d_ref_I.data(), d_ref_I.size(), handle.get_stream());
    //
    //    write_array_to_csv(host_D.data(), n, k, "/share/workspace/brute_force_dists.csv");
    //    write_array_to_csv(host_I.data(), n, k, "/share/workspace/brute_force_inds.csv");

    // What we really want are for the distances to match exactly. The
    // indices may or may not match exactly, depending upon the ordering which
    // can be nondeterministic.

    /**
     * Evaluate discrepancies for debugging
     */

    rmm::device_uvector<int> discrepancies(n, handle.get_stream());
    thrust::fill(exec_policy, discrepancies.data(),
                 discrepancies.data() + discrepancies.size(), 0);

    int res = count_discrepancies(d_ref_I.data(), d_pred_I.data(),
                                  d_ref_D.data(), d_pred_D.data(), n, k,
                                  discrepancies.data(), handle.get_stream());

    printf("res=%d\n", res);

    // Print knn indices / dists for discrepancies
    raft::print_device_vector("discrepancies",
                              discrepancies.data(), 16,
                              std::cout);
//    ASSERT_TRUE(res == 0);

    /**
         * Final Assertion
         */
//    ASSERT_TRUE(raft::devArrMatch(d_ref_D.data(), d_pred_D.data(), n * k,
//                                  raft::CompareApprox<float>(1e-7)));
            ASSERT_TRUE(
              raft::devArrMatch(d_ref_I.data(), d_pred_I.data(), n * k,
                        raft::Compare<int>()));
  }

  void SetUp() override {}

  void TearDown() override {}

 protected:
  int d = 300;
  int k = 7;
};

typedef BallCoverKNNTest<int64_t, float> BallCoverKNNTestF;

TEST_F(BallCoverKNNTestF, Fit) { basicTest(); }

}  // namespace knn
}  // namespace spatial
}  // namespace raft
