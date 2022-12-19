/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <gtest/gtest.h>
#include <optional>
#include <vector>

#include <raft/cluster/kmeans_balanced.cuh>
#include <raft/core/cudart_utils.hpp>
#include <raft/core/handle.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/stats/adjusted_rand_index.cuh>
#include <raft/util/cuda_utils.cuh>
#include <rmm/device_uvector.hpp>
#include <thrust/fill.h>

#if defined RAFT_DISTANCE_COMPILED
#include <raft/distance/specializations.cuh>
#endif

/* This test takes advantage of the fact that make_blobs generates balanced clusters.
 * It doesn't currently test whether the algorithm can make balanced clusters with an imbalanced
 * dataset.
 */

namespace raft {

template <typename T, typename IdxT>
struct KmeansBalancedInputs {
  IdxT n_rows;
  IdxT n_cols;
  IdxT n_clusters;
  raft::cluster::KMeansBalancedParams kb_params;
  T tol;
};

template <typename T, typename IdxT>
::std::ostream& operator<<(::std::ostream& os, const KmeansBalancedInputs<T, IdxT>& p)
{
  os << "{ " << p.n_rows << ", " << p.n_cols << ", " << p.n_clusters << ", " << p.kb_params.n_iters
     << static_cast<int>(p.kb_params.metric) << '}' << std::endl;
  return os;
}

template <typename T, typename LabelT, typename IdxT>
class KmeansBalancedTest : public ::testing::TestWithParam<KmeansBalancedInputs<T, IdxT>> {
 protected:
  KmeansBalancedTest()
    : stream(handle.get_stream()),
      d_labels(0, stream),
      d_labels_ref(0, stream),
      d_centroids(0, stream)
  {
  }

  void basicTest()
  {
    auto p = ::testing::TestWithParam<KmeansBalancedInputs<T, IdxT>>::GetParam();

    auto X           = raft::make_device_matrix<T, IdxT>(handle, p.n_rows, p.n_cols);
    auto blob_labels = raft::make_device_vector<IdxT, IdxT>(handle, p.n_rows);

    raft::random::make_blobs<T, IdxT>(X.data_handle(),
                                      blob_labels.data_handle(),
                                      p.n_rows,
                                      p.n_cols,
                                      p.n_clusters,
                                      stream,
                                      true,
                                      nullptr,
                                      nullptr,
                                      T(1.0),
                                      true,
                                      (T)-10.0f,
                                      (T)10.0f,
                                      (uint64_t)1234);

    d_labels.resize(p.n_rows, stream);
    d_labels_ref.resize(p.n_rows, stream);
    d_centroids.resize(p.n_clusters * p.n_cols, stream);

    raft::linalg::unaryOp(
      d_labels_ref.data(), blob_labels.data_handle(), p.n_rows, raft::cast_op<LabelT>(), stream);

    auto X_view =
      raft::make_device_matrix_view<const T, IdxT>(X.data_handle(), X.extent(0), X.extent(1));
    auto d_centroids_view =
      raft::make_device_matrix_view<T, IdxT>(d_centroids.data(), p.n_clusters, p.n_cols);
    auto d_labels_view = raft::make_device_vector_view<LabelT, IdxT>(d_labels.data(), p.n_rows);

    raft::cluster::kmeans_balanced::fit_predict(
      handle, p.kb_params, X_view, d_centroids_view, d_labels_view);

    handle.sync_stream(stream);

    score = raft::stats::adjusted_rand_index(
      d_labels_ref.data(), d_labels.data(), p.n_rows, handle.get_stream());

    if (score < 1.0) {
      std::stringstream ss;
      ss << "Expected: " << raft::arr2Str(d_labels_ref.data(), 25, "d_labels_ref", stream);
      std::cout << (ss.str().c_str()) << '\n';
      ss.str(std::string());
      ss << "Actual: " << raft::arr2Str(d_labels.data(), 25, "d_labels", stream);
      std::cout << (ss.str().c_str()) << '\n';
      std::cout << "Score = " << score << '\n';
    }
  }

  void SetUp() override { basicTest(); }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;
  rmm::device_uvector<LabelT> d_labels;
  rmm::device_uvector<LabelT> d_labels_ref;
  rmm::device_uvector<T> d_centroids;
  double score;
};

template <typename T, typename IdxT>
std::vector<KmeansBalancedInputs<T, IdxT>> get_kmeans_balanced_inputs()
{
  std::vector<KmeansBalancedInputs<T, IdxT>> out;
  KmeansBalancedInputs<T, IdxT> p;
  p.kb_params.n_iters = 20;
  p.kb_params.metric  = raft::distance::DistanceType::L2Expanded;
  p.tol               = T{0.0001};
  std::vector<std::tuple<size_t, size_t, size_t>> row_cols_k = {
    {1000, 32, 5}, {1000, 100, 20}, {10000, 32, 10}, {10000, 100, 50}, {10000, 500, 100}};
  for (auto& rck : row_cols_k) {
    p.n_rows     = static_cast<IdxT>(std::get<0>(rck));
    p.n_cols     = static_cast<IdxT>(std::get<1>(rck));
    p.n_clusters = static_cast<IdxT>(std::get<2>(rck));
    out.push_back(p);
  }
  return out;
}

const auto inputsf_i32 = get_kmeans_balanced_inputs<float, int>();
const auto inputsd_i32 = get_kmeans_balanced_inputs<double, int>();
const auto inputsf_i64 = get_kmeans_balanced_inputs<float, int64_t>();
const auto inputsd_i64 = get_kmeans_balanced_inputs<double, int64_t>();

#define KB_TEST(test_type, test_name, test_inputs)         \
  typedef RAFT_DEPAREN(test_type) test_name;               \
  TEST_P(test_name, Result) { ASSERT_TRUE(score == 1.0); } \
  INSTANTIATE_TEST_CASE_P(KmeansBalancedTests, test_name, ::testing::ValuesIn(test_inputs))

// todo: remove types which don't have specializations?
KB_TEST((KmeansBalancedTest<float, uint32_t, int>), KmeansBalancedTestFI32, inputsf_i32);
KB_TEST((KmeansBalancedTest<double, uint32_t, int>), KmeansBalancedTestDI32, inputsd_i32);
KB_TEST((KmeansBalancedTest<float, uint32_t, int64_t>), KmeansBalancedTestFI64, inputsf_i64);
KB_TEST((KmeansBalancedTest<double, uint32_t, int64_t>), KmeansBalancedTestDI64, inputsd_i64);

// Unsigned index types unsupported by CUB
// todo: throw error if user attempts to use them
// KB_TEST((KmeansBalancedTest<float, uint32_t, uint32_t>), KmeansBalancedTestFU32, inputsf_u32);
// KB_TEST((KmeansBalancedTest<double, uint32_t, uint32_t>), KmeansBalancedTestDU32, inputsd_u32);

}  // namespace raft
