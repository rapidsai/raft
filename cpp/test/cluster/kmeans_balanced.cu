/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <raft/core/resource/cuda_stream.hpp>
#include <vector>

#include <raft/cluster/kmeans_balanced.cuh>
#include <raft/core/cudart_utils.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/stats/adjusted_rand_index.cuh>
#include <raft/util/cuda_utils.cuh>
#include <rmm/device_uvector.hpp>
#include <thrust/fill.h>

/* This test takes advantage of the fact that make_blobs generates balanced clusters.
 * It doesn't currently test whether the algorithm can make balanced clusters with an imbalanced
 * dataset.
 */

namespace raft {

template <typename MathT, typename IdxT>
struct KmeansBalancedInputs {
  IdxT n_rows;
  IdxT n_cols;
  IdxT n_clusters;
  raft::cluster::kmeans_balanced_params kb_params;
  MathT tol;
};

template <typename MathT, typename IdxT>
::std::ostream& operator<<(::std::ostream& os, const KmeansBalancedInputs<MathT, IdxT>& p)
{
  os << "{ " << p.n_rows << ", " << p.n_cols << ", " << p.n_clusters << ", " << p.kb_params.n_iters
     << static_cast<int>(p.kb_params.metric) << '}' << std::endl;
  return os;
}

template <typename DataT, typename MathT, typename LabelT, typename IdxT, typename MappingOpT>
class KmeansBalancedTest : public ::testing::TestWithParam<KmeansBalancedInputs<MathT, IdxT>> {
 protected:
  KmeansBalancedTest()
    : stream(resource::get_cuda_stream(handle)),
      d_labels(0, stream),
      d_labels_ref(0, stream),
      d_centroids(0, stream)
  {
  }

  void basicTest()
  {
    MappingOpT op{};

    auto p = ::testing::TestWithParam<KmeansBalancedInputs<MathT, IdxT>>::GetParam();

    auto X           = raft::make_device_matrix<DataT, IdxT>(handle, p.n_rows, p.n_cols);
    auto blob_labels = raft::make_device_vector<IdxT, IdxT>(handle, p.n_rows);

    MathT* blobs_ptr;
    rmm::device_uvector<MathT> blobs(0, stream);
    if constexpr (!std::is_same_v<DataT, MathT>) {
      blobs.resize(p.n_rows * p.n_cols, stream);
      blobs_ptr = blobs.data();
    } else {
      blobs_ptr = X.data_handle();
    }

    raft::random::make_blobs<MathT, IdxT>(blobs_ptr,
                                          blob_labels.data_handle(),
                                          p.n_rows,
                                          p.n_cols,
                                          p.n_clusters,
                                          stream,
                                          true,
                                          nullptr,
                                          nullptr,
                                          MathT{0.1},
                                          true,
                                          MathT{-1},
                                          MathT{1},
                                          (uint64_t)1234);

    // Convert blobs dataset to DataT if necessary
    if constexpr (!std::is_same_v<DataT, MathT>) {
      raft::linalg::unaryOp(
        X.data_handle(), blobs.data(), p.n_rows * p.n_cols, op.reverse_op, stream);
    }

    d_labels.resize(p.n_rows, stream);
    d_labels_ref.resize(p.n_rows, stream);
    d_centroids.resize(p.n_clusters * p.n_cols, stream);

    raft::linalg::unaryOp(
      d_labels_ref.data(), blob_labels.data_handle(), p.n_rows, raft::cast_op<LabelT>(), stream);

    auto X_view =
      raft::make_device_matrix_view<const DataT, IdxT>(X.data_handle(), X.extent(0), X.extent(1));
    auto d_centroids_view =
      raft::make_device_matrix_view<MathT, IdxT>(d_centroids.data(), p.n_clusters, p.n_cols);
    auto d_labels_view = raft::make_device_vector_view<LabelT, IdxT>(d_labels.data(), p.n_rows);

    raft::cluster::kmeans_balanced::fit_predict(
      handle, p.kb_params, X_view, d_centroids_view, d_labels_view, op);

    resource::sync_stream(handle, stream);

    score = raft::stats::adjusted_rand_index(
      d_labels_ref.data(), d_labels.data(), p.n_rows, resource::get_cuda_stream(handle));

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
  rmm::device_uvector<MathT> d_centroids;
  double score;
};

template <typename MathT, typename IdxT>
std::vector<KmeansBalancedInputs<MathT, IdxT>> get_kmeans_balanced_inputs()
{
  std::vector<KmeansBalancedInputs<MathT, IdxT>> out;
  KmeansBalancedInputs<MathT, IdxT> p;
  p.kb_params.n_iters = 20;
  p.kb_params.metric  = raft::distance::DistanceType::L2Expanded;
  p.tol               = MathT{0.0001};
  std::vector<std::tuple<size_t, size_t, size_t>> row_cols_k = {{1000, 32, 5},
                                                                {1000, 100, 20},
                                                                {10000, 32, 10},
                                                                {10000, 100, 50},
                                                                {10000, 500, 100},
                                                                {1000000, 128, 10}};
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

/*
 * First set of tests: no conversion
 */

KB_TEST((KmeansBalancedTest<float, float, uint32_t, int, raft::identity_op>),
        KmeansBalancedTestFFU32I32,
        inputsf_i32);
KB_TEST((KmeansBalancedTest<double, double, uint32_t, int, raft::identity_op>),
        KmeansBalancedTestDDU32I32,
        inputsd_i32);
KB_TEST((KmeansBalancedTest<float, float, uint32_t, int64_t, raft::identity_op>),
        KmeansBalancedTestFFU32I64,
        inputsf_i64);
KB_TEST((KmeansBalancedTest<double, double, uint32_t, int64_t, raft::identity_op>),
        KmeansBalancedTestDDU32I64,
        inputsd_i64);
KB_TEST((KmeansBalancedTest<float, float, int, int, raft::identity_op>),
        KmeansBalancedTestFFI32I32,
        inputsf_i32);
KB_TEST((KmeansBalancedTest<float, float, int, int64_t, raft::identity_op>),
        KmeansBalancedTestFFI32I64,
        inputsf_i64);
KB_TEST((KmeansBalancedTest<float, float, int64_t, int, raft::identity_op>),
        KmeansBalancedTestFFI64I32,
        inputsf_i32);
KB_TEST((KmeansBalancedTest<float, float, int64_t, int64_t, raft::identity_op>),
        KmeansBalancedTestFFI64I64,
        inputsf_i64);

/*
 * Second set of tests: integer dataset with conversion
 */

template <typename DataT, typename MathT>
struct i2f_scaler {
  // Note: with a scaling factor of 42, and generating blobs with centers between -1 and 1 with a
  // standard deviation of 0.1, it's statistically very unlikely that we'd overflow
  const raft::compose_op<raft::div_const_op<MathT>, raft::cast_op<MathT>> op{
    raft::div_const_op<MathT>{42}, raft::cast_op<MathT>{}};
  const raft::compose_op<raft::cast_op<DataT>, raft::mul_const_op<MathT>> reverse_op{
    raft::cast_op<DataT>{}, raft::mul_const_op<MathT>{42}};

  RAFT_INLINE_FUNCTION auto operator()(const DataT& x) const { return op(x); };
};

KB_TEST((KmeansBalancedTest<int8_t, float, uint32_t, int, i2f_scaler<int8_t, float>>),
        KmeansBalancedTestFI8U32I32,
        inputsf_i32);
KB_TEST((KmeansBalancedTest<int8_t, double, uint32_t, int, i2f_scaler<int8_t, double>>),
        KmeansBalancedTestDI8U32I32,
        inputsd_i32);

}  // namespace raft
