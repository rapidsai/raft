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

#pragma once

#include <raft/distance/distance_types.hpp>
#include <raft/matrix/detail/select_k.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/spatial/knn/detail/ann_utils.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <raft_internal/neighbors/naive_knn.cuh>

#include "../test_utils.cuh"
#include <gtest/gtest.h>
#include <iostream>

namespace raft::neighbors {

struct print_dtype {
  cudaDataType_t value;
};

inline auto operator<<(std::ostream& os, const print_dtype& p) -> std::ostream&
{
  switch (p.value) {
    case CUDA_R_16F: os << "CUDA_R_16F"; break;
    case CUDA_C_16F: os << "CUDA_C_16F"; break;
    case CUDA_R_16BF: os << "CUDA_R_16BF"; break;
    case CUDA_C_16BF: os << "CUDA_C_16BF"; break;
    case CUDA_R_32F: os << "CUDA_R_32F"; break;
    case CUDA_C_32F: os << "CUDA_C_32F"; break;
    case CUDA_R_64F: os << "CUDA_R_64F"; break;
    case CUDA_C_64F: os << "CUDA_C_64F"; break;
    case CUDA_R_4I: os << "CUDA_R_4I"; break;
    case CUDA_C_4I: os << "CUDA_C_4I"; break;
    case CUDA_R_4U: os << "CUDA_R_4U"; break;
    case CUDA_C_4U: os << "CUDA_C_4U"; break;
    case CUDA_R_8I: os << "CUDA_R_8I"; break;
    case CUDA_C_8I: os << "CUDA_C_8I"; break;
    case CUDA_R_8U: os << "CUDA_R_8U"; break;
    case CUDA_C_8U: os << "CUDA_C_8U"; break;
    case CUDA_R_16I: os << "CUDA_R_16I"; break;
    case CUDA_C_16I: os << "CUDA_C_16I"; break;
    case CUDA_R_16U: os << "CUDA_R_16U"; break;
    case CUDA_C_16U: os << "CUDA_C_16U"; break;
    case CUDA_R_32I: os << "CUDA_R_32I"; break;
    case CUDA_C_32I: os << "CUDA_C_32I"; break;
    case CUDA_R_32U: os << "CUDA_R_32U"; break;
    case CUDA_C_32U: os << "CUDA_C_32U"; break;
    case CUDA_R_64I: os << "CUDA_R_64I"; break;
    case CUDA_C_64I: os << "CUDA_C_64I"; break;
    case CUDA_R_64U: os << "CUDA_R_64U"; break;
    case CUDA_C_64U: os << "CUDA_C_64U"; break;
    default: RAFT_FAIL("unreachable code");
  }
  return os;
}

struct print_metric {
  raft::distance::DistanceType value;
};

inline auto operator<<(std::ostream& os, const print_metric& p) -> std::ostream&
{
  switch (p.value) {
    case raft::distance::L2Expanded: os << "distance::L2Expanded"; break;
    case raft::distance::L2SqrtExpanded: os << "distance::L2SqrtExpanded"; break;
    case raft::distance::CosineExpanded: os << "distance::CosineExpanded"; break;
    case raft::distance::L1: os << "distance::L1"; break;
    case raft::distance::L2Unexpanded: os << "distance::L2Unexpanded"; break;
    case raft::distance::L2SqrtUnexpanded: os << "distance::L2SqrtUnexpanded"; break;
    case raft::distance::InnerProduct: os << "distance::InnerProduct"; break;
    case raft::distance::Linf: os << "distance::Linf"; break;
    case raft::distance::Canberra: os << "distance::Canberra"; break;
    case raft::distance::LpUnexpanded: os << "distance::LpUnexpanded"; break;
    case raft::distance::CorrelationExpanded: os << "distance::CorrelationExpanded"; break;
    case raft::distance::JaccardExpanded: os << "distance::JaccardExpanded"; break;
    case raft::distance::HellingerExpanded: os << "distance::HellingerExpanded"; break;
    case raft::distance::Haversine: os << "distance::Haversine"; break;
    case raft::distance::BrayCurtis: os << "distance::BrayCurtis"; break;
    case raft::distance::JensenShannon: os << "distance::JensenShannon"; break;
    case raft::distance::HammingUnexpanded: os << "distance::HammingUnexpanded"; break;
    case raft::distance::KLDivergence: os << "distance::KLDivergence"; break;
    case raft::distance::RusselRaoExpanded: os << "distance::RusselRaoExpanded"; break;
    case raft::distance::DiceExpanded: os << "distance::DiceExpanded"; break;
    case raft::distance::Precomputed: os << "distance::Precomputed"; break;
    default: RAFT_FAIL("unreachable code");
  }
  return os;
}

template <typename IdxT, typename DistT, typename CompareDist>
struct idx_dist_pair {
  IdxT idx;
  DistT dist;
  CompareDist eq_compare;
  auto operator==(const idx_dist_pair<IdxT, DistT, CompareDist>& a) const -> bool
  {
    if (idx == a.idx) return true;
    if (eq_compare(dist, a.dist)) return true;
    return false;
  }
  idx_dist_pair(IdxT x, DistT y, CompareDist op) : idx(x), dist(y), eq_compare(op) {}
};

template <typename T, typename DistT>
auto eval_neighbours(const std::vector<T>& expected_idx,
                     const std::vector<T>& actual_idx,
                     const std::vector<DistT>& expected_dist,
                     const std::vector<DistT>& actual_dist,
                     size_t rows,
                     size_t cols,
                     double eps,
                     double min_recall) -> testing::AssertionResult
{
  size_t match_count = 0;
  size_t total_count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t k = 0; k < cols; ++k) {
      size_t idx_k  = i * cols + k;  // row major assumption!
      auto act_idx  = actual_idx[idx_k];
      auto act_dist = actual_dist[idx_k];
      for (size_t j = 0; j < cols; ++j) {
        size_t idx    = i * cols + j;  // row major assumption!
        auto exp_idx  = expected_idx[idx];
        auto exp_dist = expected_dist[idx];
        idx_dist_pair exp_kvp(exp_idx, exp_dist, raft::CompareApprox<DistT>(eps));
        idx_dist_pair act_kvp(act_idx, act_dist, raft::CompareApprox<DistT>(eps));
        if (exp_kvp == act_kvp) {
          match_count++;
          break;
        }
      }
    }
  }
  double actual_recall = static_cast<double>(match_count) / static_cast<double>(total_count);
  double error_margin  = (actual_recall - min_recall) / std::max(1.0 - min_recall, eps);
  RAFT_LOG_INFO("Recall = %f (%zu/%zu), the error is %2.1f%% %s the threshold (eps = %f).",
                actual_recall,
                match_count,
                total_count,
                std::abs(error_margin * 100.0),
                error_margin < 0 ? "above" : "below",
                eps);
  if (actual_recall < min_recall - eps) {
    return testing::AssertionFailure()
           << "actual recall (" << actual_recall << ") is lower than the minimum expected recall ("
           << min_recall << "); eps = " << eps << ". ";
  }
  return testing::AssertionSuccess();
}

template <typename T, typename DistT, typename IdxT>
auto eval_distances(raft::device_resources const& handle,
                    const T* x,              // dataset, n_rows * n_cols
                    const T* queries,        // n_queries * n_cols
                    const IdxT* neighbors,   // n_queries * k
                    const DistT* distances,  // n_queries *k
                    size_t n_rows,
                    size_t n_cols,
                    size_t n_queries,
                    uint32_t k,
                    raft::distance::DistanceType metric,
                    double eps) -> testing::AssertionResult
{
  // for each vector, we calculate the actual distance to the k neighbors

  for (size_t i = 0; i < n_queries; i++) {
    auto y          = raft::make_device_matrix<T, IdxT>(handle, k, n_cols);
    auto naive_dist = raft::make_device_matrix<DistT, IdxT>(handle, 1, k);

    raft::matrix::copyRows<T, IdxT, int64_t>(
      x, k, n_cols, y.data_handle(), neighbors + i * k, k, handle.get_stream(), true);

    dim3 block_dim(16, 32, 1);
    auto grid_y =
      static_cast<uint16_t>(std::min<size_t>(raft::ceildiv<size_t>(k, block_dim.y), 32768));
    dim3 grid_dim(raft::ceildiv<size_t>(n_rows, block_dim.x), grid_y, 1);

    naive_distance_kernel<DistT, T, IdxT><<<grid_dim, block_dim, 0, handle.get_stream()>>>(
      naive_dist.data_handle(), queries + i * n_cols, y.data_handle(), 1, k, n_cols, metric);

    if (!devArrMatch(distances + i * k,
                     naive_dist.data_handle(),
                     naive_dist.size(),
                     CompareApprox<float>(eps))) {
      std::cout << n_rows << "x" << n_cols << ", " << k << std::endl;
      std::cout << "query " << i << std::endl;
      print_vector(" indices", neighbors + i * k, k, std::cout);
      print_vector("n dist", distances + i * k, k, std::cout);
      print_vector("c dist", naive_dist.data_handle(), naive_dist.size(), std::cout);

      return testing::AssertionFailure();
    }
  }
  return testing::AssertionSuccess();
}
}  // namespace raft::neighbors
