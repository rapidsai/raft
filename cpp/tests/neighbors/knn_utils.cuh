/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../test_utils.cuh"

#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <memory>

namespace raft::spatial::knn {
template <typename IdxT, typename DistT, typename compareDist>
struct idx_dist_pair {
  IdxT idx;
  DistT dist;
  compareDist eq_compare;
  bool operator==(const idx_dist_pair<IdxT, DistT, compareDist>& a) const
  {
    if (idx == a.idx) return true;
    if (eq_compare(dist, a.dist)) return true;
    return false;
  }
  idx_dist_pair(IdxT x, DistT y, compareDist op) : idx(x), dist(y), eq_compare(op) {}
};

template <typename T, typename DistT>
testing::AssertionResult devArrMatchKnnPair(const T* expected_idx,
                                            const T* actual_idx,
                                            const DistT* expected_dist,
                                            const DistT* actual_dist,
                                            size_t rows,
                                            size_t cols,
                                            const DistT eps,
                                            cudaStream_t stream = 0,
                                            bool sort_inputs    = false)
{
  size_t size = rows * cols;
  std::unique_ptr<T[]> exp_idx_h(new T[size]);
  std::unique_ptr<T[]> act_idx_h(new T[size]);
  std::unique_ptr<DistT[]> exp_dist_h(new DistT[size]);
  std::unique_ptr<DistT[]> act_dist_h(new DistT[size]);
  raft::update_host<T>(exp_idx_h.get(), expected_idx, size, stream);
  raft::update_host<T>(act_idx_h.get(), actual_idx, size, stream);
  raft::update_host<DistT>(exp_dist_h.get(), expected_dist, size, stream);
  raft::update_host<DistT>(act_dist_h.get(), actual_dist, size, stream);

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  for (size_t i(0); i < rows; ++i) {
    std::vector<std::pair<DistT, T>> actual;
    std::vector<std::pair<DistT, T>> expected;
    for (size_t j(0); j < cols; ++j) {
      auto idx      = i * cols + j;  // row major assumption!
      auto exp_idx  = exp_idx_h.get()[idx];
      auto act_idx  = act_idx_h.get()[idx];
      auto exp_dist = exp_dist_h.get()[idx];
      auto act_dist = act_dist_h.get()[idx];
      actual.push_back(std::make_pair(act_dist, act_idx));
      expected.push_back(std::make_pair(exp_dist, exp_idx));
    }
    if (sort_inputs) {
      // inputs could be unsorted here, sort for comparison
      std::sort(actual.begin(), actual.end());
      std::sort(expected.begin(), expected.end());
    }
    for (size_t j(0); j < cols; ++j) {
      auto act = actual[j];
      auto exp = expected[j];
      idx_dist_pair exp_kvp(exp.second, exp.first, raft::CompareApprox<DistT>(eps));
      idx_dist_pair act_kvp(act.second, act.first, raft::CompareApprox<DistT>(eps));
      if (!(exp_kvp == act_kvp)) {
        return testing::AssertionFailure()
               << "actual=" << act_kvp.idx << "," << act_kvp.dist << "!="
               << "expected" << exp_kvp.idx << "," << exp_kvp.dist << " @" << i << "," << j;
      }
    }
  }
  return testing::AssertionSuccess();
}
}  // namespace raft::spatial::knn
