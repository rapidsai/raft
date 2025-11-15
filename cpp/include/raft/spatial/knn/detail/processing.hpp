/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace raft {
namespace spatial {
namespace knn {

/**
 * @brief A virtual class defining pre- and post-processing
 * for metrics. This class will temporarily modify its given
 * state in `preprocess()` and undo those modifications in
 * `postprocess()`
 */

template <typename math_t>
class MetricProcessor {
 public:
  virtual void preprocess(math_t* data) {}

  virtual void revert(math_t* data) {}

  virtual void postprocess(math_t* data) {}

  virtual void set_num_queries(int k) {}

  virtual ~MetricProcessor() = default;
};

}  // namespace knn
}  // namespace spatial
}  // namespace raft
