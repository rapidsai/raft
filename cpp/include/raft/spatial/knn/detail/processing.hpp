/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
