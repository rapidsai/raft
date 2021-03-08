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

#pragma once

namespace raft {
namespace hierarchy {

enum LinkageDistance { PAIRWISE = 0, KNN_GRAPH = 1 };

/**
 * Simple POCO for consolidating linkage results. This closely
 * mirrors the trained instance variables populated in
 * Scikit-learn's AgglomerativeClustering estimator.
 * @tparam value_idx
 * @tparam value_t
 */
template <typename value_idx, typename value_t>
struct linkage_output {
  value_idx m;
  value_idx n_clusters;

  value_idx n_leaves;
  value_idx n_connected_components;

  value_idx *labels;  // size: m

  value_idx *children;  // size: (m-1, 2)
};

struct linkage_output_int_float : public linkage_output<int, float> {};
struct linkage_output__int64_float : public linkage_output<int64_t, float> {};

};  // namespace hierarchy
};  // namespace raft