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

#pragma once

#include <raft/distance/distance_type.hpp>

namespace raft::spatial::knn {

/** The base for approximate KNN index structures. */
struct index {
};

/** The base for KNN index parameters. */
struct index_params {
  /** Distance type. */
  raft::distance::DistanceType metric = distance::DistanceType::L2Expanded;
  /** The argument used by some distance metrics. */
  float metric_arg = 2.0f;
};

struct search_params {
};

};  // namespace raft::spatial::knn
