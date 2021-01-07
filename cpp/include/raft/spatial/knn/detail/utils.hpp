
/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
namespace knn {

  enum MetricType {
    METRIC_INNER_PRODUCT = 0,
    METRIC_L2,
    METRIC_L1,
    METRIC_Linf,
    METRIC_Lp,

    METRIC_Canberra = 20,
    METRIC_BrayCurtis,
    METRIC_JensenShannon,

    METRIC_Cosine = 100,
    METRIC_Correlation
  };

}  // namespace knn
}  // namespace raft
