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
namespace distance {

/** enum to tell how to compute euclidean distance */
enum DistanceType : unsigned short {
  /** evaluate as dist_ij = sum(x_ik^2) + sum(y_ij)^2 - 2*sum(x_ik * y_jk) */
  EucExpandedL2 = 0,
  /** same as above, but inside the epilogue, perform square root operation */
  EucExpandedL2Sqrt = 1,
  /** cosine distance */
  EucExpandedCosine = 2,
  /** L1 distance */
  EucUnexpandedL1 = 3,
  /** evaluate as dist_ij += (x_ik - y-jk)^2 */
  EucUnexpandedL2 = 4,
  /** same as above, but inside the epilogue, perform square root operation */
  EucUnexpandedL2Sqrt = 5,
  /** basic inner product **/
  InnerProduct = 6
};

};  // namespace distance
};  // end namespace raft
