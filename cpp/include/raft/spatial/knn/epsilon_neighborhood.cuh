/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
/**
 * This file is deprecated and will be removed in release 22.06.
 * Please use the cuh version instead.
 */

/**
 * DISCLAIMER: this file is deprecated: use epsilon_neighborhood.cuh instead
 */

#pragma once

#pragma message(__FILE__                                                    \
                  " is deprecated and will be removed in a future release." \
                  " Please use the raft::neighbors version instead.")

#include <raft/neighbors/epsilon_neighborhood.cuh>

namespace raft::spatial::knn {

using raft::neighbors::epsilon_neighborhood::eps_neighbors_l2sq;
using raft::neighbors::epsilon_neighborhood::epsUnexpL2SqNeighborhood;

}  // namespace raft::spatial::knn
