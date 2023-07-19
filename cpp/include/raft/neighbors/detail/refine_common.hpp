/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <raft/core/mdspan.hpp>
#include <raft/distance/distance_types.hpp>

namespace raft::neighbors::detail {

/** Checks whether the input data extents are compatible. */
template <typename ExtentsT>
void refine_check_input(ExtentsT dataset,
                        ExtentsT queries,
                        ExtentsT candidates,
                        ExtentsT indices,
                        ExtentsT distances,
                        distance::DistanceType metric)
{
  auto n_queries = queries.extent(0);
  auto k         = distances.extent(1);

  RAFT_EXPECTS(indices.extent(0) == n_queries && distances.extent(0) == n_queries &&
                 candidates.extent(0) == n_queries,
               "Number of rows in output indices, distances and candidates matrices must be equal"
               " with the number of rows in search matrix. Expected %d, got %d, %d, and %d",
               static_cast<int>(n_queries),
               static_cast<int>(indices.extent(0)),
               static_cast<int>(distances.extent(0)),
               static_cast<int>(candidates.extent(0)));

  RAFT_EXPECTS(indices.extent(1) == k,
               "Number of columns in output indices and distances matrices must be equal to k");

  RAFT_EXPECTS(queries.extent(1) == dataset.extent(1),
               "Number of columns must be equal for dataset and queries");

  RAFT_EXPECTS(candidates.extent(1) >= k,
               "Number of neighbor candidates must not be smaller than k (%d vs %d)",
               static_cast<int>(candidates.extent(1)),
               static_cast<int>(k));
}

}  // namespace raft::neighbors::detail
