/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "ann_types.hpp"

#include <raft/distance/distance_types.hpp>

#include <sys/types.h>

#include <cstdint>
#include <memory>
#include <type_traits>

namespace raft::neighbors::hnsw {

/**
 * @defgroup hnsw Build CAGRA index and search with hnswlib
 * @{
 */

struct search_params : ann::search_params {
  int ef;               // size of the candidate list
  int num_threads = 0;  // number of host threads to use for concurrent searches. Value of 0
                        // automatically maximizes parallelism
};

template <typename T>
struct index : ann::index {
 public:
  /**
   * @brief load a base-layer-only hnswlib index originally saved from a built CAGRA index.
   *  This is a virtual class and it cannot be used directly. To create an index, use the factory
   *  function `raft::neighbors::hnsw::from_cagra` from the header
   *  `raft/neighbors/hnsw.hpp`
   *
   * @param[in] dim dimensions of the training dataset
   * @param[in] metric distance metric to search. Supported metrics ("L2Expanded", "InnerProduct")
   */
  index(int dim, raft::distance::DistanceType metric) : dim_{dim}, metric_{metric} {}

  /**
  @brief Get underlying index
  */
  virtual auto get_index() const -> void const* = 0;

  auto dim() const -> int const { return dim_; }

  auto metric() const -> raft::distance::DistanceType { return metric_; }

  /**
  @brief Set ef for search
  */
  virtual void set_ef(int ef) const;

 private:
  int dim_;
  raft::distance::DistanceType metric_;
};

/**@}*/

}  // namespace raft::neighbors::hnsw
