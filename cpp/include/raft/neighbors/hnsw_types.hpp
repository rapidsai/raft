/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
