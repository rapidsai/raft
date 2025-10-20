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

#include "../hnsw_types.hpp"

#include <raft/core/error.hpp>
#include <raft/distance/distance_types.hpp>

#include <hnswlib/hnswlib.h>
#include <sys/types.h>

#include <cstdint>
#include <memory>
#include <type_traits>

namespace raft::neighbors::hnsw::detail {

/**
 * @addtogroup cagra_hnswlib Build CAGRA index and search with hnswlib
 * @{
 */

template <typename T>
struct hnsw_dist_t {
  using type = void;
};

template <>
struct hnsw_dist_t<float> {
  using type = float;
};

template <>
struct hnsw_dist_t<std::uint8_t> {
  using type = int;
};

template <>
struct hnsw_dist_t<std::int8_t> {
  using type = int;
};

template <typename T>
struct index_impl : index<T> {
 public:
  /**
   * @brief load a base-layer-only hnswlib index originally saved from a built CAGRA index
   *
   * @param[in] filepath path to the index
   * @param[in] dim dimensions of the training dataset
   * @param[in] metric distance metric to search. Supported metrics ("L2Expanded", "InnerProduct")
   */
  index_impl(std::string filepath, int dim, raft::distance::DistanceType metric)
    : index<T>{dim, metric}
  {
    if constexpr (std::is_same_v<T, float>) {
      if (metric == raft::distance::L2Expanded) {
        space_ = std::make_unique<hnswlib::L2Space>(dim);
      } else if (metric == raft::distance::InnerProduct) {
        space_ = std::make_unique<hnswlib::InnerProductSpace>(dim);
      }
    } else if constexpr (std::is_same_v<T, std::int8_t> or std::is_same_v<T, std::uint8_t>) {
      if (metric == raft::distance::L2Expanded) {
        space_ = std::make_unique<hnswlib::L2SpaceI<T>>(dim);
      }
    }

    RAFT_EXPECTS(space_ != nullptr, "Unsupported metric type was used");

    appr_alg_ = std::make_unique<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>>(
      space_.get(), filepath);

    appr_alg_->base_layer_only = true;
  }

  /**
  @brief Get hnswlib index
  */
  auto get_index() const -> void const* override { return appr_alg_.get(); }

  /**
  @brief Set ef for search
  */
  void set_ef(int ef) const override { appr_alg_->ef_ = ef; }

 private:
  std::unique_ptr<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>> appr_alg_;
  std::unique_ptr<hnswlib::SpaceInterface<typename hnsw_dist_t<T>::type>> space_;
};

/**@}*/

}  // namespace raft::neighbors::hnsw::detail
