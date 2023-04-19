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

#ifndef __SPARSE_DIST_H
#define __SPARSE_DIST_H

#pragma once

#include "detail/common.hpp"
#include <unordered_set>

#include <raft/core/device_csr_matrix.hpp>

#include <raft/distance/distance_types.hpp>

#include <raft/sparse/distance/detail/bin_distance.cuh>
#include <raft/sparse/distance/detail/ip_distance.cuh>
#include <raft/sparse/distance/detail/l2_distance.cuh>
#include <raft/sparse/distance/detail/lp_distance.cuh>

namespace raft {
namespace sparse {
namespace distance {

static const std::unordered_set<raft::distance::DistanceType> supportedDistance{
  raft::distance::DistanceType::L2Expanded,
  raft::distance::DistanceType::L2Unexpanded,
  raft::distance::DistanceType::L2SqrtExpanded,
  raft::distance::DistanceType::L2SqrtUnexpanded,
  raft::distance::DistanceType::InnerProduct,
  raft::distance::DistanceType::L1,
  raft::distance::DistanceType::Canberra,
  raft::distance::DistanceType::Linf,
  raft::distance::DistanceType::LpUnexpanded,
  raft::distance::DistanceType::JaccardExpanded,
  raft::distance::DistanceType::CosineExpanded,
  raft::distance::DistanceType::HellingerExpanded,
  raft::distance::DistanceType::DiceExpanded,
  raft::distance::DistanceType::CorrelationExpanded,
  raft::distance::DistanceType::RusselRaoExpanded,
  raft::distance::DistanceType::HammingUnexpanded,
  raft::distance::DistanceType::JensenShannon,
  raft::distance::DistanceType::KLDivergence};

/**
 * Compute pairwise distances between A and B, using the provided
 * input configuration and distance function.
 *
 * @tparam value_idx index type
 * @tparam value_t value type
 * @param[out] out dense output array (size A.nrows * B.nrows)
 * @param[in] input_config input argument configuration
 * @param[in] metric distance metric to use
 * @param[in] metric_arg metric argument (used for Minkowski distance)
 */
template <typename value_idx = int, typename value_t = float>
void pairwiseDistance(value_t* out,
                      detail::distances_config_t<value_idx, value_t> input_config,
                      raft::distance::DistanceType metric,
                      float metric_arg)
{
  switch (metric) {
    case raft::distance::DistanceType::L2Expanded:
      detail::l2_expanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case raft::distance::DistanceType::L2SqrtExpanded:
      detail::l2_sqrt_expanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case raft::distance::DistanceType::InnerProduct:
      detail::ip_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case raft::distance::DistanceType::L2Unexpanded:
      detail::l2_unexpanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case raft::distance::DistanceType::L2SqrtUnexpanded:
      detail::l2_sqrt_unexpanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case raft::distance::DistanceType::L1:
      detail::l1_unexpanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case raft::distance::DistanceType::LpUnexpanded:
      detail::lp_unexpanded_distances_t<value_idx, value_t>(input_config, metric_arg).compute(out);
      break;
    case raft::distance::DistanceType::Linf:
      detail::linf_unexpanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case raft::distance::DistanceType::Canberra:
      detail::canberra_unexpanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case raft::distance::DistanceType::JaccardExpanded:
      detail::jaccard_expanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case raft::distance::DistanceType::CosineExpanded:
      detail::cosine_expanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case raft::distance::DistanceType::HellingerExpanded:
      detail::hellinger_expanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case raft::distance::DistanceType::DiceExpanded:
      detail::dice_expanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case raft::distance::DistanceType::CorrelationExpanded:
      detail::correlation_expanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case raft::distance::DistanceType::RusselRaoExpanded:
      detail::russelrao_expanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case raft::distance::DistanceType::HammingUnexpanded:
      detail::hamming_unexpanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case raft::distance::DistanceType::JensenShannon:
      detail::jensen_shannon_unexpanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case raft::distance::DistanceType::KLDivergence:
      detail::kl_divergence_unexpanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;

    default: THROW("Unsupported distance: %d", metric);
  }
}

template <typename DeviceCSRMatrix,
          typename ElementType,
          typename IndexType,
          typename = std::enable_if_t<raft::is_device_csr_sparsity_preserving_v<DeviceCSRMatrix>>>
void pairwise_distance(
  raft::device_resources const& handle,
  DeviceCSRMatrix x,
  DeviceCSRMatrix y,
  raft::device_matrix_view<const ElementType, IndexType, raft::layout_c_contiguous> dist,
  raft::distance::DistanceType metric,
  float metric_arg = 2.0f)
{
  RAFT_EXPECTS(x.get_n_cols() == y.get_n_cols(), "Number of columns must be equal");
  RAFT_EXPECTS(dist.extent(0) == x.get_n_rows(),
               "Number of rows in output must be equal to "
               "number of rows in X");
  RAFT_EXPECTS(dist.extent(1) == y.get_n_rows(),
               "Number of columns in output must be equal to "
               "number of rows in Y");

  detail::distances_config_t input_config(handle);
  input_config.a_nrows   = x.get_n_rows();
  input_config.a_ncols   = x.get_n_cols();
  input_config.a_nnz     = x.get_nnz();
  input_config.a_indptr  = x.get_indptr().data();
  input_config.a_indices = x.get_indices().data();
  input_config.a_data    = x.get_elements().data();

  input_config.b_nrows   = y.get_n_rows();
  input_config.b_ncols   = y.get_n_cols();
  input_config.b_nnz     = y.get_nnz();
  input_config.b_indptr  = y.get_indptr().data();
  input_config.b_indices = y.get_indices().data();
  input_config.b_data    = y.get_elements().data();

  pairwiseDistance(dist.data_handle(), input_config, metric, metric_arg);
}

};  // namespace distance
};  // namespace sparse
};  // namespace raft

#endif