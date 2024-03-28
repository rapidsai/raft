/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <raft/core/device_csr_matrix.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/sparse/distance/detail/bin_distance.cuh>
#include <raft/sparse/distance/detail/ip_distance.cuh>
#include <raft/sparse/distance/detail/l2_distance.cuh>
#include <raft/sparse/distance/detail/lp_distance.cuh>

#include <unordered_set>

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

/**
 * @defgroup sparse_distance Sparse Pairwise Distance
 * @{
 */

/**
 * @brief Compute pairwise distances between x and y, using the provided
 * input configuration and distance function.
 *
 * @code{.cpp}
 * #include <raft/core/device_resources.hpp>
 * #include <raft/core/device_csr_matrix.hpp>
 * #include <raft/core/device_mdspan.hpp>
 *
 * int x_n_rows = 100000;
 * int y_n_rows = 50000;
 * int n_cols = 10000;
 *
 * raft::device_resources handle;
 * auto x = raft::make_device_csr_matrix<float>(handle, x_n_rows, n_cols);
 * auto y = raft::make_device_csr_matrix<float>(handle, y_n_rows, n_cols);
 *
 * ...
 * // populate data
 * ...
 *
 * auto out = raft::make_device_matrix<float>(handle, x_nrows, y_nrows);
 * auto metric = raft::distance::DistanceType::L2Expanded;
 * raft::sparse::distance::pairwise_distance(handle, x.view(), y.view(), out, metric);
 * @endcode
 *
 * @tparam DeviceCSRMatrix raft::device_csr_matrix or raft::device_csr_matrix_view
 * @tparam ElementType data-type of inputs and output
 * @tparam IndexType data-type for indexing
 *
 * @param[in] handle raft::resources
 * @param[in] x raft::device_csr_matrix_view
 * @param[in] y raft::device_csr_matrix_view
 * @param[out] dist raft::device_matrix_view dense matrix
 * @param[in] metric distance metric to use
 * @param[in] metric_arg metric argument (used for Minkowski distance)
 */
template <typename DeviceCSRMatrix,
          typename ElementType,
          typename IndexType,
          typename = std::enable_if_t<raft::is_device_csr_matrix_view_v<DeviceCSRMatrix>>>
void pairwise_distance(raft::resources const& handle,
                       DeviceCSRMatrix x,
                       DeviceCSRMatrix y,
                       raft::device_matrix_view<ElementType, IndexType, raft::row_major> dist,
                       raft::distance::DistanceType metric,
                       float metric_arg = 2.0f)
{
  auto x_structure = x.structure_view();
  auto y_structure = y.structure_view();

  RAFT_EXPECTS(x_structure.get_n_cols() == y_structure.get_n_cols(),
               "Number of columns must be equal");

  RAFT_EXPECTS(dist.extent(0) == x_structure.get_n_rows(),
               "Number of rows in output must be equal to "
               "number of rows in X");
  RAFT_EXPECTS(dist.extent(1) == y_structure.get_n_rows(),
               "Number of columns in output must be equal to "
               "number of rows in Y");

  detail::distances_config_t<IndexType, ElementType> input_config(handle);
  input_config.a_nrows   = x_structure.get_n_rows();
  input_config.a_ncols   = x_structure.get_n_cols();
  input_config.a_nnz     = x_structure.get_nnz();
  input_config.a_indptr  = const_cast<IndexType*>(x_structure.get_indptr().data());
  input_config.a_indices = const_cast<IndexType*>(x_structure.get_indices().data());
  input_config.a_data    = const_cast<ElementType*>(x.get_elements().data());

  input_config.b_nrows   = y_structure.get_n_rows();
  input_config.b_ncols   = y_structure.get_n_cols();
  input_config.b_nnz     = y_structure.get_nnz();
  input_config.b_indptr  = const_cast<IndexType*>(y_structure.get_indptr().data());
  input_config.b_indices = const_cast<IndexType*>(y_structure.get_indices().data());
  input_config.b_data    = const_cast<ElementType*>(y.get_elements().data());

  pairwiseDistance(dist.data_handle(), input_config, metric, metric_arg);
}

/** @} */  // end of sparse_distance

};  // namespace distance
};  // namespace sparse
};  // namespace raft

#endif